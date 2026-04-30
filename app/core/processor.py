"""
End-to-end video interpolation processor.

Wires together: decode -> analyze -> batch -> infer -> encode
with progress tracking, pause/resume, and cancellation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .frame_analysis import (
    AnalysisResult,
    PairType,
    analyze_video,
    build_context_indices,
)
from .interpolator import Interpolator
from .video_io import FrameDecoder, FrameEncoder, probe_video

CONTEXT_RADIUS = 3


@dataclass
class ProcessingProgress:
    """Progress info passed to callback."""
    phase: str                    # 'analyzing', 'processing', 'complete', 'cancelled', 'error'
    pairs_done: int = 0
    pairs_total: int = 0
    inferences_done: int = 0
    inferences_total: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    output_frames_written: int = 0
    fps_rate: float = 0.0
    error_message: str | None = None


ProgressCallback = Callable[[ProcessingProgress], None]


class VideoProcessor:
    """
    Processes video files through the interpolation pipeline.

    Args:
        model_path: Path to the V5 checkpoint.
        device: CUDA device.
        max_height: Max resolution for interpolation (720 default, None for native).
        codec: Output video codec.
        crf: Output quality (0=lossless, 18=high, 28=low).
        fps_multiplier: Frame rate multiplier (2=2x, 3=3x, 4=4x).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = 'cuda',
        max_height: int | None = 720,
        codec: str = 'libx264',
        crf: int = 18,
        fps_multiplier: int = 2,
    ):
        self.interpolator = Interpolator(model_path, device=device)
        self.max_height = max_height
        self.codec = codec
        self.crf = crf
        self.fps_multiplier = fps_multiplier

        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._stopped = False  # True = don't auto-advance to next item

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        precomputed_analysis: 'AnalysisResult | None' = None,
    ) -> bool:
        """
        Process a video file. Accepts optional external cancel/pause events
        so multiple threads can be independently controlled.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Use provided events or fall back to instance-level ones
        cancel = cancel_event or self._cancel
        pause = pause_event or self._pause
        if not cancel_event:
            cancel.clear()
        if not pause_event:
            pause.set()

        def report(progress: ProcessingProgress):
            if progress_callback:
                progress_callback(progress)

        try:
            return self._process_impl(input_path, output_path, report, cancel, pause, precomputed_analysis)
        except Exception as e:
            report(ProcessingProgress(phase='error', error_message=str(e)))
            raise

    def _process_impl(
        self,
        input_path: Path,
        output_path: Path,
        report: Callable[[ProcessingProgress], None],
        cancel: threading.Event | None = None,
        pause: threading.Event | None = None,
        precomputed_analysis: AnalysisResult | None = None,
    ) -> bool:
        cancel = cancel or self._cancel
        pause = pause or self._pause

        max_h = self.max_height if self.max_height and self.max_height > 0 else None

        if precomputed_analysis:
            analysis = precomputed_analysis
            print(f"  Using pre-analyzed results")
        else:
            report(ProcessingProgress(phase='analyzing'))
            analysis = analyze_video(
                str(input_path),
                max_height=max_h,
                progress_callback=lambda done, total: report(
                    ProcessingProgress(phase='analyzing', pairs_done=done, pairs_total=total)
                ),
                cancel_event=cancel,
                pause_event=pause,
            )
        print(f"\n{analysis.summary()}")

        if cancel.is_set():
            report(ProcessingProgress(phase='cancelled'))
            return False

        timesteps = self._get_timesteps()
        inferences_total = analysis.interpolate_count * len(timesteps)
        t_start = time.monotonic()

        with FrameDecoder(str(input_path), max_height=max_h) as decoder:
            batch_size = self.interpolator.find_batch_size(decoder.height, decoder.width)
            output_fps = decoder.fps * self.fps_multiplier

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with FrameEncoder(
                str(output_path),
                fps=output_fps,
                width=decoder.width,
                height=decoder.height,
                audio_source=str(input_path),
                codec=self.codec,
                crf=self.crf,
            ) as encoder:
                completed = self._stream_process(
                    decoder, encoder, analysis, batch_size,
                    timesteps, inferences_total, t_start, report,
                    cancel, pause,
                )
                final_frame_count = encoder.frames_written

        if not completed:
            if output_path.exists():
                output_path.unlink()
            report(ProcessingProgress(phase='cancelled'))
            return False

        elapsed = time.monotonic() - t_start
        report(ProcessingProgress(
            phase='complete',
            pairs_done=analysis.total_pairs,
            pairs_total=analysis.total_pairs,
            inferences_done=inferences_total,
            inferences_total=inferences_total,
            elapsed_seconds=elapsed,
            eta_seconds=0,
            output_frames_written=final_frame_count,
            fps_rate=inferences_total / max(elapsed, 0.1),
        ))
        return True

    def _get_timesteps(self) -> list[float]:
        """Compute intermediate timesteps based on fps_multiplier."""
        m = self.fps_multiplier
        return [i / m for i in range(1, m)]

    def _stream_process(
        self,
        decoder: FrameDecoder,
        encoder: FrameEncoder,
        analysis: AnalysisResult,
        batch_size: int,
        timesteps: list[float],
        inferences_total: int,
        t_start: float,
        report: Callable[[ProcessingProgress], None],
        cancel: threading.Event | None = None,
        pause: threading.Event | None = None,
    ) -> bool:
        cancel = cancel or self._cancel
        pause = pause or self._pause
        frame_cache: dict[int, np.ndarray] = {}
        decoder_iter = iter(decoder)
        frames_decoded = 0

        batch_windows: list[list[np.ndarray]] = []
        batch_pair_indices: list[int] = []

        pairs_done = 0
        inferences_done = 0
        output_frames = 0

        def decode_up_to(target_idx: int):
            nonlocal frames_decoded
            while frames_decoded <= target_idx:
                try:
                    frame = next(decoder_iter)
                    frame_cache[frames_decoded] = frame
                    frames_decoded += 1
                except StopIteration:
                    break

        def evict_before(idx: int):
            for k in list(frame_cache):
                if k < idx:
                    del frame_cache[k]

        def flush_batch():
            """Run model on batch, write frames in correct per-pair order."""
            nonlocal inferences_done, output_frames
            if not batch_windows:
                return

            if len(timesteps) == 1:
                # Single timestep: use standard batch (simpler)
                results = self.interpolator.interpolate_batch(batch_windows, timestep=timesteps[0])
                inferences_done += len(batch_windows)
                for i, pair_idx in enumerate(batch_pair_indices):
                    encoder.write_frame(frame_cache[pair_idx])
                    encoder.write_frame(results[i])
                    output_frames += 2
            else:
                # Multiple timesteps: encode once, decode per timestep (~1.7x faster)
                per_window = self.interpolator.interpolate_multi_timestep(batch_windows, timesteps)
                inferences_done += len(batch_windows) * len(timesteps)
                for i, pair_idx in enumerate(batch_pair_indices):
                    encoder.write_frame(frame_cache[pair_idx])
                    output_frames += 1
                    for frame in per_window[i]:
                        encoder.write_frame(frame)
                        output_frames += 1

            batch_windows.clear()
            batch_pair_indices.clear()

        def write_copy_pair(pair_idx: int):
            """Duplicate/scene_cut: write the same frame multiplier times."""
            nonlocal output_frames
            frame = frame_cache[pair_idx]
            for _ in range(self.fps_multiplier):
                encoder.write_frame(frame)
            output_frames += self.fps_multiplier

        def write_blend_pair(pair_idx: int):
            """Low-motion: CPU linear blend (no GPU). Near-instant."""
            nonlocal output_frames
            frame_a = frame_cache[pair_idx].astype(np.float32)
            frame_b = frame_cache[pair_idx + 1].astype(np.float32)
            encoder.write_frame(frame_cache[pair_idx])  # original
            output_frames += 1
            for t in timesteps:
                blended = ((1.0 - t) * frame_a + t * frame_b + 0.5).astype(np.uint8)
                encoder.write_frame(blended)
                output_frames += 1

        def make_progress() -> ProcessingProgress:
            elapsed = time.monotonic() - t_start
            fps_rate = inferences_done / max(elapsed, 0.1)
            eta = None
            if inferences_done > 0:
                remaining = inferences_total - inferences_done
                eta = remaining / fps_rate
            return ProcessingProgress(
                phase='processing',
                pairs_done=pairs_done,
                pairs_total=analysis.total_pairs,
                inferences_done=inferences_done,
                inferences_total=inferences_total,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                output_frames_written=output_frames,
                fps_rate=fps_rate,
            )

        for pair_idx in range(analysis.total_pairs):
            if cancel.is_set():
                return False
            pause.wait()

            pair = analysis.pairs[pair_idx]
            max_needed = min(pair_idx + CONTEXT_RADIUS + 1, analysis.frame_count - 1)
            decode_up_to(max_needed)

            if pair.pair_type == PairType.INTERPOLATE:
                context = build_context_indices(pair_idx, analysis.frame_count, analysis)
                window = [frame_cache[idx] for idx in context]
                batch_windows.append(window)
                batch_pair_indices.append(pair_idx)

                if len(batch_windows) >= batch_size:
                    flush_batch()
            elif pair.pair_type == PairType.LOW_MOTION:
                flush_batch()
                write_blend_pair(pair_idx)
            else:
                flush_batch()
                write_copy_pair(pair_idx)

            pairs_done += 1
            evict_before(max(0, pair_idx - CONTEXT_RADIUS))

            if pairs_done % 10 == 0 or pairs_done == analysis.total_pairs:
                report(make_progress())

        flush_batch()

        decode_up_to(analysis.frame_count - 1)
        if analysis.frame_count - 1 in frame_cache:
            encoder.write_frame(frame_cache[analysis.frame_count - 1])

        return True

    def cancel(self):
        """Cancel via instance-level events (fallback if no per-thread events)."""
        self._cancel.set()
        self._pause.set()

    def stop_all(self):
        """Prevent auto-advance to next queue item."""
        self._stopped = True

    def reset_stop(self):
        """Allow queue to advance again."""
        self._stopped = False

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stopped
