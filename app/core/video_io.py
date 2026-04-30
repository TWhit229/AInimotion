"""
Video I/O via FFmpeg subprocess pipes.

Decode and encode video frames as raw RGB numpy arrays without loading
entire videos into memory. Handles downscaling, audio stream copying,
and container format preservation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Prevent ffmpeg/ffprobe from spawning a visible console window on Windows
_POPEN_FLAGS: dict = {}
if sys.platform == 'win32':
    _POPEN_FLAGS['creationflags'] = subprocess.CREATE_NO_WINDOW


@dataclass
class VideoInfo:
    """Metadata from ffprobe."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    pix_fmt: str
    audio_streams: int
    subtitle_streams: int
    container: str


def probe_video(path: str | Path) -> VideoInfo:
    """
    Get video metadata via ffprobe.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If no video stream is found.
        subprocess.CalledProcessError: If ffprobe fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True, **_POPEN_FLAGS)

    data = json.loads(result.stdout)

    video_stream = None
    n_audio = 0
    n_subtitle = 0
    for stream in data.get('streams', []):
        codec_type = stream.get('codec_type', '')
        if codec_type == 'video' and video_stream is None:
            video_stream = stream
        elif codec_type == 'audio':
            n_audio += 1
        elif codec_type == 'subtitle':
            n_subtitle += 1

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # Parse frame rate (r_frame_rate is typically "num/den" but can be just "num")
    fps_str = video_stream.get('r_frame_rate', '24/1')
    parts = fps_str.split('/')
    num = int(parts[0])
    den = int(parts[1]) if len(parts) > 1 else 1
    fps = num / den if den else 24.0

    # Frame count: prefer nb_read_frames (from -count_frames), then nb_frames,
    # then estimate from duration
    frame_count = int(video_stream.get('nb_read_frames', 0))
    if frame_count == 0:
        frame_count = int(video_stream.get('nb_frames', 0))
    if frame_count == 0:
        duration = float(data.get('format', {}).get('duration', 0))
        frame_count = int(duration * fps)

    return VideoInfo(
        path=str(path),
        width=int(video_stream['width']),
        height=int(video_stream['height']),
        fps=fps,
        frame_count=frame_count,
        duration=float(data.get('format', {}).get('duration', 0)),
        codec=video_stream.get('codec_name', 'unknown'),
        pix_fmt=video_stream.get('pix_fmt', 'unknown'),
        audio_streams=n_audio,
        subtitle_streams=n_subtitle,
        container=data.get('format', {}).get('format_name', 'unknown'),
    )


def align_resolution(width: int, height: int, max_height: int = 720, alignment: int = 8) -> tuple[int, int]:
    """
    Compute target resolution: downscale to max_height (if larger), maintain
    aspect ratio, and align both dimensions to the given multiple.

    The V5 model requires dimensions divisible by 4 (two stride-2 encoder
    stages). We use 8 for extra safety with bilinear interpolation strides.
    """
    if height <= max_height:
        # No downscale needed, just align
        w = (width // alignment) * alignment
        h = (height // alignment) * alignment
        return max(w, alignment), max(h, alignment)

    scale = max_height / height
    w = int(width * scale)
    h = int(height * scale)

    # Align to multiple
    w = (w // alignment) * alignment
    h = (h // alignment) * alignment
    return max(w, alignment), max(h, alignment)


class FrameDecoder:
    """
    Decode video frames to raw RGB numpy arrays via ffmpeg pipe.

    Optionally downscales during decode (done by ffmpeg, not Python — fast).
    Use as a context manager and iterate to get frames.

    Example:
        with FrameDecoder('input.mkv', max_height=720) as decoder:
            for frame in decoder:
                # frame is np.ndarray, shape (H, W, 3), dtype uint8
                process(frame)
    """

    def __init__(self, path: str | Path, max_height: int | None = 720):
        self.path = Path(path)
        self.info = probe_video(self.path)

        # Compute output resolution
        if max_height is not None:
            self.width, self.height = align_resolution(
                self.info.width, self.info.height, max_height
            )
        else:
            self.width, self.height = align_resolution(
                self.info.width, self.info.height, max_height=self.info.height
            )

        self.frame_count = self.info.frame_count
        self.fps = self.info.fps
        self._process: subprocess.Popen | None = None
        self._frame_size = self.width * self.height * 3
        self._frames_read = 0

    @property
    def needs_downscale(self) -> bool:
        return self.width != self.info.width or self.height != self.info.height

    def __enter__(self):
        cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', str(self.path),
        ]

        # Downscale during decode if needed
        if self.needs_downscale:
            cmd += ['-vf', f'scale={self.width}:{self.height}:flags=lanczos']

        cmd += [
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-vsync', 'passthrough',
            '-',
        ]

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_POPEN_FLAGS,
        )
        self._frames_read = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._process:
            self._process.stdout.close()
            self._process.stderr.close()
            self._process.kill()
            self._process.wait()
            self._process = None

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        raw = self._process.stdout.read(self._frame_size)
        if len(raw) < self._frame_size:
            raise StopIteration
        self._frames_read += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3).copy()

    @property
    def progress(self) -> float:
        """Fraction of frames decoded (0.0 to 1.0)."""
        if self.frame_count <= 0:
            return 0.0
        return min(self._frames_read / self.frame_count, 1.0)


class FrameEncoder:
    """
    Encode raw RGB frames to a video file via ffmpeg pipe.

    Copies audio and subtitle streams from a source file if provided.
    Use as a context manager, then call write_frame() for each frame.

    Example:
        with FrameEncoder('output.mkv', fps=48, width=1280, height=720,
                          audio_source='input.mkv') as encoder:
            for frame in frames:
                encoder.write_frame(frame)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        audio_source: str | Path | None = None,
        codec: str = 'libx264',
        crf: int = 18,
        preset: str = 'medium',
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.audio_source = Path(audio_source) if audio_source else None
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self._process: subprocess.Popen | None = None
        self._frames_written = 0

    def __enter__(self):
        cmd = [
            'ffmpeg',
            '-v', 'error',
            '-y',  # overwrite output
            # Input 0: raw RGB frames from pipe
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
        ]

        if self.audio_source:
            # Input 1: source file for audio/subtitle streams
            cmd += ['-i', str(self.audio_source)]

        # Video encoding settings
        cmd += [
            '-c:v', self.codec,
            '-preset', self.preset,
            '-crf', str(self.crf),
            '-pix_fmt', 'yuv420p',
        ]

        if self.audio_source:
            # Map video from pipe, audio/subtitles from source
            cmd += [
                '-map', '0:v:0',     # video from pipe input
                '-map', '1:a?',      # all audio from source (? = skip if none)
                '-map', '1:s?',      # all subtitles from source
                '-c:a', 'copy',      # don't re-encode audio
                '-c:s', 'copy',      # don't re-encode subtitles
            ]

        cmd.append(str(self.output_path))

        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_POPEN_FLAGS,
        )
        self._frames_written = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._process:
            self._process.stdin.close()
            # Use communicate() to avoid pipe deadlock (stderr buffer can fill)
            _, stderr_bytes = self._process.communicate()
            if self._process.returncode != 0 and stderr_bytes:
                stderr = stderr_bytes.decode(errors='replace').strip()
                if stderr:
                    print(f"  [!] FFmpeg encoder warning: {stderr[:200]}")
            self._process = None

    def write_frame(self, frame: np.ndarray):
        """Write a single RGB frame. Shape must be (height, width, 3), dtype uint8."""
        assert frame.shape == (self.height, self.width, 3), \
            f"Frame shape {frame.shape} != expected ({self.height}, {self.width}, 3)"
        assert frame.dtype == np.uint8, f"Frame dtype {frame.dtype} != uint8"
        self._process.stdin.write(frame.tobytes())
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.core.video_io <input_video> [output_video]")
        print("  Decodes input, re-encodes to output (default: test_output.mp4)")
        print("  Tests the full decode -> encode pipeline.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'test_output.mp4'

    # Probe
    print(f"Probing: {input_path}")
    info = probe_video(input_path)
    print(f"  Resolution: {info.width}x{info.height}")
    print(f"  FPS: {info.fps}")
    print(f"  Frames: {info.frame_count}")
    print(f"  Duration: {info.duration:.1f}s")
    print(f"  Codec: {info.codec}")
    print(f"  Audio streams: {info.audio_streams}")
    print(f"  Subtitle streams: {info.subtitle_streams}")
    print(f"  Container: {info.container}")

    # Decode -> re-encode (passthrough test at 720p)
    target_w, target_h = align_resolution(info.width, info.height, max_height=720)
    print(f"\nDecode at {target_w}x{target_h} -> encode to {output_path}")

    with FrameDecoder(input_path, max_height=720) as decoder:
        print(f"  Decoder: {decoder.width}x{decoder.height}, {decoder.frame_count} frames")

        with FrameEncoder(output_path, fps=info.fps, width=decoder.width,
                          height=decoder.height, audio_source=input_path) as encoder:
            for i, frame in enumerate(decoder):
                encoder.write_frame(frame)
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"  Frame {i+1}/{decoder.frame_count}"
                          f" ({decoder.progress*100:.1f}%)", end='\r')

            print(f"\n  Done: {encoder.frames_written} frames written to {output_path}")
