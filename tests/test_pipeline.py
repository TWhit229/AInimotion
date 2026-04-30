"""
Comprehensive test suite for AInimotion pipeline.
Tests video I/O, frame analysis, interpolation, and end-to-end processing.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ============================================================
# Test helpers
# ============================================================

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_test_video(path, n_frames=30, width=256, height=256, fps=24, pattern='gradient'):
    """Create a test video with known content."""
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-crf', '1', '-pix_fmt', 'yuv420p',
        str(path),
    ]
    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, **kwargs)
    for i in range(n_frames):
        if pattern == 'gradient':
            r = int(255 * i / max(n_frames - 1, 1))
            frame = np.full((height, width, 3), [r, 128, 255 - r], dtype=np.uint8)
        elif pattern == 'duplicate':
            # Each frame held 2x (12fps animation)
            r = int(255 * (i // 2) / max(n_frames // 2 - 1, 1))
            frame = np.full((height, width, 3), [r, 100, 200], dtype=np.uint8)
        elif pattern == 'scene_cut':
            if i < n_frames // 2:
                frame = np.full((height, width, 3), [50, 50, 200], dtype=np.uint8)
            else:
                frame = np.full((height, width, 3), [200, 50, 50], dtype=np.uint8)
        elif pattern == 'static':
            frame = np.full((height, width, 3), [128, 128, 128], dtype=np.uint8)
        elif pattern == 'motion':
            # Moving block - creates real motion that needs GPU interpolation
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            bx = int((width - 60) * i / max(n_frames - 1, 1))
            by = int((height - 60) * i / max(n_frames - 1, 1))
            frame[by:by+60, bx:bx+60] = [200, 100, 50]
        else:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return path


# ============================================================
# Tests
# ============================================================

def test_video_io():
    print("\n=== VIDEO I/O ===")
    from app.core.video_io import probe_video, FrameDecoder, FrameEncoder, align_resolution

    with tempfile.TemporaryDirectory() as tmpdir:
        vid = make_test_video(Path(tmpdir) / 'test.mp4', n_frames=50, width=640, height=360)

        # Probe
        info = probe_video(vid)
        test("probe: width", info.width == 640)
        test("probe: height", info.height == 360)
        test("probe: fps", abs(info.fps - 24.0) < 0.1, f"got {info.fps}")
        test("probe: frame_count", info.frame_count == 50, f"got {info.frame_count}")
        test("probe: codec", info.codec == 'h264', f"got {info.codec}")

        # Probe nonexistent
        try:
            probe_video('/nonexistent/file.mp4')
            test("probe: nonexistent raises", False)
        except FileNotFoundError:
            test("probe: nonexistent raises", True)

        # Align resolution
        w, h = align_resolution(1920, 1080, max_height=720)
        test("align: 1080p->720p width", w % 8 == 0)
        test("align: 1080p->720p height", h == 720)
        w, h = align_resolution(640, 480, max_height=720)
        test("align: 480p stays", h == 480)

        # Decode
        with FrameDecoder(str(vid), max_height=None) as dec:
            frames = list(dec)
            test("decode: frame count", len(frames) == 50, f"got {len(frames)}")
            test("decode: frame shape", frames[0].shape == (360, 640, 3))
            test("decode: frame dtype", frames[0].dtype == np.uint8)

        # Decode with downscale
        with FrameDecoder(str(vid), max_height=180) as dec:
            frames = list(dec)
            test("decode downscale: height", dec.height <= 184)  # aligned to 8
            test("decode downscale: frames", len(frames) == 50, f"got {len(frames)}")

        # Encode
        out_path = Path(tmpdir) / 'output.mp4'
        with FrameDecoder(str(vid), max_height=None) as dec:
            with FrameEncoder(str(out_path), fps=48, width=dec.width, height=dec.height) as enc:
                for f in dec:
                    enc.write_frame(f)
                    enc.write_frame(f)  # double
                test("encode: frames written", enc.frames_written == 100)

        out_info = probe_video(out_path)
        test("encode: output fps", abs(out_info.fps - 48.0) < 0.1, f"got {out_info.fps}")
        test("encode: output frames", out_info.frame_count == 100, f"got {out_info.frame_count}")


def test_frame_analysis():
    print("\n=== FRAME ANALYSIS ===")
    from app.core.frame_analysis import analyze_video, build_context_indices, PairType

    with tempfile.TemporaryDirectory() as tmpdir:
        # Gradient video (all unique frames)
        vid = make_test_video(Path(tmpdir) / 'gradient.mp4', n_frames=30, pattern='gradient')
        result = analyze_video(str(vid), max_height=None)
        test("gradient: frame count", result.frame_count == 30, f"got {result.frame_count}")
        test("gradient: pairs", result.total_pairs == 29)
        test("gradient: no duplicates", result.duplicate_count == 0, f"got {result.duplicate_count}")

        # Duplicate video (12fps held)
        vid2 = make_test_video(Path(tmpdir) / 'dup.mp4', n_frames=30, pattern='duplicate')
        result2 = analyze_video(str(vid2), max_height=None)
        test("duplicate: has dupes", result2.duplicate_count > 0, f"got {result2.duplicate_count}")

        # Scene cut video
        vid3 = make_test_video(Path(tmpdir) / 'cut.mp4', n_frames=30, pattern='scene_cut')
        result3 = analyze_video(str(vid3), max_height=None)
        test("scene_cut: detected", result3.scene_cut_count >= 1, f"got {result3.scene_cut_count}")

        # Static video (all low motion)
        vid4 = make_test_video(Path(tmpdir) / 'static.mp4', n_frames=20, pattern='static')
        result4 = analyze_video(str(vid4), max_height=None)
        test("static: mostly dup/low", result4.interpolate_count < 5,
             f"interp={result4.interpolate_count}, dup={result4.duplicate_count}, low={result4.low_motion_count}")

        # Context indices: normal
        ctx = build_context_indices(5, 30, result)
        test("context: length 7", len(ctx) == 7)
        test("context: anchor pair", ctx[3] == 5 and ctx[4] == 6)

        # Context indices: start of video
        ctx0 = build_context_indices(0, 30, result)
        test("context: start mirror", all(0 <= i < 30 for i in ctx0))

        # Context indices: end of video
        ctx_end = build_context_indices(28, 30, result)
        test("context: end mirror", all(0 <= i < 30 for i in ctx_end))

        # Context indices: very short video (3 frames)
        vid_short = make_test_video(Path(tmpdir) / 'short.mp4', n_frames=3, pattern='gradient')
        res_short = analyze_video(str(vid_short), max_height=None)
        ctx_short = build_context_indices(0, 3, res_short)
        test("context: short video valid", all(0 <= i < 3 for i in ctx_short))

        # Cancel during analysis
        cancel = threading.Event()
        cancel.set()  # cancel immediately
        res_cancel = analyze_video(str(vid), max_height=None, cancel_event=cancel)
        test("cancel: stops early", res_cancel.frame_count < 30,
             f"got {res_cancel.frame_count} frames")

        # Parallel analysis (longer video)
        vid_long = make_test_video(Path(tmpdir) / 'long.mp4', n_frames=600, width=128, height=128, pattern='gradient')
        import time
        t0 = time.time()
        res_par = analyze_video(str(vid_long), max_height=None, n_workers=4)
        par_time = time.time() - t0
        test("parallel: frame count", res_par.frame_count == 600, f"got {res_par.frame_count}")
        test("parallel: pairs count", res_par.total_pairs >= 595,
             f"got {res_par.total_pairs} (expected ~599)")
        test("parallel: pairs sorted", all(res_par.pairs[i].index <= res_par.pairs[i+1].index
             for i in range(len(res_par.pairs)-1)))

        # Single thread comparison
        t0 = time.time()
        res_single = analyze_video(str(vid_long), max_height=None, n_workers=1)
        single_time = time.time() - t0
        # Parallel may not be faster for tiny videos (overhead > decode time)
        test("parallel: reasonable time", par_time < single_time * 2.0,
             f"parallel={par_time:.2f}s single={single_time:.2f}s")
        test("parallel: same frame count", res_par.frame_count == res_single.frame_count)

        # Credits detection (simulate: 400 frames action + 200 frames static)
        vid_credits = Path(tmpdir) / 'credits.mp4'
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '128x128', '-r', '24', '-i', '-',
               '-c:v', 'libx264', '-crf', '1', '-pix_fmt', 'yuv420p', str(vid_credits)]
        kwargs = {}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, **kwargs)
        # Action section: moving block
        for i in range(400):
            frame = np.full((128, 128, 3), 50, dtype=np.uint8)
            bx = int(88 * i / 399)
            frame[30:80, bx:bx+40] = [200, 120, 80]
            proc.stdin.write(frame.tobytes())
        # Credits section: nearly static (simulates text on dark background)
        for i in range(200):
            frame = np.full((128, 128, 3), 20, dtype=np.uint8)
            # Tiny text-like change each frame
            frame[60 + (i % 5), 40:90] = 180
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()

        res_credits = analyze_video(str(vid_credits), max_height=None, n_workers=1)
        test("credits: detected", res_credits.credits_start is not None,
             f"credits_start={res_credits.credits_start}")
        if res_credits.credits_start is not None:
            test("credits: after action", res_credits.credits_start >= 350,
                 f"starts at {res_credits.credits_start} (expected ~400)")
        test("credits: count > 0", res_credits.credits_count > 0,
             f"got {res_credits.credits_count}")

        # Action-only mode detection
        from app.core.frame_analysis import detect_action_regions, apply_action_only_mode, PairInfo

        # Simulate: calm(200) → action(300) → calm(100) → action(200) → calm(200)
        test_pairs = []
        for i in range(200):
            test_pairs.append(PairInfo(index=i, pair_type=PairType.INTERPOLATE, mse=30.0))
        for i in range(300):
            test_pairs.append(PairInfo(index=200+i, pair_type=PairType.INTERPOLATE, mse=400.0 + np.random.rand()*200))
        for i in range(100):
            test_pairs.append(PairInfo(index=500+i, pair_type=PairType.INTERPOLATE, mse=25.0))
        for i in range(200):
            test_pairs.append(PairInfo(index=600+i, pair_type=PairType.INTERPOLATE, mse=350.0 + np.random.rand()*150))
        for i in range(200):
            test_pairs.append(PairInfo(index=800+i, pair_type=PairType.INTERPOLATE, mse=20.0))

        regions = detect_action_regions(test_pairs, fps=24.0, link_seconds=15.0)
        test("action: regions found", len(regions) >= 1, f"got {len(regions)}")

        # The two action bursts (200-500 and 600-800) should merge if link_seconds=15
        # because the gap is 100 frames = ~4.2 seconds < 15 seconds
        if len(regions) >= 1:
            test("action: merged nearby bursts", len(regions) <= 2,
                 f"got {len(regions)} (expected 1-2 merged)")

        # With tight linking (2s) and no extension, they should stay separate
        regions_tight = detect_action_regions(test_pairs, fps=24.0, link_seconds=2.0, extend_seconds=0.0)
        test("action tight: separate bursts", len(regions_tight) >= 2,
             f"got {len(regions_tight)}")

        # Apply action-only mode
        from app.core.frame_analysis import AnalysisResult
        test_result = AnalysisResult(frame_count=1001, pairs=test_pairs.copy())
        apply_action_only_mode(test_result, fps=24.0, link_seconds=15.0)
        calm_count = sum(1 for p in test_result.pairs if p.pair_type == PairType.CALM)
        interp_count = sum(1 for p in test_result.pairs if p.pair_type == PairType.INTERPOLATE)
        test("action-only: has calm", calm_count > 0, f"got {calm_count}")
        test("action-only: has interp", interp_count > 0, f"got {interp_count}")
        test("action-only: calm < total", calm_count < len(test_result.pairs))
        # Calm should be non-action sections (with extension eating into edges, ~200+ calm)
        test("action-only: significant calm", calm_count > 100,
             f"got {calm_count} calm out of {len(test_result.pairs)}")


def test_interpolator():
    print("\n=== INTERPOLATOR ===")
    from app.core.interpolator import Interpolator

    interp = Interpolator('model/ainimotion.pt', use_compile=False)

    # Basic inference
    H, W = 64, 64
    window = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(7)]

    result = interp.interpolate_single(window)
    test("single: shape", result.shape == (H, W, 3))
    test("single: dtype", result.dtype == np.uint8)
    test("single: range", 0 <= result.min() and result.max() <= 255)

    # Batch inference
    batch = [window, window]
    results = interp.interpolate_batch(batch)
    test("batch: count", len(results) == 2)
    test("batch: shape", results[0].shape == (H, W, 3))

    # Different timestep
    r05 = interp.interpolate_single(window)
    r03 = interp.interpolate_batch([window], timestep=0.3)[0]
    test("timestep: different output", not np.array_equal(r05, r03))

    # Multi-timestep
    multi = interp.interpolate_multi_timestep([window], [0.25, 0.5, 0.75])
    test("multi_timestep: outer len", len(multi) == 1)
    test("multi_timestep: inner len", len(multi[0]) == 3)
    test("multi_timestep: shapes", all(f.shape == (H, W, 3) for f in multi[0]))

    # Batch size detection
    bs = interp.find_batch_size(64, 64)
    test("batch_size: positive", bs >= 1)
    test("batch_size: cached", interp.find_batch_size(64, 64) == bs)

    # Different resolution invalidates cache
    bs2 = interp.find_batch_size(128, 128)
    test("batch_size: new res", bs2 >= 1)


def test_processor():
    print("\n=== PROCESSOR (end-to-end) ===")
    from app.core.processor import VideoProcessor

    processor = VideoProcessor('model/ainimotion.pt', max_height=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small test video
        vid = make_test_video(Path(tmpdir) / 'input.mp4', n_frames=12, width=64, height=64)
        out = Path(tmpdir) / 'output.mp4'

        # Normal processing
        progress_log = []
        def on_progress(p):
            progress_log.append(p.phase)

        success = processor.process(str(vid), str(out), progress_callback=on_progress)
        test("process: success", success)
        test("process: output exists", out.exists())
        test("process: phases", 'analyzing' in progress_log and 'processing' in progress_log,
             f"phases: {set(progress_log)}")

        # Verify output
        from app.core.video_io import probe_video
        out_info = probe_video(out)
        test("process: output fps doubled", abs(out_info.fps - 48.0) < 1.0, f"got {out_info.fps}")
        expected_frames = 12 * 2 - 1  # 2x interpolation
        test("process: frame count", abs(out_info.frame_count - expected_frames) <= 2,
             f"expected ~{expected_frames}, got {out_info.frame_count}")

        # Cancel mid-processing
        # Moving block pattern — creates real motion that needs GPU interpolation
        vid2 = make_test_video(Path(tmpdir) / 'input2.mp4', n_frames=200, width=128, height=128, pattern='motion')
        out2 = Path(tmpdir) / 'output2.mp4'
        cancel = threading.Event()
        pause = threading.Event()
        pause.set()  # NOT paused — must be set or processor blocks forever

        def cancel_after_delay():
            time.sleep(0.5)
            cancel.set()

        t = threading.Thread(target=cancel_after_delay)
        t.start()
        success2 = processor.process(str(vid2), str(out2),
                                      cancel_event=cancel, pause_event=pause)
        # Make sure pause event is set (not paused)
        t.join()
        test("cancel: not success", not success2)
        test("cancel: output cleaned", not out2.exists())

        # Precomputed analysis
        from app.core.frame_analysis import analyze_video
        analysis = analyze_video(str(vid), max_height=None)
        out3 = Path(tmpdir) / 'output3.mp4'
        success3 = processor.process(str(vid), str(out3), precomputed_analysis=analysis)
        test("precomputed: success", success3)
        test("precomputed: output exists", out3.exists())


def test_analysis_cache():
    print("\n=== ANALYSIS CACHE ===")
    from app.core.analysis_cache import AnalysisCache

    with tempfile.TemporaryDirectory() as tmpdir:
        vid = make_test_video(Path(tmpdir) / 'test.mp4', n_frames=20, width=64, height=64)

        cache = AnalysisCache(max_height=None)

        # Submit and wait for result
        cache.submit(str(vid))
        result = None
        for _ in range(100):  # wait up to 10 seconds
            result = cache.get(str(vid))
            if result:
                break
            time.sleep(0.1)

        test("cache: result available", result is not None)
        if result:
            test("cache: frame count", result.frame_count == 20)

        # Duplicate submit
        cache.submit(str(vid))  # should not crash
        test("cache: duplicate submit ok", True)

        # Evict
        cache.evict(str(vid))
        test("cache: evicted", cache.get(str(vid)) is None)

        # Shutdown
        cache.shutdown()
        test("cache: shutdown ok", True)


def test_make_output_path():
    print("\n=== OUTPUT PATH ===")
    from app.ui.main_window import make_output_path

    p = make_output_path(Path('/v/test.mkv'), None, 2, 24.0)
    test("path: 2x 24fps", p.name == 'test_48fps.mkv')

    p = make_output_path(Path('/v/test.mkv'), None, 3, 24.0)
    test("path: 3x 24fps", p.name == 'test_72fps.mkv')

    p = make_output_path(Path('/v/test.mkv'), None, 4, 23.976)
    test("path: 4x 23.976", p.name == 'test_95fps.mkv')

    p = make_output_path(Path('/v/test.mkv'), Path('/out'), 2, 30.0)
    test("path: custom dir", str(p.parent).replace('\\', '/') == '/out')

    p = make_output_path(Path('/v/test.mkv'), None, 2, None)
    test("path: no fps info", p.name == 'test_2x.mkv')


def test_format_time():
    print("\n=== TIME FORMATTING ===")
    from app.ui.queue_item_widget import fmt_time

    test("fmt: seconds", fmt_time(30) == '30s')
    test("fmt: minutes", fmt_time(90) == '1m30s')
    test("fmt: hours", fmt_time(3661) == '1h01m')
    test("fmt: zero", fmt_time(0) == '0s')


def test_ui_imports():
    print("\n=== UI IMPORTS ===")
    from app.ui.theme import DARK_THEME, VIDEO_EXTENSIONS, STATUS_COLORS

    test("theme: not empty", len(DARK_THEME) > 100)
    test("extensions: mkv", '.mkv' in VIDEO_EXTENSIONS)
    test("extensions: mp4", '.mp4' in VIDEO_EXTENSIONS)
    test("colors: complete", 'complete' in STATUS_COLORS)
    test("colors: error", 'error' in STATUS_COLORS)

    from app.ui.queue_item_widget import ItemState
    test("states: all exist", len(ItemState) == 6)


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  AInimotion Test Suite")
    print("=" * 60)

    test_video_io()
    test_frame_analysis()
    test_format_time()
    test_make_output_path()
    test_ui_imports()
    test_analysis_cache()
    test_interpolator()
    test_processor()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
