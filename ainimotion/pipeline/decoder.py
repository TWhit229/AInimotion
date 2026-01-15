"""
FFmpeg-based video decoder for frame extraction.

Extracts frames from video files using FFmpeg subprocess calls.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class VideoInfo:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    duration: float  # seconds
    frame_count: int
    codec: str
    has_audio: bool
    
    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)


def get_ffmpeg_path() -> str:
    """Get path to FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found in PATH. Please install FFmpeg: "
            "https://ffmpeg.org/download.html"
        )
    return ffmpeg


def get_ffprobe_path() -> str:
    """Get path to FFprobe executable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(
            "FFprobe not found in PATH. Please install FFmpeg: "
            "https://ffmpeg.org/download.html"
        )
    return ffprobe


def get_video_info(video_path: str | Path) -> VideoInfo:
    """
    Get video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoInfo with video metadata
        
    Raises:
        RuntimeError: If ffprobe fails or video is invalid
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    ffprobe = get_ffprobe_path()
    
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")
    
    # Find video stream
    video_stream = None
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True
    
    if video_stream is None:
        raise RuntimeError(f"No video stream found in: {video_path}")
    
    # Parse FPS (can be "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 24.0
    else:
        fps = float(fps_str)
    
    # Get duration from format or stream
    duration = float(data.get("format", {}).get("duration", 0))
    if duration == 0:
        duration = float(video_stream.get("duration", 0))
    
    # Calculate frame count
    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        frame_count = int(nb_frames)
    else:
        frame_count = int(duration * fps)
    
    return VideoInfo(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        codec=video_stream.get("codec_name", "unknown"),
        has_audio=has_audio,
    )


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    frame_format: str = "png",
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """
    Extract frames from video using FFmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Output frame rate (None = keep original)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        frame_format: Output format ('png' or 'jpg')
        progress_callback: Optional callback(current_frame, total_frames)
        
    Returns:
        List of paths to extracted frames, sorted by frame number
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info for progress tracking
    info = get_video_info(video_path)
    
    ffmpeg = get_ffmpeg_path()
    
    # Build FFmpeg command
    cmd = [ffmpeg, "-y"]
    
    # Input options
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    cmd.extend(["-i", str(video_path)])
    
    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
        else:
            duration = end_time
        cmd.extend(["-t", str(duration)])
    
    # Output options
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])
    
    # Frame format settings
    if frame_format == "jpg":
        cmd.extend(["-qscale:v", "2"])  # High quality JPEG
    
    # Output pattern
    output_pattern = output_dir / f"frame_%08d.{frame_format}"
    cmd.append(str(output_pattern))
    
    # Run FFmpeg
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    _, stderr = process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {stderr}")
    
    # Collect output frames
    frames = sorted(output_dir.glob(f"frame_*.{frame_format}"))
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    
    return frames


def extract_frames_to_memory(
    video_path: str | Path,
    fps: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[bytes]:
    """
    Extract frames to memory as raw bytes (PNG format).
    
    Warning: This can use a lot of memory for long videos!
    
    Args:
        video_path: Path to input video
        fps: Output frame rate (None = keep original)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        
    Returns:
        List of PNG frame data as bytes
    """
    video_path = Path(video_path)
    ffmpeg = get_ffmpeg_path()
    
    # Build command to output raw frames to pipe
    cmd = [ffmpeg]
    
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    cmd.extend(["-i", str(video_path)])
    
    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
        else:
            duration = end_time
        cmd.extend(["-t", str(duration)])
    
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])
    
    # Output as image sequence to pipe (using image2pipe format)
    cmd.extend([
        "-f", "image2pipe",
        "-c:v", "png",
        "-"
    ])
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
    
    # Parse PNG frames from raw output
    # PNG files start with magic bytes: 89 50 4E 47 0D 0A 1A 0A
    png_magic = b'\x89PNG\r\n\x1a\n'
    frames = []
    data = result.stdout
    
    # Find all PNG start positions
    positions = []
    pos = 0
    while True:
        idx = data.find(png_magic, pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    
    # Extract each frame
    for i, start in enumerate(positions):
        if i + 1 < len(positions):
            end = positions[i + 1]
        else:
            end = len(data)
        frames.append(data[start:end])
    
    return frames
