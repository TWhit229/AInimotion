"""
FFmpeg-based video encoder for frame encoding.

Encodes frames back to video files using FFmpeg subprocess calls.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Literal

from .decoder import get_ffmpeg_path, get_video_info


CodecType = Literal["h264", "h265", "copy"]
PresetType = Literal["ultrafast", "fast", "medium", "slow", "veryslow"]


def encode_frames(
    frame_dir: str | Path,
    output_path: str | Path,
    fps: float,
    codec: CodecType = "h264",
    preset: PresetType = "medium",
    crf: int = 18,
    frame_pattern: str = "frame_%08d.png",
    audio_source: str | Path | None = None,
    progress_callback=None,
) -> Path:
    """
    Encode frames to video using FFmpeg.
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output video file path
        fps: Output frame rate
        codec: Video codec ('h264', 'h265', or 'copy')
        preset: Encoding preset (speed/quality tradeoff)
        crf: Constant Rate Factor (0-51, lower = better quality)
        frame_pattern: Frame filename pattern (printf style)
        audio_source: Optional video to copy audio from
        progress_callback: Optional progress callback
        
    Returns:
        Path to output video
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg = get_ffmpeg_path()
    
    # Build FFmpeg command
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / frame_pattern),
    ]
    
    # Add audio source if provided
    if audio_source is not None:
        cmd.extend(["-i", str(audio_source)])
    
    # Video codec settings
    if codec == "h264":
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",  # Compatibility
        ])
    elif codec == "h265":
        cmd.extend([
            "-c:v", "libx265",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1",  # Apple compatibility
        ])
    else:
        cmd.extend(["-c:v", "copy"])
    
    # Audio settings
    if audio_source is not None:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",  # Video from first input
            "-map", "1:a?",   # Audio from second input (if exists)
        ])
    
    # Output
    cmd.append(str(output_path))
    
    # Run FFmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")
    
    return output_path


def remux_audio(
    video_source: str | Path,
    video_with_audio: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Remux audio from one video to another.
    
    Takes the video stream from video_source and audio from video_with_audio.
    
    Args:
        video_source: Video file with desired video stream
        video_with_audio: Video file with desired audio stream
        output_path: Output file path
        
    Returns:
        Path to output video
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg = get_ffmpeg_path()
    
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_source),
        "-i", str(video_with_audio),
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a?",
        "-shortest",
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg remux failed: {result.stderr}")
    
    return output_path


def encode_with_nvenc(
    frame_dir: str | Path,
    output_path: str | Path,
    fps: float,
    codec: Literal["h264", "h265"] = "h264",
    preset: Literal["fast", "medium", "slow"] = "medium",
    cq: int = 20,
    frame_pattern: str = "frame_%08d.png",
    audio_source: str | Path | None = None,
) -> Path:
    """
    Encode frames using NVIDIA NVENC hardware encoder.
    
    Much faster than CPU encoding but requires NVIDIA GPU.
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output video file path
        fps: Output frame rate
        codec: Video codec ('h264' or 'h265')
        preset: Encoding preset
        cq: Constant Quality value (0-51, lower = better)
        frame_pattern: Frame filename pattern
        audio_source: Optional video to copy audio from
        
    Returns:
        Path to output video
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg = get_ffmpeg_path()
    
    # Map codec to NVENC encoder
    encoder = "h264_nvenc" if codec == "h264" else "hevc_nvenc"
    
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / frame_pattern),
    ]
    
    if audio_source is not None:
        cmd.extend(["-i", str(audio_source)])
    
    # NVENC settings
    cmd.extend([
        "-c:v", encoder,
        "-preset", preset,
        "-cq", str(cq),
        "-pix_fmt", "yuv420p",
    ])
    
    if audio_source is not None:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a?",
        ])
    
    cmd.append(str(output_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # NVENC might not be available, provide helpful error
        if "nvenc" in result.stderr.lower():
            raise RuntimeError(
                "NVENC encoding failed. Make sure you have an NVIDIA GPU "
                "with NVENC support and up-to-date drivers."
            )
        raise RuntimeError(f"FFmpeg NVENC encoding failed: {result.stderr}")
    
    return output_path
