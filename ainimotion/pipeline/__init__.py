"""
Pipeline module for video I/O operations.

Provides FFmpeg-based frame extraction and encoding.
"""

from .decoder import extract_frames, get_video_info, VideoInfo
from .encoder import encode_frames, remux_audio
from .pipeline import EnhancePipeline

__all__ = [
    "extract_frames",
    "get_video_info", 
    "VideoInfo",
    "encode_frames",
    "remux_audio",
    "EnhancePipeline",
]
