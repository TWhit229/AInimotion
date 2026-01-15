"""
Video enhancement pipeline orchestrator.

Coordinates decode -> process -> encode workflow.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Literal
from dataclasses import dataclass

from .decoder import extract_frames, get_video_info, VideoInfo
from .encoder import encode_frames, remux_audio


@dataclass
class PipelineConfig:
    """Configuration for the enhancement pipeline."""
    
    # Output settings
    target_fps: float | None = None  # None = 2x source
    target_scale: float | None = None  # None = no scaling
    
    # Encoding settings
    codec: Literal["h264", "h265"] = "h264"
    preset: Literal["ultrafast", "fast", "medium", "slow", "veryslow"] = "medium"
    crf: int = 18
    use_nvenc: bool = False
    
    # Processing settings
    frame_format: str = "png"  # 'png' or 'jpg' for temp frames
    keep_temp: bool = False
    temp_dir: Path | None = None


class EnhancePipeline:
    """
    Main video enhancement pipeline.
    
    Orchestrates the workflow:
    1. Decode input video to frames
    2. Process frames (interpolation, upscaling, etc.)
    3. Encode processed frames back to video
    4. Remux audio from original
    
    Args:
        config: Pipeline configuration
    """
    
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._temp_dir: Path | None = None
        self._input_info: VideoInfo | None = None
    
    @property
    def input_info(self) -> VideoInfo | None:
        """Get info about current input video."""
        return self._input_info
    
    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        frame_processor: Callable[[list[Path]], list[Path]] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> Path:
        """
        Run the full enhancement pipeline.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            frame_processor: Optional function to process frames
                Takes list of input frame paths, returns list of output frame paths.
                If None, frames are passed through unchanged.
            progress_callback: Optional callback(stage, current, total)
                Stages: 'decode', 'process', 'encode'
                
        Returns:
            Path to output video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Get input video info
        self._input_info = get_video_info(input_path)
        
        # Calculate target FPS
        if self.config.target_fps is not None:
            target_fps = self.config.target_fps
        else:
            target_fps = self._input_info.fps * 2  # Default: 2x FPS
        
        # Create temp directory
        if self.config.temp_dir is not None:
            self._temp_dir = Path(self.config.temp_dir)
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ainimotion_"))
        
        frames_dir = self._temp_dir / "frames"
        processed_dir = self._temp_dir / "processed"
        frames_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)
        
        try:
            # Stage 1: Decode
            if progress_callback:
                progress_callback("decode", 0, self._input_info.frame_count)
            
            input_frames = extract_frames(
                input_path,
                frames_dir,
                fps=None,  # Keep original FPS for decoding
                frame_format=self.config.frame_format,
            )
            
            if progress_callback:
                progress_callback("decode", len(input_frames), len(input_frames))
            
            # Stage 2: Process frames
            if frame_processor is not None:
                if progress_callback:
                    progress_callback("process", 0, len(input_frames))
                
                output_frames = frame_processor(input_frames)
                
                if progress_callback:
                    progress_callback("process", len(output_frames), len(output_frames))
            else:
                # Passthrough mode: copy frames to processed dir
                output_frames = []
                for i, frame in enumerate(input_frames):
                    dest = processed_dir / f"frame_{i:08d}.{self.config.frame_format}"
                    shutil.copy2(frame, dest)
                    output_frames.append(dest)
            
            # Stage 3: Encode
            if progress_callback:
                progress_callback("encode", 0, len(output_frames))
            
            # Determine frame pattern based on output frames
            frame_pattern = f"frame_%08d.{self.config.frame_format}"
            
            # Create temp output without audio
            temp_output = self._temp_dir / "temp_video.mp4"
            
            encode_frames(
                processed_dir if frame_processor else frames_dir,
                temp_output,
                fps=target_fps,
                codec=self.config.codec,
                preset=self.config.preset,
                crf=self.config.crf,
                frame_pattern=frame_pattern,
            )
            
            # Stage 4: Remux audio from original
            if self._input_info.has_audio:
                remux_audio(temp_output, input_path, output_path)
            else:
                shutil.move(str(temp_output), str(output_path))
            
            if progress_callback:
                progress_callback("encode", len(output_frames), len(output_frames))
            
            return output_path
            
        finally:
            # Cleanup temp directory
            if not self.config.keep_temp and self._temp_dir is not None:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    def passthrough(
        self,
        input_path: str | Path,
        output_path: str | Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> Path:
        """
        Run pipeline in passthrough mode (decode + encode, no processing).
        
        Useful for testing the pipeline without any enhancement.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback
            
        Returns:
            Path to output video
        """
        return self.process(
            input_path,
            output_path,
            frame_processor=None,
            progress_callback=progress_callback,
        )
