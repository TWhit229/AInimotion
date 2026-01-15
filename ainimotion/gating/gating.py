"""
Gating processor for frame interpolation.

Wraps an interpolation model and applies gating decisions
to determine whether to interpolate or duplicate frames.
"""

from pathlib import Path
from typing import Callable, Sequence
import shutil

import torch

from .classifier import (
    FrameType,
    GatingThresholds,
    GatingDecision,
    classify_pair,
    classify_sequence,
    summarize_gating,
)


class GatingProcessor:
    """
    Frame processor with gating for anime-safe interpolation.
    
    For each frame pair:
    - CUT: Duplicate first frame (no interpolation)
    - HOLD: Duplicate first frame (no interpolation)
    - MOTION: Call interpolator to generate middle frame
    
    Args:
        thresholds: Gating thresholds for classification
        interpolator: Optional function(frame1, frame2) -> middle_frame
            If None, always duplicates (passthrough mode)
        debug: If True, print gating decisions
    """
    
    def __init__(
        self,
        thresholds: GatingThresholds | None = None,
        interpolator: Callable | None = None,
        debug: bool = False,
    ):
        self.thresholds = thresholds or GatingThresholds.anime_default()
        self.interpolator = interpolator
        self.debug = debug
        
        # Statistics
        self.stats = {
            "cuts": 0,
            "holds": 0,
            "motion": 0,
            "interpolated": 0,
            "duplicated": 0,
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def process_pair(
        self,
        frame1: str | Path | torch.Tensor,
        frame2: str | Path | torch.Tensor,
    ) -> tuple[GatingDecision, torch.Tensor | Path | None]:
        """
        Process a single frame pair.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Tuple of (decision, interpolated_frame_or_None)
            If decision is CUT/HOLD, returns None (duplicate frame1)
            If decision is MOTION and interpolator exists, returns interpolated frame
        """
        decision = classify_pair(frame1, frame2, self.thresholds)
        
        # Update stats
        if decision.frame_type == FrameType.CUT:
            self.stats["cuts"] += 1
        elif decision.frame_type == FrameType.HOLD:
            self.stats["holds"] += 1
        else:
            self.stats["motion"] += 1
        
        if self.debug:
            print(f"  {decision.frame_type.value}: sim={decision.similarity:.4f} ({decision.confidence})")
        
        # Decide whether to interpolate
        if decision.frame_type == FrameType.MOTION and self.interpolator is not None:
            self.stats["interpolated"] += 1
            interpolated = self.interpolator(frame1, frame2)
            return decision, interpolated
        else:
            self.stats["duplicated"] += 1
            return decision, None
    
    def process_sequence(
        self,
        input_frames: Sequence[Path],
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """
        Process a sequence of frames with gating and 2x frame rate.
        
        For input frames [F1, F2, F3, ...], produces output:
        [F1, M1, F2, M2, F3, ...] where M is interpolated or duplicated.
        
        Args:
            input_frames: List of input frame paths
            output_dir: Directory for output frames
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of output frame paths (2x - 1 input length)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reset_stats()
        output_frames = []
        total = len(input_frames)
        
        for i, frame_path in enumerate(input_frames):
            frame_path = Path(frame_path)
            
            # Copy original frame
            out_idx = len(output_frames)
            out_path = output_dir / f"frame_{out_idx:08d}{frame_path.suffix}"
            shutil.copy2(frame_path, out_path)
            output_frames.append(out_path)
            
            # Generate intermediate frame (except for last frame)
            if i < total - 1:
                next_frame = input_frames[i + 1]
                decision, interpolated = self.process_pair(frame_path, next_frame)
                
                out_idx = len(output_frames)
                mid_path = output_dir / f"frame_{out_idx:08d}{frame_path.suffix}"
                
                if interpolated is not None:
                    # Save interpolated frame
                    if isinstance(interpolated, torch.Tensor):
                        from PIL import Image
                        import numpy as np
                        
                        if interpolated.dim() == 4:
                            interpolated = interpolated.squeeze(0)
                        img = (interpolated.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img).save(mid_path)
                    else:
                        shutil.copy2(interpolated, mid_path)
                else:
                    # Duplicate first frame
                    shutil.copy2(frame_path, mid_path)
                
                output_frames.append(mid_path)
            
            if progress_callback is not None:
                progress_callback(i + 1, total)
        
        if self.debug:
            summary = self.get_stats_summary()
            print(f"\nGating Summary:")
            print(f"  Total pairs: {summary['total']}")
            print(f"  Cuts: {summary['cuts']} ({summary['cuts_pct']:.1f}%)")
            print(f"  Holds: {summary['holds']} ({summary['holds_pct']:.1f}%)")
            print(f"  Motion: {summary['motion']} ({summary['motion_pct']:.1f}%)")
            print(f"  Interpolated: {self.stats['interpolated']}")
            print(f"  Duplicated: {self.stats['duplicated']}")
        
        return output_frames
    
    def get_stats_summary(self) -> dict:
        """Get summary statistics for processed frames."""
        total = self.stats["cuts"] + self.stats["holds"] + self.stats["motion"]
        if total == 0:
            return {"total": 0}
        
        return {
            "total": total,
            "cuts": self.stats["cuts"],
            "holds": self.stats["holds"],
            "motion": self.stats["motion"],
            "cuts_pct": self.stats["cuts"] / total * 100,
            "holds_pct": self.stats["holds"] / total * 100,
            "motion_pct": self.stats["motion"] / total * 100,
            "interpolated": self.stats["interpolated"],
            "duplicated": self.stats["duplicated"],
        }
