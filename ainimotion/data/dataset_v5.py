"""
V5 Multi-Frame Dataset.

Loads 9-frame sequences from the extracted training data.
During training, the middle frame (index 4) is the ground truth,
and the 7 surrounding frames (indices 0-3, 5-8) are context.

wait - actually looking at the design more carefully:
We have 7 context frames and want to interpolate between
frames[3] and frames[4]. The GT would be the frame that
was between them in the original video.

So we extract 9 frames: 0,1,2,3, GT, 4,5,6,7
And remap to: context = [0,1,2,3,5,6,7,8], GT = 4

Features:
  - Random 256x256 crops (with random flip augmentation)
  - Temporal order augmentation (reverse sequence randomly)
  - Motion-weighted sampling (optional)
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class V5SequenceDataset(Dataset):
    """
    Dataset of 9-frame anime sequences for V5 training.
    
    Directory structure:
      root/
        series_ep01_seg000_seq000001_m0.0234/
          frame_000.png  (context: 3 before anchor)
          frame_001.png  (context: 2 before anchor)
          frame_002.png  (context: 1 before anchor)
          frame_003.png  (anchor F_i)
          frame_004.png  (GROUND TRUTH — model never sees this)
          frame_005.png  (anchor F_{i+1})
          frame_006.png  (context: 1 after anchor)
          frame_007.png  (context: 2 after anchor)
          frame_008.png  (unused — safety margin)
    
    The model receives 7 frames:
      [0, 1, 2, 3, 5, 6, 7]  →  remapped to model indices [0..6]
      
      Model anchors: model[3] = video frame 3 (F_i)
                      model[4] = video frame 5 (F_{i+1})
      GT: video frame 4 (between anchors)
    
    Args:
        root: Directory containing sequence subdirectories
        crop_size: Random crop size (default: 256)
        augment: Whether to apply augmentation (default: True)
    """
    
    def __init__(
        self,
        root: str,
        crop_size: int = 256,
        augment: bool = True,
    ):
        self.root = root
        self.crop_size = crop_size
        self.augment = augment
        
        # Find all sequence directories
        self.sequences = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith('_')
        ])
        
        # Parse motion scores from directory names for weighted sampling
        self.motion_scores = []
        for seq_dir in self.sequences:
            name = os.path.basename(seq_dir)
            try:
                # Format: series_ep01_seg000_seq000001_m0.0234
                m_score = float(name.split('_m')[-1])
            except (ValueError, IndexError):
                m_score = 0.01
            self.motion_scores.append(m_score)
        
        if not self.sequences:
            raise RuntimeError(f"No sequences found in {root}")
        
        print(f"V5Dataset: {len(self.sequences)} sequences from {root}")
        if self.motion_scores:
            print(f"  Motion scores: min={min(self.motion_scores):.4f}, "
                  f"max={max(self.motion_scores):.4f}, "
                  f"mean={sum(self.motion_scores)/len(self.motion_scores):.4f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def _load_frame(self, path: str) -> Image.Image:
        """Load a frame as PIL Image."""
        return Image.open(path).convert('RGB')
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_dir = self.sequences[idx]
        
        # Load all 9 frames
        frame_paths = sorted([
            os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
            if f.endswith('.png')
        ])
        
        if len(frame_paths) < 9:
            # Pad with duplicates if needed
            while len(frame_paths) < 9:
                frame_paths.append(frame_paths[-1])
        
        frames = [self._load_frame(fp) for fp in frame_paths[:9]]
        
        # Get common size
        w, h = frames[0].size
        
        # Random crop
        cs = self.crop_size
        if h > cs and w > cs:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top, left = 0, 0
            cs = min(h, w)
        
        frames = [TF.crop(f, top, left, cs, cs) for f in frames]
        
        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                frames = [TF.hflip(f) for f in frames]
            
            # Random vertical flip
            if random.random() > 0.5:
                frames = [TF.vflip(f) for f in frames]
            
            # Random temporal reversal
            if random.random() > 0.5:
                frames = frames[::-1]
                
            # Random rotation (0, 90, 180, 270)
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                frames = [TF.rotate(f, rot) for f in frames]
                
            # Color jitter (applied consistently to all frames in sequence)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue = random.uniform(-0.05, 0.05)
                frames = [TF.adjust_brightness(f, brightness) for f in frames]
                frames = [TF.adjust_contrast(f, contrast) for f in frames]
                frames = [TF.adjust_saturation(f, saturation) for f in frames]
                frames = [TF.adjust_hue(f, hue) for f in frames]
        
        # Convert to tensors [0, 1]
        frames = [TF.to_tensor(f) for f in frames]
        
        # Split: context (7 frames) and GT (1 frame)
        # Context: video indices [0,1,2,3, 5,6,7] → model indices [0..6]
        # GT: video index [4] (between anchor pair)
        # Frame 8 is unused (extraction safety margin)
        context = frames[:4] + frames[5:8]  # 7 frames
        gt = frames[4]
        
        return {
            'context': torch.stack(context, dim=0),  # (7, 3, H, W)
            'gt': gt,  # (3, H, W)
            'motion_score': self.motion_scores[idx],
        }
    
    def get_sampler_weights(self, epoch: int = 0) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler.
        Higher motion = higher weight.
        Uses curriculum learning (easy first, then hard).
        """
        scores = torch.tensor(self.motion_scores)
        difficulty_ramp = min(1.0, epoch / 200.0)
        
        base_weights = torch.sqrt(scores + 1e-6)
        max_score = base_weights.max()
        inv_weights = max_score - base_weights + 0.01
        
        weights = base_weights * difficulty_ramp + inv_weights * (1.0 - difficulty_ramp)
        
        # Normalize
        weights = weights / weights.sum() * len(weights)
        return weights


if __name__ == '__main__':
    import sys
    
    root = sys.argv[1] if len(sys.argv) > 1 else 'training_data/v5'
    
    if os.path.exists(root):
        ds = V5SequenceDataset(root, crop_size=256)
        sample = ds[0]
        print(f"Context: {sample['context'].shape}")
        print(f"GT: {sample['gt'].shape}")
        print(f"Motion: {sample['motion_score']:.4f}")
    else:
        print(f"Dataset root not found: {root}")
        print("Run extract_sequences.py first to create training data.")
