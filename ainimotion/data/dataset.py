"""
PyTorch Dataset for loading training triplets.

Loads (F1, F3) as input and F2 as target for VFI training.
"""

import random
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class TripletDataset(Dataset):
    """
    Dataset for loading VFI training triplets.
    
    Each sample is a directory containing:
        - f1.png: First frame (input)
        - f2.png: Middle frame (target)
        - f3.png: Third frame (input)
    
    Args:
        root_dir: Path to directory containing triplet subdirectories
        transform: Optional transform to apply to all frames
        augment: Whether to apply training augmentations (crop, flip, etc.)
        crop_size: Size of random crops during augmentation (H, W)
        max_samples: Maximum number of triplets to use (None = use all)
        difficulty_file: Path to difficulty_scores.json (from precompute_difficulty.py)
        min_motion: Minimum motion threshold (exclude trivial samples)
        max_motion: Maximum motion threshold (exclude extreme/scene-cut samples)
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        augment: bool = True,
        crop_size: tuple[int, int] = (256, 256),
        max_samples: int | None = None,
        temporal_augment: bool = True,  # Swap frame1/frame3 for 2x data
        difficulty_file: str | Path | None = None,
        min_motion: float = 0.0,
        max_motion: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.crop_size = crop_size
        self.temporal_augment = temporal_augment
        
        # Find all triplet directories (support both naming conventions and formats)
        # Our format: f1.jpg/f1.png  |  ATD-12K format: frame1.jpg/frame1.png
        all_triplet_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (
                (d / "f1.png").exists() or (d / "f1.jpg").exists() or
                (d / "frame1.png").exists() or (d / "frame1.jpg").exists()
            )
        ])
        
        if len(all_triplet_dirs) == 0:
            raise ValueError(f"No triplets found in {root_dir}")
        
        total_before = len(all_triplet_dirs)
        
        # Filter by difficulty if scores are available
        if difficulty_file is not None:
            difficulty_file = Path(difficulty_file)
            if difficulty_file.exists():
                import json
                with open(difficulty_file) as f:
                    self.difficulty_scores = json.load(f)
                
                # Filter by motion range
                filtered = []
                for d in all_triplet_dirs:
                    score = self.difficulty_scores.get(d.name, None)
                    if score is not None and min_motion <= score <= max_motion:
                        filtered.append(d)
                
                skipped = total_before - len(filtered)
                print(
                    f"Difficulty filter [{min_motion:.3f}, {max_motion:.3f}]: "
                    f"{len(filtered):,} kept, {skipped:,} filtered "
                    f"({skipped/total_before*100:.1f}% removed)"
                )
                all_triplet_dirs = filtered
            else:
                print(f"WARNING: difficulty_file not found: {difficulty_file}")
                self.difficulty_scores = {}
        else:
            self.difficulty_scores = {}
        
        # Randomly sample if max_samples is set
        if max_samples is not None and max_samples < len(all_triplet_dirs):
            random.seed(42)  # Reproducible sampling
            self.triplet_dirs = random.sample(all_triplet_dirs, max_samples)
            self.triplet_dirs.sort()  # Keep sorted for consistency
            print(f"Sampled {max_samples:,} triplets from {len(all_triplet_dirs):,} total")
        else:
            self.triplet_dirs = all_triplet_dirs
    
    def __len__(self) -> int:
        return len(self.triplet_dirs)
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load an image as RGB PIL Image."""
        return Image.open(path).convert("RGB")
    
    def _apply_augmentation(
        self,
        f1: Image.Image,
        f2: Image.Image,
        f3: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """
        Apply synchronized augmentations to all three frames.
        
        Augmentations:
            - Random crop
            - Horizontal flip (50% chance)
            - Vertical flip (50% chance)
            - Temporal flip (50% chance) - swaps F1 and F3
        """
        # Get dimensions
        w, h = f1.size
        crop_h, crop_w = self.crop_size
        
        # Random crop (same location for all frames)
        if h > crop_h and w > crop_w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            
            f1 = TF.crop(f1, top, left, crop_h, crop_w)
            f2 = TF.crop(f2, top, left, crop_h, crop_w)
            f3 = TF.crop(f3, top, left, crop_h, crop_w)
        
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            f1 = TF.hflip(f1)
            f2 = TF.hflip(f2)
            f3 = TF.hflip(f3)
        
        # Vertical flip (50% chance)
        if random.random() < 0.5:
            f1 = TF.vflip(f1)
            f2 = TF.vflip(f2)
            f3 = TF.vflip(f3)
        
        # Temporal flip (50% chance) - swap F1 and F3
        # This teaches the model symmetry in time
        if random.random() < 0.5:
            f1, f3 = f3, f1
        
        return f1, f2, f3
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor."""
        tensor = TF.to_tensor(img)  # [0, 1] range
        return tensor
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - 'frame1': (C, H, W) tensor, first frame (input)
                - 'frame3': (C, H, W) tensor, third frame (input)  
                - 'frame2': (C, H, W) tensor, middle frame (target)
                - 'inputs': (2, C, H, W) tensor, stacked inputs
        """
        # Try to load the triplet, skip corrupted ones
        try:
            triplet_dir = self.triplet_dirs[idx]
            
            # Load images (auto-detect naming convention and format)
            # Our format: f1.jpg/f1.png  |  ATD-12K format: frame1.jpg/frame1.png
            if (triplet_dir / "f1.png").exists():
                f1 = self._load_image(triplet_dir / "f1.png")
                f2 = self._load_image(triplet_dir / "f2.png")
                f3 = self._load_image(triplet_dir / "f3.png")
            elif (triplet_dir / "f1.jpg").exists():
                f1 = self._load_image(triplet_dir / "f1.jpg")
                f2 = self._load_image(triplet_dir / "f2.jpg")
                f3 = self._load_image(triplet_dir / "f3.jpg")
            elif (triplet_dir / "frame1.png").exists():
                f1 = self._load_image(triplet_dir / "frame1.png")
                f2 = self._load_image(triplet_dir / "frame2.png")
                f3 = self._load_image(triplet_dir / "frame3.png")
            else:  # frame1.jpg (ATD-12K train set)
                f1 = self._load_image(triplet_dir / "frame1.jpg")
                f2 = self._load_image(triplet_dir / "frame2.jpg")
                f3 = self._load_image(triplet_dir / "frame3.jpg")
            
        except Exception as e:
            # If corrupted, return a random different sample
            new_idx = random.randint(0, len(self) - 1)
            if new_idx == idx:
                new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        
        # Apply augmentations
        if self.augment:
            f1, f2, f3 = self._apply_augmentation(f1, f2, f3)
        
        # Temporal augmentation: randomly swap frame1 and frame3
        # This effectively 2x the training data since motion is bidirectional
        if self.temporal_augment and random.random() > 0.5:
            f1, f3 = f3, f1  # Swap! Middle frame (target) stays the same
        
        # Apply custom transform if provided
        if self.transform is not None:
            f1 = self.transform(f1)
            f2 = self.transform(f2)
            f3 = self.transform(f3)
        
        # Convert to tensors
        f1_t = self._to_tensor(f1)
        f2_t = self._to_tensor(f2)
        f3_t = self._to_tensor(f3)
        
        return {
            "frame1": f1_t,
            "frame2": f2_t,  # Target
            "frame3": f3_t,
            "inputs": torch.stack([f1_t, f3_t], dim=0),  # (2, C, H, W)
        }


class InferenceDataset(Dataset):
    """
    Dataset for inference on sequential frames.
    
    Takes a list of frame paths and returns consecutive pairs
    for interpolation.
    
    Args:
        frame_paths: List of paths to sequential frames
        transform: Optional transform to apply
    """
    
    def __init__(
        self,
        frame_paths: list[Path],
        transform: Callable | None = None,
    ):
        self.frame_paths = frame_paths
        self.transform = transform
    
    def __len__(self) -> int:
        # Number of interpolatable pairs
        return max(0, len(self.frame_paths) - 1)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """
        Get a frame pair for interpolation.
        
        Returns:
            Dictionary with:
                - 'frame1': (C, H, W) tensor
                - 'frame2': (C, H, W) tensor
                - 'idx': Index of the pair
        """
        f1_path = self.frame_paths[idx]
        f2_path = self.frame_paths[idx + 1]
        
        f1 = Image.open(f1_path).convert("RGB")
        f2 = Image.open(f2_path).convert("RGB")
        
        if self.transform is not None:
            f1 = self.transform(f1)
            f2 = self.transform(f2)
        
        f1_t = TF.to_tensor(f1)
        f2_t = TF.to_tensor(f2)
        
        return {
            "frame1": f1_t,
            "frame2": f2_t,
            "inputs": torch.stack([f1_t, f2_t], dim=0),
            "idx": idx,
        }


def create_dataloader(
    root_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = True,
    crop_size: tuple[int, int] = (256, 256),
    shuffle: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    max_samples: int | None = None,
    difficulty_file: str | Path | None = None,
    min_motion: float = 0.0,
    max_motion: float = 1.0,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        root_dir: Path to triplet directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply augmentations
        crop_size: Size of random crops
        shuffle: Whether to shuffle data
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs (faster)
        max_samples: Maximum number of triplets to use (None = use all)
        difficulty_file: Path to difficulty_scores.json
        min_motion: Min motion threshold for filtering
        max_motion: Max motion threshold for filtering
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TripletDataset(
        root_dir=root_dir,
        augment=augment,
        crop_size=crop_size,
        max_samples=max_samples,
        difficulty_file=difficulty_file,
        min_motion=min_motion,
        max_motion=max_motion,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
    )

