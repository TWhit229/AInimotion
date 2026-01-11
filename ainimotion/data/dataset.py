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
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        augment: bool = True,
        crop_size: tuple[int, int] = (256, 256),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.crop_size = crop_size
        
        # Find all triplet directories (support both PNG and JPEG)
        self.triplet_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (
                (d / "f1.png").exists() or (d / "f1.jpg").exists()
            )
        ])
        
        if len(self.triplet_dirs) == 0:
            raise ValueError(f"No triplets found in {root_dir}")
    
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
        triplet_dir = self.triplet_dirs[idx]
        
        # Load images (auto-detect PNG or JPEG)
        ext = "png" if (triplet_dir / "f1.png").exists() else "jpg"
        f1 = self._load_image(triplet_dir / f"f1.{ext}")
        f2 = self._load_image(triplet_dir / f"f2.{ext}")
        f3 = self._load_image(triplet_dir / f"f3.{ext}")
        
        # Apply augmentations
        if self.augment:
            f1, f2, f3 = self._apply_augmentation(f1, f2, f3)
        
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
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TripletDataset(
        root_dir=root_dir,
        augment=augment,
        crop_size=crop_size,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
