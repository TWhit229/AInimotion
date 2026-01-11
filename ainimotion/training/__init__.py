"""Training infrastructure components."""

from .losses import VFILoss, PerceptualLoss, EdgeWeightedL1Loss
from .discriminator import PatchDiscriminator

__all__ = [
    "VFILoss",
    "PerceptualLoss", 
    "EdgeWeightedL1Loss",
    "PatchDiscriminator",
]
