"""
LayeredInterpolator - Main model combining all components.

Orchestrates: FPN → Scene Gate → Background/Foreground paths → Compositor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeaturePyramid
from .scene_gate import SceneGate
from .background_flow import BackgroundFlowNet
from .foreground_flow import AdaCoFNet
from .compositor import Compositor


class LayeredInterpolator(nn.Module):
    """
    Layer-Separated Deformable Interpolation Model.
    
    Architecture:
        1. Feature Pyramid Network extracts multi-scale features
        2. Scene Gate detects hard cuts (returns frame1 if detected)
        3. Background path: 8×8 affine grid for camera motion
        4. Foreground path: AdaCoF kernels for character motion
        5. Compositor: soft alpha blending + refinement
    
    Args:
        base_channels: Base feature channels for FPN
        kernel_size: AdaCoF kernel size (K)
        grid_size: Background affine grid size
        use_refinement: Whether to use refinement network
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        kernel_size: int = 7,
        grid_size: int = 8,
        use_refinement: bool = True,
    ):
        super().__init__()
        
        # Feature extraction
        self.fpn = FeaturePyramid(in_channels=3, base_channels=base_channels)
        feat_channels = self.fpn.out_channels
        
        # Scene gate
        self.scene_gate = SceneGate(corr_channels=81, threshold=0.15)
        
        # Background motion (rigid)
        self.background_net = BackgroundFlowNet(
            feat_channels=feat_channels,
            grid_size=grid_size,
        )
        
        # Foreground motion (deformable)
        self.foreground_net = AdaCoFNet(
            feat_channels=feat_channels,
            kernel_size=kernel_size,
        )
        
        # Layer compositing
        self.compositor = Compositor(
            feat_channels=feat_channels,
            use_refinement=use_refinement,
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Interpolate a frame between frame1 and frame2.
        
        Args:
            frame1: (B, 3, H, W) first frame, values in [0, 1]
            frame2: (B, 3, H, W) second frame, values in [0, 1]
            timestep: Interpolation time, 0.0 = frame1, 1.0 = frame2
            
        Returns:
            Dictionary with:
                - 'output': (B, 3, H, W) interpolated frame
                - 'alpha': (B, 1, H, W) foreground/background mask
                - 'is_scene_cut': (B,) boolean tensor
                - 'background': (B, 3, H, W) background canvas
                - 'foreground': (B, 3, H, W) foreground synthesis
        """
        b, c, h, w = frame1.shape
        
        # Extract features
        fpn_output = self.fpn(frame1, frame2)
        feat1 = fpn_output['feat1']
        feat2 = fpn_output['feat2']
        corr = fpn_output['corr']
        
        # Check for scene cuts
        is_scene_cut, confidence = self.scene_gate(corr, return_confidence=True)
        
        # Background path: rigid motion
        bg_output = self.background_net(feat1, feat2, corr, target_size=(h, w))
        bg_flow = bg_output['flow']
        
        # Create background canvas by stitching from both frames
        # Flow at t=0.5 means warp each frame by half
        background = self.background_net.stitch_background(
            frame1, frame2,
            flow_1to2=bg_flow,
            flow_2to1=-bg_flow,  # Approximate reverse flow
        )
        
        # Foreground path: deformable motion
        fg_output = self.foreground_net(
            frame1, frame2,
            feat1, feat2, corr,
        )
        foreground = fg_output['output']
        
        # Composite layers
        comp_output = self.compositor(
            background=background,
            foreground=foreground,
            feat1=feat1,
            feat2=feat2,
            corr=corr,
            frame1=frame1,
            frame2=frame2,
        )
        
        output = comp_output['output']
        alpha = comp_output['alpha']
        
        # Handle scene cuts: return frame1 instead of interpolated
        if is_scene_cut.any():
            # Expand for broadcasting
            mask = is_scene_cut.view(b, 1, 1, 1).expand_as(output)
            output = torch.where(mask, frame1, output)
        
        return {
            'output': output,
            'alpha': alpha,
            'is_scene_cut': is_scene_cut,
            'scene_confidence': confidence,
            'background': background,
            'foreground': foreground,
            'bg_flow': bg_flow,
        }
    
    def interpolate_multi(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_frames: int = 1,
    ) -> list[torch.Tensor]:
        """
        Interpolate multiple frames between frame1 and frame2.
        
        Args:
            frame1: (B, 3, H, W) first frame
            frame2: (B, 3, H, W) second frame
            num_frames: Number of intermediate frames to generate
            
        Returns:
            List of (B, 3, H, W) interpolated frames
        """
        results = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            output = self.forward(frame1, frame2, timestep=t)
            results.append(output['output'])
        return results


def build_model(
    config: dict | None = None,
    pretrained: str | None = None,
) -> LayeredInterpolator:
    """
    Build a LayeredInterpolator model.
    
    Args:
        config: Optional configuration dict
        pretrained: Optional path to pretrained weights
        
    Returns:
        Initialized model
    """
    if config is None:
        config = {}
    
    model = LayeredInterpolator(
        base_channels=config.get('base_channels', 32),
        kernel_size=config.get('kernel_size', 7),
        grid_size=config.get('grid_size', 8),
        use_refinement=config.get('use_refinement', True),
    )
    
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


# Quick test
if __name__ == '__main__':
    # Test forward pass
    model = LayeredInterpolator()
    
    # Random input
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x1, x2)
    
    print("Input shape:", x1.shape)
    print("Output shape:", output['output'].shape)
    print("Alpha shape:", output['alpha'].shape)
    print("Scene cut:", output['is_scene_cut'])
    print("Confidence:", output['scene_confidence'])
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
