"""
LayeredInterpolatorV3 - Main model combining all v3 components.

v3 improvements over v2:
  1. Vectorized AdaCoF sampler (single grid_sample, ~5x faster)
  2. Swap-input reverse flow (no naive -flow negation)
  3. Lightweight flow residual refinement
  4. Timestep conditioning (arbitrary-time interpolation)
  5. Configurable correlation displacement (default d=6, 169 channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeaturePyramid
from .scene_gate import SceneGate
from .background_flow import BackgroundFlowNet
from .foreground_flow import AdaCoFNet
from .compositor import Compositor


class LayeredInterpolatorV3(nn.Module):
    """
    Layer-Separated Deformable Interpolation Model (v3).
    
    Architecture:
        1. Feature Pyramid Network extracts multi-scale features
        2. Scene Gate detects hard cuts (returns frame1 if detected)
        3. Background path: 8×8 affine grid with swap-input reverse flow
        4. Foreground path: Vectorized AdaCoF with timestep conditioning
        5. Compositor: soft alpha blending + refinement
    
    Args:
        base_channels: Base feature channels for FPN
        kernel_size: AdaCoF kernel size (K)
        grid_size: Background affine grid size
        use_refinement: Whether to use refinement network
        max_displacement: Correlation volume displacement (default 6)
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        kernel_size: int = 5,
        grid_size: int = 8,
        use_refinement: bool = True,
        max_displacement: int = 6,
    ):
        super().__init__()
        self.max_displacement = max_displacement
        
        # Feature extraction (v3: configurable displacement)
        self.fpn = FeaturePyramid(
            in_channels=3,
            base_channels=base_channels,
            max_displacement=max_displacement,
        )
        feat_channels = self.fpn.out_channels
        corr_channels = self.fpn.corr_channels
        
        # Scene gate (v3: updated corr_channels)
        self.scene_gate = SceneGate(
            corr_channels=corr_channels,
            threshold=0.15,
        )
        
        # Background motion (v3: swap-input + flow refiner)
        self.background_net = BackgroundFlowNet(
            feat_channels=feat_channels,
            grid_size=grid_size,
            corr_channels=corr_channels,
        )
        
        # Foreground motion (v3: vectorized AdaCoF + timestep)
        self.foreground_net = AdaCoFNet(
            feat_channels=feat_channels,
            kernel_size=kernel_size,
            corr_channels=corr_channels,
        )
        
        # Layer compositing (v3: updated corr_channels)
        self.compositor = Compositor(
            feat_channels=feat_channels,
            use_refinement=use_refinement,
            corr_channels=corr_channels,
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Interpolate a frame between frame1 and frame2.
        
        v3: timestep is actually used for arbitrary-time interpolation.
        
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
                - 'bg_flow_fwd': (B, 2, H, W) forward background flow
                - 'bg_flow_bwd': (B, 2, H, W) backward background flow
        """
        b, c, h, w = frame1.shape
        
        # Extract features
        fpn_output = self.fpn(frame1, frame2)
        feat1 = fpn_output['feat1']
        feat2 = fpn_output['feat2']
        corr = fpn_output['corr']
        
        # Check for scene cuts
        is_scene_cut, confidence = self.scene_gate(corr, return_confidence=True)
        
        # Background path: bidirectional flow (v3: swap-input, no negation)
        bg_output = self.background_net(feat1, feat2, corr, target_size=(h, w))
        flow_fwd = bg_output['flow_fwd']
        flow_bwd = bg_output['flow_bwd']
        
        # Stitch background (v3: timestep-aware warping)
        background = self.background_net.stitch_background(
            frame1, frame2,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            timestep=timestep,
        )
        
        # Foreground path: deformable motion (v3: vectorized + timestep)
        fg_output = self.foreground_net(
            frame1, frame2,
            feat1, feat2, corr,
            timestep=timestep,
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
            mask = is_scene_cut.view(b, 1, 1, 1).expand_as(output)
            output = torch.where(mask, frame1, output)
        
        return {
            'output': output,
            'alpha': alpha,
            'is_scene_cut': is_scene_cut,
            'scene_confidence': confidence,
            'background': background,
            'foreground': foreground,
            'bg_flow_fwd': flow_fwd,
            'bg_flow_bwd': flow_bwd,
        }
    
    def interpolate_multi(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_frames: int = 1,
    ) -> list[torch.Tensor]:
        """
        Interpolate multiple frames between frame1 and frame2.
        
        v3: Actually works now — timestep is wired through the network.
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
) -> LayeredInterpolatorV3:
    """
    Build a LayeredInterpolatorV3 model.
    
    Args:
        config: Optional configuration dict
        pretrained: Optional path to pretrained weights
        
    Returns:
        Initialized model
    """
    if config is None:
        config = {}
    
    model = LayeredInterpolatorV3(
        base_channels=config.get('base_channels', 32),
        kernel_size=config.get('kernel_size', 5),
        grid_size=config.get('grid_size', 8),
        use_refinement=config.get('use_refinement', True),
        max_displacement=config.get('max_displacement', 6),
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
    model = LayeredInterpolatorV3(base_channels=16, kernel_size=3, grid_size=4)
    
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    
    with torch.no_grad():
        # Test t=0.5
        output = model(x1, x2, timestep=0.5)
        print("Input shape:", x1.shape)
        print("Output shape:", output['output'].shape)
        print("Alpha shape:", output['alpha'].shape)
        print("BG flow fwd:", output['bg_flow_fwd'].shape)
        print("BG flow bwd:", output['bg_flow_bwd'].shape)
        
        # Test t=0.25 (arbitrary time)
        output2 = model(x1, x2, timestep=0.25)
        print("Output t=0.25:", output2['output'].shape)
        
        # Test multi-frame interpolation
        multi = model.interpolate_multi(x1, x2, num_frames=3)
        print(f"Multi-frame: {len(multi)} frames, shape {multi[0].shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
