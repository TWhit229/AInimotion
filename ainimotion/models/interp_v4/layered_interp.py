"""
LayeredInterpolatorV4 - Main model with Motion Complexity Routing.

v4 improvements over v3:
  1. Motion Complexity Router — per-pixel routing between warp and synthesis
  2. Deformable Cross-Attention Synthesis Decoder — generates content for
     non-rigid motion regions where pixel warping fails
  3. Edge-Guided Synthesis — Sobel edge maps as structural guides for the
     synthesis path, preserving anime line art
  4. All v3 features: vectorized AdaCoF, swap-input flow, timestep conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FeaturePyramid
from .scene_gate import SceneGate
from .background_flow import BackgroundFlowNet
from .foreground_flow import AdaCoFNet
from .compositor import Compositor
from .edge_extractor import EdgeExtractor
from .motion_router import MotionComplexityRouter
from .synthesis_decoder import SynthesisDecoder


class LayeredInterpolatorV4(nn.Module):
    """
    Layer-Separated Deformable Interpolation Model with
    Motion Complexity Routing (v4).
    
    Architecture:
        1. Feature Pyramid Network extracts multi-scale features
        2. Scene Gate detects hard cuts (returns frame1 if detected)
        3. Edge Extractor produces Sobel edge maps for both frames
        4. Background Flow estimates camera/background motion (bidirectional)
        5. Motion Complexity Router predicts per-pixel routing map
        6. Two foreground paths run in parallel:
           a. AdaCoF (warp path) — sharp, for simple translational motion
           b. Synthesis Decoder (attention path) — flexible, for non-rigid motion
        7. Compositor routes and blends all layers into final output
    
    Args:
        base_channels: Base feature channels for FPN
        kernel_size: AdaCoF kernel size
        grid_size: Background affine grid size
        use_refinement: Whether to use refinement U-Net in compositor
        max_displacement: Correlation volume displacement
        n_attn_layers: Number of deformable attention layers in synthesis decoder
        n_attn_points: Number of attention sampling points per pixel
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        kernel_size: int = 5,
        grid_size: int = 8,
        use_refinement: bool = True,
        max_displacement: int = 6,
        n_attn_layers: int = 1,
        n_attn_points: int = 9,
    ):
        super().__init__()
        
        # Feature extraction (shared, from v3)
        self.fpn = FeaturePyramid(
            base_channels=base_channels,
            max_displacement=max_displacement,
        )
        
        feat_channels = self.fpn.out_channels
        corr_channels = self.fpn.corr_channels
        
        # Scene cut detection
        self.scene_gate = SceneGate(corr_channels=corr_channels)
        
        # Edge extraction (v4: anime line art)
        self.edge_extractor = EdgeExtractor(normalize=True)
        
        # Background path (from v3)
        self.background_net = BackgroundFlowNet(
            feat_channels=feat_channels,
            grid_size=grid_size,
            corr_channels=corr_channels,
        )
        
        # Foreground path A: AdaCoF warp (from v3)
        self.foreground_net = AdaCoFNet(
            feat_channels=feat_channels,
            kernel_size=kernel_size,
            corr_channels=corr_channels,
        )
        
        # v4: Motion Complexity Router (operates at scale 1 for VRAM)
        self.motion_router = MotionComplexityRouter(
            corr_channels=corr_channels,
            feat_channels=feat_channels[1],  # Scale 1 channels
        )
        
        # v4: Foreground path B: Synthesis Decoder (scale 1 = 1/4 res)
        self.synthesis_decoder = SynthesisDecoder(
            feat_channels=feat_channels[1],  # Scale 1 channels
            n_attn_layers=n_attn_layers,
            n_points=n_attn_points,
        )
        
        # Layer compositor (v4: MCR-aware)
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
                - 'foreground': (B, 3, H, W) routed foreground
                - 'fg_warp': (B, 3, H, W) AdaCoF output
                - 'fg_synth': (B, 3, H, W) synthesis decoder output
                - 'routing_map': (B, 1, H, W) motion complexity map
                - 'edge_map': (B, 1, H, W) combined edge map
                - 'bg_flow_fwd': (B, 2, H, W) forward background flow
                - 'bg_flow_bwd': (B, 2, H, W) backward background flow
        """
        b, c, h, w = frame1.shape
        
        # 1. Extract features
        fpn_output = self.fpn(frame1, frame2)
        feat1 = fpn_output['feat1']
        feat2 = fpn_output['feat2']
        corr = fpn_output['corr']
        
        # 2. Scene cut detection
        is_scene_cut, confidence = self.scene_gate(corr, return_confidence=True)
        
        # 3. Edge extraction (v4)
        edge1 = self.edge_extractor(frame1)  # (B, 1, H, W)
        edge2 = self.edge_extractor(frame2)
        # Combined edge map (average for refinement context)
        edge_map = (edge1 + edge2) / 2
        
        # 4. Background path
        bg_output = self.background_net(feat1, feat2, corr, target_size=(h, w))
        flow_fwd = bg_output['flow_fwd']
        flow_bwd = bg_output['flow_bwd']
        
        background = self.background_net.stitch_background(
            frame1, frame2,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            timestep=timestep,
        )
        
        # 5. Foreground path A: AdaCoF warp (v3)
        fg_warp_output = self.foreground_net(
            frame1, frame2,
            feat1, feat2, corr,
            timestep=timestep,
        )
        fg_warp = fg_warp_output['output']
        
        # 6. Motion Complexity Router (v4, scale 1 for VRAM)
        routing_map = self.motion_router(
            corr[1], feat1[1], feat2[1],
            target_size=(h, w),
        )
        
        # 7. Foreground path B: Synthesis Decoder (v4, scale 1)
        fg_synth = self.synthesis_decoder(
            feat1[1], feat2[1],
            flow_fwd, flow_bwd,
            edge1, edge2,
            timestep=timestep,
        )
        
        # 8. Composite with routing
        comp_output = self.compositor(
            background=background,
            fg_warp=fg_warp,
            fg_synth=fg_synth,
            routing_map=routing_map,
            edge_map=edge_map,
            feat1=feat1,
            feat2=feat2,
            corr=corr,
            frame1=frame1,
            frame2=frame2,
        )
        
        output = comp_output['output']
        alpha = comp_output['alpha']
        foreground = comp_output['foreground']
        
        # Handle scene cuts
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
            'fg_warp': fg_warp,
            'fg_synth': fg_synth,
            'routing_map': routing_map,
            'edge_map': edge_map,
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
        
        Args:
            frame1: (B, 3, H, W) first frame
            frame2: (B, 3, H, W) second frame
            num_frames: Number of intermediate frames
            
        Returns:
            List of interpolated frame tensors
        """
        timesteps = [i / (num_frames + 1) for i in range(1, num_frames + 1)]
        results = []
        for t in timesteps:
            out = self.forward(frame1, frame2, timestep=t)
            results.append(out['output'])
        return results


def build_model(
    config: dict | None = None,
    pretrained: str | None = None,
) -> LayeredInterpolatorV4:
    """
    Build a LayeredInterpolatorV4 model.
    
    Args:
        config: Optional configuration dict
        pretrained: Optional path to pretrained weights
        
    Returns:
        Initialized model
    """
    config = config or {}
    
    model = LayeredInterpolatorV4(
        base_channels=config.get('base_channels', 32),
        kernel_size=config.get('kernel_size', 5),
        grid_size=config.get('grid_size', 8),
        use_refinement=config.get('use_refinement', True),
        max_displacement=config.get('max_displacement', 6),
        n_attn_layers=config.get('n_attn_layers', 2),
        n_attn_points=config.get('n_attn_points', 16),
    )
    
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        if 'generator_state_dict' in checkpoint:
            state = checkpoint['generator_state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint
        
        # Load with strict=False to allow new v4 components
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [v4] New components (not in pretrained): {len(missing)} params")
        if unexpected:
            print(f"  [v4] Unexpected keys: {len(unexpected)}")
    
    return model


# Quick test
if __name__ == '__main__':
    model = LayeredInterpolatorV4(base_channels=16, kernel_size=3, grid_size=4)
    
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    
    with torch.no_grad():
        # Test t=0.5
        out = model(x1, x2)
        print(f"Output shape: {out['output'].shape}")
        print(f"Routing map: min={out['routing_map'].min():.3f}, max={out['routing_map'].max():.3f}")
        print(f"Edge map shape: {out['edge_map'].shape}")
        print(f"FG warp shape: {out['fg_warp'].shape}")
        print(f"FG synth shape: {out['fg_synth'].shape}")
        
        # Test different timesteps
        out_025 = model(x1, x2, timestep=0.25)
        out_075 = model(x1, x2, timestep=0.75)
        print(f"t=0.25: {out_025['output'].shape}, t=0.75: {out_075['output'].shape}")
        
        # Test multi-frame
        multi = model.interpolate_multi(x1, x2, num_frames=3)
        print(f"Multi-frame: {len(multi)} frames, shape {multi[0].shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
