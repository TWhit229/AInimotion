"""
LayeredInterpolatorV5 - Multi-Frame Anime Interpolation.

V5 architecture takes 7 frames as context (+-3 from anchor pair)
and uses temporal attention to borrow pixels across time,
solving the blurring problem from V1-V4.

Pipeline:
  1. Shared Spatial Encoder (per-frame features at 3 scales)
  2. Temporal Attention Fusion (cross-frame at 1/4 res)
  3. Scene Gate (hard cut detection)
  4. Background Path (affine trajectory + occlusion-aware blend)
  5. Foreground Warp Branch (RIFE-style coarse-to-fine)
  6. Foreground Synthesis Branch (deformable cross-temporal attention)
  7. Motion Router (per-pixel warp vs synthesis)
  8. Layer Compositor (learned alpha blend)
  9. Edge-Guided Refinement (line art sharpening)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_encoder import SpatialEncoder
from .temporal_attention import TemporalFusion, compute_correlation
from .scene_gate import SceneGate
from .background_path import BackgroundPath
from .warp_branch import WarpBranch
from .synthesis_branch import SynthesisBranch, compute_sobel_edges
from .motion_router import MotionRouter
from .compositor import LayerCompositor, EdgeGuidedRefinement


class LayeredInterpolatorV5(nn.Module):
    """
    Multi-Frame Layered Interpolation Model (V5).
    
    Takes 7 context frames and produces an interpolated frame between
    the two anchor frames (idx 3 and 4 in the 7-frame window).
    
    Args:
        base_channels: Base feature channels for spatial encoder (default: 64)
        n_temporal_layers: Temporal attention layers (default: 2)
        n_attn_heads: Temporal attention heads (default: 4)
        window_size: Windowed attention spatial window (default: 8)
        n_synth_layers: Synthesis branch attention layers (default: 2)
        n_synth_points: Synthesis branch sampling points (default: 9)
        max_offset: Maximum deformable offset in pixels at 1/4 res (default: 32)
    """
    
    def __init__(
        self,
        base_channels: int = 64,
        n_temporal_layers: int = 2,
        n_attn_heads: int = 4,
        window_size: int = 8,
        n_synth_layers: int = 2,
        n_synth_points: int = 9,
        max_offset: int = 32,
    ):
        super().__init__()
        C = base_channels
        
        # 1. Shared spatial encoder
        self.encoder = SpatialEncoder(base_channels=C)
        feat_ch = self.encoder.out_channels  # [C, 2C, 4C]
        
        # 2. Temporal attention fusion at 1/4 resolution
        self.temporal_fusion = TemporalFusion(
            channels=feat_ch[2],  # 4C at 1/4 res
            n_layers=n_temporal_layers,
            n_heads=n_attn_heads,
            window_size=window_size,
        )
        
        # 3. Scene gate
        self.scene_gate = SceneGate()
        
        # 4. Background path
        self.background_path = BackgroundPath(
            feat_channels=feat_ch[2],
        )
        
        # 5. Warp branch (operates on raw frames, self-contained)
        self.warp_branch = WarpBranch()
        
        # 6. Synthesis branch
        self.synthesis_branch = SynthesisBranch(
            feat_channels=feat_ch[2],
            n_attn_layers=n_synth_layers,
            n_points=n_synth_points,
            max_offset=max_offset,
        )
        
        # 7. Motion router
        self.motion_router = MotionRouter(feat_channels=feat_ch[2])
        
        # 8. Layer compositor
        self.compositor = LayerCompositor(feat_channels=feat_ch[2])
        
        # 9. Edge-guided refinement
        self.refinement = EdgeGuidedRefinement()
    
    @property
    def n_context_frames(self) -> int:
        """Number of context frames expected (7 = 3 past + 2 anchor + 2 future + 1)."""
        return 7
    
    def forward(
        self,
        frames: list[torch.Tensor] | torch.Tensor,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Interpolate between anchor frames using multi-frame context.
        
        Args:
            frames: Either:
              - List of 7 tensors, each (B, 3, H, W)
              - Single tensor (B, 7, 3, H, W) or (B, 21, H, W)
            timestep: Interpolation time (0.0 = frame_i, 1.0 = frame_i+1)
            
        Returns:
            Dictionary with:
              - 'output': (B, 3, H, W) interpolated frame
              - 'alpha': (B, 1, H, W) foreground/background mask
              - 'is_scene_cut': (B,) boolean tensor
              - 'background': (B, 3, H, W) background estimate
              - 'fg_warp': (B, 3, H, W) warp branch output
              - 'fg_synth': (B, 3, H, W) synthesis branch output
              - 'routing_map': (B, 1, H, W) motion complexity map
              - 'edge_map': (B, 1, H, W) combined edge map
        """
        # Parse input
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 5:  # (B, 7, 3, H, W)
                frames = [frames[:, i] for i in range(frames.shape[1])]
            elif frames.dim() == 4:  # (B, 21, H, W)
                frames = [frames[:, i*3:(i+1)*3] for i in range(7)]
        
        assert len(frames) == 7, f"Expected 7 frames, got {len(frames)}"
        
        B, _, H, W = frames[0].shape
        
        # Anchor frames (center pair)
        frame_i = frames[3]   # F_i
        frame_ip1 = frames[4]  # F_{i+1}
        
        # ============= 1. Spatial Encoding (all 7 frames) =============
        # Encode each frame independently with shared weights
        all_feats = [self.encoder(f) for f in frames]
        # all_feats[t] = [scale0, scale1, scale2] for frame t
        
        # Extract 1/4-res features for temporal fusion
        feats_quarter = [feats[2] for feats in all_feats]  # list of (B, 4C, H/4, W/4)
        
        # ============= 2. Temporal Attention Fusion =============
        fused_feats = self.temporal_fusion(feats_quarter)
        # fused_feats: list of 7 tensors (B, 4C, H/4, W/4)
        
        # Anchor features (after temporal fusion)
        feat_i = fused_feats[3]
        feat_ip1 = fused_feats[4]
        
        # ============= 3. Scene Gate =============
        corr = compute_correlation(feat_i, feat_ip1)  # (B, 1, H/4, W/4)
        is_scene_cut, confidence = self.scene_gate(corr, return_confidence=True)
        
        # ============= 4. Edge Extraction =============
        edge_i = compute_sobel_edges(frame_i)
        edge_ip1 = compute_sobel_edges(frame_ip1)
        edge_map = (edge_i + edge_ip1) / 2
        
        # ============= 5. Background Path =============
        bg_output = self.background_path(
            frame_i, frame_ip1, feat_i, feat_ip1, timestep=timestep
        )
        background = bg_output['background']
        
        # ============= 6. Foreground Warp Branch =============
        warp_output = self.warp_branch(
            frame_i, frame_ip1, 
            feat1=feat_i, feat2=feat_ip1, 
            timestep=timestep
        )
        fg_warp = warp_output['output']
        
        # ============= 7. Motion Router =============
        routing_map = self.motion_router(
            corr, feat_i, feat_ip1, target_size=(H, W)
        )
        
        # ============= 8. Foreground Synthesis Branch =============
        # Mean of anchor features as query
        anchor_feat = (feat_i + feat_ip1) / 2
        fg_synth = self.synthesis_branch(
            anchor_feat, fused_feats, edge_i, edge_ip1
        )
        
        # ============= 9. Route Foreground =============
        foreground = (1 - routing_map) * fg_warp + routing_map * fg_synth
        
        # ============= 10. Composite Layers =============
        comp_output = self.compositor(
            foreground, background, frame_i, frame_ip1, edge_map
        )
        composite = comp_output['composite']
        alpha = comp_output['alpha']
        
        # ============= 11. Edge-Guided Refinement =============
        output = self.refinement(composite, edge_i, edge_ip1)
        
        # ============= 12. Handle Scene Cuts =============
        # Use unconditional soft gating to avoid data-dependent control flow
        # that would break torch.compile and block gradient flow
        scene_mask = confidence.view(B, 1, 1, 1)
        output = (1 - scene_mask) * output + scene_mask * frame_i
        
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
            'flow_fwd': warp_output['flow_fwd'],
            'flow_bwd': warp_output['flow_bwd'],
        }


def build_model(
    config: dict | None = None,
    pretrained: str | None = None,
) -> LayeredInterpolatorV5:
    """
    Build a LayeredInterpolatorV5 model.
    
    Args:
        config: Optional configuration dict with keys:
          - base_channels (int, default: 64)
          - n_temporal_layers (int, default: 2)
          - n_attn_heads (int, default: 4)
          - window_size (int, default: 8)
          - n_synth_layers (int, default: 2)
          - n_synth_points (int, default: 9)
          - max_offset (int, default: 32)
        pretrained: Optional path to checkpoint
        
    Returns:
        Initialized model
    """
    config = config or {}
    
    model = LayeredInterpolatorV5(
        base_channels=config.get('base_channels', 64),
        n_temporal_layers=config.get('n_temporal_layers', 2),
        n_attn_heads=config.get('n_attn_heads', 4),
        window_size=config.get('window_size', 8),
        n_synth_layers=config.get('n_synth_layers', 2),
        n_synth_points=config.get('n_synth_points', 9),
        max_offset=config.get('max_offset', 32),
    )
    
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        if 'generator_state_dict' in checkpoint:
            state = checkpoint['generator_state_dict']
        elif 'generator' in checkpoint:
            state = checkpoint['generator']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint
        
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [v5] Missing keys (random init): {len(missing)} params")
            for k in missing[:5]:
                print(f"        - {k}")
        if unexpected:
            print(f"  [v5] Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"        - {k}")
    
    return model


# Quick test
if __name__ == '__main__':
    model = LayeredInterpolatorV5(base_channels=32)  # smaller for testing
    
    B = 1
    frames = [torch.randn(B, 3, 64, 64) for _ in range(7)]
    
    with torch.no_grad():
        out = model(frames)
        print(f"Output: {out['output'].shape}")
        print(f"Background: {out['background'].shape}")
        print(f"FG Warp: {out['fg_warp'].shape}")
        print(f"FG Synth: {out['fg_synth'].shape}")
        print(f"Routing: min={out['routing_map'].min():.3f}, max={out['routing_map'].max():.3f}")
        print(f"Alpha: {out['alpha'].shape}")
        print(f"Scene cut: {out['is_scene_cut']}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # Test with tensor input
    frames_tensor = torch.stack(frames, dim=1)  # (B, 7, 3, 64, 64)
    with torch.no_grad():
        out2 = model(frames_tensor)
        print(f"\nTensor input test: {out2['output'].shape}")
