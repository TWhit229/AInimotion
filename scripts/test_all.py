"""Comprehensive test suite for AInimotion."""
import sys

def main():
    print("=" * 60)
    print("AInimotion Comprehensive Test Suite")
    print("=" * 60)

    # Test 1: Pipeline module imports
    print("\n[1/6] Testing pipeline module imports...")
    try:
        from ainimotion.pipeline import extract_frames, get_video_info, encode_frames, EnhancePipeline
        print("  OK: Pipeline module imports work")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test 2: Gating module imports
    print("\n[2/6] Testing gating module imports...")
    try:
        from ainimotion.gating import compute_ssim, compute_mad, classify_pair, FrameType, GatingProcessor
        print("  OK: Gating module imports work")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test 3: Model imports
    print("\n[3/6] Testing model imports...")
    try:
        from ainimotion.models.interp import LayeredInterpolator
        print("  OK: LayeredInterpolator imports")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test 4: SSIM/MAD with real frames
    print("\n[4/6] Testing similarity on real frames...")
    try:
        f1 = "D:/Triplets/triplet_00000002/f1.jpg"
        f2 = "D:/Triplets/triplet_00000002/f2.jpg"
        ssim = compute_ssim(f1, f2)
        mad = compute_mad(f1, f2)
        print(f"  SSIM: {ssim:.4f}")
        print(f"  MAD: {mad:.4f}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test 5: Classification
    print("\n[5/6] Testing frame classification...")
    try:
        decision = classify_pair(f1, f2)
        print(f"  Type: {decision.frame_type.value}")
        print(f"  Similarity: {decision.similarity:.4f}")
        print(f"  Should interpolate: {decision.should_interpolate()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test 6: Model forward pass
    print("\n[6/6] Testing model forward pass (CPU)...")
    try:
        import torch
        model = LayeredInterpolator(base_channels=16, kernel_size=5, grid_size=4)
        dummy_f1 = torch.randn(1, 3, 256, 256)
        dummy_f3 = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_f1, dummy_f3)
        print(f"  Input shape: {dummy_f1.shape}")
        print(f"  Output shape: {output['output'].shape}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
