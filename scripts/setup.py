#!/usr/bin/env python3
"""
AInimotion Setup & Environment Check Script

Checks for all required dependencies and optionally installs them.
Run this after cloning the repo to verify your environment is ready.

Usage:
    python scripts/setup.py           # Check only
    python scripts/setup.py --install # Check and install missing packages
"""

import subprocess
import sys
import shutil
from pathlib import Path


# Required Python packages
REQUIRED_PACKAGES = {
    # Core ML
    "torch": "torch",
    "torchvision": "torchvision",
    # Data processing
    "PIL": "pillow",
    "skimage": "scikit-image",
    "numpy": "numpy",
    # Training utilities
    "tqdm": "tqdm",
    "yaml": "pyyaml",
    "tensorboard": "tensorboard",
}

# Minimum Python version
MIN_PYTHON = (3, 10)


def print_status(name: str, ok: bool, message: str = ""):
    """Print a status line with checkmark or X."""
    icon = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    
    # Windows might not support ANSI colors in all terminals
    try:
        print(f"  {color}{icon}{reset} {name}", end="")
    except:
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}", end="")
    
    if message:
        print(f" — {message}")
    else:
        print()


def check_python_version() -> bool:
    """Check Python version is 3.10+."""
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    version_str = f"{current[0]}.{current[1]}"
    min_str = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    
    if ok:
        print_status("Python", True, f"v{version_str}")
    else:
        print_status("Python", False, f"v{version_str} (need {min_str}+)")
    
    return ok


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        # Get version
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            version_line = result.stdout.split("\n")[0]
            version = version_line.split(" ")[2] if "version" in version_line else "unknown"
            print_status("FFmpeg", True, f"v{version}")
            return True
        except:
            print_status("FFmpeg", True, ffmpeg_path)
            return True
    else:
        print_status("FFmpeg", False, "not found in PATH")
        print("\n    Install FFmpeg:")
        if sys.platform == "win32":
            print("      winget install ffmpeg")
            print("      # or: choco install ffmpeg")
        elif sys.platform == "darwin":
            print("      brew install ffmpeg")
        else:
            print("      sudo apt install ffmpeg")
        print()
        return False


def check_cuda() -> tuple[bool, str]:
    """Check CUDA availability via PyTorch."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_status("CUDA", True, f"{device_name} ({vram_gb:.1f}GB, CUDA {cuda_version})")
            return True, device_name
        else:
            print_status("CUDA", False, "not available (will use CPU)")
            return False, ""
    except ImportError:
        print_status("CUDA", False, "PyTorch not installed yet")
        return False, ""


def check_package(import_name: str, pip_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_packages() -> list[str]:
    """Check all required packages, return list of missing ones."""
    missing = []
    
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if check_package(import_name, pip_name):
            print_status(pip_name, True)
        else:
            print_status(pip_name, False, "not installed")
            missing.append(pip_name)
    
    return missing


def install_packages(packages: list[str], use_cuda: bool = True) -> bool:
    """Install missing packages via pip."""
    if not packages:
        return True
    
    print(f"\nInstalling {len(packages)} package(s)...")
    
    # Handle PyTorch specially for CUDA
    torch_packages = [p for p in packages if p in ("torch", "torchvision")]
    other_packages = [p for p in packages if p not in ("torch", "torchvision")]
    
    # Install PyTorch with CUDA if needed
    if torch_packages:
        if use_cuda and sys.platform != "darwin":  # CUDA not on Mac
            print("  Installing PyTorch with CUDA support...")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ]
        else:
            print("  Installing PyTorch (CPU)...")
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ Failed to install PyTorch")
            print(result.stderr)
            return False
        print("  ✓ PyTorch installed")
    
    # Install other packages
    if other_packages:
        print(f"  Installing: {', '.join(other_packages)}")
        cmd = [sys.executable, "-m", "pip", "install"] + other_packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ Failed to install packages")
            print(result.stderr)
            return False
        print("  ✓ Packages installed")
    
    return True


def test_model_import() -> bool:
    """Test that the model can be imported."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from ainimotion.models.interp import LayeredInterpolator
        print_status("Model import", True)
        return True
    except Exception as e:
        print_status("Model import", False, str(e))
        return False


def test_forward_pass() -> bool:
    """Test a forward pass through the model."""
    try:
        import torch
        from ainimotion.models.interp import LayeredInterpolator
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LayeredInterpolator(base_channels=32).to(device)
        
        # Small test input
        x1 = torch.randn(1, 3, 128, 128, device=device)
        x2 = torch.randn(1, 3, 128, 128, device=device)
        
        with torch.no_grad():
            output = model(x1, x2)
        
        params = sum(p.numel() for p in model.parameters())
        print_status("Forward pass", True, f"OK ({params:,} params, {device})")
        return True
    except Exception as e:
        print_status("Forward pass", False, str(e))
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AInimotion setup check")
    parser.add_argument("--install", action="store_true", help="Install missing packages")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("  AInimotion Environment Check")
    print("="*50 + "\n")
    
    all_ok = True
    
    # Check Python
    print("System:")
    if not check_python_version():
        print("\n✗ Python 3.10+ required. Please upgrade.\n")
        sys.exit(1)
    
    # Check FFmpeg
    if not check_ffmpeg():
        all_ok = False
    
    # Check packages
    print("\nPython Packages:")
    missing = check_packages()
    
    if missing and args.install:
        if not install_packages(missing):
            all_ok = False
        else:
            # Re-check after install
            print("\nRe-checking packages:")
            missing = check_packages()
    elif missing:
        all_ok = False
        print(f"\n  Run with --install to install missing packages")
    
    # Check CUDA (after potential torch install)
    print("\nGPU:")
    cuda_ok, gpu_name = check_cuda()
    
    # Test model
    if not missing:
        print("\nModel:")
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        if test_model_import():
            test_forward_pass()
    
    # Summary
    print("\n" + "="*50)
    if all_ok and not missing:
        print("  ✓ All checks passed! Ready to train.")
        if cuda_ok:
            print(f"  GPU: {gpu_name}")
    else:
        print("  ✗ Some checks failed. See above for details.")
        if missing:
            print(f"  Missing packages: {', '.join(missing)}")
            print(f"  Run: python scripts/setup.py --install")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
