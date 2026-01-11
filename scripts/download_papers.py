#!/usr/bin/env python3
"""
Download frame interpolation research papers from arXiv.
Papers are saved to the literature/ folder.
"""

import os
import urllib.request
from pathlib import Path

# Paper metadata: (filename, arxiv_id, title)
PAPERS = [
    # Foundations
    ("FlowNet_2015.pdf", "1504.06852", "FlowNet: Learning Optical Flow with CNNs"),
    ("PWC-Net_2018.pdf", "1709.02371", "PWC-Net: CNNs for Optical Flow"),
    
    # Core Interpolation
    ("SuperSloMo_2018.pdf", "1712.00080", "Super SloMo: High Quality Intermediate Frames"),
    ("RIFE_2022.pdf", "2011.06294", "RIFE: Real-Time Intermediate Flow Estimation"),
    ("IFRNet_2022.pdf", "2205.14620", "IFRNet: Intermediate Feature Refine Network"),
    ("FILM_2022.pdf", "2202.04901", "FILM: Frame Interpolation for Large Motion"),
    ("AMT_2023.pdf", "2304.09790", "AMT: All-Pairs Multi-Field Transforms"),
    
    # Alternatives
    ("AdaCoF_2020.pdf", "1907.10244", "AdaCoF: Adaptive Collaboration of Flows"),
    ("CAIN_2020.pdf", "2002.12569", "CAIN: Channel Attention Is All You Need"),
    ("VFIformer_2022.pdf", "2205.07230", "VFIformer: Video Frame Interpolation with Transformer"),
]


def download_paper(filename: str, arxiv_id: str, title: str, output_dir: Path) -> bool:
    """Download a single paper from arXiv."""
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"  ✓ Already exists: {filename}")
        return True
    
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  ↓ Downloading: {title}...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Saved: {filename}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {filename} - {e}")
        return False


def main():
    # Determine output directory (literature/ folder in project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "literature"
    
    # Create literature folder if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {len(PAPERS)} papers to: {output_dir}\n")
    
    success_count = 0
    for filename, arxiv_id, title in PAPERS:
        if download_paper(filename, arxiv_id, title, output_dir):
            success_count += 1
    
    print(f"\nDone! Downloaded {success_count}/{len(PAPERS)} papers.")
    print(f"Papers saved to: {output_dir}")


if __name__ == "__main__":
    main()
