"""Analyze triplet folders to identify valid vs invalid entries."""
from pathlib import Path
from collections import Counter

def main():
    triplets_dir = Path("D:/Triplets")
    all_dirs = [d for d in triplets_dir.iterdir() if d.is_dir()]

    categories = Counter()
    sample_invalid = {}

    for d in all_dirs:
        has_f1_png = (d / "f1.png").exists()
        has_f2_png = (d / "f2.png").exists()
        has_f3_png = (d / "f3.png").exists()
        has_f1_jpg = (d / "f1.jpg").exists()
        has_f2_jpg = (d / "f2.jpg").exists()
        has_f3_jpg = (d / "f3.jpg").exists()
        
        has_f1 = has_f1_png or has_f1_jpg
        has_f2 = has_f2_png or has_f2_jpg
        has_f3 = has_f3_png or has_f3_jpg
        
        if has_f1 and has_f2 and has_f3:
            categories["VALID (has f1, f2, f3)"] += 1
        else:
            # Determine what's missing
            missing = []
            if not has_f1:
                missing.append("f1")
            if not has_f2:
                missing.append("f2")
            if not has_f3:
                missing.append("f3")
            key = "INVALID: missing " + ", ".join(missing)
            categories[key] += 1
            if key not in sample_invalid and len(sample_invalid) < 5:
                files = [f.name for f in list(d.iterdir())[:5]]
                sample_invalid[key] = (d.name, files)

    print("=== TRIPLET ANALYSIS REPORT ===")
    print(f"Total directories: {len(all_dirs):,}")
    print()
    print("Breakdown by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(all_dirs) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    print()
    
    total_invalid = sum(c for k, c in categories.items() if "INVALID" in k)
    print(f"Summary:")
    print(f"  Valid triplets: {categories.get('VALID (has f1, f2, f3)', 0):,}")
    print(f"  Invalid folders: {total_invalid:,}")
    print()
    
    if sample_invalid:
        print("Sample invalid folders (showing contents):")
        for cat, (name, files) in sample_invalid.items():
            print(f"  {cat}")
            print(f"    Folder: {name}")
            print(f"    Contains: {files}")

if __name__ == "__main__":
    main()
