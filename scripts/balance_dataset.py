"""
Balance training dataset: cap each anime series at N sequences using stratified sampling.

Groups sequence directories by anime series, divides each series into 3 motion
bins (low/medium/high), and samples equally from each bin to ensure the model
sees the full range of motion complexity.

Usage:
  # Dry run (just show what would happen)
  python scripts/balance_dataset.py --dirs D:\Training C:\Projects\AInimotion\training_val --cap 10000

  # Actually delete
  python scripts/balance_dataset.py --dirs D:\Training C:\Projects\AInimotion\training_val --cap 10000 --delete
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict


# Map raw directory prefixes -> canonical anime names
SERIES_MAP = {
    "Jujutsu_Kaisen_1080p_Dual_Audio_BDRip_x265-EMBER": "Jujutsu_Kaisen",
    "Jujutsu_Kaisen_S03E10_Tokyo_No_1_Colony_Part_4_1080p_NF_WEB-": "Jujutsu_Kaisen",
    "Demon_Slayer_S04_1080p_Blu-Ray_10-Bit_Dual-Audio_LPCM_x265-i": "Demon_Slayer",
    "Demon_Slayer_Kimetsu_no_Yaiba_Infinity_Castle_2025_Dual_YG": "Demon_Slayer",
    "Judas_Kimi_no_Na_Wa_Your_Name_BD_2160p_4K_UHDHEVC_x265_10bit": "Your_Name",
    "SPY_x_FAMILY_S03E09_Anyas_Era_Has_Come_1080p_NF_WEB-DL_AAC2_": "SPY_x_FAMILY",
    "SPY_x_FAMILY_S03E10_Austins_Troubles-A_Normal_Mixer-Moon_Lan": "SPY_x_FAMILY",
    "SPY_x_FAMILY_S03E12_Battle_to_the_Death_in_the_Sewers_1080p_": "SPY_x_FAMILY",
    "SPY_x_FAMILY_S03E13_A_World_Where_We_Cannot_Survive_1080p_NF": "SPY_x_FAMILY",
    "SubsPlease_Spy_x_Family_-_50_1080p_42C2C63B": "SPY_x_FAMILY",
    "Violet_Evergarden_S01MoviesOVA_1080p_Dual_Audio_BDRip_10_bit": "Violet_Evergarden",
    "MTBB_Mob_Psycho_100_S2_BD_1080p": "Mob_Psycho_100",
}

BIN_NAMES = ["LOW", "MED", "HIGH"]


def get_series(dirname: str) -> str | None:
    """Map a sequence directory name to its canonical anime series."""
    prefix = re.split(r'_ep\d', dirname)[0]
    return SERIES_MAP.get(prefix)


def get_motion(dirname: str) -> float:
    """Extract motion score from directory name suffix like _m0.0347."""
    m = re.search(r'_m([\d.]+)$', dirname)
    return float(m.group(1)) if m else 0.0


def scan_directories(dir_paths: list[str]) -> dict[str, list[tuple[str, float]]]:
    """
    Scan directories and group sequences by anime series.
    Returns {series: [(full_path, motion_score), ...]}
    """
    groups = defaultdict(list)

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            print(f"  Skipping (not found): {dir_path}")
            continue

        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if not os.path.isdir(full):
                continue
            if name.startswith('_'):  # skip _tmp_frames etc
                continue

            series = get_series(name)
            if series is None:
                print(f"  WARNING: Unknown series for '{name}' -- will be SKIPPED")
                continue

            motion = get_motion(name)
            groups[series].append((full, motion))

    return groups


def stratified_sample(
    seqs: list[tuple[str, float]], cap: int, n_bins: int = 3, seed: int = 42
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[dict]]:
    """
    Stratified sampling: divide sequences into motion bins, sample equally.

    Returns (keep_list, delete_list, bin_stats).
    """
    if len(seqs) <= cap:
        stats = [{"name": "ALL", "total": len(seqs), "kept": len(seqs),
                  "min_m": min(m for _, m in seqs), "max_m": max(m for _, m in seqs)}]
        return seqs, [], stats

    # Sort by motion to find tercile boundaries
    sorted_seqs = sorted(seqs, key=lambda x: x[1])
    motions = [m for _, m in sorted_seqs]

    # Split into n_bins equal-count bins
    bin_size = len(sorted_seqs) // n_bins
    bins = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(sorted_seqs)
        bins.append(sorted_seqs[start:end])

    # Sample per_bin from each, with remainder going to middle bins first
    per_bin = cap // n_bins
    remainder = cap - (per_bin * n_bins)

    rng = random.Random(seed)
    keep_set = set()
    bin_stats = []

    for b, bin_seqs in enumerate(bins):
        # Distribute remainder: extra samples go to middle bin first, then high, then low
        alloc = per_bin
        if remainder > 0 and b == 1:  # middle bin gets first extra
            alloc += 1
            remainder -= 1
        elif remainder > 0 and b == 2:  # high bin gets second extra
            alloc += 1
            remainder -= 1
        elif remainder > 0:
            alloc += 1
            remainder -= 1

        n_sample = min(alloc, len(bin_seqs))
        sampled = rng.sample(bin_seqs, n_sample)
        for s in sampled:
            keep_set.add(s[0])  # track by path

        bin_motions = [m for _, m in bin_seqs]
        bin_stats.append({
            "name": BIN_NAMES[b],
            "total": len(bin_seqs),
            "kept": n_sample,
            "min_m": min(bin_motions),
            "max_m": max(bin_motions),
        })

    keep = [s for s in seqs if s[0] in keep_set]
    delete = [s for s in seqs if s[0] not in keep_set]

    return keep, delete, bin_stats


def balance(dir_paths: list[str], cap: int, do_delete: bool = False):
    print(f"\nScanning: {dir_paths}")
    groups = scan_directories(dir_paths)

    print(f"\n{'='*70}")
    print(f"  BALANCING PLAN -- cap = {cap:,} per anime (stratified sampling)")
    print(f"  Bins: LOW / MED / HIGH motion (~{cap//3:,} each)")
    print(f"{'='*70}\n")

    total_keep = 0
    total_delete = 0

    for series in sorted(groups):
        seqs = groups[series]
        keep, delete, bin_stats = stratified_sample(seqs, cap)

        n_total = len(seqs)
        n_keep = len(keep)
        n_delete = len(delete)
        total_keep += n_keep
        total_delete += n_delete

        status = "[OK] UNDER CAP" if n_delete == 0 else f"[DEL] DELETE {n_delete:,}"
        print(f"  {series}:  (total: {n_total:,})")
        for bs in bin_stats:
            print(f"    {bs['name']:>4}: {bs['kept']:>5,} kept / {bs['total']:>6,} avail"
                  f"   motion [{bs['min_m']:.4f} - {bs['max_m']:.4f}]")
        print(f"    --> Keep: {n_keep:,}  |  Delete: {n_delete:,}  {status}")
        print()

        if do_delete and delete:
            print(f"    Deleting {n_delete:,} sequences...")
            for i, (path, _) in enumerate(delete):
                shutil.rmtree(path)
                if (i + 1) % 5000 == 0:
                    print(f"      {i+1}/{n_delete} deleted...")
            print(f"    [DONE] Deleted {n_delete:,} sequences")
            print()

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"    Total sequences found: {total_keep + total_delete:,}")
    print(f"    Keeping:  {total_keep:,}")
    print(f"    Deleting: {total_delete:,}")
    if not do_delete and total_delete > 0:
        print(f"\n    ** DRY RUN -- nothing was deleted.")
        print(f"    Re-run with --delete to actually remove sequences.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Balance training dataset (stratified)')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Directories containing sequence folders')
    parser.add_argument('--cap', type=int, default=10000,
                        help='Max sequences per anime series (default: 10000)')
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete excess sequences (default: dry run)')
    args = parser.parse_args()
    balance(args.dirs, args.cap, args.delete)
