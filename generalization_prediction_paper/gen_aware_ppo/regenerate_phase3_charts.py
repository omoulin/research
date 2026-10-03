"""Regenerate the final phase-3 comparison chart for every experiment under data/.

Scans for directories containing gen_curve_genppo_??.npy files, pairs each with
its baseline curves (parent dir, or itself for the older flat layout), and
re-saves the final comparison_after_NN.pdf via main._compare.

Usage:
    python regenerate_phase3_charts.py
"""
from __future__ import annotations

import glob
import os

from main import _compare


def find_genppo_dirs(root: str) -> list[str]:
    hits = glob.glob(os.path.join(root, "**", "gen_curve_genppo_00.npy"), recursive=True)
    return sorted({os.path.dirname(h) for h in hits})


def resolve_phase3_base(genppo_dir: str) -> str | None:
    parent = os.path.dirname(genppo_dir)
    if glob.glob(os.path.join(parent, "gen_curve_baseline_??.npy")):
        return parent
    if glob.glob(os.path.join(genppo_dir, "gen_curve_baseline_??.npy")):
        return genppo_dir
    return None


def main() -> None:
    genppo_dirs = find_genppo_dirs("data")
    print(f"Found {len(genppo_dirs)} phase-3 result directories.\n")

    regenerated, skipped = 0, 0
    for genppo_dir in genppo_dirs:
        phase3_base = resolve_phase3_base(genppo_dir)
        if phase3_base is None:
            print(f"[skip] {genppo_dir}  (no matching baseline curves found)")
            skipped += 1
            continue

        gen_files = sorted(glob.glob(os.path.join(genppo_dir, "gen_curve_genppo_??.npy")))
        run_index = len(gen_files) - 1
        print(f"[chart] {genppo_dir}  (baseline={phase3_base}, n_runs={len(gen_files)})")
        _compare(phase3_base, genppo_dir, run_index=run_index)
        regenerated += 1

    print(f"\nDone. Regenerated {regenerated} charts, skipped {skipped}.")


if __name__ == "__main__":
    main()
