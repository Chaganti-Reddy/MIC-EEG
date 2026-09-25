"""
Full pipeline orchestrator — runs WGAN-GP + CNN for all 9 subjects,
performs exhaustive 9×9 cross-subject evaluation, then generates all
aggregate figures and summary tables.

Usage
-----
  python run_all.py                    # all 9 subjects
  python run_all.py --subjects 1 2 3   # specific subjects
  python run_all.py --skip_wgan        # CNN only (synthetic already exists)
  python run_all.py --analyze_only     # jump straight to plots

Outputs
-------
  outputs/synthetic/synthetic_combined_s0N.npz  (one per subject)
  outputs/models/cnn_s0N_best.weights.h5
  outputs/figures/                              (per-subject + aggregate)
  metrics/cnn_metrics_s0N.json
  metrics/cross_subject_matrix.csv
  metrics/summary_table.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS_DIR  = ROOT / "metrics"
FIGURES_DIR  = ROOT / "outputs" / "figures"
SYNTHETIC_DIR = ROOT / "outputs" / "synthetic"

for d in [METRICS_DIR, FIGURES_DIR, SYNTHETIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_script(script: str, extra_args: list[str] | None = None) -> bool:
    """Run a src/ script as a subprocess. Returns True on success."""
    cmd = [sys.executable, str(ROOT / "src" / script)] + (extra_args or [])
    print(f"\n  $ {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  FAILED  (exit {result.returncode})  [{elapsed:.0f}s]")
        return False
    print(f"  OK  [{elapsed:.0f}s]")
    return True


def build_cross_matrix(subjects: list[int]) -> pd.DataFrame:
    """
    Build a |subjects|×|subjects| accuracy matrix from saved JSON metrics.
    Rows = train subject, Cols = test subject.
    Diagonal = in-subject accuracy.
    """
    n = len(subjects)
    mat = np.full((n, n), np.nan)

    for i, train_s in enumerate(subjects):
        jpath = METRICS_DIR / f"cnn_metrics_s{train_s:02d}.json"
        if not jpath.exists():
            continue
        with open(jpath) as fh:
            m = json.load(fh)
        mat[i, i] = m.get("accuracy", np.nan)

        for j, test_s in enumerate(subjects):
            if i == j:
                continue
            cpath = METRICS_DIR / f"cross_s{train_s:02d}_to_s{test_s:02d}.json"
            if cpath.exists():
                with open(cpath) as fh:
                    cm = json.load(fh)
                mat[i, j] = cm.get("accuracy", np.nan)

    labels = [f"S{s}" for s in subjects]
    return pd.DataFrame(mat * 100, index=labels, columns=labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete WGAN-GP + CNN pipeline for all subjects"
    )
    parser.add_argument("--subjects",     type=int, nargs="+",
                        default=list(range(1, 10)),
                        help="Which subjects to process (default: 1-9)")
    parser.add_argument("--wgan_epochs",  type=int, default=None,
                        help="Override WGAN epoch count")
    parser.add_argument("--cnn_epochs",   type=int, default=None,
                        help="Override CNN epoch count")
    parser.add_argument("--skip_wgan",    action="store_true",
                        help="Skip WGAN training (use existing synthetic files)")
    parser.add_argument("--skip_cnn",     action="store_true",
                        help="Skip CNN training (only run analysis)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip training and jump straight to analysis")
    parser.add_argument("--cross_all",    action="store_true", default=True,
                        help="Run full 9×9 cross-subject evaluation (default: on)")
    parser.add_argument("--gpu",          action="store_true",
                        help="Enable GPU training flag")
    args = parser.parse_args()

    subjects = args.subjects
    print(f"\n{'='*65}")
    print(f"  Motor Imagery Classification — Full Pipeline")
    print(f"  Subjects : {subjects}")
    print(f"{'='*65}\n")

    failures: list[str] = []
    t0 = time.time()

    if not args.analyze_only:
        if not args.skip_wgan:
            print(f"\n{'─'*65}")
            print(f"  PHASE 1 — WGAN-GP Training  ({len(subjects)} subjects)")
            print(f"{'─'*65}")
            for s in subjects:
                synth_path = SYNTHETIC_DIR / f"synthetic_combined_s{s:02d}.npz"
                if synth_path.exists():
                    print(f"\n  [S{s:02d}] Synthetic already exists — skipping WGAN")
                    continue
                print(f"\n  [S{s:02d}] Training WGAN-GP …")
                wgan_args = [f"--subject", str(s)]
                if args.wgan_epochs:
                    wgan_args += ["--epochs", str(args.wgan_epochs)]
                if args.gpu:
                    wgan_args += ["--gpu"]
                ok = run_script("train_wgan.py", wgan_args)
                if not ok:
                    failures.append(f"WGAN S{s}")

        if not args.skip_cnn:
            print(f"\n{'─'*65}")
            print(f"  PHASE 2 — CNN Training  ({len(subjects)} subjects)")
            print(f"{'─'*65}")
            for s in subjects:
                print(f"\n  [S{s:02d}] Training CNN …")
                cnn_args = ["--subject", str(s)]
                if args.cnn_epochs:
                    cnn_args += ["--epochs", str(args.cnn_epochs)]
                ok = run_script("train_cnn.py", cnn_args)
                if not ok:
                    failures.append(f"CNN S{s}")

            if args.cross_all and len(subjects) > 1:
                print(f"\n{'─'*65}")
                print(f"  PHASE 3 — Cross-Subject Evaluation  ({len(subjects)}×{len(subjects)})")
                print(f"{'─'*65}")
                for train_s in subjects:
                    for test_s in subjects:
                        if train_s == test_s:
                            continue
                        out_path = METRICS_DIR / f"cross_s{train_s:02d}_to_s{test_s:02d}.json"
                        if out_path.exists():
                            print(f"  [S{train_s}→S{test_s}] Already exists — skipping")
                            continue
                        print(f"\n  [S{train_s}→S{test_s}] Cross evaluation …")
                        ok = run_script("cross_eval.py", [
                            "--train_subject", str(train_s),
                            "--test_subject",  str(test_s),
                        ])
                        if not ok:
                            failures.append(f"Cross S{train_s}→S{test_s}")

    print(f"\n{'─'*65}")
    print(f"  PHASE 4 — Aggregate Analysis & Figures")
    print(f"{'─'*65}")
    ok = run_script("analyze_results.py", ["--subjects"] + [str(s) for s in subjects])
    if not ok:
        failures.append("analyze_results")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Pipeline complete  [{elapsed/60:.1f} min]")
    if failures:
        print(f"  Failed steps: {', '.join(failures)}")
    else:
        print(f"  All steps succeeded")
    print(f"  Figures  → outputs/figures/")
    print(f"  Metrics  → metrics/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
