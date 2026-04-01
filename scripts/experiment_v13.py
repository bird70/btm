"""
experiment_v13.py — Multi-seed CV stability analysis for pipeline configurations.

Runs the same pipeline config across N seeds, reporting mean±std of weighted F1.
This provides a reliable comparison between feature sets, immune to single-seed noise.

Usage:
  python scripts/experiment_v13.py [--seeds 10] [--submit]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON = sys.executable
TRAIN_CMD = [
    PYTHON,
    "-m",
    "benthic_model.cli",
    "train",
    "--bathymetry-tif",
    "data/MBES/bathymetry.tif",
    "--backscatter-tif",
    "data/MBES/backscatter.tif",
]
PREDICT_CMD = [
    PYTHON,
    "-m",
    "benthic_model.cli",
    "predict",
    "--bathymetry-tif",
    "data/MBES/bathymetry.tif",
    "--backscatter-tif",
    "data/MBES/backscatter.tif",
]


def run_pipeline(train_csv: str, config: str, seed: int) -> dict:
    """Run a pipeline training and return metrics dict."""
    cmd = TRAIN_CMD + [
        "--train-csv",
        train_csv,
        "--config",
        config,
        "--run-type",
        "candidate",
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Extract run_id from output
    for line in result.stdout.splitlines():
        if "run_id=" in line:
            run_id = line.split("run_id=")[1].strip()
            metrics_path = Path("artifacts/runs") / run_id / "metrics.json"
            with open(metrics_path) as f:
                return json.load(f)
    raise RuntimeError(f"Training failed: {result.stderr}")


def predict_and_submit(run_id: str, test_csv: str, output_path: str) -> str:
    """Generate predictions and create submission CSV."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = PREDICT_CMD + [
        "--run-id",
        run_id,
        "--test-csv",
        test_csv,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed: {result.stderr}")

    pred_path = Path("artifacts/predictions") / f"{run_id}_test_predictions.csv"
    cmd_sub = [
        PYTHON,
        "-m",
        "benthic_model.cli",
        "make-submission",
        "--predictions",
        str(pred_path),
        "--sample-submission",
        "data/sample_submission.csv",
        "--output",
        output_path,
    ]
    result = subprocess.run(cmd_sub, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Submission failed: {result.stderr}")
    return output_path


def multi_seed_cv(
    train_csv: str,
    config: str,
    seeds: list[int],
    label: str,
) -> dict:
    """Run pipeline across multiple seeds and report statistics."""
    results = []
    for seed in seeds:
        print(f"  [{label}] Seed {seed}...", end=" ", flush=True)
        m = run_pipeline(train_csv, config, seed)
        cv = m["weighted_f1"]
        sgam = m["per_class_f1"]["SGAM"]
        results.append(
            {
                "seed": seed,
                "run_id": m["run_id"],
                "cv": cv,
                "sgam": sgam,
                "per_class": m["per_class_f1"],
            }
        )
        print(f"CV={cv:.4f} SGAM={sgam:.4f}")

    cvs = [r["cv"] for r in results]
    sgams = [r["sgam"] for r in results]
    summary = {
        "label": label,
        "train_csv": train_csv,
        "config": config,
        "n_seeds": len(seeds),
        "cv_mean": np.mean(cvs),
        "cv_std": np.std(cvs),
        "cv_min": np.min(cvs),
        "cv_max": np.max(cvs),
        "sgam_mean": np.mean(sgams),
        "sgam_std": np.std(sgams),
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--submit", action="store_true", help="Submit best to Kaggle")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    print(f"=== Multi-seed CV analysis ({args.seeds} seeds) ===\n")

    # Define experiment configurations
    experiments = [
        {
            "label": "R04-baseline",
            "train_csv": "data/train_btm.csv",
            "config": "configs/rf-btm-fine.yaml",
        },
        {
            "label": "S10-noxy",
            "train_csv": "data/train_btm.csv",
            "config": "configs/rf-btm-s10-noxy.yaml",
        },
    ]

    all_summaries = []
    for exp in experiments:
        print(f"\n--- {exp['label']} ---")
        summary = multi_seed_cv(exp["train_csv"], exp["config"], seeds, exp["label"])
        all_summaries.append(summary)
        print(
            f"  => CV={summary['cv_mean']:.4f} ± {summary['cv_std']:.4f}"
            f"  (min={summary['cv_min']:.4f}, max={summary['cv_max']:.4f})"
            f"  SGAM={summary['sgam_mean']:.4f} ± {summary['sgam_std']:.4f}"
        )

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(
        f"{'Label':<20} {'CV mean':>9} {'CV std':>8} {'CV min':>8} {'CV max':>8} {'SGAM mean':>10}"
    )
    print("-" * 75)
    for s in all_summaries:
        print(
            f"{s['label']:<20} {s['cv_mean']:>9.4f} {s['cv_std']:>8.4f}"
            f" {s['cv_min']:>8.4f} {s['cv_max']:>8.4f} {s['sgam_mean']:>10.4f}"
        )

    # Save results
    os.makedirs("reports/metrics", exist_ok=True)
    with open("reports/metrics/experiment_v13_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print("\nResults saved to reports/metrics/experiment_v13_results.json")

    # Determine if S10-noxy improves mean CV
    if len(all_summaries) >= 2:
        baseline = all_summaries[0]
        candidate = all_summaries[1]
        delta = candidate["cv_mean"] - baseline["cv_mean"]
        print(f"\nDelta ({candidate['label']} - {baseline['label']}): {delta:+.4f}")
        if delta > 0.005:
            print("=> SIGNIFICANT improvement; candidate for submission")
        elif delta > 0:
            print("=> Marginal improvement; not sufficient for submission")
        else:
            print("=> No improvement; baseline remains best")


if __name__ == "__main__":
    main()
