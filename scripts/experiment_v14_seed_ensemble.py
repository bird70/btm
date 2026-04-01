"""
experiment_v14_seed_ensemble.py — Majority-vote seed ensemble for 019-pipeline-cv-improvement

Loads predictions from 5 R04 seed models and computes a majority-vote ensemble.

Usage:
    python scripts/experiment_v14_seed_ensemble.py [--dry-run] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# Seed models from branch 018/019 (R04 config, train_btm.csv)
SEED_RUN_IDS: dict[int, str] = {
    42:   "candidate-20260401021038",
    123:  "candidate-20260401021045",
    456:  "candidate-20260401021121",
    789:  "candidate-20260401021128",
    2026: "candidate-20260401021136",
}

# CV scores for each seed (used as tiebreaker)
SEED_CV: dict[int, float] = {
    42:   0.8024,
    123:  0.7626,
    456:  0.7890,
    789:  0.8101,
    2026: 0.7815,
}

PREDICTIONS_DIR = Path("artifacts/predictions")
DEFAULT_OUTPUT = Path("data/submission_v14_seed_ensemble.csv")
ALLOWED_CLASSES = {"ALG", "FMAT", "NVB", "SGAM", "SGZ"}


def majority_vote(votes: list[str], seed_cv: dict[int, float] | None = None) -> str:
    """Return the class winning the most votes; use seed CV as tiebreaker.

    Args:
        votes: List of class strings, one per seed (in seed order matching SEED_RUN_IDS).
        seed_cv: Mapping seed->CV score used to break ties.

    Returns:
        Winning class string.
    """
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [cls for cls, cnt in counts.items() if cnt == max_count]
    if len(winners) == 1:
        return winners[0]
    # Tiebreaker: pick class from the highest-CV seed
    if seed_cv is not None:
        seeds_ordered = sorted(seed_cv.keys(), key=lambda s: seed_cv[s], reverse=True)
        seed_list = list(SEED_RUN_IDS.keys())
        for seed in seeds_ordered:
            idx = seed_list.index(seed)
            if idx < len(votes) and votes[idx] in winners:
                return votes[idx]
    return winners[0]


def load_predictions(predictions_dir: Path, run_ids: dict[int, str]) -> pd.DataFrame:
    """Load per-seed prediction CSVs and return a wide DataFrame (index=ID, cols=seeds)."""
    frames: dict[int, pd.Series] = {}
    for seed, run_id in run_ids.items():
        path = predictions_dir / f"{run_id}_test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        df = pd.read_csv(path)
        frames[seed] = df.set_index("ID")["class"]
    return pd.DataFrame(frames)


def build_ensemble(
    vote_matrix: pd.DataFrame,
    seed_cv: dict[int, float],
) -> pd.DataFrame:
    """Apply majority vote per sample; return DataFrame with columns: ID, class, votes, total."""
    results = []
    seed_list = list(vote_matrix.columns)
    for sample_id, row in vote_matrix.iterrows():
        votes = row.tolist()
        winner = majority_vote(votes, seed_cv={s: seed_cv[s] for s in seed_list})
        winning_count = Counter(votes)[winner]
        results.append({
            "ID": sample_id,
            "class": winner,
            "votes": winning_count,
            "total": len(votes),
        })
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Majority-vote seed ensemble for benthic habitat predictions"
    )
    parser.add_argument("--dry-run", action="store_true", help="Load files and report, no output written")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--predictions-dir", type=Path, default=PREDICTIONS_DIR,
        help="Directory containing prediction CSVs"
    )
    args = parser.parse_args(argv)

    print(f"Seed ensemble: loading {len(SEED_RUN_IDS)} prediction files from {args.predictions_dir}")

    if args.dry_run:
        # Validate files exist and report counts
        found = 0
        for seed, run_id in SEED_RUN_IDS.items():
            path = args.predictions_dir / f"{run_id}_test_predictions.csv"
            status = "OK" if path.exists() else "MISSING"
            print(f"  seed={seed:>5} {run_id}: {status}")
            if path.exists():
                found += 1
        print(f"Found {found}/{len(SEED_RUN_IDS)} prediction files")
        return 0

    # Load predictions
    vote_matrix = load_predictions(args.predictions_dir, SEED_RUN_IDS)
    print(f"Loaded predictions for {len(vote_matrix)} test samples")

    # Build ensemble
    ensemble_df = build_ensemble(vote_matrix, SEED_CV)

    # Validate
    assert len(ensemble_df) == 98, f"Expected 98 samples, got {len(ensemble_df)}"
    invalid = set(ensemble_df["class"].unique()) - ALLOWED_CLASSES
    assert not invalid, f"Invalid class labels: {invalid}"

    # Confidence report
    vote_dist = ensemble_df["votes"].value_counts().sort_index()
    print("Vote confidence distribution:")
    for n_votes, count in vote_dist.items():
        print(f"  {n_votes}/5 agreement: {count} samples")

    print(f"\nEnsemble class distribution:")
    print(ensemble_df["class"].value_counts().to_string())

    # Compare to R04 predictions (seed=42)
    r04_path = args.predictions_dir / f"{SEED_RUN_IDS[42]}_test_predictions.csv"
    if r04_path.exists():
        r04 = pd.read_csv(r04_path).set_index("ID")["class"]
        merged = ensemble_df.set_index("ID")["class"].rename("ens").to_frame()
        merged["r04"] = r04
        changed = (merged["ens"] != merged["r04"]).sum()
        print(f"\nEnsemble vs R04 (seed=42): {len(merged)-changed}/{len(merged)} agree ({(len(merged)-changed)/len(merged):.1%}), {changed} changed")

    # Write output
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df[["ID", "class"]].to_csv(out_path, index=False)
    print(f"\nEnsemble submission written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
