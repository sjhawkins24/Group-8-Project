"""Command-line helper to train the hybrid AP ranking model.

Usage:
    python train_hybrid_model.py [--merged PATH] [--elo PATH]

The script trains both the regression and classification pipelines, saves the
artifacts under ``artifacts/`` and prints a concise performance summary using
5-fold time-series cross validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from models.elo_hybrid import (
    DATA_MERGED_DEFAULT,
    ELO_SEASON_DEFAULT,
    build_dataset,
    ensure_models,
    evaluate_time_series,
    save_models,
    train_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid ranking model")
    parser.add_argument(
        "--merged",
        type=Path,
        default=DATA_MERGED_DEFAULT,
        help="Path to mergedTrainingData.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--elo",
        type=Path,
        default=ELO_SEASON_DEFAULT,
        help="Path to elo_ratings_by_season.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if artifacts already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = build_dataset(args.merged, args.elo)
    print(f"Prepared dataset with {len(dataset.features)} samples and {len(dataset.features.columns)} features")

    metrics = evaluate_time_series(dataset)
    reg_mae, reg_sd = metrics["reg_mae"]
    clf_acc, clf_sd = metrics["clf_acc"]

    print("\nTime-series CV (5 folds)")
    print(f"  Regression MAE:  {reg_mae:.3f} ± {reg_sd:.3f}")
    print(f"  Classification:  {clf_acc:.3f} ± {clf_sd:.3f}")

    if args.force:
        models = train_models(dataset)
        save_models(models)
        print("\nArtifacts retrained and saved under artifacts/")
    else:
        models = ensure_models(args.merged, args.elo)
        save_models(models)
        print("\nArtifacts trained and saved under artifacts/ (existing files overwritten)")


if __name__ == "__main__":
    main()
