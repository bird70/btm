from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from benthic_model.data.ingest import (
    enforce_competition_data_whitelist,
    parse_and_validate_metadata,
)
from benthic_model.evaluation.compare import (
    compare_run_to_baseline,
    write_comparison_report,
    write_machine_readable_metric_summary,
)
from benthic_model.evaluation.reproducibility import write_reproducibility_report
from benthic_model.inference.predict import predict_with_run
from benthic_model.models.train import train_and_register_run
from benthic_model.submission.writer import load_allowed_classes, write_submission

Handler = Callable[[argparse.Namespace], int]


def _train_handler(args: argparse.Namespace) -> int:
    try:
        train_csv = Path(args.train_csv)
        data_root = train_csv.parent
        metadata_path = data_root / "METADATA.MD"

        enforce_competition_data_whitelist(train_csv, data_root)
        enforce_competition_data_whitelist(args.bathymetry_tif, data_root)
        enforce_competition_data_whitelist(args.backscatter_tif, data_root)
        parse_and_validate_metadata(metadata_path)

        command = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "train"
        metadata = train_and_register_run(
            train_csv=args.train_csv,
            bathymetry_tif=args.bathymetry_tif,
            backscatter_tif=args.backscatter_tif,
            config_path=args.config,
            run_type=args.run_type,
            seed=args.seed,
            command=command,
        )
        print(f"Training completed. run_id={metadata.run_id}")
        return 0
    except Exception as exc:
        print(f"Train failed: {exc}")
        return 1


def _evaluate_handler(args: argparse.Namespace) -> int:
    try:
        if args.fold_scheme != "spatial_blocked":
            raise ValueError(
                "Model selection must use fold-scheme spatial_blocked. "
                f"Received: {args.fold_scheme}"
            )

        payload = compare_run_to_baseline(
            candidate_run_id=args.run_id,
            fold_scheme=args.fold_scheme,
            override=False,
        )

        report_path = Path("reports") / "metrics" / f"{args.run_id}_comparison.md"
        write_comparison_report(payload, report_path)

        json_path = Path("reports") / "metrics" / f"{args.run_id}_metrics.json"
        csv_path = Path("reports") / "metrics" / f"{args.run_id}_per_class.csv"
        write_machine_readable_metric_summary(payload, json_path, csv_path)

        write_reproducibility_report(args.run_id, tolerance=0.01)

        print(f"Evaluation completed. report={report_path}")
        return 0
    except Exception as exc:
        print(f"Evaluate failed: {exc}")
        return 1


def _predict_handler(args: argparse.Namespace) -> int:
    try:
        output_path = predict_with_run(
            run_id=args.run_id,
            test_csv=args.test_csv,
            bathymetry_tif=args.bathymetry_tif,
            backscatter_tif=args.backscatter_tif,
        )
        print(f"Prediction completed. output={output_path}")
        return 0
    except Exception as exc:
        print(f"Predict failed: {exc}")
        return 1


def _make_submission_handler(args: argparse.Namespace) -> int:
    try:
        allowed_classes = load_allowed_classes()
        output_path = write_submission(
            predictions_path=args.predictions,
            sample_submission_path=args.sample_submission,
            output_path=args.output,
            allowed_classes=allowed_classes,
        )
        print(f"Submission written: {output_path}")
        return 0
    except Exception as exc:
        print(f"Make-submission failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benthic-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a baseline or candidate model run")
    train.add_argument("--train-csv", required=True)
    train.add_argument("--bathymetry-tif", required=True)
    train.add_argument("--backscatter-tif", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-type", choices=["baseline", "candidate"], default="baseline")
    train.set_defaults(handler=_train_handler)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained run")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument(
        "--fold-scheme",
        choices=["spatial_blocked", "stratified_random"],
        default="spatial_blocked",
    )
    evaluate.set_defaults(handler=_evaluate_handler)

    predict = subparsers.add_parser("predict", help="Generate test predictions")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--test-csv", required=True)
    predict.add_argument("--bathymetry-tif", required=True)
    predict.add_argument("--backscatter-tif", required=True)
    predict.set_defaults(handler=_predict_handler)

    submission = subparsers.add_parser("make-submission", help="Write validated submission CSV")
    submission.add_argument("--predictions", required=True)
    submission.add_argument("--sample-submission", required=True)
    submission.add_argument("--output", required=True)
    submission.set_defaults(handler=_make_submission_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
