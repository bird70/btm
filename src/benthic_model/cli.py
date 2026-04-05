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
from benthic_model.segmentation.benchmark import run_segmentation_benchmark
from benthic_model.segmentation.io import write_mask_artifacts
from benthic_model.segmentation.masking import build_centered_window_mask
from benthic_model.segmentation.stacking import run_hybrid_stacking
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


def _segmentation_build_masks_handler(args: argparse.Namespace) -> int:
    try:
        import pandas as pd

        train = pd.read_csv(args.train_csv)
        mask, metadata = build_centered_window_mask(
            train,
            class_col=args.class_col,
            window_size=args.window_size,
        )
        from pathlib import Path

        run_dir = Path(args.output_dir)
        write_mask_artifacts(run_dir, mask, metadata)
        print(f"Mask artifacts written to: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Segmentation-build-masks failed: {exc}")
        return 1


def _segmentation_benchmark_handler(args: argparse.Namespace) -> int:
    try:
        run_dir = run_segmentation_benchmark(args.config)
        print(f"Segmentation benchmark completed. run_dir={run_dir}")
        return 0
    except Exception as exc:
        print(f"Segmentation-benchmark failed: {exc}")
        return 1


def _hybrid_stack_handler(args: argparse.Namespace) -> int:
    try:
        metrics_path, decision_path, prediction_path = run_hybrid_stacking(
            baseline_run=args.baseline_run,
            seg_run=args.seg_run,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            seg_base_dir=args.seg_base_dir,
            random_state=args.seed,
        )
        print(
            "Hybrid stack completed. "
            f"metrics={metrics_path} decision={decision_path} predictions={prediction_path}"
        )
        return 0
    except Exception as exc:
        print(f"Hybrid-stack failed: {exc}")
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

    segmentation_masks = subparsers.add_parser(
        "segmentation-build-masks",
        help="Create segmentation masks from labeled points",
    )
    segmentation_masks.add_argument("--train-csv", required=True)
    segmentation_masks.add_argument("--output-dir", required=True)
    segmentation_masks.add_argument("--class-col", default="class")
    segmentation_masks.add_argument("--window-size", type=int, default=5)
    segmentation_masks.set_defaults(handler=_segmentation_build_masks_handler)

    segmentation_benchmark = subparsers.add_parser(
        "segmentation-benchmark",
        help="Benchmark configured segmentation candidates",
    )
    segmentation_benchmark.add_argument("--config", required=True)
    segmentation_benchmark.set_defaults(handler=_segmentation_benchmark_handler)

    hybrid_stack = subparsers.add_parser(
        "hybrid-stack",
        help="Stack segmentation outputs with baseline signals",
    )
    hybrid_stack.add_argument("--baseline-run", required=True)
    hybrid_stack.add_argument("--seg-run", required=True)
    hybrid_stack.add_argument("--train-csv", required=True)
    hybrid_stack.add_argument("--test-csv", required=True)
    hybrid_stack.add_argument("--seg-base-dir", default="artifacts/segmentation")
    hybrid_stack.add_argument("--seed", type=int, default=42)
    hybrid_stack.set_defaults(handler=_hybrid_stack_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
