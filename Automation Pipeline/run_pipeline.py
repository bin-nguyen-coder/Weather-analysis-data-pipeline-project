"""Local end-to-end pipeline runner (no Airflow required).

Executes every stage of the weather pipeline in the documented order:

    Ingestion -> Cleaning -> QualityChecks -> Transform (metrics)
    -> FeaturePrep (scaling) -> Aggregation (SQL)
    -> FeatureEngineering (windows) -> ML Train -> Predict -> EDA

Usage examples
--------------
    python "Automation Pipeline/run_pipeline.py"                 # run all steps
    python "Automation Pipeline/run_pipeline.py" --only ingest   # single step
    python "Automation Pipeline/run_pipeline.py" --skip eda      # skip a step

A JSON run report is written to ``data/pipeline_run.json``.

NOTE: the folder name contains a space, so Airflow / imports never target it
directly -- the script is launched with ``python path/run_pipeline.py`` and
imports the other (cleanly named) pipeline packages from the repo root.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from more_itertools import only

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger("run_pipeline")


# --------------------------------------------------------------------------- #
# Stage definitions (ordered)
# --------------------------------------------------------------------------- #
def _step_ingest(config: Dict[str, Any]) -> Dict[str, Any]:
    from Ingestion import db_upsert

    count = db_upsert.fetch_and_upsert(config)
    return {"ingested_rows": count}


def _step_clean(config: Dict[str, Any]) -> Dict[str, Any]:
    from Cleaning import clean_observations

    return clean_observations.run(config)


def _step_quality(config: Dict[str, Any]) -> Dict[str, Any]:
    from Cleaning import quality_checks

    return quality_checks.run(config)


def _step_transform(config: Dict[str, Any]) -> Dict[str, Any]:
    from Transform import calculate_heat_index

    return calculate_heat_index.run(config)


def _step_feature_prep(config: Dict[str, Any]) -> Dict[str, Any]:
    from Transform import transform_features

    return transform_features.run(config)


def _step_aggregation(config: Dict[str, Any]) -> Dict[str, Any]:
    from Aggregation import run_aggregations

    return run_aggregations.run(config)


def _step_feature_engineering(config: Dict[str, Any]) -> Dict[str, Any]:
    from Train import feature_engineering

    return feature_engineering.run(config)


def _step_train(config: Dict[str, Any]) -> Dict[str, Any]:
    from Train import train_lstm

    return train_lstm.run(config)


def _step_predict(config: Dict[str, Any]) -> Dict[str, Any]:
    from Forecast import predict

    return predict.run(config)


def _step_eda(config: Dict[str, Any]) -> Dict[str, Any]:
    from EDA import generate_reports

    return generate_reports.run(config)


# (name, human label, callable)
STEPS: List[tuple] = [
    ("ingest", "Ingestion", _step_ingest),
    ("clean", "Cleaning", _step_clean),
    ("quality", "QualityChecks", _step_quality),
    ("transform", "Transform", _step_transform),
    ("feature_prep", "FeaturePrep", _step_feature_prep),
    ("aggregation", "Aggregation", _step_aggregation),
    ("feature_engineering", "FeatureEngineering", _step_feature_engineering),
    ("train", "MLTrain", _step_train),
    ("predict", "Predict", _step_predict),
    ("eda", "EDA", _step_eda),
]


def _resolve_steps(
    only: Optional[List[str]], skip: Optional[List[str]]
) -> List[tuple]:
    selected = list(STEPS)
    if only:
        selected = [item for item in STEPS if item[0] in only]
    if skip:
        selected = [item for item in selected if item[0] not in skip]
    # Lấy danh sách tên bước từ phần tử đầu tiên của mỗi tuple trong STEPS
    step_names = [step[0] for step in STEPS]
    missing = [name for name in (only or []) if name not in step_names]
    if only and missing:
        raise ValueError(
            f"Unknown step(s): {missing}. Available: {[s[0] for s in STEPS]}"
        )
    return selected


def run_pipeline(
    only: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute the pipeline stages and return the aggregated run report."""
    config = load_config()
    selected = _resolve_steps(only, skip)

    run_report: Dict[str, Any] = {
        "started_at": _now_iso(),
        "stages": {},
    }
    logger.info("Starting weather pipeline (%d stages).", len(selected))

    for name, label, callable_ in selected:
        stage_start = _now_iso()
        logger.info("=" * 72)
        logger.info("Stage [%s]: %s %s", name, label, stage_start)
        try:
            result = callable_(config)
            run_report["stages"][name] = {
                "status": "success",
                "started_at": stage_start,
                "finished_at": _now_iso(),
                "result": _serialize(result),
            }
            logger.info("Stage %s finished OK -> %s", label, _serialize(result))
        except Exception as exc:  # noqa: BLE001 - fail fast on any stage error
            run_report["stages"][name] = {
                "status": "failed",
                "started_at": stage_start,
                "finished_at": _now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            run_report["finished_at"] = _now_iso()
            run_report["overall_status"] = "failed"
            _write_run_report(config, run_report)
            logger.exception("Stage %s FAILED", label)
            raise

    run_report["finished_at"] = _now_iso()
    run_report["overall_status"] = "success"
    _write_run_report(config, run_report)
    logger.info("Pipeline completed successfully.")
    return run_report


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_run_report(config: Dict[str, Any], report: Dict[str, Any]) -> Path:
    path = Path(config["paths"]["run_report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Run report written to %s", path)
    return path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the weather data pipeline end-to-end locally."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        help="Run only these stages (e.g. --only ingest clean).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="STEP",
        help="Skip these stages (e.g. --skip train eda).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_arguments()
    report = run_pipeline(only=args.only, skip=args.skip)
    print(json.dumps(report, indent=2, default=str))
