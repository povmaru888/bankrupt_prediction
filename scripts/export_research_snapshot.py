"""Export the research dashboard's display contract as a static snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bankruptcy_prediction.ui.pages.research_dashboard import (  # noqa: E402
    CANDIDATE,
    CHARTS,
    SHAP_TOP_FEATURES,
    _active_version,
    load_class_weight_comparison,
    load_final_test_comparison,
    load_objective_comparison,
    load_training_comparison,
)

OUTPUT_DIR = ROOT / "deploy" / "research_snapshot"


def _records(frame):
    return json.loads(frame.reset_index().to_json(orient="records"))


def export_snapshot(output_dir: Path = OUTPUT_DIR) -> Path:
    """Write images first, then atomically publish the JSON manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    final_test, xgboost, logistic = load_final_test_comparison(ROOT)
    training = load_training_comparison(ROOT)
    class_weight = load_class_weight_comparison(ROOT)
    objectives = load_objective_comparison(ROOT)

    images = []
    for title, relative_name in CHARTS:
        source = ROOT / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing research chart: {source}")
        destination = image_dir / source.name
        shutil.copy2(source, destination)
        images.append(
            {
                "title": title,
                "file": f"images/{destination.name}",
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate": CANDIDATE,
        "active_model": _active_version(ROOT),
        "final_test_comparison": _records(final_test),
        "final_test_thresholds": {
            "xgboost": float(xgboost["threshold"]),
            "logistic_regression": float(logistic["threshold"]),
        },
        "training_comparison": _records(training),
        "class_weight_comparison": _records(class_weight),
        "objective_comparison": _records(objectives),
        "images": images,
        "shap_top_features": [
            {
                "rank": rank,
                "english": english,
                "chinese": chinese,
                "explanation": explanation,
            }
            for rank, english, chinese, explanation in SHAP_TOP_FEATURES
        ],
    }

    handle, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix="research_snapshot.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        target = output_dir / "snapshot.json"
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


if __name__ == "__main__":
    result = export_snapshot()
    print(f"Wrote {result.relative_to(ROOT)} ({result.stat().st_size:,} bytes)")
