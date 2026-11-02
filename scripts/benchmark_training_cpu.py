"""Time representative fits on one training fold; never evaluate held-out data."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import platform
import time
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.training.logistic_pipeline import build_logistic_pipeline
from bankruptcy_prediction.training.split_manager import create_split
from bankruptcy_prediction.training.xgboost_pipeline import build_xgboost_pipeline


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in (
            "total_physical", "available_physical", "total_page", "available_page",
            "total_virtual", "available_virtual", "extended_virtual",
        )
    ]


root = Path(__file__).resolve().parents[1]
data = load_training_data(root / "Database/bankrupt_data.csv")
data.features.columns = [item.name for item in build_feature_schema(data).features]
split = create_split(data.labels.to_numpy(), seed=42)
X = data.features.iloc[list(split.train_indices)]
y = data.labels.iloc[list(split.train_indices)]
fold, _ = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
memory = MemoryStatus()
memory.length = ctypes.sizeof(memory)
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
report = {
    "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
    "ram_gib": round(memory.total_physical / 2**30, 2),
    "available_ram_gib": round(memory.available_physical / 2**30, 2),
    "fold_rows": len(fold), "input_features": X.shape[1], "fits": [],
    "held_out_evaluated": False,
}
print(json.dumps(report), flush=True)
models = [
    ("logistic_standard", build_logistic_pipeline().set_params(scaler=StandardScaler())),
    ("logistic_robust", build_logistic_pipeline().set_params(scaler=RobustScaler())),
    ("xgboost_300_trees", build_xgboost_pipeline()),
]
for name, model in models:
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X.iloc[fold], y.iloc[fold])
    record = {"name": name, "seconds": round(time.perf_counter() - start, 3),
              "warnings": sorted({str(item.message) for item in caught})}
    report["fits"].append(record)
    print(json.dumps(record), flush=True)
(root / "reports/local_cpu_benchmark.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
