"""Build a source-only Colab handoff and executable notebook (no private data)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "colab"


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = OUT / "bankruptcy_colab_source.zip"
    paths = [ROOT / name for name in (
        "pyproject.toml", "requirements.txt", "requirements-dev.txt", ".gitignore", "ruff.toml",
        ".github/workflows/ci.yml", "scripts/start_api.ps1", "scripts/start_ui.ps1",
        "reports/acceptance_report.md", "docs/RELEASE_CHECKLIST.md",
    )]
    for folder, pattern in (("src", "*.py"), ("config", "*.yaml"), ("tests", "*.py")):
        paths.extend(sorted((ROOT / folder).rglob(pattern)))
    paths.extend(sorted((ROOT / "tests" / "fixtures").glob("*.csv")))
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    cells = []

    def md(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

    def code(source: str) -> None:
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    md("""# 企業破產預測：Colab 訓練交接
依序執行每格，不使用「全部執行」。本包是既有專案快照，不代表系統驗收完成。

**預算 10 運算單元，非自動停扣上限。** 先記下帳戶餘額，訓練中定期查看消耗。
目前兩個模型均為 CPU 實作：請選 CPU 執行階段，勿為此開 GPU／TPU。
Colab 不保證比本機更快；勿為趕時間修改搜尋次數、切分或種子。
完成／停止後，下載結果，再透過 Colab「中斷連線並刪除執行階段」釋放資源。

需要本筆記本、bankruptcy_colab_source.zip、原始 bankrupt_data.csv。
不需要 Google 密碼或權杖，不掛載整個 Drive。上傳資料只供本次執行；
執行階段是暫存，斷線或重建可能遺失結果。請勿公開分享含輸出的筆記本。
模型信心分數僅供分析優先順序使用，不是未來一年破產機率。
""")
    md("## 1. 上傳程式 ZIP（此格只選 ZIP）")
    code(f'''
        import hashlib, json, os, pathlib, platform, subprocess, sys, time, zipfile
        from google.colab import files
        uploaded = files.upload()
        assert list(uploaded) == ["bankruptcy_colab_source.zip"], "請只上傳指定 ZIP"
        blob = uploaded["bankruptcy_colab_source.zip"]
        assert hashlib.sha256(blob).hexdigest() == "{digest}", "程式包校驗失敗"
        ROOT = pathlib.Path("/content/bankruptcy_project")
        ROOT.mkdir(exist_ok=True)
        import io
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for info in archive.infolist():
                target = (ROOT / info.filename).resolve()
                assert target.is_relative_to(ROOT.resolve()), "不安全路徑"
            archive.extractall(ROOT)
        del uploaded, blob
        os.chdir(ROOT)
        EVIDENCE = ROOT / "colab_evidence"
        EVIDENCE.mkdir(exist_ok=True)
        print("程式包 SHA256: {digest}")
    ''')
    md("## 2. 安裝隔離環境\n使用獨立 venv，不改動 Colab 內建套件。若固定版本無法安裝，停止並回傳錯誤，不自行升降版本。")
    code('''
        subprocess.run([sys.executable, "-m", "venv", ".colab_venv"], check=True)
        PYTHON = str(ROOT / ".colab_venv/bin/python")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        def run_logged(args, name):
            with (EVIDENCE / name).open("w") as log:
                process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)
                        log.flush()
                    result = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if result:
                    raise RuntimeError(f"執行失敗，請回傳 {name}，不要繼續")
        run_logged([PYTHON, "-m", "pip", "install", "-r", "requirements-dev.txt"], "install.log")
        run_logged([PYTHON, "-m", "pip", "check"], "dependency_check.log")
        versions = subprocess.check_output([PYTHON, "-m", "pip", "freeze"], text=True)
        (EVIDENCE / "packages.txt").write_text(versions)
        (EVIDENCE / "environment.json").write_text(json.dumps({
            "platform": platform.platform(), "python": sys.version,
            "cpu_count": os.cpu_count(),
            "cpu_info": pathlib.Path("/proc/cpuinfo").read_text(),
            "memory": pathlib.Path("/proc/meminfo").read_text(),
            "budget_units": 10, "device": "cpu",
        }, indent=2))
        ENVIRONMENT_OK = True
    ''')
    md("## 3. 上傳 CSV（此格只選 CSV）")
    code('''
        assert ENVIRONMENT_OK
        uploaded = files.upload()
        assert len(uploaded) == 1, "請只上傳原始 CSV"
        filename, data = next(iter(uploaded.items()))
        assert filename.lower().endswith(".csv")
        (ROOT / "Database").mkdir(exist_ok=True)
        (ROOT / "Database/bankrupt_data.csv").write_bytes(data)
        (EVIDENCE / "data_sha256.txt").write_text(hashlib.sha256(data).hexdigest())
        del uploaded, data
    ''')
    md("## 4. 小型試跑：只使用 train partition\n只驗證載入、fit、predict、SHAP 相容性；不選模型、不計算 final test 指標、不產生正式模型包。試跑的迭代上限不能當成正式收斂證據。")
    smoke = '''
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.training.split_manager import create_split
from bankruptcy_prediction.training.logistic_pipeline import build_logistic_pipeline
from bankruptcy_prediction.training.xgboost_pipeline import build_xgboost_pipeline
from bankruptcy_prediction.explanation.shap_service import explain
from sklearn.model_selection import train_test_split
import numpy as np
d = load_training_data("Database/bankrupt_data.csv")
d.features.columns = [field.name for field in build_feature_schema(d).features]
s = create_split(d.labels.to_numpy(), seed=42)
indices, _ = train_test_split(list(s.train_indices), train_size=400,
                            stratify=d.labels.iloc[list(s.train_indices)], random_state=42)
X, y = d.features.iloc[indices], d.labels.iloc[indices]
for family, model in (("logistic_regression", build_logistic_pipeline(max_iter=50)),
                      ("xgboost", build_xgboost_pipeline())):
    if family == "xgboost":
        model.set_params(classifier__n_estimators=5, classifier__max_depth=2)
    model.fit(X, y)
    assert np.isfinite(model.predict_proba(X.iloc[:3])).all()
    explain(family, model, X.iloc[:3])
    print(family, "smoke passed", flush=True)
print("No validation/test scores computed.")
'''
    code('assert ENVIRONMENT_OK\nrun_logged([PYTHON, "-m", "pytest", "-q"], "pytest.log")\nrun_logged([PYTHON, "-c", ' + repr(smoke) +
         '], "smoke.log")\nSMOKE_OK = True')
    md("""## 5. 人工預算確認與正式訓練
先查看目前餘額，確認距最初餘額消耗未達 10 單元。
本格執行兩模型各 30 組、5-fold 搜尋 → 5-fold sigmoid 校準 → validation 選模與 F2 門檻 → final test 一次 → 候選模型包。
過程可能長時間沒有輸出；請勿重複啟動。達預算時按停止，下載日誌並釋放執行階段。

**重跑保護僅限同一執行階段**。重建執行階段、遺失標記或中途失敗後，不可自行再次正式訓練：
先回傳日誌判斷是否已評估 final test。不可根據 final test 結果調參。
此版沒有跨工作階段 checkpoint／預算 API，也不能保證中斷後保留模型。
""")
    code('''
        assert SMOKE_OK
        marker = EVIDENCE / "formal_started.json"
        assert not marker.exists(), "已啟動過正式訓練，請回傳日誌，不可直接重跑"
        confirm = input("已確認餘額且接受正式測試只執行一次，請輸入 START：")
        assert confirm == "START", "未啟動"
        with marker.open("x") as output:
            json.dump({"started_unix": time.time(), "budget_units": 10}, output)
        start = time.monotonic()
        try:
            run_logged([PYTHON, "-m", "bankruptcy_prediction.cli", "train",
                        "--data", "Database/bankrupt_data.csv",
                        "--config", "config/training.yaml",
                        "--service-config", "config/service.yaml",
                        "--models-root", "models"], "formal_training.log")
        finally:
            (EVIDENCE / "elapsed_seconds.txt").write_text(str(time.monotonic() - start))
        print("流程已結束：請檢查 quality_gate_passed，不能以 exit code 判定達標。")
        print("未自動啟用模型；Windows CPU/UI/效能驗收仍待完成。")
    ''')
    md("## 6. 下載證據與模型包（成功或失敗都執行）\n請將 ZIP 回傳本對話供檢查，勿自行 activate。下載後中斷並刪除執行階段。ZIP 不包含原始 CSV。")
    code('''
        result_path = pathlib.Path("/content/bankruptcy_colab_results.zip")
        with zipfile.ZipFile(result_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in (EVIDENCE, ROOT / "models"):
                if folder.exists():
                    for path in sorted(folder.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(ROOT).as_posix())
        files.download(str(result_path))
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"step-{index:02d}"
    (OUT / "bankruptcy_training_colab.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "SHA256.txt").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    print(f"Built {len(paths)} source files and {len(cells)} notebook cells in {OUT}")


if __name__ == "__main__":
    build()
