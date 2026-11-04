# Research snapshot app

This entrypoint is a read-only presentation of already-computed research
results. It does not import or start the scoring API, load model binaries,
train models, or run inference.

Refresh the snapshot from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\export_research_snapshot.py
```

Preview locally on port 8503:

```powershell
.\.venv\Scripts\python.exe -m streamlit run deploy\research_app.py --server.address 127.0.0.1 --server.port 8503
# 或執行腳本：
powershell -ExecutionPolicy Bypass -File scripts\start_research_ui.ps1
```

If deployed later, use `deploy/research_app.py` as the main file and
`deploy/requirements.txt` as the minimal dependency contract.
