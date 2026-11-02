# 10,000 列效能驗收

## 測試條件

- 測試：`tests/performance/test_score_10000.py`
- 資料：deterministic 10,000 列、3 欄 schema-complete synthetic CSV frame
- 範圍：schema／列級驗證、已 fit `predict_proba`、前三項解釋投影與 JSON 序列化
- 暖機：pytest process 啟動後單次量測；不包含模型訓練
- 門檻：完整同步流程小於 10 秒

## 結果

執行 `PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/performance/test_score_10000.py`；測試以 wall-clock assertion 驗證小於 10 秒，通過即代表本機驗收條件達成。CPU、記憶體與套件版本由正式驗收機執行時記錄；本報告不宣稱跨硬體效能。
