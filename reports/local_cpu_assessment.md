# 本機 CPU 訓練評估 — 2026-10-22

決策：使用本機 CPU 正式訓練，不啟用 Colab 或消耗其運算單元。

本機 Intel Core i7-8665U，8 邏輯處理器，15.81 GiB 實體記憶體；試跑開始時可用約 6.92 GiB。
固定種子 42，取 64% 訓練 partition 的第一個 5-fold 訓練折，共 3,491 列、95 個輸入特徵。
沒有在 validation 或 final test 上計算分數，也未依測試資料調整搜尋設定。

| 代表性設定 | 單次 fit 耗時 | 警告 |
| --- | ---: | --- |
| Logistic Regression + StandardScaler | 9.667 秒 | 無 |
| Logistic Regression + RobustScaler | 12.189 秒 | 達 5,000 次迭代仍未收斂 |
| XGBoost，300 棵樹、預設深度 6 | 2.301 秒 | 無 |

兩模型各 30 組 × 5 折，共 300 次 CV fit，另有 refit 與校準。
按代表性 fit 線性外推約 30–40 分鐘，但搜尋包含不同樹數、深度、C、L1/L2，
加上散熱降頻與系統負載，這不是完成時間保證。
單次完整訓練折能正常完成，現有資料規模沒有顯示必須使用 GPU；不以 GPU 取代模型收斂問題的調查。

正式訓練保留原本 n_jobs=1 與依序執行，不同時執行多個 task。
啟用進度日誌與搜尋警告彙整，不改變模型、超參數搜尋空間或測試門檻。
模型品質仍需正式結果證明，模型啟用仍需人工檢視。

量測原始資料：`reports/local_cpu_benchmark.json`；量測腳本：`scripts/benchmark_training_cpu.py`。
