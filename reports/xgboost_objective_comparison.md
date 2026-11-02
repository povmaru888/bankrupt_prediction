# XGBoost objective 實驗比較

日期：2026-10-22

比較模型：`binary:logistic`、`rank:pairwise`、`reg:pseudohubererror`。

資料治理：固定 seed 42 與既有 64%／16%／20% 分層切分；兩個新模型只用 training 的 30 組 × 5-fold 搜尋，再用既有 validation 比較；**未存取 final test**。

## Validation 排序表現

Validation 共 1,091 列、35 筆正例；Top 10% 依規則取 `ceil(1091 × 10%) = 110` 列。

| Objective | Average Precision | Recall@Top10% | Precision@Top10% | NDCG@Top10% | Top10 命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `binary:logistic`（既有） | 0.53660 | **85.71%** | **27.27%** | 0.79817 | **30/35** |
| `rank:pairwise` | **0.57585** | 82.86% | 26.36% | 0.79913 | 29/35 |
| `reg:pseudohubererror` | 0.53633 | **85.71%** | **27.27%** | **0.80077** | **30/35** |

相對既有 binary 模型：

- Rank 的 AP 增加 0.03925，約 +7.3%，但 Top 10% 少找到 1 筆正例，Recall@Top10% 下降 2.86 個百分點。
- Pseudo-Huber 的 AP 低 0.00027，實質相同；Top 10% 命中相同，NDCG 高 0.00260，差距很小。

## Training CV 與耗時

| Objective | 最佳 5-fold CV AP | 實驗總耗時 |
| --- | ---: | ---: |
| `binary:logistic` | 0.44479 | 既有正式訓練；本次未重跑 |
| `rank:pairwise` | **0.45455** | 660.31 秒（約 11.0 分鐘） |
| `reg:pseudohubererror` | 0.42664 | 916.59 秒（約 15.3 分鐘） |

## 最佳超參數

### `rank:pairwise`

- `n_estimators=660`
- `max_depth=8`
- `learning_rate=0.0281451`
- `subsample=0.618666`
- `colsample_bytree=0.996885`
- `min_child_weight=9`
- `reg_alpha=0.000528411`
- `reg_lambda=0.0998401`
- `lambdarank_num_pair_per_sample=4`

資料沒有公司群組或 query ID，因此每個 CV fit／全 training refit 被視為一個 ranking query。這是可執行的實驗定義，但它不是來自業務資料的自然 query 分組。

### `reg:pseudohubererror`

- `n_estimators=379`
- `max_depth=9`
- `learning_rate=0.0224398`
- `subsample=0.759530`
- `colsample_bytree=0.868854`
- `min_child_weight=6`
- `reg_alpha=0.000170886`
- `reg_lambda=82.625806`
- `huber_slope=3.336101`

每個 training fold 依該 fold 的負例／正例比例建立正例 sample weight，沒有使用 validation 或 test 的類別比例。

## 判讀與建議

若只看 validation AP，`rank:pairwise` 最好；若核心營運目標是固定 Top 10% 名單的破產涵蓋率，既有 binary 與 Pseudo-Huber 都比 rank 多命中 1 筆。Pseudo-Huber 相對 binary 沒有實質改善。

目前不建議直接取代 production candidate：

1. Binary 已用 final test 一次並通過 MVP 門檻；新模型沒有可使用的未見測試集，不能以既有 final test 再做選模。
2. Rank 與 Huber 輸出是排序分數，不是經 sigmoid calibration 的模型信心分數；本實驗套用單調 sigmoid 只為 sklearn 介面，排序指標不受影響，但不能直接沿用 binary 的門檻或 Brier Score。
3. Rank 的 validation AP 改善伴隨 Top10 少命中一筆；是否值得取代需由新的外部 holdout 或未來時點資料驗證。

因此維持既有 `binary:logistic` XGBoost 為 production candidate；保存 rank 與 Huber 作為實驗模型。取得全新外部測試資料後，再以預先固定的 AP、Recall@Top10%、NDCG@Top10% 規則做一次最終比較。

## 可重現證據

- 比較資料：`experiments/xgboost-objectives-20261022/comparison.json`
- 模型：`xgboost_rank_pairwise.joblib`、`xgboost_pseudo_huber.joblib`
- 完整日誌：`reports/xgboost_objective_experiments.log`
- 執行程式：`scripts/train_xgboost_objective_experiments.py`
- 測試結果：226 passed；模型已在新程序獨立載入並重算 validation 指標一致。

本實驗的分數僅供分析優先順序使用，不代表未來一年破產機率或因果關係。
