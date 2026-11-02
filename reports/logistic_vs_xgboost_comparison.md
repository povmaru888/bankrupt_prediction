# Logistic Regression 基準模型與 XGBoost 比較

日期：2026-10-22

資料：6,819 列、220 筆正例；固定 seed 42；分層 64%／16%／20% 切分。

比較原則：兩模型均只用 training partition 搜尋及校準，再用同一 validation partition 選模；final test 僅評估入選模型一次。

## 基準定義

Logistic Regression 是本專案的**基準模型（baseline model）**：結構簡單、可解釋，用來判斷較複雜模型是否帶來實質增益。
另有一個純統計基準是正例盛行率 3.2258%，用於 MVP 的 Average Precision 倍率門檻；兩者不要混為一談。

## 同條件比較

| 項目 | Logistic Regression（基準） | XGBoost | 判讀 |
| --- | ---: | ---: | --- |
| 搜尋設定 | 30 組 × 5-fold | 30 組 × 5-fold | 相同預算與 scoring |
| 最佳 training-CV Average Precision | 0.4075 | 0.4448 | XGBoost +0.0373，約 +9.1% |
| 5-fold sigmoid 校準後 validation AP | 0.4235 | 0.5366 | XGBoost +0.1131，約 +26.7% |
| validation AP 選模差距 | — | +0.1131 | 大於 0.01，依規則選 XGBoost |
| 失敗候選組數 | 0 | 0 | 兩者均完成 30 組 |
| CV fit 日誌時間合計 | 1,847.4 秒 | 408.3 秒 | 本次 XGBoost 約快 4.5 倍 |
| 平均每個 CV fit | 12.32 秒 | 2.72 秒 | 僅為本機本次搜尋量測 |
| 收斂／fit 警告 | 57 次未收斂警告 | 0 | Logistic 的部分參數達 5,000 iterations |
| 模型能力 | 線性主效應 | 非線性與交互作用 | XGBoost 表達力較高 |
| 解釋方式 | Linear SHAP／係數方向 | Tree SHAP | 均為模型貢獻，不代表因果 |

最佳 Logistic Regression 設定為 StandardScaler、L1、`C=0.0467378671`、`class_weight=balanced`。
最佳 XGBoost 設定為 956 棵樹、深度 9、learning rate 0.0161496、`scale_pos_weight=balanced`，其餘參數見正式模型報告。

## Final test（僅入選 XGBoost）

| 指標 | XGBoost |
| --- | ---: |
| Average Precision | 0.5204 |
| AP／盛行率倍率 | 16.13 倍 |
| Precision | 33.33% |
| Recall | 65.91% |
| F1 | 44.27% |
| Brier Score | 0.02215 |
| Recall@Top10% | 75.00% |
| False positives | 58 |
| F2 validation 預設門檻 | 0.0497944 |

沒有 Logistic Regression 的 final-test 數值，這是刻意的治理設計，不是資料遺漏：final test 只允許在選模及門檻固定後對入選模型評估一次。現在補測 Logistic Regression 會把 final test 變成模型比較資料，違反「final test 只評估一次」與防洩漏要求。

## 結論

XGBoost 在校準後 validation AP 上相對 Logistic Regression 提升約 26.7%，差距明顯超過 0.01 的簡單模型優先規則；本次也沒有 fit 警告，且搜尋更快。因此應保留 Logistic Regression 作為基準與稽核參考，候選 active model 採 XGBoost。

此結論只代表模型信心排序能力與目前固定資料切分結果；不代表因果關係或公司未來一年破產機率。正式啟用前仍須完成模型包人工檢視及 Windows 端到端驗收。

證據：`models/candidate-20261022T105726Z-155ceb0b/model_report.html`、`reports/local_formal_training.log`、候選模型包的 `manifest.json` 與 `metrics.json`。
