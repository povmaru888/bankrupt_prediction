# 企業破產預測系統驗收報告（Draft）

## 執行環境

- 日期：2026-10-22
- Windows／Python 虛擬環境；完整測試：223 passed
- `ruff check src tests`：通過
- `ruff format --check src tests`：通過

## 需求追溯

| 需求 | 實作元件 | 介面 | 測試證據 |
| --- | --- | --- | --- |
| F01 | training loader、schema、row quality、audit logger | `load_training_data`、`/api/v1/score` | data、security、API workflow |
| F02 | split manager、orchestrator | offline `train` | training split、acceptance |
| F03 | imputer、range monitor、pipelines | fitted model pipeline | features、no leakage |
| F04 | model search、model selection | offline `train` | model search、selection |
| F05 | logistic／XGBoost class weighting | offline pipeline | training pipeline tests |
| F06 | threshold selector、response formatter | `threshold` query、UI slider | scoring、response、UI |
| F07 | scoring service、metrics | `/api/v1/score` | service、acceptance matrix |
| F08 | SHAP service、case page | score response、case page | explanation、UI |
| F09 | result contracts、CSV exporter、temp workspace | JSON／CSV score | export、API、security |
| F10 | registry、health/model、overview/performance pages | `/health`、`/api/v1/model` | registry、API、UI |

## 判定

目前判定：**Not Ready for Release**。程式與自動化驗收已完成；正式 30 組／5-fold 訓練曾啟動但因長時間搜尋中止，`models/` 仍無候選包或 active model。final-test 品質門檻、人工 activate 與 Windows／瀏覽器實機證據仍是 Release 阻擋項。完成這些外部狀態後，才可將 checklist 的未放行項目改為 `[x]`。
