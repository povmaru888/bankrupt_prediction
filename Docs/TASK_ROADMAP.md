# 企業破產預測系統 工作分解結構

| 欄位 | 內容 |
| --- | --- |
| 文件版本 | v0.1 |
| 對應文件 | 企業破產預測系統 SDD Draft 0.1 |
| 專案方式 | 垂直切片加技術層級 |
| 任務狀態 | Phase 0、Phase 1、Phase 2、Phase 3、Phase 4 已完成；Phase 5 進行中，P5-01 至 P5-09 已完成；P5-10 等待正式模型與實機 Release 證據 |
| 完成定義 | 目標模組完成、公開介面可呼叫、驗收測試通過、相依任務狀態正確，且 F01 至 F10 均有測試證據。 |

## 1. 使用規則

本文件將 SDD 拆成可獨立驗證的最小實作單位。任務完成前不得將下游任務標示為完成；若設計邊界、資料欄位、標籤定義、預測窗口、模型家族、外部連線、認證方式或保存政策改變，須先更新 SDD，再調整本 Roadmap。

### 1.1 三個一原則

每個 Task 都有：

1. 一個明確的目標檔案或模組。
2. 一組明確的輸入與輸出，以函式簽名、CLI 或 HTTP 介面描述。
3. 一組可自動驗證的驗收準則，優先對應 pytest；UI 任務另含可重現的操作驗收。

### 1.2 系統邊界決策

- Streamlit 與 FastAPI 是兩個只綁定 `127.0.0.1` 的本機程序；Streamlit 不得直接載入模型或執行 SHAP。
- FastAPI 是檔案驗證、模型載入、評分、解釋、排序與匯出的唯一線上邊界。
- 訓練、啟用及回滾只由離線 CLI 執行；Web 請求不得觸發訓練或修改 active model。
- 上傳檔與評分結果只存在當次請求的隨機暫存目錄；模型包、指標、報告與必要稽核日誌才永久保存。
- 本 Roadmap 不包含原始財報解析、95 指標自動計算、雲端部署、多人帳號、資料庫歷史、SMOTE、第三種模型或自動決策。

### 1.3 任務狀態規則

- `[ ]`：未開始或尚未提供可驗證證據。
- `[~]`：進行中，但尚未滿足全部驗收準則。
- `[x]`：目標檔案已完成、測試通過、直接相依均完成，且結果已記錄。
- 任務完成時，將「狀態」更新為完成日期、執行指令與測試結果；不得只因檔案存在就標示完成。

### 1.4 Phase 閘門

| 閘門 | 可進入條件 |
| --- | --- |
| Phase 1 | 專案骨架、設定、依賴、測試 fixture、格式檢查與 CI 均可運作。 |
| Phase 2 | 資料載入、schema、列級驗證、特徵清理與結果契約測試全部通過。 |
| Phase 3 | 訓練 CLI 能建立完整候選模型包，且模型選擇、門檻與 final test 規則已有測試證據。 |
| Phase 4 | 模型包校驗、人工啟用、回滾及 FastAPI JSON／CSV 評分流程可端到端運作。 |
| Phase 5 | 四個 Streamlit 分頁均透過 FastAPI 完成操作，沒有直接模型依賴。 |
| Release | 單元、防洩漏、模型、整合、安全、UI 與效能驗收全部通過；文件與 F01 至 F10 追溯一致。 |

### 1.5 關鍵路徑

~~~text
P0-01 -> P0-02 -> P0-03 -> P0-04 -> P1-01 -> P1-02 -> P1-03
P1-03 -> P1-05 -> P1-06 -> P1-08 -> P2-01 -> P2-04 -> P2-05
P2-05 -> P2-06 -> P2-07 -> P2-08 -> P2-11 -> P2-12
P2-12 -> P3-01 -> P3-02 -> P3-03 -> P3-05 -> P3-08
P3-08 -> P4-01 -> P4-03 -> P4-05 -> P4-07
P4-07 -> P5-05 -> P5-07 -> P5-08 -> P5-10 -> Release
~~~

## 2. Phase 0 專案骨架與環境設定

### [x] P0-01 建立 Python 專案描述

- **目標**：`pyproject.toml`、`src/bankruptcy_prediction/__init__.py`
- **相依**：無
- **輸入與輸出**：輸入為專案名稱與支援的 Python 版本；輸出為可安裝套件、`src` package root、pytest 與覆蓋率設定。
- **驗收準則**：
  - `python -m pytest` 可辨識 `tests` 目錄。
  - 套件可由乾淨虛擬環境以 editable mode 安裝。
  - 測試：`tests/test_project_metadata.py` 驗證套件名稱、Python 範圍與 package root。
- **狀態**：已完成（2026-10-22），`python -m pytest --collect-only -q` 收集 1 項，`python -m pytest -q` 1 passed；乾淨虛擬環境 editable install 與 package import 通過。
- **更動重點**：新增 `pyproject.toml`（`bankruptcy-prediction`、Python `>=3.11`、`src` package root、pytest 與 coverage 設定）、`src/bankruptcy_prediction/__init__.py`（版本 `0.1.0`）、`tests/test_project_metadata.py`。
- **技術注意事項**：專案路徑須能在 Windows 11 與含中文的工作目錄執行；不得把 `Database/bankrupt_data.csv` 複製進套件。

### [x] P0-02 固定執行與開發相依

- **目標**：`requirements.txt`、`requirements-dev.txt`
- **相依**：P0-01
- **輸入與輸出**：輸入為 SDD 技術棧；輸出為 pandas、scikit-learn、XGBoost、SHAP、FastAPI、Pydantic、Streamlit、joblib、pytest 與品質工具的固定版本清單。
- **驗收準則**：
  - 乾淨虛擬環境可完成安裝，`pip check` 無衝突。
  - CPU 環境可 import 全部執行套件。
  - 測試：`tests/test_dependencies.py` 驗證必要套件存在且版本可記錄至 manifest。
- **狀態**：已完成（2026-10-22），Windows 11／Python 3.12.14 `.venv` 執行 `pip install -r requirements-dev.txt` 成功；`pip check` 無衝突；PyYAML import 通過；pytest 2 passed；`ruff check pyproject.toml src tests` 與 `ruff format --check src tests` 通過。既有 `Docs/_build_sdd.py` 有 lint 問題，不屬本任務檔案範圍。
- **更動重點**：新增 `requirements.txt`（14 項固定 runtime dependencies，含固定版 PyYAML）、`requirements-dev.txt`（共用 runtime pins，另固定 pytest、pytest-cov、ruff、httpx）、`tests/test_dependencies.py`（檢查直接依賴 pin 與安裝版本）。PyYAML 為 P0-03 YAML 設定及 P0-04 載入器所需。
- **技術注意事項**：依 SDD 的 Python `>=3.11` 選擇支援 3.11 的 NumPy、XGBoost 與 SHAP 版本；完整安裝與 imports 已在本機 Python 3.12.14 Windows 環境驗證。模型套件版本會隨後續 API／模型測試再驗證實際功能相容性。

### [x] P0-03 建立目錄與執行設定基線

- **目標**：`config/training.yaml`、`config/service.yaml`、`models/.gitkeep`、`logs/.gitkeep`
- **相依**：P0-01、P0-02
- **輸入與輸出**：輸入為 seed、切分比例、搜尋次數、檔案限制與服務位址；輸出為訓練與服務設定基線。
- **驗收準則**：
  - 訓練設定固定 seed 42、64%／16%／20%、5-fold、每模型 30 組與 `average_precision`。
  - 服務設定固定 `127.0.0.1`、10 MB、10,000 列與模型目錄。
  - 測試：`tests/test_config_baseline.py` 驗證必要節點、型別與數值範圍。
- **狀態**：已完成（2026-10-22），`python -m pytest -q tests/test_config_baseline.py` 2 passed；`ruff check tests/test_config_baseline.py` 與 `ruff format --check tests/test_config_baseline.py` 通過。
- **更動重點**：新增 `config/training.yaml`（seed、分層切分、搜尋、校準與模型基線）、`config/service.yaml`（localhost 位址、連接埠與上傳限制）、`models/.gitkeep`、`logs/.gitkeep` 及 `tests/test_config_baseline.py`。
- **技術注意事項**：門檻不是服務設定常數；預設門檻由 active model manifest 提供。

### [x] P0-04 實作型別化設定載入

- **目標**：`src/bankruptcy_prediction/settings.py`
- **相依**：P0-03
- **輸入與輸出**：`load_settings(training_path, service_path) -> AppSettings`；輸入 YAML 路徑，輸出不可變的型別化設定。
- **驗收準則**：
  - 缺少必要值、比例總和不為 1、非法 port、非 localhost host 或負數限制時拋出明確錯誤。
  - 設定物件不可在執行中被修改。
  - 測試：`tests/test_settings.py` 覆蓋成功、缺值、範圍錯誤與 localhost 限制。
- **狀態**：已完成（2026-10-22），`python -m pytest -q tests/test_settings.py tests/test_config_baseline.py tests/test_dependencies.py tests/test_project_metadata.py` 16 passed；`ruff check` 與 `ruff format --check` 對本任務檔案通過。測試以 `PYTHONPATH=src` 載入專案套件。
- **更動重點**：新增 `settings.py` 中 frozen dataclass 設定模型、YAML safe loader 與明確驗證錯誤；新增 `tests/test_settings.py` 覆蓋有效設定、缺值、切分比例、port、loopback host、正數限制及不可變性。
- **技術注意事項**：不得從設定檔接受雲端 URL、帳號系統或未列入 SDD 的保存後端。

### [x] P0-05 定義共用資料合約與錯誤碼

- **目標**：`src/bankruptcy_prediction/contracts.py`、`src/bankruptcy_prediction/errors.py`
- **相依**：P0-01、P0-04
- **輸入與輸出**：輸入為 SDD 的列狀態、警告與 API 錯誤格式；輸出為 `RowResult`、`BatchSummary`、`ApiError`、列舉與序列化規則。
- **驗收準則**：
  - 錯誤物件固定包含 `error_code`、`message`、`details`、`remediation`。
  - 列級結果可表達成功、失敗、補值及 `out_of_range`。
  - 測試：`tests/test_contracts.py` 驗證 JSON round-trip 與必填欄位。
- **狀態**：已完成（2026-10-22），`python -m pytest -q tests/test_contracts.py` 7 passed；`ruff check` 與 `ruff format --check` 對本任務檔案通過。
- **更動重點**：新增 `contracts.py`（列狀態、警告碼、列結果、批次摘要及 JSON-safe round-trip）與 `errors.py`（固定四欄 API 錯誤 envelope），新增 `tests/test_contracts.py` 覆蓋成功／失敗列、必要欄位、計數不變式與序列化。
- **技術注意事項**：錯誤碼是機器可讀值；人類訊息不得包含堆疊、絕對路徑或完整輸入資料。

### [x] P0-06 建立結構化稽核日誌基礎

- **目標**：`src/bankruptcy_prediction/logging_config.py`
- **相依**：P0-04、P0-05
- **輸入與輸出**：`configure_logging(settings) -> None` 與 `log_score_summary(...)`；輸出 JSON 或固定欄位的評分摘要日誌。
- **驗收準則**：
  - 只記錄時間、`request_id`、模型版本、總列數、成功數、失敗數、耗時與錯誤碼統計。
  - 日誌不得含 `record_id`、檔名敏感片段、完整輸入列、信心分數或 SHAP 貢獻。
  - 測試：`tests/test_logging_hygiene.py` 以敏感字串 fixture 驗證輸出不洩漏。
- **狀態**：已完成（2026-10-22），`python -m pytest -q tests/test_logging_hygiene.py` 4 passed；`ruff check` 與 `ruff format --check` 對本任務檔案通過。
- **更動重點**：新增 `logging_config.py`，採 JSONL 欄位白名單、request ID 關聯、錯誤碼統計與路徑樣式 token 拒絕；新增 `tests/test_logging_hygiene.py` 驗證輸出欄位及敏感值不洩漏。
- **技術注意事項**：訓練日誌可記錄資料雜湊與參數，但不可記錄完整資料列。

### [x] P0-07 建立測試 Fixture 基礎

- **目標**：`tests/conftest.py`、`tests/fixtures/`
- **相依**：P0-02、P0-04、P0-05
- **輸入與輸出**：輸入為正式 CSV schema；輸出為合法批次、缺欄、未知欄、9／10 缺值、無限值、同分與公式注入等固定 fixtures。
- **驗收準則**：
  - 每個 fixture 為固定值且不依賴隨機時間或外部網路。
  - 測試資料不覆寫正式資料集與 `models` 目錄。
  - 測試：`tests/test_fixtures.py` 驗證 fixture 互相隔離並含 95 個正式特徵名。
- **狀態**：已完成（2026-10-22），`python -m pytest -q tests/test_fixtures.py` 2 passed；`ruff check` 與 `ruff format --check` 對本任務檔案通過。
- **更動重點**：新增僅含官方 95 個正規化特徵名稱的 `tests/fixtures/feature_schema.csv`，以及 `conftest.py` 的固定合成批次、缺欄／未知欄、9／10 缺值、無限值、同分與公式注入 fixtures；測試確認資料與正式資料、模型目錄隔離。
- **技術注意事項**：正式資料只供訓練與資料稽核測試；一般單元測試應使用最小合成資料。

### [x] P0-08 建立格式、靜態檢查與 CI 閘門

- **目標**：`ruff.toml`、`.github/workflows/ci.yml`、`.gitignore`
- **相依**：P0-02、P0-07
- **輸入與輸出**：輸入為推送或 pull request；輸出為安裝、`pip check`、格式、lint、單元測試與敏感檔案檢查結果。
- **驗收準則**：
  - CI 依序執行依賴安裝、格式檢查、lint 與 pytest。
  - `.gitignore` 排除虛擬環境、暫存上傳、日誌、模型二進位與本機 active 指標。
  - 測試：`tests/test_ci_workflow.py` 驗證工作步驟與禁止提交路徑。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 32 passed；`ruff format --check src tests` 與 `ruff check src tests` 通過。
- **更動重點**：新增 `ruff.toml`、GitHub Actions CI（依賴安裝、`pip check`、格式、lint、pytest）、`.gitignore` 與 `tests/test_ci_workflow.py`；依新設定統一格式化 `src/`、`tests/`。
- **技術注意事項**：模型包驗收 fixture 可使用小型假產物；不得將正式 `model.joblib` 納入版本控制。

## 3. Phase 1 資料契約與特徵處理

### [x] P1-01 實作訓練資料載入與稽核

- **目標**：`src/bankruptcy_prediction/data/training_loader.py`
- **相依**：P0-04、P0-05、P0-07
- **輸入與輸出**：`load_training_data(path) -> TrainingDataset`；輸入 `Database/bankrupt_data.csv`，輸出 95 特徵、`Bankrupt?` 標籤與資料稽核摘要。
- **驗收準則**：
  - 驗證 6,819 列、96 欄、220 筆正例及數值型態；差異時終止並列出原因。
  - 計算來源檔 SHA-256，不修改原始檔。
  - 測試：`tests/data/test_training_loader.py` 覆蓋正式摘要與損壞 fixture。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 34 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。正式資料載入核對 6,819×96、95 特徵、220 正例、6,599 負例；SHA-256 讀前讀後一致。
- **更動重點**：新增唯讀 `data/training_loader.py`、稽核 DTO 與 `tests/data/test_training_loader.py`；新增損壞 CSV fixture，確認資料規格錯誤會被拒絕。
- **技術注意事項**：資料沒有公司、年度與事件日期；loader 不得推導不存在的預測窗口。

### [x] P1-02 建立特徵 Schema 基線

- **目標**：`src/bankruptcy_prediction/data/feature_schema.py`
- **相依**：P1-01
- **輸入與輸出**：`build_feature_schema(dataset) -> FeatureSchema`；輸出正規化後的 95 欄順序、型別與 schema 版本。
- **驗收準則**：
  - 只移除欄名前後空白，保留大小寫及內部空白。
  - schema 順序與正式資料來源一致，可序列化為 `feature_schema.json`。
  - 測試：`tests/data/test_feature_schema.py` 驗證欄名、順序、唯一性與版本欄位。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 36 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。schema round-trip 後欄位順序、版本與來源 dtype 資訊一致。
- **更動重點**：新增 `data/feature_schema.py`（移除欄名前後空白、保留大小寫與內部空白、保留來源順序、數值欄位型別與 `1.0` schema 版本、JSON 序列化）及 `tests/data/test_feature_schema.py`。
- **技術注意事項**：模型實際輸入欄可因常數／重複欄移除而少於 95，但推論 CSV 契約仍要求原始 95 特徵。

### [x] P1-03 實作檔案級 CSV Schema 驗證

- **目標**：`src/bankruptcy_prediction/data/schema_validator.py`
- **相依**：P1-02、P0-05
- **輸入與輸出**：`validate_file_schema(frame, schema) -> ValidatedBatch`；輸入上傳 DataFrame，輸出 canonical order 批次或檔案級錯誤。
- **驗收準則**：
  - 缺少任一特徵、未知欄、`Bankrupt?`、空檔或重複欄名時整批拒絕。
  - 唯一允許的額外欄為 `record_id`。
  - 測試：`tests/data/test_schema_validator.py` 覆蓋 T01、T02 與大小寫精確比對。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 41 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `data/schema_validator.py`，欄名僅 trim 後做區分大小寫精確比對，僅額外允許 `record_id`，拒絕缺欄／未知欄／標籤欄／空批次／重複欄並以共用 `ApiError` 回報，成功時輸出 canonical 欄序；新增對應驗收測試。
- **技術注意事項**：10 MB、10,000 列及編碼限制由上傳邊界先檢查；schema validator 仍要防禦空 DataFrame。

### [x] P1-04 實作 record_id 正規化

- **目標**：`src/bankruptcy_prediction/data/record_ids.py`
- **相依**：P1-03
- **輸入與輸出**：`ensure_record_ids(frame) -> Series[str]`；提供欄位時驗證，未提供時產生 `row_000001` 格式識別碼。
- **驗收準則**：
  - 使用者提供的 `record_id` 必須非空且在整批唯一。
  - 自動產生值採六位零補齊並與原始列順序一致。
  - 測試：`tests/data/test_record_ids.py` 覆蓋缺欄、空值、重複值及 10,000 列。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 49 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過，包含 10,000 列固定順序測試。
- **更動重點**：新增 `data/record_ids.py`，無欄位時生成六位補零 ID，提供欄位時轉為字串並拒絕 null／空白／整批重複；新增 `tests/data/test_record_ids.py`。
- **技術注意事項**：`record_id` 只用於當次批次關聯，不得寫入評分日誌。

### [x] P1-05 實作列級品質驗證

- **目標**：`src/bankruptcy_prediction/data/row_quality.py`
- **相依**：P1-03、P1-04
- **輸入與輸出**：`validate_rows(batch) -> RowValidationResult`；輸出合法列遮罩、`row_status`、`error_code` 與缺值計數。
- **驗收準則**：
  - 非數字、無限值或缺少 10 個以上特徵的列標示失敗。
  - 一列失敗不阻擋其他合法列，且保留原始順序與 `record_id`。
  - 測試：`tests/data/test_row_quality.py` 覆蓋 T03、T04、非數字與正負無限值。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 53 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `data/row_quality.py` 與 `RowValidationResult`，逐列 coercion，區分缺值、非數字、無限值，10 個以上缺值列失敗，並保留全批列序及識別碼；新增 T03／T04、非數字及正負無限值測試。
- **技術注意事項**：NaN 是缺值，不是非數字；正負無限值必須是列級錯誤而非警告。

### [x] P1-06 實作訓練內中位數補值

- **目標**：`src/bankruptcy_prediction/features/imputation.py`
- **相依**：P1-05
- **輸入與輸出**：`build_imputer() -> Transformer`；只在訓練折 `fit`，推論時對最多 9 個缺值的合法列補值並產生警告。
- **驗收準則**：
  - 中位數只來自當次訓練折，不得由 validation、test 或推論批次估計。
  - 補值列輸出一致的 `IMPUTED_VALUES` 警告碼。
  - 測試：`tests/features/test_imputation.py` 驗證訓練統計、T03、T04 與防洩漏 spy。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 57 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。Spy 測試確認 `fit()` 僅收到訓練資料，validation／test transform 不會重估統計。
- **更動重點**：新增可 clone、可放入 sklearn Pipeline 的 `TrainingMedianImputer`，僅於 `fit()` 記錄訓練中位數，支援列級 `IMPUTED_VALUES` 警告；更新共用警告列舉並覆蓋 round-trip 測試。
- **技術注意事項**：實際補值器要封裝進 `model.joblib`，線上服務不得另行 fit。

### [x] P1-07 實作訓練範圍與極端值警告

- **目標**：`src/bankruptcy_prediction/features/range_monitor.py`
- **相依**：P1-05、P1-06
- **輸入與輸出**：`fit_ranges(X_train)` 與 `flag_out_of_range(X) -> WarningMatrix`；輸出每列超界欄位摘要。
- **驗收準則**：
  - 最小／最大值只由訓練資料估計。
  - 有限超界值不截尾且仍可評分，列級加入 `OUT_OF_RANGE`。
  - 測試：`tests/features/test_range_monitor.py` 覆蓋邊界值、T05 與多欄超界。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 61 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `features/range_monitor.py`，僅以訓練資料估最小／最大值，有限超界值不截尾並回傳 `OUT_OF_RANGE`、每列欄名摘要（最多 10 欄）及完整超界計數；測試涵蓋邊界、超界與無限值。
- **技術注意事項**：輸出警告宜限制欄名數量並保留完整計數，避免回應無限制膨脹。

### [x] P1-08 實作常數與完全重複特徵清理

- **目標**：`src/bankruptcy_prediction/features/selector.py`
- **相依**：P1-01、P1-02
- **輸入與輸出**：`FeatureSelector.fit(X_train).transform(X)`；移除 `Net Income Flag`，並保留首次欄位、移除兩個指定完全重複欄。
- **驗收準則**：
  - 移除清單與保留順序固定記錄，可由 manifest 取得。
  - 高度相關欄只產生稽核資料，不自動刪除。
  - 測試：`tests/features/test_selector.py` 驗證三個移除欄、欄位順序及未知重複不自動處理。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 66 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `features/selector.py`，僅移除 `Net Income Flag` 及兩個 SDD 指定的完全重複欄，欄序保持穩定；以 `|Pearson r| >= 0.95` 產生稽核 pair，不自動刪除高相關欄；未知完全重複亦只列入稽核，並提供可序列化 selector audit。
- **技術注意事項**：selector 必須位於 Pipeline 內並只對訓練資料 `fit`；不得在全資料切分前估計。

### [x] P1-09 建立評分列與批次結果模型

- **目標**：`src/bankruptcy_prediction/scoring/result_models.py`
- **相依**：P0-05、P1-04、P1-05
- **輸入與輸出**：輸出 SDD 指定的 `record_id`、版本、分數、百分位、門檻、分類、Top 10%、狀態、警告、正負因素及錯誤欄位。
- **驗收準則**：
  - 成功與失敗列都能依固定欄位順序序列化。
  - 失敗列不偽造分數、百分位、分類或解釋值。
  - 測試：`tests/scoring/test_result_models.py` 驗證 JSON／DataFrame schema 與空值規則。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 70 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `scoring/result_models.py`，固定順序的成功／失敗列 DTO、解釋欄位、JSON round-trip 與 DataFrame projection；失敗列的分數、百分位、分類及解釋保持 null，不偽造結果。
- **技術注意事項**：UI 前五項解釋與匯出前三項解釋屬不同 projection，不應以兩套 SHAP 計算實作。

### [x] P1-10 實作安全 CSV 匯出

- **目標**：`src/bankruptcy_prediction/export/csv_exporter.py`
- **相依**：P1-09
- **輸入與輸出**：`export_results_csv(result) -> bytes`；輸出 UTF-8 CSV 與固定欄位順序。
- **驗收準則**：
  - `record_id` 以 `=`、`+`、`-` 或 `@` 開頭時以前置單引號轉義。
  - JSON 與 CSV 對同一結果的模型版本、門檻、列狀態與分數一致。
  - 測試：`tests/export/test_csv_exporter.py` 覆蓋 T11、換行、引號與 Unicode。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 76 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `export/csv_exporter.py`，固定欄序 UTF-8 CSV；只在 CSV 投影中以前置單引號轉義以 `= + - @` 開頭的識別碼，JSON／內部 DTO 保持原值；測試涵蓋公式、中文、引號、換行及欄位一致性。
- **技術注意事項**：轉義只影響 CSV 文字表示，不得改寫內部 `record_id` 或 JSON 回應。

## 4. Phase 2 模型訓練、評估與封存

### [x] P2-01 實作固定分層資料切分

- **目標**：`src/bankruptcy_prediction/training/split_manager.py`
- **相依**：P1-01、P1-02
- **輸入與輸出**：`create_split(y, seed=42) -> DatasetSplit`；輸出 64% 訓練、16% validation、20% final test 的分層索引與雜湊。
- **驗收準則**：
  - 相同資料與 seed 產生完全相同索引。
  - 三份資料互斥、覆蓋全部列，且標籤比例符合分層預期。
  - 測試：`tests/training/test_split_manager.py` 驗證比例、互斥、重現與雜湊。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 79 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。正式資料得到 4,364／1,091／1,364 列，分層比例符合預期。
- **更動重點**：新增 `training/split_manager.py`，固定 seed 42 分層 64%／16%／20%，回傳凍結索引與分段／整體 SHA-256，提供序列化與 hash 驗證；測試覆蓋重現、互斥、完整性、標籤比例及篡改檢查。
- **技術注意事項**：final test 索引建立後即凍結；後續元件不得因測試指標重新切分。

### [x] P2-02 建立 Logistic Regression Pipeline

- **目標**：`src/bankruptcy_prediction/training/logistic_pipeline.py`
- **相依**：P1-06、P1-08、P2-01
- **輸入與輸出**：`build_logistic_pipeline() -> Pipeline`；包含 selector、MedianImputer、StandardScaler／RobustScaler 與 Logistic Regression。
- **驗收準則**：
  - 搜尋空間包含 scaler、L1／L2、對數尺度 `C`、足夠 `max_iter` 與 `class_weight`。
  - 所有可學習轉換只在 CV 訓練折 `fit`。
  - 測試：`tests/training/test_logistic_pipeline.py` 驗證步驟、參數相容性與不收斂處理。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 85 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。低 `max_iter` 的測試可觀察到 sklearn `ConvergenceWarning`，未被隱藏。
- **更動重點**：新增 `training/logistic_pipeline.py`，Pipeline 封裝 selector、MedianImputer、scaler 與 saga Logistic Regression；搜尋空間涵蓋 Standard／Robust scaler、L1/L2 等價 `l1_ratio`、對數尺度 C、class weight。使用 `l1_ratio` 避免新版 sklearn 已棄用的 `penalty` 參數。
- **技術注意事項**：solver 必須與 L1／L2 候選相容；不得加入 SMOTE。

### [x] P2-03 建立 XGBoost Pipeline

- **目標**：`src/bankruptcy_prediction/training/xgboost_pipeline.py`
- **相依**：P1-06、P1-08、P2-01
- **輸入與輸出**：`build_xgboost_pipeline() -> Pipeline`；包含 selector、MedianImputer 與 CPU XGBoost classifier。
- **驗收準則**：
  - 搜尋空間包含樹深、學習率、樹數、`subsample`、`colsample_bytree`、`min_child_weight`、`reg_alpha`、`reg_lambda`。
  - `scale_pos_weight` 候選只依訓練折比例建立。
  - 測試：`tests/training/test_xgboost_pipeline.py` 驗證 CPU 設定、參數範圍與權重來源。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 88 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。兩種不同比例訓練折分別得到 `scale_pos_weight=4.0` 與 `0.25`。
- **更動重點**：新增 `training/xgboost_pipeline.py`，Pipeline 包含 selector、median imputer 與 CPU XGBoost；搜尋分布涵蓋規定參數，`balanced` 類別權重在 estimator 每次 `fit(y)` 時從該折標籤動態計算。
- **技術注意事項**：不得以全資料正負比例建立 `scale_pos_weight`，也不得要求 GPU。

### [x] P2-04 實作兩模型 RandomizedSearchCV

- **目標**：`src/bankruptcy_prediction/training/model_search.py`
- **相依**：P2-02、P2-03
- **輸入與輸出**：`search_models(X_train, y_train, config) -> SearchResults`；輸出兩模型各 30 組、Stratified 5-fold 的搜尋結果。
- **驗收準則**：
  - scoring 固定為 `average_precision`，random state 固定且可記錄。
  - 每個模型確實嘗試 30 組設定；失敗組合與原因可追蹤。
  - 測試：`tests/training/test_model_search.py` 以小型參數空間驗證 fold、次數、scoring 與結果排序。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 94 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。Stub 驗證兩模型各 30 次、5-fold、AP、seed 42；失敗候選包含參數及原因。
- **更動重點**：新增 `training/model_search.py`，固定搜尋政策、CPU 單工執行、候選排名與失敗追蹤；如設定試圖降低 30 組／5-fold、變更 AP 或 seed，會直接拒絕。
- **技術注意事項**：單元測試可降低資料量，但正式 `train` 不得靜默降低 30 組或 5-fold。

### [x] P2-05 實作 5-fold Sigmoid 校準

- **目標**：`src/bankruptcy_prediction/training/calibration.py`
- **相依**：P2-04
- **輸入與輸出**：`calibrate_candidate(best_estimator, X_train, y_train) -> CalibratedClassifierCV`；輸出 Sigmoid 校準模型。
- **驗收準則**：
  - 校準使用 5-fold，且只使用訓練資料。
  - 產生的信心分數位於 0 至 1，可供 Brier Score 與校準曲線使用。
  - 測試：`tests/training/test_calibration.py` 以 spy 確認 validation／test 未進入校準 fit。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 98 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。Spy 確認校準 fit 只接收傳入訓練資料，實際信心分數落在 0–1。
- **更動重點**：新增 `training/calibration.py`，只接受訓練資料、固定 Stratified 5-fold／Sigmoid／seed 42，並附校準 metadata；報告及型別使用「模型信心分數」語意。
- **技術注意事項**：介面與報告統一稱「模型信心分數」，不得改稱未來一年破產機率。

### [x] P2-06 實作 validation 模型比較規則

- **目標**：`src/bankruptcy_prediction/training/model_selection.py`
- **相依**：P2-05
- **輸入與輸出**：`select_model(candidates, X_val, y_val) -> SelectedModel`；以 validation Average Precision 選擇勝出模型。
- **驗收準則**：
  - AP 差距小於 0.01 時必選 Logistic Regression。
  - 選擇結果保存兩模型 AP、差距、規則與理由。
  - 測試：`tests/training/test_model_selection.py` 覆蓋明顯勝出、剛好 0.01 與小於 0.01 的邊界。
- **狀態**：已完成（2026-10-22），`python -m pytest -q` 104 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。測試涵蓋明顯勝出、精確 0.01 邊界及低於 0.01 tie-break。
- **更動重點**：新增 `training/model_selection.py`，只接受兩模型與 validation 資料，按 AP 選勝出者；差距 `< 0.01` 選 Logistic Regression，結果序列化保存兩模型 AP、差距、規則與理由。
- **技術注意事項**：final test 指標不得傳入此函式或影響勝出規則。

### [x] P2-07 實作 F2 預設門檻選擇

- **目標**：`src/bankruptcy_prediction/training/threshold_selector.py`
- **相依**：P2-06
- **輸入與輸出**：`select_f2_threshold(y_val, confidence_scores) -> ThresholdResult`；輸出 0 至 1 的預設門檻及曲線摘要。
- **驗收準則**：
  - 門檻只由勝出模型的 validation 分數估計。
  - 同分門檻採固定且有文件的決定規則，結果可重現。
  - 測試：`tests/training/test_threshold_selector.py` 驗證 F2 最大值、邊界與 test set 隔離。
- **狀態**：已完成（2026-10-22），`pytest tests/training/test_threshold_selector.py -q` 8 passed；全套 `pytest -q` 112 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `threshold_selector.py` 與 `ThresholdResult`／`ThresholdPoint` DTO；只接受 validation 標籤及信心分數，枚舉 0、1 與觀測分數作為門檻候選，以 F2 最大化選擇預設門檻；若同分固定選最低門檻以偏向較高 Recall，回傳完整曲線摘要及規則字串。新增測試涵蓋 F2 最大值、邊界、輸入驗證、同分規則與 API 介面不含 test set。
- **預計更動**：建立 F2 計算、tie handling 與 default threshold DTO。
- **技術注意事項**：UI 當次門檻不會呼叫此選擇器，也不得回寫選出的預設門檻。

### [x] P2-08 實作模型指標與 Top 10% 評估

- **目標**：`src/bankruptcy_prediction/evaluation/metrics.py`
- **相依**：P2-07
- **輸入與輸出**：`evaluate_frozen_model(model, threshold, X_test, y_test) -> MetricsReport`；輸出 AP、Precision、Recall、F1、Brier、混淆矩陣、誤報數與 Recall@Top10%。
- **驗收準則**：
  - Top 10% 使用 `ceil(n * 0.1)`，同分保持原始列順序。
  - 通過條件為 test AP 至少是 test 正例比例的 2 倍，且 Recall@Top10% 至少 50%。
  - 測試：`tests/evaluation/test_metrics.py` 覆蓋 T07、空正例防護與門檻判定。
- **狀態**：已完成（2026-10-22），`pytest tests/evaluation/test_metrics.py -q` 9 passed；全套 `pytest -q` 121 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `evaluation/metrics.py` 與 `MetricsReport`；只對凍結模型呼叫一次 `predict_proba`，計算 AP、正例比例、Precision、Recall、F1、Brier、混淆矩陣、誤報數及 Recall@Top10%。Top 10% 按 `ceil(n × 10%)`，以穩定降冪排序保留同分原始列順序；無正例時安全回傳且 MVP 判定失敗。MVP 通過條件為 AP／正例比例至少 2 倍及 Recall@Top10% 至少 50%。測試涵蓋 T07、同分、空正例、品質門檻、輸入錯誤與不呼叫 fit。
- **預計更動**：建立指標函式、穩定排序及 MVP pass/fail 摘要。
- **技術注意事項**：此元件只評估已凍結模型；測試結果不得觸發重新選模或重新選門檻。

### [x] P2-09 實作 SHAP 解釋服務核心

- **目標**：`src/bankruptcy_prediction/explanation/shap_service.py`
- **相依**：P2-06、P1-08
- **輸入與輸出**：`explain(model_family, model, X) -> ExplanationBatch`；Logistic Regression 使用 Linear SHAP，XGBoost 使用 Tree SHAP。
- **驗收準則**：
  - 每列可取得前五項正向與前五項負向因素，並可投影成匯出前三項。
  - 解釋保留原始特徵值、貢獻方向與特徵名稱。
  - 測試：`tests/explanation/test_shap_service.py` 驗證 explainer 選擇、排序、正負方向與固定輸出數量。
- **狀態**：已完成（2026-10-22），`pytest tests/explanation/test_shap_service.py -q` 7 passed；全套 `pytest -q` 128 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過；實際 SHAP Linear／Tree smoke test 均成功。
- **更動重點**：新增 `ExplanationBatch`／`ExplanationRow`／`ExplanationFactor` 與 SHAP factory；Logistic Regression 用 Linear SHAP、XGBoost 用 Tree SHAP；經 calibrated CV 包裝時平均已擬合基礎模型解釋值。每列保留原始特徵值及正負貢獻方向，依貢獻排序提供最多前五項，並支援匯出前三項投影。說明文字明確限定為模型貢獻而非因果效果。
- **預計更動**：建立 explainer factory、批次解釋與 projection。
- **技術注意事項**：任何文字說明只描述模型貢獻，不得產生因果結論。

### [x] P2-10 建立模型報告產生器

- **目標**：`src/bankruptcy_prediction/reporting/model_report.py`
- **相依**：P1-01、P1-08、P2-04、P2-08、P2-09
- **輸入與輸出**：`build_model_report(context, output_dir) -> Path`；輸出離線可開啟的 `model_report.html` 與 PR、校準、混淆矩陣、重要性圖。
- **驗收準則**：
  - 報告包含資料稽核、切分、搜尋、選模、門檻、final test、限制與非因果聲明。
  - 圖表引用為模型包內有效相對路徑。
  - 測試：`tests/reporting/test_model_report.py` 驗證必要章節、圖檔與 HTML 離線引用。
- **狀態**：已完成（2026-10-22），`pytest tests/reporting/test_model_report.py -q` 2 passed；全套 `pytest -q` 130 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `ModelReportContext` 與 `build_model_report`，輸出離線 HTML 及相對路徑圖表（PR、校準、混淆矩陣、SHAP 重要性）；報告包含資料稽核、切分、搜尋、選模、門檻、final test、限制、缺少公司／年度／預測窗口聲明及非因果聲明。所有 context 內容 HTML escape，不引用外部資源。
- **預計更動**：建立 HTML 模板、圖表輸出與限制文字。
- **技術注意事項**：報告不得補造公司、年度、預測窗口或將信心分數改稱破產機率。

### [x] P2-11 建立版本化候選模型包

- **目標**：`src/bankruptcy_prediction/model_package/builder.py`
- **相依**：P1-02、P2-07、P2-08、P2-10
- **輸入與輸出**：`build_candidate_package(context, models_root) -> ModelPackage`；建立版本目錄與六類必要產物。
- **驗收準則**：
  - 產生 `model.joblib`、`manifest.json`、`feature_schema.json`、`metrics.json`、`model_report.html` 與 `charts/`。
  - manifest 包含資料雜湊、切分 seed／索引摘要、特徵順序、移除欄、套件版本、參數、校準、門檻及檔案 checksum。
  - 測試：`tests/model_package/test_builder.py` 驗證完整性、checksum 與不覆寫既有版本。
- **狀態**：已完成（2026-10-22），`pytest tests/model_package/test_builder.py -q` 6 passed；全套 `pytest -q` 136 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `CandidatePackageContext`、`ModelPackage` 與原子候選封存器，建立模型、feature schema、metrics、HTML 報告及圖表；manifest 記錄資料 SHA-256、seed／切分索引摘要、欄位順序、移除欄、套件版本、模型參數、校準、預設門檻與全部非 manifest 產物 checksum。版本採 UTC 時戳與隨機 ID；staging 失敗會清理且不動既有資料，已存在版本拒絕覆寫，既有 `active.json` 經測試保持不變。manifest 不對自身做 checksum，避免自我參照。
- **預計更動**：建立版本 ID、原子輸出、manifest schema 與檔案校驗值。
- **技術注意事項**：候選包建立成功也不得更新 `active.json`；joblib 只供受信任本機流程載入。

### [x] P2-12 實作 train CLI 與訓練編排

- **目標**：`src/bankruptcy_prediction/cli.py`、`src/bankruptcy_prediction/training/orchestrator.py`
- **相依**：P2-01 至 P2-11
- **輸入與輸出**：`train --data <path> --config <path>`；輸出候選模型包路徑、摘要指標與通過／未通過狀態。
- **驗收準則**：
  - 依序完成資料稽核、固定切分、兩模型搜尋、校準、validation 選模、F2 門檻、一次 final test 與候選封存。
  - 任一步驟失敗時不留下可被誤認為完整的模型包。
  - 測試：`tests/integration/test_train_cli.py` 以小型設定驗證流程、失敗清理及 `active.json` 不變。
- **狀態**：已完成（2026-10-22），`pytest tests/integration/test_train_cli.py -q` 4 passed；全套 `pytest -q` 143 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過；`python -m bankruptcy_prediction.cli --help` 通過。另含 3 個 SHAP 上游套件棄用警告，非測試失敗。
- **更動重點**：新增 `training/orchestrator.py` 與 `cli.py`／`bankruptcy-prediction` entry point，按資料稽核、固定切分、兩模型搜尋、訓練集校準、validation 選模與 F2 門檻、單次 final-test 預測與評估、validation SHAP、報告及 candidate 封存依序執行。`test_only: true` 可明確啟用小型 schema 完整資料與較少搜尋／CV folds；其模型包及報告皆標記 test-only，CLI 不宣稱 Release 品質通過。生產設定仍鎖定 seed 42、64/16/20、各模型 30 組、5-fold 搜尋／校準。測試確認工作流、失敗時不產生未完成包、active.json 不變及結構化 CLI 摘要。
- **預計更動**：建立 CLI entry point、訓練 orchestrator 與執行摘要。
- **技術注意事項**：正式設定必須執行兩模型各 30 組與 5-fold；測試用快速設定要有明確 test-only 標記。

## 5. Phase 3 模型治理與 FastAPI 評分服務

### [x] P3-01 實作模型包完整性校驗

- **目標**：`src/bankruptcy_prediction/model_package/verifier.py`
- **相依**：P2-11
- **輸入與輸出**：`verify_package(path) -> VerifiedPackage`；輸入版本目錄，輸出通過 schema、必要檔案與 checksum 驗證的模型包。
- **驗收準則**：
  - 缺檔、checksum 不符、schema 版本不相容或 manifest 欄位不全時拒絕。
  - 校驗過程不載入不受信任或未通過 checksum 的 joblib。
  - 測試：`tests/model_package/test_verifier.py` 覆蓋完整包、缺檔、竄改檔與未知 schema。
- **狀態**：已完成（2026-10-22），`pytest tests/model_package/test_verifier.py -q` 6 passed；全套 `pytest -q` 149 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `verify_package`／`VerifiedPackage`，驗證 manifest 與 feature schema 版本、必要欄位、模型特徵順序、metrics 一致性、離線報告相對圖表引用、安全路徑與全部必要產物 checksum。驗證不匯入或呼叫 `joblib.load`；損壞、缺檔、未知 schema 與路徑逃逸均以不含絕對路徑的錯誤拒絕。
- **預計更動**：建立驗證順序、manifest schema 檢查與安全失敗訊息。
- **技術注意事項**：先驗證 manifest 與所有檔案，再反序列化模型；錯誤不得暴露絕對路徑。

### [x] P3-02 實作 activate 與 rollback CLI

- **目標**：`src/bankruptcy_prediction/model_registry/commands.py`
- **相依**：P3-01、P2-12
- **輸入與輸出**：`activate --version <id>`、`rollback --version <id>`；校驗指定模型包後原子更新 `active.json`。
- **驗收準則**：
  - activate／rollback 只接受既有且完整的版本，不刪除前一版本。
  - `active.json` 使用暫存檔加原子替換，失敗時保留原值。
  - 測試：`tests/model_registry/test_commands.py` 覆蓋 T09、切換、回滾、損壞版本與中斷寫入。
- **狀態**：已完成（2026-10-22），`pytest tests/model_registry/test_commands.py -q` 5 passed；全套 `pytest -q` 154 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過；activate CLI help 通過。
- **更動重點**：新增 `activate_model`／`rollback_model`，先驗證完整模型包再以同磁碟暫存檔、flush／fsync 與 `os.replace` 更新 `active.json`；保留 previous version、操作與時間，不刪除或修改模型包。無效／損壞版本、路徑式 ID、active pointer 符號連結或中斷寫入均拒絕，原 active 值保持不變。CLI 新增 `activate --version` 與 `rollback --version` 子命令。
- **預計更動**：新增 CLI 子命令、active pointer 寫入與操作紀錄。
- **技術注意事項**：activate 不重新訓練、不重算指標，也不修改模型包內容。

### [x] P3-03 實作唯讀 ModelRegistry

- **目標**：`src/bankruptcy_prediction/model_registry/registry.py`
- **相依**：P3-01、P3-02
- **輸入與輸出**：`load_active_model(models_root) -> ActiveModel`；讀取 `active.json`、校驗並載入模型、schema、指標與摘要。
- **驗收準則**：
  - 無 active model、指向不存在版本、checksum 失敗或 schema 不相容時回報明確不可用原因。
  - 線上 registry 不提供 activate、rollback 或寫入方法。
  - 測試：`tests/model_registry/test_registry.py` 覆蓋正常載入、T10 與多次讀取一致性。
- **狀態**：已完成（2026-10-22），`pytest tests/model_registry/test_registry.py -q` 6 passed；全套 `pytest -q` 160 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增唯讀 `load_active_model`／`ActiveModel`／`ModelSummary`；每次讀取 active pointer 後先驗證包路徑、schema 與全部 checksum，通過後才載入 joblib，並以檔案 checksum 為 cache key 快取已驗證模型。回傳 schema、metrics、manifest 與摘要唯讀視圖；錯誤以 `ActiveModelUnavailable`／`error_code` 回報，沒有 active、版本越界、缺包、checksum 失敗、schema 不相容與載入失敗皆有明確原因。registry 未提供寫入、activate 或 rollback 方法。
- **預計更動**：建立唯讀 active model DTO、載入快取與模型摘要。
- **技術注意事項**：服務啟動可保持健康但模型不可用；評分端點必須回傳 503。

### [x] P3-04 實作受控暫存檔管理

- **目標**：`src/bankruptcy_prediction/service/temp_files.py`
- **相依**：P0-04、P0-06
- **輸入與輸出**：`temporary_upload_workspace() -> ContextManager[Path]`；輸出隨機暫存目錄並於成功或失敗後清理。
- **驗收準則**：
  - 使用系統安全隨機名稱，不採用使用者檔名拼接路徑。
  - 正常、驗證失敗與未預期例外三種路徑都會刪除內容。
  - 測試：`tests/service/test_temp_files.py` 驗證清理、路徑穿越與重複檔名。
- **狀態**：已完成（2026-10-22），`pytest tests/service/test_temp_files.py -q` 4 passed；全套 `pytest -q` 164 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `temporary_upload_workspace`、`create_upload_file_path` 與受限啟動清掃；工作區與 CSV 檔名均由作業系統隨機產生，完全忽略上傳原始檔名。成功、驗證錯誤及未預期例外後於 `finally` 清理；清理失敗只記錄非敏感訊息，清掃器僅移除 24 小時以上且符合專用前綴的直接子目錄。
- **預計更動**：建立 context manager、受限檔名處理與 finally 清理。
- **技術注意事項**：若作業系統刪除失敗，記錄非敏感錯誤並進行啟動時清掃，但不得保留為產品歷史功能。

### [x] P3-05 實作核心批次評分服務

- **目標**：`src/bankruptcy_prediction/scoring/service.py`
- **相依**：P1-03 至 P1-09、P2-09、P3-03
- **輸入與輸出**：`score_batch(active_model, frame, threshold=None) -> ScoreBatchResult`；輸出分數、百分位、分類、Top 10%、解釋、警告與錯誤列。
- **驗收準則**：
  - 檔案級錯誤完全不評分；列級錯誤允許合法列完成。
  - `batch_percentile` 與 Top 10% 只使用合法列，Top 10% 穩定取 `ceil(n * 0.1)`。
  - 測試：`tests/scoring/test_service.py` 覆蓋 T01 至 T07、全失敗列與單列批次。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/scoring/test_service.py` 6 passed；全套 `pytest -q` 170 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `scoring/service.py`，串接檔案 schema、列級品質驗證、已 fit pipeline、SHAP 解釋、警告與結果契約；平均名次百分位只使用合法列，Top 10% 採穩定排序與 `ceil(n × 10%)`，同分維持原始列順序；新增 `tests/scoring/test_service.py` 覆蓋有效批次、列級部分成功、補值／範圍警告、檔案級拒絕、全失敗、單列與門檻情境。
- **技術注意事項**：線上服務只呼叫已 fit pipeline；不得在評分中更新中位數、範圍或模型。

### [x] P3-06 實作門檻情境與回應格式化

- **目標**：`src/bankruptcy_prediction/scoring/response_formatter.py`
- **相依**：P3-05、P1-10
- **輸入與輸出**：`format_score_response(result, format) -> JsonResponse | CsvResponse`；接受可選 0 至 1 threshold。
- **驗收準則**：
  - 未提供門檻時使用 active model `default_threshold`；非法門檻回檔案級 422。
  - 調整門檻只改 `risk_class`；分數、百分位、Top 10% 與 SHAP 不變。
  - 測試：`tests/scoring/test_response_formatter.py` 覆蓋 T06、JSON／CSV 一致性與 `threshold_used`。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/scoring/test_response_formatter.py` 8 passed；全套 `pytest -q` 178 passed；`ruff check` 與 `ruff format --check` 對本任務檔案通過。
- **更動重點**：新增 `scoring/response_formatter.py`，提供 JSON／CSV 回應物件、可選情境門檻、統一錯誤 envelope 與公式安全 CSV；調整門檻只重算 `risk_class` 與 `threshold_used`，不修改來源結果、信心分數、百分位、Top 10% 或解釋。
- **技術注意事項**：不得把使用者情境門檻寫回 manifest、active pointer 或程序全域狀態。

### [x] P3-07 建立 FastAPI 應用殼層與錯誤映射

- **目標**：`src/bankruptcy_prediction/api/app.py`、`src/bankruptcy_prediction/api/error_handlers.py`
- **相依**：P0-04 至 P0-06、P3-03、P3-04
- **輸入與輸出**：`create_app(settings) -> FastAPI`；輸出統一錯誤格式、request ID 與啟動／關閉生命週期。
- **驗收準則**：
  - 422、413、503、500 均轉換為 `error_code`、`message`、`details`、`remediation`。
  - 500 不回傳堆疊、內部路徑或上傳內容。
  - 測試：`tests/api/test_error_handlers.py` 驗證狀態碼、request ID 與敏感資訊隔離。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/api/test_error_handlers.py` 3 passed；全套 `pytest -q` 181 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `api/app.py`、`api/error_handlers.py` 與 `api/errors.py`；建立 request ID middleware、localhost-safe app factory、422／413／503／500 統一錯誤 envelope，並隔離內部路徑、堆疊與上傳內容。
- **技術注意事項**：執行命令與部署文件必須明確綁定 `127.0.0.1`，不得預設 `0.0.0.0`。

### [x] P3-08 實作 health 與 model 端點

- **目標**：`src/bankruptcy_prediction/api/routes/health.py`、`src/bankruptcy_prediction/api/routes/model.py`
- **相依**：P3-03、P3-07
- **輸入與輸出**：`GET /health` 與 `GET /api/v1/model`；輸出服務狀態與 active model 摘要。
- **驗收準則**：
  - health 在模型不可用時仍可呼叫，並回傳 `active_model_loaded=false` 與非敏感原因。
  - model 端點回傳版本、建立時間、預設門檻、特徵數及主要 final test 指標。
  - 測試：`tests/api/test_health_model.py` 覆蓋有模型、無模型與損壞模型。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/api/test_health_model.py` 2 passed；全套 `pytest -q` 183 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `/health` 與 `/api/v1/model` 唯讀端點；health 在無 active model 時仍回報 degraded，model 端點只輸出版本、建立時間、門檻、特徵數及主要指標，不暴露模型包或本機路徑。
- **技術注意事項**：不得透過端點下載 joblib、完整 manifest 或本機路徑。

### [x] P3-09 實作 score JSON／CSV 端點

- **目標**：`src/bankruptcy_prediction/api/routes/score.py`
- **相依**：P3-04 至 P3-07
- **輸入與輸出**：`POST /api/v1/score?format=json|csv`；輸入 multipart CSV 與可選 threshold，輸出同步評分結果。
- **驗收準則**：
  - 超過 10 MB 或 10,000 列回 413；schema／threshold 錯誤回 422；無 active model 回 503。
  - 部分列失敗仍回 200，摘要與各列狀態一致。
  - 測試：`tests/api/test_score_endpoint.py` 覆蓋 JSON、CSV、部分成功、超限、空檔與模型不可用。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/api/test_score_route.py` 2 passed；全套 `pytest -q` 185 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `api/routes/score.py`，以 multipart CSV、10 MB／10,000 列限制、受控 request workspace、同步批次評分與 JSON／CSV 回應串接；成功、超限與解析失敗路徑均清理暫存，CSV 回應使用附件下載標頭。
- **技術注意事項**：檔案大小應在完整載入記憶體前受限；回應完成或中斷後都須清理暫存。

### [x] P3-10 串接評分稽核日誌

- **目標**：`src/bankruptcy_prediction/service/audit_logger.py`
- **相依**：P0-06、P3-07、P3-09
- **輸入與輸出**：`record_score_request(summary) -> None`；在每次評分成功或失敗後寫入摘要日誌。
- **驗收準則**：
  - 日誌包含請求時間、request ID、版本、列數、成功／失敗數、耗時與錯誤碼統計。
  - 對含敏感 `record_id`、檔名及特徵值的請求，日誌仍不出現原值。
  - 測試：`tests/service/test_audit_logger.py` 驗證成功、422、503、500 與敏感字串掃描。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/service/test_audit_logger.py` 1 passed；全套 `pytest -q` 186 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `service/audit_logger.py`，將評分摘要接入安全 JSONL 稽核日誌；score endpoint 記錄時間、模型版本、列數、成功／失敗數、耗時與錯誤碼統計，不記錄識別碼、完整特徵、分數或 SHAP。
- **技術注意事項**：日誌失敗不得讓成功評分結果變成失敗，但必須以安全方式回報維運警告。

## 6. Phase 4 Streamlit 使用者介面

### Dashboard hotfix（2026-11-02）

- [x] H1 修復 Streamlit 啟動路徑與獨立入口。
- [x] H2 修復 FastAPI 啟動設定並校驗、啟用既有模型。
- [x] H3 以實際成果檔恢復研究資料與圖表。
- [x] H4 保留評分結果並完成端到端驗收。
- [x] H5 依明確授權補評估 Logistic Regression final test，並顯示雙模型八指標比較。
- [x] H6 將 Training CV AP 與 Validation AP 拆成兩張雙模型比較圖並移除原合併圖。
- [x] H7 美化 AP 圖表：XGBoost 深藍、細長柱、統一尺度與柱頂數值。
- [x] H8 以相同 held-out final test 重繪 Logistic Regression 與 XGBoost PR curve，標示 AP、盛行率基準、樣本數及事後授權比較，並通過完整回歸測試（240 passed）。
- [x] H9 依 TN／FP／FN／TP 格式重繪 XGBoost confusion matrix，標示實際數量、類別定義、final-test 與門檻，並通過完整回歸測試（242 passed）。
- [x] H10 強化 XGBoost calibration 圖：保留 0–1 全尺度、增加 0–0.25 放大圖、分數直方圖及十個分箱樣本／正例標示，並通過完整回歸測試（244 passed）。

### [x] P4-01 建立 Streamlit 應用殼層與 API Client

- **目標**：`src/bankruptcy_prediction/ui/app.py`、`src/bankruptcy_prediction/ui/api_client.py`
- **相依**：P3-08、P3-09
- **輸入與輸出**：輸入 FastAPI localhost base URL；輸出四分頁導覽、共用模型摘要與型別化 API client。
- **驗收準則**：
  - UI 只透過 HTTP 呼叫 FastAPI，不 import joblib、SHAP 或訓練模組。
  - FastAPI 不可用或無 active model 時顯示可操作的錯誤訊息，評分控制停用。
  - 測試：`tests/ui/test_app_shell.py` 驗證四分頁、client 邊界與失敗狀態。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_app_shell.py` 2 passed；全套 `pytest -q` 188 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `ui/app.py` 與 `ui/api_client.py`；UI 僅透過 localhost FastAPI，提供四分頁導覽、API 連線錯誤提示、無 active model 時停用評分控制與記憶體內 session 選頁狀態。
- **技術注意事項**：UI 與 API 都只綁定 localhost；MVP 不建立登入、cookie 帳號或外部連線。

### [x] P4-02 實作總覽與模型資訊頁

- **目標**：`src/bankruptcy_prediction/ui/pages/overview.py`
- **相依**：P4-01
- **輸入與輸出**：輸入 `GET /api/v1/model` 回應；輸出限制聲明、版本、日期、預設門檻、AP 與 Recall@Top10%。
- **驗收準則**：
  - 全頁使用「模型信心分數」，不出現「破產機率」字樣。
  - 顯示資料缺少公司、年度與預測窗口，以及「僅供分析優先順序使用」。
  - 測試：`tests/ui/test_overview_page.py` 驗證必要欄位與禁用文案。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_overview_page.py` 1 passed；全套 `pytest -q` 189 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `ui/pages/overview.py`，顯示 active model 的版本、建立時間、門檻、AP、Recall@Top10% 與限制聲明；頁面只使用「模型信心分數」，並明確說明缺少公司、年度與預測窗口及分析用途限制。
- **技術注意事項**：只顯示模型包保存的凍結指標，不在 UI 重算 final test。

### [x] P4-03 實作批次上傳與驗證摘要

- **目標**：`src/bankruptcy_prediction/ui/pages/batch_scoring.py`
- **相依**：P4-01、P3-09
- **輸入與輸出**：輸入 CSV 與可選 threshold；輸出總列數、成功數、失敗數、警告及錯誤摘要。
- **驗收準則**：
  - 上傳前顯示 10 MB、10,000 列、95 特徵與 `record_id` 規則。
  - 422、413、503 以可修正訊息顯示，不暴露堆疊。
  - 測試：`tests/ui/test_batch_upload.py` 驗證合法、部分成功與檔案級錯誤呈現。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_batch_upload.py tests/ui/test_app_shell.py` 4 passed；全套 `pytest -q` 191 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `ui/pages/batch_scoring.py`，顯示 10 MB／10,000 列／95 特徵／record_id 規則，透過 API client 上傳並呈現摘要；422、413、503 轉為可操作訊息，無 active model 時停用控制。
- **技術注意事項**：Streamlit 不自行重做 schema 驗證；前端提示不能取代 FastAPI 的權威判定。

### [x] P4-04 實作門檻情境、分布與高風險名單

- **目標**：`src/bankruptcy_prediction/ui/components/scoring_results.py`
- **相依**：P4-03
- **輸入與輸出**：輸入成功評分批次；輸出門檻滑桿、分數分布、高風險表格、Top 10% 名單及下載按鈕。
- **驗收準則**：
  - 調整門檻只更新分類與高風險數，不改分數、百分位或 Top 10%。
  - 下載內容與畫面顯示相同 `threshold_used`、版本與列狀態。
  - 測試：`tests/ui/test_scoring_results.py` 覆蓋 T06、T07、空成功列與下載一致性。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_scoring_results.py` 2 passed；全套 `pytest -q` 193 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過。
- **更動重點**：新增 `ui/components/scoring_results.py`，以 API 回應顯示分數分布、高風險名單、Top 10% 與下載；門檻調整重新呼叫 API，保留同批次 Top 10% 與模型版本／列狀態一致性。
- **技術注意事項**：若門檻變更需重新呼叫 API，session state 必須保留同一上傳批次且不得在磁碟建立歷史。

### [x] P4-05 實作個案解釋頁

- **目標**：`src/bankruptcy_prediction/ui/pages/case_explanation.py`
- **相依**：P4-03、P2-09
- **輸入與輸出**：輸入當次成功評分結果與 `record_id` 選擇；輸出分數、分類、原始值及前五項正負向貢獻。
- **驗收準則**：
  - 只能選擇當次合法列；失敗列顯示錯誤而非空白解釋。
  - 顯示補值／out_of_range 警告與非因果解釋聲明。
  - 測試：`tests/ui/test_case_explanation.py` 驗證選取、正負因素、原始值與限制文案。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_case_explanation.py` 2 passed；全套 `pytest -q` 195 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `ui/pages/case_explanation.py`，可依 record_id 顯示當次上傳的原始值、模型信心分數、分類與正／負向貢獻；空成功列顯示提示，並明確標示貢獻不代表因果關係。
- **技術注意事項**：畫面不得將 SHAP 貢獻描述為造成破產的原因。

### [x] P4-06 實作模型效能頁

- **目標**：`src/bankruptcy_prediction/ui/pages/model_performance.py`
- **相依**：P4-01、P2-10、P3-08
- **輸入與輸出**：輸入模型摘要與模型包圖表；輸出 PR、校準、混淆矩陣、Recall@Top10%、切分及限制。
- **驗收準則**：
  - 指標、圖表與 `metrics.json`／`model_report.html` 保存結果一致。
  - 明確標示 64%／16%／20%、seed 42、final test 只評估一次及 MVP 門檻。
  - 測試：`tests/ui/test_model_performance.py` 驗證必要圖表、指標與限制說明。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_model_performance.py` 1 passed；全套 `pytest -q` 196 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `ui/pages/model_performance.py`，顯示 API 保存的 AP、Recall@Top10%、MVP 結果、64%／16%／20% 切分、seed 42、final test 一次性評估及門檻限制；不在 UI 即時計算替代圖表。
- **技術注意事項**：若圖表缺失或 checksum 不符，頁面應顯示模型包錯誤，不得即時計算替代圖。

### [x] P4-07 完成跨分頁狀態與 UI 端到端流程

- **目標**：`src/bankruptcy_prediction/ui/session_state.py`、`tests/ui/test_end_to_end.py`
- **相依**：P4-02 至 P4-06
- **輸入與輸出**：輸入同一 Streamlit 工作階段的上傳與評分結果；輸出四分頁共用且不跨瀏覽器保存的狀態。
- **驗收準則**：
  - 批次評分後可直接在個案解釋頁選取成功列，模型版本與門檻一致。
  - 重新啟動 session 後不保留上傳內容或結果歷史。
  - 測試：以 Streamlit 測試工具或等價 browser test 驗證四分頁、門檻、Top 10%、下載與限制聲明。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui/test_end_to_end.py` 1 passed；全套 `pytest -q` 197 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `ui/session_state.py`，集中管理當次批次結果、CSV bytes 與檔名；批次評分、門檻結果與個案解釋共用同一 session，並提供明確清理函式，不建立磁碟歷史。
- **技術注意事項**：session state 只保存記憶體內的當次結果；不得新增 SQLite 或其他歷史資料庫。

## 7. Phase 5 測試、安全、效能與發行驗收

### [x] P5-01 完成資料與評分單元測試矩陣

- **目標**：`tests/data/`、`tests/features/`、`tests/scoring/`
- **相依**：P1-01 至 P1-10、P3-05、P3-06
- **輸入與輸出**：輸入 SDD 邊界案例；輸出欄名、schema、缺值、補值、極端值、欄位移除、指標、Top 10% 與門檻分類測試證據。
- **驗收準則**：
  - T01 至 T07、T11 均有明確自動測試與預期值。
  - 錯誤碼、警告碼、列順序與 JSON／CSV schema 均固定。
  - 執行：`pytest tests/data tests/features tests/scoring tests/export -q` 全部通過。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/data tests/features tests/scoring tests/export` 68 passed；全套 `pytest -q` 206 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/scoring/test_acceptance_matrix.py`，建立 T01 至 T07、T11 驗收矩陣索引，並確認固定結果欄位、補值／範圍警告碼與資料／特徵／評分／匯出測試套件均可執行。
- **技術注意事項**：測試不得依賴已存在的 active model 或前次執行暫存。

### [x] P5-02 完成防洩漏測試

- **目標**：`tests/training/test_no_leakage.py`
- **相依**：P2-01 至 P2-07
- **輸入與輸出**：輸入可識別 train／validation／test 的 sentinel 資料；輸出所有 `fit` 與選擇步驟的資料來源證據。
- **驗收準則**：
  - Imputer、Scaler、selector、範圍、類別權重與校準只使用訓練折。
  - 模型選擇與門檻只使用 validation；final test 不出現在 fit 或選擇流程。
  - 執行：`pytest tests/training/test_no_leakage.py -q` 全部通過。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/training/test_no_leakage.py` 2 passed；全套 `pytest -q` 208 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/training/test_no_leakage.py`，驗證 Imputer／RangeMonitor 僅使用 train 統計、validation 門檻選擇不接收 test、final test 評估只呼叫 frozen model 且不 fit。
- **技術注意事項**：這是 Release 阻擋測試；不得以人工程式審查取代。

### [x] P5-03 完成模型可重現與品質門檻測試

- **目標**：`tests/model/test_training_acceptance.py`
- **相依**：P2-08、P2-12
- **輸入與輸出**：輸入固定資料、設定與 seed；輸出切分、搜尋、勝出規則、門檻與 final test 指標驗收報告。
- **驗收準則**：
  - 固定設定重跑時切分一致，主要指標在允許容差內等價。
  - 驗證兩模型各 30 組、5-fold、AP tie-breaker 與 final test 一次性規則。
  - 正式候選模型須達 test AP 基準 2 倍及 Recall@Top10% 50%，未達時 Release 失敗但仍可保存未通過候選包。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/model/test_training_acceptance.py` 2 passed；全套 `pytest -q` 210 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/model/test_training_acceptance.py`，驗證 seed 42、64%／16%／20% 切分、30 組／5-fold／Average Precision 搜尋基線，以及 final frozen metrics 的 AP 與 Recall@Top10% MVP 品質閘門。
- **技術注意事項**：不可為了讓測試通過而用 test 指標調參；品質未達應回到需求或資料檢討。

### [x] P5-04 完成 CLI 與模型生命週期整合測試

- **目標**：`tests/integration/test_model_lifecycle.py`
- **相依**：P2-12、P3-01 至 P3-03
- **輸入與輸出**：輸入 train、activate、rollback 命令序列；輸出候選包、active pointer 與回滾狀態。
- **驗收準則**：
  - train 不更新 active；activate 校驗後更新；rollback 可切回既有版本。
  - 損壞包、未知版本與原子寫入失敗均保留原 active model。
  - 執行：`pytest tests/integration/test_model_lifecycle.py -q` 通過 T09、T10。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/integration/test_model_lifecycle.py tests/integration/test_train_cli.py` 通過；全套 `pytest -q` 212 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/integration/test_model_lifecycle.py`，驗證 train／activate／rollback CLI 命令存在、未知版本失敗時不建立或覆寫 active pointer；既有 test-only train 整合測試同時覆蓋候選包與 active 保留。
- **技術注意事項**：整合測試不得操作正式 `models` 目錄。

### [x] P5-05 完成 FastAPI 端到端整合測試

- **目標**：`tests/integration/test_api_workflow.py`
- **相依**：P3-07 至 P3-10、P5-04
- **輸入與輸出**：輸入 active model 與多種 multipart CSV；輸出 health、model、score JSON／CSV 與日誌結果。
- **驗收準則**：
  - 覆蓋全成功、部分成功、422、413、503、500 與 request ID。
  - JSON／CSV 列級結果、版本及門檻一致，暫存目錄在每種結果後為空。
  - 執行：`pytest tests/integration/test_api_workflow.py -q` 全部通過。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/integration/test_api_workflow.py` 2 passed；全套 `pytest -q` 214 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/integration/test_api_workflow.py`，驗證 health／model 與 score JSON／CSV、request ID、422／413／503、受控 active model 及所有路徑的暫存清理。
- **技術注意事項**：測試 500 時應注入可預期例外，不得依賴真實記憶體不足或破壞系統環境。

### [x] P5-06 完成 Streamlit UI 驗收測試

- **目標**：`tests/ui/test_acceptance.py`
- **相依**：P4-07、P5-05
- **輸入與輸出**：輸入模擬 API 回應與實際 localhost 測試環境；輸出四分頁操作驗收證據。
- **驗收準則**：
  - 四分頁、門檻即時調整、Top 10% 不變、個案解釋、限制聲明及下載一致性全部通過。
  - 全 UI 掃描不含「破產機率」、「將會破產」或「不會破產」等禁用表達。
  - 無 active model 時所有評分控制停用並顯示可理解提示。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/ui` 全部通過；全套 `pytest -q` 216 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/ui/test_acceptance.py`，驗證四分頁、Top 10% 不受門檻語意影響、限制聲明及全 UI 禁用文案掃描；既有 UI 測試覆蓋上傳、下載、個案解釋與無 active model 停用。
- **技術注意事項**：驗收應同時涵蓋 Microsoft Edge 或專案指定的 Windows 瀏覽器；不得只測函式回傳。

### [x] P5-07 完成安全與韌性測試

- **目標**：`tests/security/`
- **相依**：P3-01、P3-04、P3-07、P3-09、P3-10、P1-10
- **輸入與輸出**：輸入超大檔、惡意檔名、路徑穿越、公式注入、損壞模型、無 active model 與中斷請求；輸出安全測試報告。
- **驗收準則**：
  - Streamlit／FastAPI 啟動設定只綁定 `127.0.0.1`。
  - 路徑穿越、CSV 公式注入、checksum 失敗與敏感日誌掃描全部被阻擋。
  - 所有成功與失敗路徑均清理暫存，且錯誤可由 request ID 追蹤。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/security tests/service tests/export` 通過；全套 `pytest -q` 219 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/security/test_security.py`，驗證 localhost 綁定、受控暫存路徑與中斷清理、路徑穿越防護及 CSV 公式注入；既有 checksum、錯誤與日誌測試納入安全矩陣。
- **技術注意事項**：測試只能在隔離暫存目錄破壞檔案；不得修改正式模型包或使用廣泛刪除命令。

### [x] P5-08 完成 10,000 列效能驗收

- **目標**：`tests/performance/test_score_10000.py`、`reports/performance_acceptance.md`
- **相依**：P3-05、P3-09、P5-05
- **輸入與輸出**：輸入 10,000 列有效資料與 active model；輸出三次同步評分時間、CPU、RAM、Python 與套件版本。
- **驗收準則**：
  - 每次完整驗證、評分、前三項解釋與序列化在 10 秒內完成。
  - 測試包含 API 邊界而非只量測模型 `predict_proba`。
  - 報告保存資料生成方式、暖機條件、三次結果與是否通過。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/performance/test_score_10000.py` 1 passed（完整流程 wall-clock < 10 秒）；全套 `pytest -q` 220 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `tests/performance/test_score_10000.py` 與 `reports/performance_acceptance.md`，以 deterministic 10,000 列量測 schema／列驗證、評分、前三項解釋及 JSON 序列化，不省略解釋步驟。
- **技術注意事項**：若未達標，優先分析 SHAP 批次化、序列化與資料轉換；不得省略解釋以偽造通過。

### [x] P5-09 建立 Windows 啟動腳本與 Runbook

- **目標**：`scripts/start_api.ps1`、`scripts/start_ui.ps1`、`docs/RUNBOOK.md`
- **相依**：P3-08、P3-09、P4-07、P5-07
- **輸入與輸出**：輸入虛擬環境、設定與 models root；輸出依序檢查模型、啟動 FastAPI、啟動 Streamlit 及故障排除步驟。
- **驗收準則**：
  - 啟動腳本明確綁定 `127.0.0.1`，FastAPI 預設 port 8000、Streamlit 預設 port 8502。
  - Runbook 涵蓋無模型、模型損壞、port 佔用、回滾、日誌位置與暫存清理。
  - 測試：`tests/test_launch_scripts.py` 驗證參數、啟動順序與無 `0.0.0.0`。
- **狀態**：已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_launch_scripts.py` 2 passed；全套 `pytest -q` 222 passed；`ruff check src tests` 通過。
- **更動重點**：新增 `scripts/start_api.ps1`、`scripts/start_ui.ps1` 與 `docs/RUNBOOK.md`；啟動固定 loopback、API 8000／UI 8502，並記錄模型啟用／回滾、checksum、port、無模型與暫存清理處置。
- **技術注意事項**：MVP 不建立 Windows Service、容器、雲端部署或自動對外開放防火牆。

### [~] P5-10 完成 Release 驗收與需求追溯

- **目標**：`docs/RELEASE_CHECKLIST.md`、`reports/acceptance_report.md`
- **相依**：P5-01 至 P5-09
- **輸入與輸出**：輸入全部測試、模型包、UI 與效能證據；輸出 Release 判定及 F01 至 F10 的需求—元件—介面—測試對照。
- **驗收準則**：
  - F01 至 F10 每項都有可執行元件、介面與通過的測試 ID。
  - active model 已人工啟用且通過 final test 門檻；報告、API、UI 與下載內容一致。
  - 全量測試、lint、依賴檢查、安全測試與效能測試皆有日期、環境與結果紀錄。
- **狀態**：文件與自動化追溯已完成（2026-10-22），`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_release_checklist.py` 1 passed；全套 `pytest -q` 223 passed；`ruff check src tests` 與 `ruff format --check src tests` 通過；Release 閘門阻擋中。
- **更動重點**：新增 `docs/RELEASE_CHECKLIST.md`、`reports/acceptance_report.md` 與 `tests/test_release_checklist.py`，完成 F01 至 F10 需求—元件—介面—測試追溯；明確列出尚缺正式模型、人工 activate、Windows／瀏覽器實機證據，未將 Not Ready for Release 誤標為通過。
- **技術注意事項**：單有模型指標或畫面 Demo 不構成完成；任何未通過項目須明確標為阻擋或經核准的限制。

## 8. 依賴索引

| 任務 | 直接依賴 |
| --- | --- |
| P1-01 | P0-04、P0-05、P0-07 |
| P1-02 至 P1-10 | 依各任務列出的 schema、驗證、特徵及輸出上游任務 |
| P2-01 | P1-01、P1-02 |
| P2-02、P2-03 | P1-06、P1-08、P2-01 |
| P2-04 至 P2-12 | 依序依賴搜尋、校準、選模、門檻、評估、報告與模型包 |
| P3-01 | P2-11 |
| P3-02、P3-03 | P3-01 與既有候選模型包 |
| P3-05、P3-06 | Phase 1 資料契約、P2-09 解釋及 P3-03 active model |
| P3-07 至 P3-10 | P3-03 至 P3-06 與 Phase 0 共用錯誤／日誌 |
| P4-01 至 P4-07 | P3-08、P3-09；頁面只透過 API client 取得資料 |
| P5-01 至 P5-03 | 對應資料、訓練、評估元件完成 |
| P5-04、P5-05 | CLI／模型治理及 FastAPI 全流程完成 |
| P5-06 | P4-07、P5-05 |
| P5-07、P5-08 | 線上評分流程完成且有可用 active model |
| P5-09 | API、UI、安全驗收完成 |
| P5-10 | P5-01 至 P5-09 全部有結果證據 |

## 9. 第一個可展示的垂直切片

第一個 Demo 不等待完整 UI 與正式長時間訓練，可依下列順序建立最小但符合架構邊界的切片：

~~~text
P0-01 -> P0-02 -> P0-03 -> P0-04 -> P0-05 -> P0-07
-> P1-01 -> P1-02 -> P1-03 -> P1-04 -> P1-05 -> P1-06 -> P1-08 -> P1-09
-> P2-01 -> P2-02 -> P2-04 -> P2-05 -> P2-06 -> P2-07 -> P2-08 -> P2-09 -> P2-11 -> P2-12
-> P3-01 -> P3-02 -> P3-03 -> P3-04 -> P3-05 -> P3-06 -> P3-07 -> P3-08 -> P3-09
-> P4-01 -> P4-02 -> P4-03 -> P4-04 -> P4-05
~~~

展示條件：

1. 使用小型但 schema 完整的 CSV 上傳至 Streamlit。
2. FastAPI 回傳模型信心分數、百分位、分類、Top 10%、警告與前三項解釋。
3. UI 能顯示模型摘要、批次結果與一筆個案解釋，並下載安全 CSV。
4. 候選模型須先由 CLI 建立並人工 activate；Demo 不得繞過 active model 或讓 UI 直接載入模型。
5. Demo 可使用 test-only 快速搜尋設定，但不得據此宣稱正式模型已達 MVP 品質或 10 秒效能門檻。

## 10. F01 至 F10 任務覆蓋

| 需求 | 實作任務 | 主要驗收任務 |
| --- | --- | --- |
| F01 資料稽核 | P1-01、P1-03、P1-05、P3-10 | P5-01、P5-05 |
| F02 分層 train／validation／test | P2-01、P2-12 | P5-02、P5-03 |
| F03 缺值、縮放與特徵 Pipeline | P1-06 至 P1-08、P2-02、P2-03 | P5-01、P5-02 |
| F04 Logistic Regression 與 XGBoost | P2-02 至 P2-06 | P5-03 |
| F05 類別權重 | P2-02、P2-03、P2-04 | P5-02、P5-03 |
| F06 門檻選擇與情境調整 | P2-07、P3-06、P4-04 | P5-01、P5-06 |
| F07 分數、百分位與排序 | P2-08、P3-05、P4-04 | P5-01、P5-05 |
| F08 SHAP 個案解釋 | P2-09、P3-05、P4-05 | P5-05、P5-06、P5-08 |
| F09 批次結果與匯出 | P1-09、P1-10、P3-06、P3-09 | P5-01、P5-05、P5-07 |
| F10 模型版本、指標、限制與報告 | P2-10 至 P2-12、P3-01 至 P3-03、P4-02、P4-06 | P5-03、P5-04、P5-10 |

## 11. Release 完成定義

MVP 完成必須同時滿足：

- F01 至 F10 都有實作、介面與測試證據。
- active model 由人工 activate，且 final test 的 Average Precision 至少為正例比例基準的 2 倍、Recall@Top10% 至少 50%。
- Streamlit 與 FastAPI 在 Windows 11 CPU 環境以 localhost 完成端到端評分。
- 10,000 列的驗證、評分、前三項解釋與序列化在驗收機 10 秒內完成。
- 上傳與結果暫存均清理，評分日誌不含 `record_id`、完整財務特徵、分數或 SHAP 貢獻。
- 模型包、`model_report.html`、`metrics.json`、API、四個 UI 分頁與下載檔的版本、門檻、指標及限制聲明一致。
- 所有介面只使用「模型信心分數」及「僅供分析優先順序使用」限制說明，不宣稱未來窗口破產機率或因果關係。

## 12. Colab 交接補充（2026-10-22）

此清單只記錄交接工作，不代表既有 Phase 5 已取得正式訓練或本機驗收證據。

- [x] C01 檢查訓練入口，修正原始欄名前後空白造成 SHAP 名稱不一致，通過回歸測試。
- [x] C02 建立不含原始 CSV 的程式 ZIP 與逐步 Colab 筆記本；驗證語法、封裝內容及 train-only 兩模型／SHAP 試跑。
- [ ] C03 使用者於 Colab 安裝固定依賴並執行測試與試跑；目前只有 Windows 本機驗證。
- [ ] C04 在 10 單元預算內執行正式搜尋、校準、選模、門檻及一次 final test，回傳完整證據。
- [ ] C05 檢查真實品質指標、人工決定啟用，再完成 Windows CPU／UI／效能驗收。

交接檔案：`artifacts/colab/`。目前實作使用 CPU，不因配置 GPU 自動加速；10 單元需人工監控，無自動扣費上限。正式失敗不得盲目重跑最終測試。

### 執行路徑更新：先使用本機 CPU

使用者於 2026-10-22 指示先評估本機；C03、C04 暫緩，不消耗 Colab 單元。

- [x] L01 完成本機 CPU 訓練折耗時及記憶體評估，結果見 `reports/local_cpu_assessment.md`。
- [x] L02 本機正式訓練完成；兩模型各 30 組、5-fold 已依序執行，XGBoost 入選，候選模型包 `candidate-20261022T105726Z-155ceb0b` 通過完整性校驗及 MVP 品質門檻；證據記錄於 `reports/local_formal_training.log`。
- [ ] L03 檢查收斂警告、正式指標與模型包，人工確認啟用及 Windows 驗收。

### XGBoost objective 比較實驗（2026-10-22）

- [x] E01 定義比較協定：固定原 training／validation、30 組 × 5-fold，不再存取 final test。
- [x] E02 完成 `rank:pairwise` 搜尋、全 training refit、validation 評估與可攜模型保存。
- [x] E03 完成 `reg:pseudohubererror` 搜尋、fold-local 類別權重、全 training refit、validation 評估與可攜模型保存。
- [x] E04 以既有 `binary:logistic` 在相同 validation 比較 AP、Recall／Precision@Top10% 及 NDCG@Top10%，獨立重載重算一致。
- [x] E05 完成比較報告及完整回歸測試（226 passed）；production candidate 未變更。
