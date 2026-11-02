# 企業破產預測系統 Runbook

## 啟動

1. 確認 `.venv` 已安裝依賴，並在專案根目錄執行 `scripts\start_api.ps1`。
2. 另開 PowerShell 執行 `scripts\start_ui.ps1`。
3. API 固定綁定 `127.0.0.1:8000`，Streamlit 固定綁定 `127.0.0.1:8501`；不對外開放。
4. 以 `GET http://127.0.0.1:8000/health` 確認服務狀態。

## 常見故障

- 無 active model：health 仍可用但評分控制停用；執行離線 CLI `activate --version <id>`。
- 模型損壞或 checksum 失敗：保留目前 active pointer，檢查模型包後以 `rollback --version <id>` 切回既有版本。
- port 佔用：以 `-Port` 指定另一個本機 port，並同步更新 UI 的 `-ApiPort`；仍只允許 loopback。
- 上傳暫存：每次請求完成或中斷後自動清理；啟動時只清掃專用前綴且超過 24 小時的暫存目錄。
- CSV 錯誤：依 API 回傳的 error_code、details 與 remediation 修正，不在 UI 端自行推導 schema。

## 模型生命週期

`train` 只建立候選包，不更新 active；人工檢視後執行 `activate --version <id>`。需要回復時執行 `rollback --version <id>`。所有命令都應指定隔離的 models root 進行測試。
