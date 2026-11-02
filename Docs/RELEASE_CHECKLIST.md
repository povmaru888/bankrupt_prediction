# Release Checklist

## 自動化閘門

- [x] F01 資料稽核：loader、schema、列品質與稽核日誌測試
- [x] F02 固定分層：64%／16%／20%、seed 42 測試
- [x] F03 缺值、縮放與 Pipeline：特徵處理與防洩漏測試
- [x] F04 兩個候選模型：Logistic Regression／XGBoost 搜尋測試
- [x] F05 類別權重：模型 pipeline 與設定測試
- [x] F06 門檻：validation F2、最低門檻 tie-breaker 與情境 API 測試
- [x] F07 分數、平均名次百分位與 Top 10% 測試
- [x] F08 SHAP 個案解釋與非因果文案測試
- [x] F09 JSON／CSV 結果、公式注入與暫存清理測試
- [x] F10 模型版本、指標、限制、health/model endpoint 測試
- [x] lint、format、全量 pytest
- [x] 10,000 列效能 assertion 小於 10 秒

## 尚未放行項目

- [ ] 正式（非 test-only）模型候選包已建立並通過 final-test MVP 門檻
- [ ] 人工 `activate --version <id>` 已執行並存在 active pointer
- [ ] 正式 Windows 11 驗收機已記錄 CPU、RAM、套件版本與三次效能結果
- [ ] Edge／瀏覽器端到端操作證據與簽核

Release 只有在上述四項完成後才能標記為通過；test-only 模型不可宣稱正式品質。
