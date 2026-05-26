# VeriPromiseESG 2026 — ESG 承諾驗證競賽

本專案為 [VeriPromiseESG 2026](https://veripromiseesg.github.io/) 競賽之參賽程式碼，目標是從企業 ESG 永續報告書段落中，自動驗證承諾的存在性、時程、佐證品質等多個面向。

---

## 🏆 競賽說明

- **競賽平台**：[AI CUP](https://www.aicup.tw/) / [AIDEA](https://www.aidea-web.tw/aicup_veripromiseesg)
- **任務類型**：多任務文本分類（Multi-Task Classification）
- **評分指標**：加權 Macro F1

---

## 📋 任務說明

給定一段來自企業 ESG 報告書的中文文字段落，模型需同時預測以下 4 個欄位：

| 欄位 | 說明 | 類別 | 評分權重 |
|------|------|------|--------|
| `promise_status` | 是否包含承諾 | Yes / No | **20%** |
| `verification_timeline` | 承諾預期完成時程 | already / within_2_years / between_2_and_5_years / more_than_5_years / N/A | **15%** |
| `evidence_status` | 是否有佐證 | Yes / No / N/A | **30%** |
| `evidence_quality` | 佐證語意品質 | Clear / Not Clear / Misleading / N/A | **35%** |

---

## 📁 資料說明

每筆資料為 JSON 格式，主要欄位如下：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `data` | str | ESG 報告書原始段落文本 |
| `URL` | str | 來源超連結 |
| `page_number` | int | 報告書頁碼 |
| `ESG_type` | str | 文本主題（E / S / G） |
| `promise_status` | str | 承諾狀態標籤 |
| `promise_string` | str | 承諾對應語句 |
| `verification_timeline` | str | 承諾時程標籤 |
| `evidence_status` | str | 證據狀態標籤 |
| `evidence_string` | str | 證據對應語句 |
| `evidence_quality` | str | 證據品質標籤 |

> ⚠️ 資料檔案幫你們放在`data/` 底下。

---

## 📂 專案結構

```
ESG_compitition/
├── README.md              # 本說明文件
├── .gitignore             # Git 忽略規則，
├── requirements.txt       # Python 套件需求（各 branch 自行維護）
│
├── data/                  # 資料集
│   ├── vpesg_4k_train_1000.json
│   └── vpesg_4k_train_1000.csv
│
├── notebooks/             # 實驗 Notebooks
│   ├── baseline.ipynb     # 主辦方提供的 baseline
│   └── exp_YYYY-MM-DD_<實驗名>.ipynb
│
└── results/               # 實驗結果
    └── YYYY-MM-DD_<實驗名>/
        ├── config.json    # 超參數設定
        ├── metrics.json   # 評估分數
        └── training_curve.png
```

---

## 🌿 Branch 策略

| Branch | 說明 |
|--------|------|
| `main` | 穩定版，存放 baseline 與原始資料初始狀態 |
| `dev` | 整合各最佳結果 |
| `<name>/exp` | 各成員個人實驗分支 |

---

## 🚀 環境安裝

```bash
pip install -r requirements.txt
```

### ⚠️ 首次 clone 後請額外執行（一次就好）

```bash
pip install nbstripout
nbstripout --install
```

> **為什麼需要這個？** `nbstripout` 會在每次 `git commit` 前自動清除 notebook 的執行輸出（outputs、execution count），避免不同人跑完 notebook 後產生大量無意義的 merge 衝突。本機執行結果**不受影響**。

---

## 📊 實驗結果紀錄

每次實驗請在 `results/YYYY-MM-DD_<實驗名>/` 中存放以下檔案：
- `config.json`：模型名稱、超參數
- `metrics.json`：各欄位 Macro F1 與加權分數
- `training_curve.png`：訓練曲線圖

---

## 🔗 相關連結

- [VeriPromiseESG 競賽官網](https://veripromiseesg.github.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SemEval-2025 Task 6 論文](https://aclanthology.org/2025.semeval-1.321/)
