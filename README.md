# Smart Factory Alert Agent

智慧工廠設備異常偵測與告警系統

## 專案簡介

使用機器學習模型偵測工廠感測器異常，提供即時告警。包含資料生成、模型訓練、評估與互動式儀表板。

## 專案結構

```
Smart-Factory-Alert-Agent/
├── app.py                   # Streamlit 儀表板
├── config.yaml              # 系統配置檔
├── requirements.txt         # Python 套件依賴
│
├── data/
│   └── generate_data.py     # 產生合成感測器資料
│
├── src/                     # 核心程式模組
│   ├── preprocess.py        # 資料前處理
│   ├── train_model.py       # 模型訓練
│   ├── evaluate.py          # 模型評估
│   ├── agent.py             # 告警代理
│   ├── dashboard.py         # Dashboard UI
│   ├── utils.py
│   └── noise.py
│
└── scripts/
    └── cli_simulator.py     # CLI 模擬器
```

## 快速開始

### 1. 環境設置

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Windows)
.venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 產生資料

```bash
# 產生合成感測器資料
python data/generate_data.py --modes fuzzy no_fuzzy -n 300
```

### 3. 訓練模型

```bash
# 訓練所有模型
python -m src.train_model
```

### 4. 啟動儀表板

```bash
# 啟動 Streamlit 網頁介面
streamlit run app.py
```

瀏覽器訪問：http://localhost:8501

### 5. 使用 CLI 模擬器

```bash
# 執行命令列模擬器
python scripts/cli_simulator.py
```

## 主要功能

- **資料生成**：使用三層信號合成技術產生逼真的感測器時間序列資料
- **異常偵測**：訓練 Logistic Regression、Random Forest、Gradient Boosting 三種模型
- **評估指標**：Accuracy、Precision、Recall、F1、AUC、MSE
- **告警分級**：NORMAL（正常）/ WARNING（警告）/ CRITICAL（危急）
- **互動儀表板**：即時監控與告警建議

## 技術棧

- **Python 3.10+**
- **scikit-learn** - 機器學習框架
- **pandas / numpy** - 資料處理
- **Streamlit** - Web 儀表板
- **matplotlib / seaborn** - 視覺化

## 授權

MIT License
