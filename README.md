# Smart Factory Alert Agent

本專案示範一個端到端的智慧工廠設備異常偵測與告警系統，包含資料生成、前處理、監督式模型訓練、規則式 AI 代理，以及使用 Streamlit 的視覺化介面。

專案結構

```
smart_factory_alert_agent/
│
├── data/
│   └── generate_data.py
│   └── sensor_data.csv (由 generate_data.py 產生)
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── agent.py
│   └── evaluate.py
│
├── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

快速開始

1. 建議建立虛擬環境並安裝套件：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 產生測試資料：

```bash
python data/generate_data.py
```

3. 訓練模型並產生評估報告：

```bash
python -m src.train_model
```

4. 啟動 Streamlit 應用：

```bash
streamlit run app.py
```

AI 代理邏輯

代理使用模型輸出（異常機率）並套用下列簡單規則：

```
if prob >= 0.8: CRITICAL -> Stop machine immediately and notify engineer
elif prob >= 0.6: WARNING -> Schedule maintenance inspection
else: NORMAL -> No action required
```

工具加速說明

- 使用 ChatGPT / GPT-5-mini 與 GitHub Copilot 協助系統設計與範例程式碼生成。
- Scikit-learn 用於模型訓練與評估；Streamlit 用於快速建立互動式儀表板。

下一步與截圖

- 在啟動 Streamlit 後，可擷取「Data Viewer」、「Anomaly Detection」、「Agent Logs」與「Metrics」畫面的截圖作為 Demo。
