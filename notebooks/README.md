# 🛰️ NASA SMAP/MSL Telemetry Anomaly Detection

Unsupervised anomaly detection on NASA satellite sensor telemetry data using machine learning. Built as a research-level project with a live interactive dashboard.

---

## 📌 Project Overview

Spacecraft like NASA's SMAP and MSL generate thousands of telemetry channels continuously. Detecting anomalies in these signals is critical for mission safety. This project applies unsupervised ML to detect anomalies **without needing labeled training data**.

The same approach directly maps to **network intrusion detection** and **SOC analyst workflows** in cybersecurity.

---

## 📊 Dataset

- **Source**: NASA SMAP/MSL Anomaly Detection Dataset
- **Channels**: 82 telemetry channels (55 SMAP + 27 MSL)
- **Features**: 25 per channel
- **Average anomaly rate**: 12.29%
- **Labels**: `labeled_anomalies.csv` with exact anomaly timestamps

---

## 🧠 Models

| Model | Approach | Avg F1 | Avg AUC |
|---|---|---|---|
| Baseline Threshold | Mean + 3σ | 0.00 | - |
| Isolation Forest | Unsupervised tree ensemble | 0.0377 | 0.5749 |
| MLP Autoencoder | Reconstruction error | 0.1935 | 0.6342 |
| Sliding Window AE | Windowed reconstruction (w=50) | **0.9111** | **0.9579** |

### Best Channel Results (D-4)
- F1: 0.9563 | AUC: 0.9994

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn streamlit
```

### 2. Run the dashboard
```bash
streamlit run notebooks/app.py
```

### 3. Run individual scripts
```bash
python notebooks/explore.py          # Phase 1: Data exploration
python notebooks/baseline.py         # Phase 2: Baseline model
python notebooks/isolation_forest.py # Phase 3: Isolation Forest
python notebooks/evaluation.py       # Phase 4: Evaluation
python notebooks/autoencoder.py      # Phase 5: MLP Autoencoder
python notebooks/lstm_autoencoder.py # Phase 5b: Sliding Window AE
python notebooks/run_all_channels.py # Run on all 82 channels
```

---

## 📁 Project Structure

---

## 📈 Dashboard Features

- **Detection Tab**: Select any channel + model, run live detection, view signal plot + ROC + PR curves
- **Dataset Stats Tab**: 82-channel overview, anomaly % bar chart, full stats table
- **Model Comparison Tab**: F1 score comparison across all channels, top 10 leaderboard

---

## 🔗 Cybersecurity Connection

The unsupervised anomaly detection pipeline built here directly applies to:
- Network intrusion detection (NIDS)
- SOC alert triage
- IoT sensor monitoring

This project is part of a broader Security Analyst portfolio including a [Network Intrusion Detection System](https://github.com/apoorvparmar46/network-ids) built with Scapy + scikit-learn.

---

## 👨‍💻 Author

**Apoorv Parmar**  
BTech CS — COER University, Roorkee  
GitHub: [github.com/apoorvparmar46](https://github.com/apoorvparmar46)