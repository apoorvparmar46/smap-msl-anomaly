import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"
RESULTS_PATH = r"C:\Users\Acer\smap-msl-anomaly\notebooks\all_channels_results.csv"

labels = pd.read_csv(LABELS_PATH)
all_channels = sorted(labels['chan_id'].tolist())
all_results = pd.read_csv(RESULTS_PATH)

st.set_page_config(page_title="SMAP Anomaly Detection", layout="wide")
st.title("🛰️ NASA SMAP/MSL Telemetry Anomaly Detection")
st.markdown("Unsupervised anomaly detection on NASA satellite sensor data using Isolation Forest and Autoencoder.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Dataset Stats", "🏆 Model Comparison"])

# ── TAB 1: Detection ──
with tab1:
    st.sidebar.header("Configuration")
    channel = st.sidebar.selectbox("Select Channel", all_channels)
    model_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "Autoencoder"])
    run_btn = st.sidebar.button("Run Detection")

    if run_btn:
        with st.spinner("Loading data..."):
            train_data = np.load(os.path.join(TRAIN_DIR, f"{channel}.npy"))
            test_data = np.load(os.path.join(TEST_DIR, f"{channel}.npy"))
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)
            chan_row = labels[labels['chan_id'] == channel].iloc[0]
            anomaly_ranges = eval(chan_row['anomaly_sequences'])
            ground_truth = np.zeros(test_data.shape[0], dtype=int)
            for start, end in anomaly_ranges:
                ground_truth[start:end] = 1

        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Isolation Forest":
                clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                clf.fit(train_scaled)
                scores = -clf.decision_function(test_scaled)
                preds = (clf.predict(test_scaled) == -1).astype(int)
            else:
                ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation='relu', max_iter=100, random_state=42)
                ae.fit(train_scaled, train_scaled)
                train_pred = ae.predict(train_scaled)
                test_pred = ae.predict(test_scaled)
                train_mse = np.mean(np.power(train_scaled - train_pred, 2), axis=1)
                scores = np.mean(np.power(test_scaled - test_pred, 2), axis=1)
                threshold = np.percentile(train_mse, 95)
                preds = (scores > threshold).astype(int)

        f1 = f1_score(ground_truth, preds, zero_division=0)
        precision = precision_score(ground_truth, preds, zero_division=0)
        recall = recall_score(ground_truth, preds, zero_division=0)
        fpr, tpr, _ = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)
        prec_curve, rec_curve, _ = precision_recall_curve(ground_truth, scores)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 Score", f"{f1:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("AUC", f"{roc_auc:.4f}")

        st.subheader("Signal with Anomaly Regions")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(test_scaled[:, 0], label="Signal (scaled)", alpha=0.7)
        ax.fill_between(range(len(ground_truth)), -3, 3,
                        where=ground_truth.astype(bool),
                        alpha=0.3, color='orange', label="True Anomaly")
        ax.fill_between(range(len(preds)), -3, 3,
                        where=preds.astype(bool),
                        alpha=0.3, color='red', label="Predicted Anomaly")
        ax.legend()
        ax.set_title(f"Channel {channel} - {model_choice}")
        st.pyplot(fig)

        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.subheader("ROC Curve")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.4f}")
            ax2.plot([0, 1], [0, 1], 'navy', lw=1, linestyle='--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)

        with col_pr:
            st.subheader("Precision-Recall Curve")
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.plot(rec_curve, prec_curve, color='green', lw=2)
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve")
            st.pyplot(fig3)

        st.success("Done!")

# ── TAB 2: Dataset Stats ──
with tab2:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Channels", len(all_results))
    col2.metric("SMAP Channels", len(all_results[all_results['spacecraft'] == 'SMAP']))
    col3.metric("MSL Channels", len(all_results[all_results['spacecraft'] == 'MSL']))
    col4.metric("Avg Anomaly %", f"{all_results['anomaly_pct'].mean():.2f}%")

    st.subheader("Anomaly % per Channel")
    fig4, ax4 = plt.subplots(figsize=(14, 4))
    ax4.bar(all_results['channel'], all_results['anomaly_pct'], color='steelblue')
    ax4.set_xlabel("Channel")
    ax4.set_ylabel("Anomaly %")
    ax4.set_title("Anomaly Percentage per Channel")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("Full Channel Stats")
    st.dataframe(all_results, use_container_width=True)

# ── TAB 3: Model Comparison ──
with tab3:
    st.subheader("Model Performance Across All Channels")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg IF F1", f"{all_results['if_f1'].mean():.4f}")
    col2.metric("Avg AE F1", f"{all_results['ae_f1'].mean():.4f}")
    col3.metric("Avg IF AUC", f"{all_results['if_auc'].mean():.4f}")
    col4.metric("Avg AE AUC", f"{all_results['ae_auc'].mean():.4f}")

    st.subheader("F1 Score Comparison per Channel")
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    x = range(len(all_results))
    ax5.bar([i - 0.2 for i in x], all_results['if_f1'], width=0.4, label='Isolation Forest', color='steelblue')
    ax5.bar([i + 0.2 for i in x], all_results['ae_f1'], width=0.4, label='Autoencoder', color='darkorange')
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(all_results['channel'], rotation=90, fontsize=7)
    ax5.set_ylabel("F1 Score")
    ax5.set_title("Isolation Forest vs Autoencoder - F1 Score per Channel")
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)

    st.subheader("Top 10 Channels by Autoencoder F1")
    st.dataframe(all_results.nlargest(10, 'ae_f1')[['channel', 'spacecraft', 'ae_f1', 'ae_auc', 'if_f1', 'if_auc']], use_container_width=True)