import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"

labels = pd.read_csv(LABELS_PATH)
all_channels = sorted(labels['chan_id'].tolist())

results = []

for i, channel in enumerate(all_channels):
    print(f"[{i+1}/{len(all_channels)}] Processing {channel}...")

    train_path = os.path.join(TRAIN_DIR, f"{channel}.npy")
    test_path = os.path.join(TEST_DIR, f"{channel}.npy")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"  Skipping {channel} - file not found")
        continue

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Ground truth
    chan_row = labels[labels['chan_id'] == channel].iloc[0]
    anomaly_ranges = eval(chan_row['anomaly_sequences'])
    ground_truth = np.zeros(test_data.shape[0], dtype=int)
    for start, end in anomaly_ranges:
        ground_truth[start:end] = 1

    anomaly_pct = ground_truth.sum() / len(ground_truth) * 100

    # Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(train_scaled)
    if_scores = -clf.decision_function(test_scaled)
    if_preds = (clf.predict(test_scaled) == -1).astype(int)
    if_f1 = f1_score(ground_truth, if_preds, zero_division=0)
    if_auc = auc(*roc_curve(ground_truth, if_scores)[:2])

    # Autoencoder
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation='relu', max_iter=100, random_state=42)
    ae.fit(train_scaled, train_scaled)
    train_pred = ae.predict(train_scaled)
    test_pred = ae.predict(test_scaled)
    train_mse = np.mean(np.power(train_scaled - train_pred, 2), axis=1)
    ae_scores = np.mean(np.power(test_scaled - test_pred, 2), axis=1)
    threshold = np.percentile(train_mse, 95)
    ae_preds = (ae_scores > threshold).astype(int)
    ae_f1 = f1_score(ground_truth, ae_preds, zero_division=0)
    ae_auc = auc(*roc_curve(ground_truth, ae_scores)[:2])

    results.append({
        "channel": channel,
        "spacecraft": chan_row['spacecraft'],
        "train_len": train_data.shape[0],
        "test_len": test_data.shape[0],
        "n_features": train_data.shape[1],
        "anomaly_pct": round(anomaly_pct, 2),
        "if_f1": round(if_f1, 4),
        "if_auc": round(if_auc, 4),
        "ae_f1": round(ae_f1, 4),
        "ae_auc": round(ae_auc, 4),
    })

    print(f"  IF F1: {if_f1:.4f} | AE F1: {ae_f1:.4f}")

df = pd.DataFrame(results)
df.to_csv(r"C:\Users\Acer\smap-msl-anomaly\notebooks\all_channels_results.csv", index=False)

print("\n=== SUMMARY ===")
print(f"Total channels processed: {len(df)}")
print(f"Avg IF  F1: {df['if_f1'].mean():.4f} | Avg IF  AUC: {df['if_auc'].mean():.4f}")
print(f"Avg AE  F1: {df['ae_f1'].mean():.4f} | Avg AE  AUC: {df['ae_auc'].mean():.4f}")
print("\nTop 5 channels by AE F1:")
print(df.nlargest(5, 'ae_f1')[['channel', 'ae_f1', 'ae_auc']])
print("\nResults saved to all_channels_results.csv")