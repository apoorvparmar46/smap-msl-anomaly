import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"

labels = pd.read_csv(LABELS_PATH)

# Run on multiple channels
channels = ["A-1", "D-1", "P-1", "T-1", "E-1"]
results = []

for channel in channels:
    train_data = np.load(os.path.join(TRAIN_DIR, f"{channel}.npy"))
    test_data = np.load(os.path.join(TEST_DIR, f"{channel}.npy"))

    # Normalize
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Train Isolation Forest on clean training data
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(train_scaled)

    # Predict on test (-1 = anomaly, 1 = normal)
    preds_raw = clf.predict(test_scaled)
    preds = (preds_raw == -1).astype(int)

    # Ground truth
    chan_row = labels[labels['chan_id'] == channel]
    if chan_row.empty:
        continue
    anomaly_ranges = eval(chan_row.iloc[0]['anomaly_sequences'])
    ground_truth = np.zeros(test_data.shape[0], dtype=int)
    for start, end in anomaly_ranges:
        ground_truth[start:end] = 1

    f1 = f1_score(ground_truth, preds, zero_division=0)
    precision = precision_score(ground_truth, preds, zero_division=0)
    recall = recall_score(ground_truth, preds, zero_division=0)
    results.append({"channel": channel, "f1": f1, "precision": precision, "recall": recall})
    print(f"{channel} → F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(test_scaled[:, 0], label="Signal (scaled)", alpha=0.7)
    plt.fill_between(range(len(ground_truth)), 0, 1,
                     where=ground_truth.astype(bool),
                     transform=plt.gca().get_xaxis_transform(),
                     alpha=0.3, color='orange', label="True Anomaly")
    plt.fill_between(range(len(preds)), 0, 1,
                     where=preds.astype(bool),
                     transform=plt.gca().get_xaxis_transform(),
                     alpha=0.3, color='red', label="Predicted Anomaly")
    plt.title(f"Isolation Forest - Channel {channel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf"C:\Users\Acer\smap-msl-anomaly\notebooks\if_{channel}.png")
    plt.close()
    print(f"  Plot saved for {channel}")

# Summary table
print("\n--- Summary ---")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv(r"C:\Users\Acer\smap-msl-anomaly\notebooks\if_results.csv", index=False)
print("\nResults saved to if_results.csv")