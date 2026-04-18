import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, f1_score

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"

# Load
labels = pd.read_csv(LABELS_PATH)
channel = "A-1"
train_data = np.load(os.path.join(TRAIN_DIR, f"{channel}.npy"))
test_data = np.load(os.path.join(TEST_DIR, f"{channel}.npy"))

# Use feature 0 only for baseline
train_feat = train_data[:, 0]
test_feat = test_data[:, 0]

# Threshold = mean + 3*std from training data
mean = np.mean(train_feat)
std = np.std(train_feat)
threshold = mean + 3 * std
print(f"Mean: {mean:.4f}, Std: {std:.4f}, Threshold: {threshold:.4f}")

# Predict anomalies
preds = (test_feat > threshold).astype(int)

# Build ground truth from labels
chan_label = labels[labels['chan_id'] == channel].iloc[0]
anomaly_ranges = eval(chan_label['anomaly_sequences'])
total_len = test_data.shape[0]
ground_truth = np.zeros(total_len, dtype=int)
for start, end in anomaly_ranges:
    ground_truth[start:end] = 1

print(f"\nTotal anomaly points: {ground_truth.sum()} / {total_len}")
print(f"Predicted anomalies:  {preds.sum()} / {total_len}")
print(f"\nF1 Score: {f1_score(ground_truth, preds):.4f}")
print("\nClassification Report:")
print(classification_report(ground_truth, preds))

# Plot
plt.figure(figsize=(14, 4))
plt.plot(test_feat, label="Signal", alpha=0.7)
plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
plt.fill_between(range(total_len), 0, 1,
                 where=ground_truth.astype(bool),
                 transform=plt.gca().get_xaxis_transform(),
                 alpha=0.3, color='orange', label="True Anomaly")
plt.title(f"Baseline Threshold Detection - Channel {channel}")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\baseline_result.png")
plt.show()
print("Done.")