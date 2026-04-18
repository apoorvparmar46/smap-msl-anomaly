import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"

labels = pd.read_csv(LABELS_PATH)
channel = "D-1"

train_data = np.load(os.path.join(TRAIN_DIR, f"{channel}.npy"))
test_data = np.load(os.path.join(TEST_DIR, f"{channel}.npy"))

# Normalize
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Autoencoder using MLPRegressor (encode then decode)
print("Training autoencoder...")
autoencoder = MLPRegressor(
    hidden_layer_sizes=(16, 8, 16),
    activation='relu',
    max_iter=100,
    random_state=42,
    verbose=True
)
autoencoder.fit(train_scaled, train_scaled)
print("Training done.")

# Reconstruction error
train_pred = autoencoder.predict(train_scaled)
test_pred = autoencoder.predict(test_scaled)
train_mse = np.mean(np.power(train_scaled - train_pred, 2), axis=1)
test_mse = np.mean(np.power(test_scaled - test_pred, 2), axis=1)

# Threshold = 95th percentile of train error
threshold = np.percentile(train_mse, 95)
print(f"Threshold: {threshold:.6f}")

preds = (test_mse > threshold).astype(int)

# Ground truth
chan_row = labels[labels['chan_id'] == channel].iloc[0]
anomaly_ranges = eval(chan_row['anomaly_sequences'])
ground_truth = np.zeros(test_data.shape[0], dtype=int)
for start, end in anomaly_ranges:
    ground_truth[start:end] = 1

# Metrics
f1 = f1_score(ground_truth, preds, zero_division=0)
precision = precision_score(ground_truth, preds, zero_division=0)
recall = recall_score(ground_truth, preds, zero_division=0)
fpr, tpr, _ = roc_curve(ground_truth, test_mse)
roc_auc = auc(fpr, tpr)

print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {roc_auc:.4f}")

# Plot
plt.figure(figsize=(14, 4))
plt.plot(test_mse, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
plt.fill_between(range(len(ground_truth)), 0, test_mse.max(),
                 where=ground_truth.astype(bool),
                 alpha=0.3, color='orange', label="True Anomaly")
plt.title(f"Autoencoder Reconstruction Error - Channel {channel}")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\autoencoder_result.png")
plt.show()
print("Done.")