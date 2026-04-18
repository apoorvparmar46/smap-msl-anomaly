import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
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

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data).astype(np.float32)
test_scaled = scaler.transform(test_data).astype(np.float32)

# Sliding window
WINDOW = 50

def make_windows(data, window):
    return np.array([data[i:i+window] for i in range(len(data) - window)])

train_windows = make_windows(train_scaled, WINDOW)
test_windows = make_windows(test_scaled, WINDOW)

# Flatten windows for MLP
train_flat = train_windows.reshape(len(train_windows), -1)
test_flat = test_windows.reshape(len(test_windows), -1)

print(f"Train windows: {train_flat.shape}")
print(f"Test windows:  {test_flat.shape}")

# Simple numpy autoencoder (manually trained with gradient descent)
np.random.seed(42)
input_dim = train_flat.shape[1]
hidden_dim = 64
latent_dim = 16
lr = 0.001
epochs = 30
batch_size = 128

# Initialize weights
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, latent_dim) * 0.01
b2 = np.zeros(latent_dim)
W3 = np.random.randn(latent_dim, hidden_dim) * 0.01
b3 = np.zeros(hidden_dim)
W4 = np.random.randn(hidden_dim, input_dim) * 0.01
b4 = np.zeros(input_dim)

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

def forward(X):
    h1 = relu(X @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)
    h3 = relu(h2 @ W3 + b3)
    out = h3 @ W4 + b4
    return out, h1, h2, h3

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

print("Training sliding window autoencoder...")
for epoch in range(epochs):
    idx = np.random.permutation(len(train_flat))
    total_loss = 0
    for i in range(0, len(train_flat), batch_size):
        batch = train_flat[idx[i:i+batch_size]]
        out, h1, h2, h3 = forward(batch)
        loss = mse_loss(out, batch)
        total_loss += loss

        # Backprop
        dout = 2 * (out - batch) / batch.shape[0]
        dW4 = h3.T @ dout
        db4 = dout.sum(axis=0)
        dh3 = dout @ W4.T * relu_grad(h3)
        dW3 = h2.T @ dh3
        db3 = dh3.sum(axis=0)
        dh2 = dh3 @ W3.T * relu_grad(h2)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ W2.T * relu_grad(h1)
        dW1 = batch.T @ dh1
        db1 = dh1.sum(axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        W3 -= lr * dW3; b3 -= lr * db3
        W4 -= lr * dW4; b4 -= lr * db4

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Reconstruction error
train_pred, *_ = forward(train_flat)
test_pred, *_ = forward(test_flat)
train_mse = np.mean((train_flat - train_pred) ** 2, axis=1)
test_mse = np.mean((test_flat - test_pred) ** 2, axis=1)

threshold = np.percentile(train_mse, 95)
preds = (test_mse > threshold).astype(int)

# Ground truth
chan_row = labels[labels['chan_id'] == channel].iloc[0]
anomaly_ranges = eval(chan_row['anomaly_sequences'])
ground_truth_full = np.zeros(test_data.shape[0], dtype=int)
for start, end in anomaly_ranges:
    ground_truth_full[start:end] = 1
ground_truth = ground_truth_full[WINDOW:]

# Metrics
f1 = f1_score(ground_truth, preds, zero_division=0)
precision = precision_score(ground_truth, preds, zero_division=0)
recall = recall_score(ground_truth, preds, zero_division=0)
fpr, tpr, _ = roc_curve(ground_truth, test_mse)
roc_auc = auc(fpr, tpr)
print(f"\nF1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {roc_auc:.4f}")

# Plot
plt.figure(figsize=(14, 4))
plt.plot(test_mse, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
plt.fill_between(range(len(ground_truth)), 0, test_mse.max(),
                 where=ground_truth.astype(bool),
                 alpha=0.3, color='orange', label="True Anomaly")
plt.title(f"Sliding Window Autoencoder - Channel {channel}")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\lstm_result.png")
plt.show()
print("Done.")