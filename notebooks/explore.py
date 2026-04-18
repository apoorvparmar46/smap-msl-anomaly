import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = r"C:\Users\Acer\smap-msl-anomaly\data\data\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = r"C:\Users\Acer\smap-msl-anomaly\data\labeled_anomalies.csv"

# Load labels
labels = pd.read_csv(LABELS_PATH)
print("Labels shape:", labels.shape)
print(labels.head(10))

# Load one channel
channel = "A-1"
train_data = np.load(os.path.join(TRAIN_DIR, f"{channel}.npy"))
test_data = np.load(os.path.join(TEST_DIR, f"{channel}.npy"))

print(f"\nChannel: {channel}")
print(f"Train shape: {train_data.shape}")
print(f"Test shape:  {test_data.shape}")

# Plot first feature of train
plt.figure(figsize=(14, 4))
plt.plot(train_data[:, 0], label="Train", alpha=0.7)
plt.title(f"Channel {channel} - Feature 0 (Train)")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\train_signal.png")
plt.show()
print("Plot saved.")