import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

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

# Isolation Forest - get anomaly scores
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(train_scaled)
scores = -clf.decision_function(test_scaled)  # higher = more anomalous
preds = (clf.predict(test_scaled) == -1).astype(int)

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
print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(ground_truth, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Isolation Forest - Channel {channel}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\roc_curve.png")
plt.show()

# Confusion Matrix
cm = confusion_matrix(ground_truth, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - Channel {channel}")
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\smap-msl-anomaly\notebooks\confusion_matrix.png")
plt.show()
print("Plots saved.")