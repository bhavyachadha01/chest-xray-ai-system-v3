import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

CLASSES = [
    "atelectasis",
    "cardiomegaly",
    "emphysema",
    "opacity",
    "pneumothorax",
    "pleural_effusion",
    "nodule",
    "scarring"
]

PROB_PATH = "results/classifier_v2_probabilities.npy"
LABEL_PATH = "results/classifier_v2_labels.npy"

y_prob = np.load(PROB_PATH)
y_true = np.load(LABEL_PATH)

results = []

print("\nTUNING THRESHOLDS")
print("=" * 75)

for i, class_name in enumerate(CLASSES):

    true = y_true[:, i]
    prob = y_prob[:, i]

    best_threshold = 0.50
    best_f1 = -1
    best_precision = 0
    best_recall = 0

    # Search thresholds from 0.10 to 0.90
    for threshold in np.arange(0.10, 0.91, 0.01):

        pred = (prob >= threshold).astype(int)

        precision = precision_score(
            true, pred, zero_division=0
        )

        recall = recall_score(
            true, pred, zero_division=0
        )

        f1 = f1_score(
            true, pred, zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    if len(np.unique(true)) == 2:
        auc = roc_auc_score(true, prob)
    else:
        auc = float("nan")

    results.append({
        "class": class_name,
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "roc_auc": auc
    })

    print(
        f"{class_name:20s} "
        f"Threshold={best_threshold:.2f} | "
        f"Precision={best_precision:.4f} | "
        f"Recall={best_recall:.4f} | "
        f"F1={best_f1:.4f} | "
        f"AUC={auc:.4f}"
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/classifier_v2_thresholds.csv",
    index=False
)

print("\n" + "=" * 75)
print("THRESHOLD TUNING COMPLETE")
print("=" * 75)

print(
    "\nSaved to: "
    "results/classifier_v2_thresholds.csv"
)