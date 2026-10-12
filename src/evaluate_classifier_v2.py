import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================
# SETTINGS
# ==========================================

VAL_CSV = "processed/classifier_v2_val.csv"
MODEL_PATH = "models/best_classifier_v2.pth"

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

BATCH_SIZE = 8
THRESHOLD = 0.5
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# IMAGE PREPROCESSING
# MUST MATCH TRAINING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

class XRayDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = transform(image)

        labels = torch.tensor(
            row[CLASSES].to_numpy(dtype="float32"),
            dtype=torch.float32
        )

        return image, labels


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

dataset = XRayDataset(VAL_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Device:", device)
print("Validation images:", len(dataset))


# ==========================================
# CREATE MODEL
# ==========================================

# weights=None because we are loading our
# own trained checkpoint immediately afterward.
model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(CLASSES)
)


# ==========================================
# LOAD TRAINED V2 CHECKPOINT
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("\nLoaded model:", MODEL_PATH)
print(
    "Saved epoch:",
    checkpoint.get("epoch", "unknown")
)
print(
    "Saved validation loss:",
    checkpoint.get("validation_loss", "unknown")
)


# ==========================================
# RUN PREDICTIONS
# ==========================================

all_labels = []
all_probabilities = []

print("\nRunning evaluation...")

with torch.no_grad():

    for batch_number, (images, labels) in enumerate(
        loader,
        start=1
    ):

        images = images.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        all_labels.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        if batch_number % 25 == 0:
            print(
                f"Processed {batch_number}/{len(loader)} batches"
            )


y_true = np.concatenate(
    all_labels,
    axis=0
).astype(int)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
)

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ==========================================
# OVERALL METRICS
# ==========================================

# Exact-match accuracy:
# An X-ray counts as correct only if ALL 8
# labels exactly match.
exact_accuracy = accuracy_score(
    y_true,
    y_pred
)

micro_precision = precision_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

try:
    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        average="macro"
    )
except ValueError:
    macro_auc = np.nan


print("\n" + "=" * 70)
print("OVERALL CLASSIFIER V2 RESULTS")
print("=" * 70)

print(f"Threshold:            {THRESHOLD:.2f}")
print(f"Exact-match Accuracy: {exact_accuracy:.4f}")

print("\nMICRO AVERAGES")
print(f"Precision:             {micro_precision:.4f}")
print(f"Recall:                {micro_recall:.4f}")
print(f"F1-score:              {micro_f1:.4f}")

print("\nMACRO AVERAGES")
print(f"Precision:             {macro_precision:.4f}")
print(f"Recall:                {macro_recall:.4f}")
print(f"F1-score:              {macro_f1:.4f}")

if np.isnan(macro_auc):
    print("ROC-AUC:               N/A")
else:
    print(f"ROC-AUC:               {macro_auc:.4f}")


# ==========================================
# PER-CLASS METRICS
# ==========================================

rows = []

print("\n" + "=" * 90)
print("PER-CLASS RESULTS")
print("=" * 90)

for i, class_name in enumerate(CLASSES):

    true_class = y_true[:, i]
    pred_class = y_pred[:, i]
    prob_class = y_prob[:, i]

    accuracy = accuracy_score(
        true_class,
        pred_class
    )

    precision = precision_score(
        true_class,
        pred_class,
        zero_division=0
    )

    recall = recall_score(
        true_class,
        pred_class,
        zero_division=0
    )

    f1 = f1_score(
        true_class,
        pred_class,
        zero_division=0
    )

    # AUC requires both positive and negative
    # examples to be present.
    if len(np.unique(true_class)) == 2:
        auc = roc_auc_score(
            true_class,
            prob_class
        )
    else:
        auc = np.nan

    cm = confusion_matrix(
        true_class,
        pred_class,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    rows.append({
        "class": class_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })


results_df = pd.DataFrame(rows)

print(
    results_df[
        [
            "class",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================
# CONFUSION MATRICES
# ==========================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for _, row in results_df.iterrows():

    print(
        f"\n{row['class'].upper()}"
    )

    print(
        f"TN={int(row['tn'])}  "
        f"FP={int(row['fp'])}  "
        f"FN={int(row['fn'])}  "
        f"TP={int(row['tp'])}"
    )


# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/classifier_v2_metrics.csv",
    index=False
)

np.save(
    "results/classifier_v2_probabilities.npy",
    y_prob
)

np.save(
    "results/classifier_v2_labels.npy",
    y_true
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "Metrics saved to: "
    "results/classifier_v2_metrics.csv"
)

print(
    "Probabilities saved for threshold tuning."
)
