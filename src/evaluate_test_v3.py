from pathlib import Path

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = (
    PROJECT_ROOT
    / "processed"
    / "classifier_v3_test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_classifier_v3.pth"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "v3"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASSES
# ============================================================

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


# ============================================================
# LOCKED VALIDATION-TUNED THRESHOLDS
# ============================================================

THRESHOLDS = np.array([
    0.666,  # atelectasis
    0.582,  # cardiomegaly
    0.380,  # emphysema
    0.540,  # opacity
    0.202,  # pneumothorax
    0.723,  # pleural_effusion
    0.381,  # nodule
    0.577,  # scarring
])


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(
            csv_path
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(
            row["image_path"]
        )

        if not image_path.is_absolute():

            image_path = (
                PROJECT_ROOT
                /
                image_path
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        )

        labels = torch.tensor(
            row[
                CLASSES
            ].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# DATA
# ============================================================

dataset = XRayDataset(
    TEST_CSV
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = models.densenet121(
    weights=None
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(CLASSES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# INFERENCE
# ============================================================

print("=" * 80)
print("FINAL CLASSIFIER V3 TEST EVALUATION")
print("=" * 80)

print("Device:", device)
print("Test images:", len(dataset))
print("Checkpoint epoch:", checkpoint.get("epoch"))

print()
print("LOCKED THRESHOLDS")

for name, threshold in zip(
    CLASSES,
    THRESHOLDS
):

    print(
        f"{name:20s}: "
        f"{threshold:.3f}"
    )


print()
print("Running untouched test-set inference...")


all_labels = []
all_probabilities = []


with torch.no_grad():

    for batch_number, (
        images,
        labels
    ) in enumerate(
        loader,
        start=1
    ):

        images = images.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.append(
            labels.numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        if (
            batch_number % 20 == 0
            or
            batch_number == len(loader)
        ):

            print(
                f"Processed "
                f"{batch_number}/"
                f"{len(loader)} batches"
            )


# ============================================================
# ARRAYS
# ============================================================

y_true = np.concatenate(
    all_labels,
    axis=0
).astype(int)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
)

y_pred = (
    y_prob >= THRESHOLDS
).astype(int)


# ============================================================
# OVERALL METRICS
# ============================================================

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


# ============================================================
# MACRO ROC-AUC
# ============================================================

auc_values = []

for i in range(
    len(CLASSES)
):

    if len(
        np.unique(
            y_true[:, i]
        )
    ) == 2:

        auc_values.append(
            roc_auc_score(
                y_true[:, i],
                y_prob[:, i]
            )
        )


macro_auc = (
    float(np.mean(auc_values))
    if auc_values
    else float("nan")
)


# ============================================================
# DISPLAY OVERALL
# ============================================================

print()
print("=" * 80)
print("FINAL V3 TEST RESULTS")
print("=" * 80)

print(
    f"Exact-match Accuracy: "
    f"{exact_accuracy:.4f}"
)

print()

print("MICRO AVERAGES")

print(
    f"Precision: {micro_precision:.4f}"
)

print(
    f"Recall:    {micro_recall:.4f}"
)

print(
    f"F1-score:  {micro_f1:.4f}"
)

print()

print("MACRO AVERAGES")

print(
    f"Precision: {macro_precision:.4f}"
)

print(
    f"Recall:    {macro_recall:.4f}"
)

print(
    f"F1-score:  {macro_f1:.4f}"
)

print(
    f"ROC-AUC*:  {macro_auc:.4f}"
)

print(
    "* Macro AUC averages classes with "
    "both positive and negative test examples."
)


# ============================================================
# PER CLASS
# ============================================================

rows = []

print()
print("=" * 100)
print("FINAL V3 TEST PER-CLASS RESULTS")
print("=" * 100)


for i, class_name in enumerate(
    CLASSES
):

    true = y_true[:, i]
    pred = y_pred[:, i]
    prob = y_prob[:, i]

    accuracy = accuracy_score(
        true,
        pred
    )

    precision = precision_score(
        true,
        pred,
        zero_division=0
    )

    recall = recall_score(
        true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        true,
        pred,
        zero_division=0
    )

    if len(
        np.unique(true)
    ) == 2:

        auc = roc_auc_score(
            true,
            prob
        )

    else:

        auc = float("nan")


    tn, fp, fn, tp = confusion_matrix(
        true,
        pred,
        labels=[0, 1]
    ).ravel()


    rows.append({
        "class": class_name,
        "threshold": THRESHOLDS[i],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    })


results_df = pd.DataFrame(
    rows
)


print(
    results_df[
        [
            "class",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}"
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print()
print("=" * 80)
print("FINAL TEST CONFUSION MATRICES")
print("=" * 80)


for row in rows:

    print()

    print(
        row["class"].upper()
    )

    print(
        f"TN={row['tn']}  "
        f"FP={row['fp']}  "
        f"FN={row['fn']}  "
        f"TP={row['tp']}"
    )


# ============================================================
# NORMAL / ABNORMAL IMAGE LEVEL
# ============================================================

true_abnormal = (
    y_true.sum(axis=1) > 0
)

pred_abnormal = (
    y_pred.sum(axis=1) > 0
)

normal_mask = ~true_abnormal


normal_total = int(
    normal_mask.sum()
)

normal_correct = int(
    (
        normal_mask
        &
        (~pred_abnormal)
    ).sum()
)

normal_false_positive = (
    normal_total
    -
    normal_correct
)


abnormal_total = int(
    true_abnormal.sum()
)

abnormal_detected = int(
    (
        true_abnormal
        &
        pred_abnormal
    ).sum()
)


normal_specificity = (
    normal_correct
    /
    normal_total
    if normal_total
    else 0
)


abnormal_detection_rate = (
    abnormal_detected
    /
    abnormal_total
    if abnormal_total
    else 0
)


print()
print("=" * 80)
print("IMAGE-LEVEL NORMAL / ABNORMAL SUMMARY")
print("=" * 80)

print(
    "Normal test images:",
    normal_total
)

print(
    "Normal correctly not flagged:",
    normal_correct
)

print(
    "Normal false-positive images:",
    normal_false_positive
)

print(
    f"Normal-image specificity: "
    f"{normal_specificity:.4f}"
)

print()

print(
    "Abnormal test images:",
    abnormal_total
)

print(
    "Abnormal images with >=1 predicted finding:",
    abnormal_detected
)

print(
    f"Abnormal-image detection rate: "
    f"{abnormal_detection_rate:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_path = (
    RESULT_DIR
    / "classifier_v3_test_metrics.csv"
)

prob_path = (
    RESULT_DIR
    / "classifier_v3_test_probabilities.npy"
)

label_path = (
    RESULT_DIR
    / "classifier_v3_test_labels.npy"
)

prediction_path = (
    RESULT_DIR
    / "classifier_v3_test_predictions.npy"
)


results_df.to_csv(
    metrics_path,
    index=False
)

np.save(
    prob_path,
    y_prob
)

np.save(
    label_path,
    y_true
)

np.save(
    prediction_path,
    y_pred
)


print()
print("=" * 80)
print("FINAL V3 TEST EVALUATION COMPLETE")
print("=" * 80)

print(
    "Metrics:",
    metrics_path
)

print(
    "Probabilities:",
    prob_path
)

print(
    "Labels:",
    label_path
)

print(
    "Predictions:",
    prediction_path
)

print()
print(
    "IMPORTANT: These are now final held-out "
    "test results. Do not retune thresholds using "
    "this test set."
)