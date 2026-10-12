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

VAL_CSV = (
    PROJECT_ROOT
    / "processed"
    / "classifier_v3_val.csv"
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
# SETTINGS
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

BATCH_SIZE = 8
NUM_WORKERS = 0
DEFAULT_THRESHOLD = 0.50


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
# DATA LOADER
# ============================================================

dataset = XRayDataset(
    VAL_CSV
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


# ============================================================
# MODEL
# ============================================================

model = models.densenet121(
    weights=None
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(CLASSES)
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    device
)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

print("=" * 75)
print("CLASSIFIER V3 VALIDATION EVALUATION")
print("=" * 75)

print(
    "Device:",
    device
)

print(
    "Validation images:",
    len(dataset)
)

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch"
    )
)

print(
    "Checkpoint validation loss:",
    checkpoint.get(
        "validation_loss"
    )
)

print()
print(
    "Running V3 validation inference..."
)


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

        images = images.to(
            device
        )

        logits = model(
            images
        )

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
    y_prob
    >=
    DEFAULT_THRESHOLD
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


try:

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        average="macro"
    )

except ValueError:

    macro_auc = float("nan")


print()
print("=" * 75)
print("V3 OVERALL VALIDATION RESULTS")
print("=" * 75)

print(
    f"Threshold:            "
    f"{DEFAULT_THRESHOLD:.2f}"
)

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
    f"ROC-AUC:   {macro_auc:.4f}"
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

rows = []

print()
print("=" * 95)
print("V3 PER-CLASS VALIDATION RESULTS")
print("=" * 95)


for index, class_name in enumerate(
    CLASSES
):

    true = y_true[
        :,
        index
    ]

    pred = y_pred[
        :,
        index
    ]

    prob = y_prob[
        :,
        index
    ]


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


    try:

        auc = roc_auc_score(
            true,
            prob
        )

    except ValueError:

        auc = float("nan")


    tn, fp, fn, tp = confusion_matrix(
        true,
        pred,
        labels=[
            0,
            1
        ]
    ).ravel()


    rows.append({

        "class":
            class_name,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            auc,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)
    })


results_df = pd.DataFrame(
    rows
)


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
        float_format=lambda value:
            f"{value:.4f}"
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print()
print("=" * 75)
print("CONFUSION MATRICES")
print("=" * 75)


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
# SAVE
# ============================================================

metrics_path = (
    RESULT_DIR
    / "classifier_v3_validation_metrics.csv"
)

probability_path = (
    RESULT_DIR
    / "classifier_v3_val_probabilities.npy"
)

label_path = (
    RESULT_DIR
    / "classifier_v3_val_labels.npy"
)


results_df.to_csv(
    metrics_path,
    index=False
)

np.save(
    probability_path,
    y_prob
)

np.save(
    label_path,
    y_true
)


print()
print("=" * 75)
print("V3 VALIDATION EVALUATION COMPLETE")
print("=" * 75)

print(
    "Metrics:",
    metrics_path
)

print(
    "Probabilities:",
    probability_path
)

print(
    "Labels:",
    label_path
)

print()
print(
    "DO NOT evaluate the V3 test set yet."
)

print(
    "Next step: tune thresholds using these "
    "validation probabilities."
)