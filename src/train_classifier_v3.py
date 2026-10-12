import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    PROJECT_ROOT
    / "processed"
    / "classifier_v3_train.csv"
)

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
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 8

# Three epochs first.
# Partial fine-tuning on CPU is slower than V2.
NUM_EPOCHS = 3

# New classifier head gets the larger learning rate.
CLASSIFIER_LR = 0.0002

# Pretrained DenseNet features get a much smaller LR.
FEATURE_LR = 0.00002

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("CHEST X-RAY CLASSIFIER V3")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )


if not VAL_CSV.exists():

    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
#
# Mild augmentation only.
# We do NOT use horizontal flipping because left/right
# orientation can be clinically meaningful.

train_transform = transforms.Compose([

    transforms.Resize(
        (240, 240)
    ),

    transforms.RandomCrop(
        (224, 224)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
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


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
#
# No random augmentation during validation.

val_transform = transforms.Compose([

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

    def __init__(
        self,
        csv_path,
        image_transform
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.image_transform = (
            image_transform
        )


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[
            idx
        ]


        # ----------------------------------------------------
        # IMAGE PATH
        # ----------------------------------------------------

        image_path = Path(
            row["image_path"]
        )


        if not image_path.is_absolute():

            image_path = (
                PROJECT_ROOT
                /
                image_path
            )


        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        image = self.image_transform(
            image
        )


        # ----------------------------------------------------
        # LABEL VECTOR
        # ----------------------------------------------------

        labels = torch.tensor(
            row[
                CLASSES
            ].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32
        )


        return (
            image,
            labels
        )


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = XRayDataset(
    TRAIN_CSV,
    train_transform
)

val_dataset = XRayDataset(
    VAL_CSV,
    val_transform
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_df = train_dataset.df


positive_counts = torch.tensor(
    train_df[
        CLASSES
    ].sum().values,
    dtype=torch.float32
)


negative_counts = (
    len(train_df)
    -
    positive_counts
)


print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)


for name, count in zip(
    CLASSES,
    positive_counts.tolist()
):

    print(
        f"{name:20s}: "
        f"{int(count)}"
    )


# ============================================================
# CLASS IMBALANCE HANDLING
# ============================================================

pos_weight = (
    negative_counts
    /
    positive_counts.clamp(
        min=1
    )
)


# Prevent extremely rare classes from producing
# unreasonably large loss weights.
pos_weight = pos_weight.clamp(
    max=30.0
)


print()
print("=" * 70)
print("POSITIVE CLASS WEIGHTS")
print("=" * 70)


for name, weight in zip(
    CLASSES,
    pos_weight.tolist()
):

    print(
        f"{name:20s}: "
        f"{weight:.2f}"
    )


pos_weight = pos_weight.to(
    device
)


# ============================================================
# LOAD PRETRAINED DENSENET121
# ============================================================

print()
print("=" * 70)
print("LOADING DENSENET121")
print("=" * 70)


model = models.densenet121(
    weights=models.DenseNet121_Weights.DEFAULT
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

input_features = (
    model.classifier.in_features
)


model.classifier = nn.Linear(
    input_features,
    len(CLASSES)
)


# ============================================================
# FREEZE MOST OF DENSENET
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# UNFREEZE FINAL DENSENET BLOCK
# ============================================================
#
# V2 trained only the final classifier head.
#
# V3 allows the last DenseNet block to adapt to chest X-rays.

for parameter in (
    model.features.denseblock4.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.features.norm5.parameters()
):

    parameter.requires_grad = True


# Ensure classifier is trainable.

for parameter in (
    model.classifier.parameters()
):

    parameter.requires_grad = True


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    device
)


# ============================================================
# PARAMETER COUNTS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# OPTIMIZER
# ============================================================
#
# Use separate learning rates:
#
# DenseNet block = small LR
# classifier head = larger LR

optimizer = torch.optim.Adam([

    {
        "params":
            model.features.denseblock4.parameters(),

        "lr":
            FEATURE_LR
    },

    {
        "params":
            model.features.norm5.parameters(),

        "lr":
            FEATURE_LR
    },

    {
        "params":
            model.classifier.parameters(),

        "lr":
            CLASSIFIER_LR
    }

])


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    loader,
    training=True
):

    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0


    for batch_number, (
        images,
        labels
    ) in enumerate(
        loader,
        start=1
    ):


        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        if training:

            optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        with torch.set_grad_enabled(
            training
        ):


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            # ------------------------------------------------
            # BACKPROP
            # ------------------------------------------------

            if training:

                loss.backward()


                # Gradient clipping for stability.

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )


                optimizer.step()


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        total_loss += (
            loss.item()
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            training
            and
            batch_number % 25 == 0
        ):

            print(
                f"Batch "
                f"{batch_number}/"
                f"{len(loader)} "
                f"| Loss: "
                f"{loss.item():.4f}",
                flush=True
            )


    average_loss = (
        total_loss
        /
        len(loader)
    )


    return average_loss


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        PROJECT_ROOT / "models",
        exist_ok=True
    )


    best_val_loss = float(
        "inf"
    )


    print()
    print("=" * 70)
    print("STARTING CLASSIFIER V3 TRAINING")
    print("=" * 70)

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Classifier learning rate:",
        CLASSIFIER_LR
    )

    print(
        "Feature learning rate:",
        FEATURE_LR
    )


    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):


        start_time = time.time()


        print()
        print("=" * 70)

        print(
            f"EPOCH "
            f"{epoch}/"
            f"{NUM_EPOCHS}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        train_loss = run_epoch(
            train_loader,
            training=True
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss = run_epoch(
            val_loader,
            training=False
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        elapsed_minutes = (
            time.time()
            -
            start_time
        ) / 60


        print()

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Epoch Time: "
            f"{elapsed_minutes:.1f} minutes"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:


            best_val_loss = (
                val_loss
            )


            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "classes":
                    CLASSES,

                "num_classes":
                    len(CLASSES),

                "validation_loss":
                    val_loss,

                "train_loss":
                    train_loss,

                "pos_weight":
                    pos_weight.detach().cpu(),

                "architecture":
                    "DenseNet121",

                "classifier_lr":
                    CLASSIFIER_LR,

                "feature_lr":
                    FEATURE_LR,

                "batch_size":
                    BATCH_SIZE,

                "training_csv":
                    str(TRAIN_CSV),

                "validation_csv":
                    str(VAL_CSV),

                "fine_tuned_blocks": [
                    "denseblock4",
                    "norm5",
                    "classifier"
                ]
            }


            torch.save(
                checkpoint,
                MODEL_PATH
            )


            print()

            print(
                "BEST V3 MODEL SAVED:"
            )

            print(
                MODEL_PATH
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("V3 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
    