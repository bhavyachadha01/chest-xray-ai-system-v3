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
# SETTINGS
# ============================================================

BATCH_SIZE = 8

# Start with 3 epochs because we are partially fine-tuning
# DenseNet on CPU.
NUM_EPOCHS = 3

CLASSIFIER_LR = 0.0002
FEATURE_LR = 0.00002

NUM_WORKERS = 0

torch.manual_seed(42)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Training augmentation:
# keep it mild for chest radiographs.

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


# Validation must NOT use random augmentation.

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
        transform
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.transform = transform


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
        ).convert(
            "RGB"
        )


        image = self.transform(
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
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
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


# ============================================================
# POSITIVE CLASS WEIGHTS
# ============================================================

pos_weight = (
    negative_counts
    /
    positive_counts.clamp(
        min=1
    )
)


# Rare classes can otherwise receive extremely large weights.
# Keep the same general protection used in V2.

pos_weight = pos_weight.clamp(
    max=30
)


print()
print(
    "Positive counts:"
)


for name, count in zip(
    CLASSES,
    positive_counts.tolist()
):

    print(
        f"{name:20s}: "
        f"{int(count)}"
    )


print()
print(
    "Positive class weights:"
)


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
# MODEL
# ============================================================

print()
print(
    "Loading pretrained DenseNet121..."
)


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
# FREEZE FEATURE EXTRACTOR
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# PARTIAL FINE-TUNING
# ============================================================

# Unfreeze only the final DenseNet block.
# This lets the visual encoder adapt to chest X-rays
# without making CPU training as expensive as full fine-tuning.

for parameter in (
    model.features.denseblock4.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.features.norm5.parameters()
):

    parameter.requires_grad = True


# Classifier must be trainable.

for parameter in model.classifier.parameters():

    parameter.requires_grad = True


model = model.to(
    device
)


# ============================================================
# PRINT TRAINABLE PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
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

# Smaller learning rate for pretrained DenseNet features.
# Larger learning rate for the newly initialized classifier.

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


        images = images.to(
            device
        )


        labels = labels.to(
            device
        )


        if training:

            optimizer.zero_grad()


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


            if training:

                loss.backward()


                # Helps prevent unusually large gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )


                optimizer.step()


        total_loss += (
            loss.item()
        )


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


    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# MAIN TRAINING
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
    print(
        "=" * 70
    )

    print(
        "STARTING CLASSIFIER V3 TRAINING"
    )

    print(
        "=" * 70
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):


        start_time = time.time()


        print()
        print(
            "=" * 70
        )

        print(
            f"EPOCH "
            f"{epoch}/"
            f"{NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # TRAIN
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

        if (
            val_loss
            <
            best_val_loss
        ):


            best_val_loss = (
                val_loss
            )


            torch.save({

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

                "classifier_lr":
                    CLASSIFIER_LR,

                "feature_lr":
                    FEATURE_LR,

                "architecture":
                    "DenseNet121",

                "fine_tuned_blocks":
                    [
                        "denseblock4",
                        "norm5",
                        "classifier"
                    ],

                "training_csv":
                    str(TRAIN_CSV),

                "validation_csv":
                    str(VAL_CSV)

            }, MODEL_PATH)


            print()
            print(
                "BEST V3 MODEL SAVED:"
            )

            print(
                MODEL_PATH
            )


    print()
    print(
        "=" * 70
    )

    print(
        "V3 TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Saved model:",
        MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()