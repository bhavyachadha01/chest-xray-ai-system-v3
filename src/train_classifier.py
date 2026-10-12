import os
import time

import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "processed/disease_labels.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_classifier.pth"
)

CLASSES = [
    "atelectasis",
    "cardiomegaly",
    "emphysema",
    "opacity",
    "pneumothorax",
    "pleural_effusion",
    "nodule",
    "scarring",
]

NUM_CLASSES = len(CLASSES)

# CPU-friendly settings
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_WORKERS = 0

VALIDATION_SPLIT = 0.15

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
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

class DiseaseDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.transform = transform

        for disease in CLASSES:

            if disease not in self.df.columns:

                raise ValueError(
                    f"Missing disease column: "
                    f"{disease}"
                )


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_path = str(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        labels = torch.tensor(
            [
                float(row[disease])
                for disease in CLASSES
            ],
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        "\nClassifier dataset not found:\n"
        f"{CSV_PATH}\n\n"
        "Run first:\n"
        "python src/prepare_classifier_data.py"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("DISEASE CLASSIFIER TRAINING")
print("=" * 60)

print(
    "\nDevice:",
    device
)

dataset = DiseaseDataset(
    CSV_PATH,
    transform=transform
)

print(
    "Total samples:",
    len(dataset)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

validation_size = int(
    len(dataset)
    *
    VALIDATION_SPLIT
)

training_size = (
    len(dataset)
    -
    validation_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(

    dataset,

    [
        training_size,
        validation_size
    ],

    generator=generator
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
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
# MODEL
# ============================================================

print(
    "\nCreating DenseNet121 classifier..."
)

model = models.densenet121(
    weights=models.DenseNet121_Weights.DEFAULT
)


# Freeze DenseNet convolutional features.
# This keeps CPU training much faster.

for parameter in model.features.parameters():

    parameter.requires_grad = False


# Replace original ImageNet classifier.

input_features = (
    model.classifier.in_features
)

model.classifier = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(
    device
)


# ============================================================
# PARAMETERS
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
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

# BCEWithLogitsLoss is appropriate because
# one X-ray can have MULTIPLE findings.

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    start_time = time.time()

    for batch_number, (
        images,
        labels
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        if batch_number % 50 == 0:

            elapsed = (
                time.time()
                -
                start_time
            ) / 60

            average_loss = (
                total_loss
                /
                batch_number
            )

            print(
                f"Batch "
                f"{batch_number}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f} "
                f"| Average: "
                f"{average_loss:.4f} "
                f"| Time: "
                f"{elapsed:.1f} min"
            )

    return (
        total_loss
        /
        len(train_loader)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
            )

    return (
        total_loss
        /
        len(val_loader)
    )


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING LOOP
# ============================================================

best_validation_loss = float(
    "inf"
)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print()
    print("=" * 60)

    print(
        f"EPOCH "
        f"{epoch}/"
        f"{NUM_EPOCHS}"
    )

    print("=" * 60)

    epoch_start = time.time()

    train_loss = train_epoch()

    print(
        "\nRunning validation..."
    )

    validation_loss = validate()

    epoch_minutes = (
        time.time()
        -
        epoch_start
    ) / 60

    print()
    print("-" * 60)

    print(
        f"EPOCH {epoch} RESULTS"
    )

    print("-" * 60)

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_minutes:.2f} minutes"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        validation_loss
        <
        best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )

        checkpoint = {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "classes":
                CLASSES,

            "num_classes":
                NUM_CLASSES,

            "validation_loss":
                validation_loss,

            "train_loss":
                train_loss
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            "\nBest classifier saved:"
        )

        print(
            MODEL_PATH
        )

    else:

        print(
            "\nValidation did not improve."
        )

        print(
            "Best validation loss:",
            f"{best_validation_loss:.4f}"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("CLASSIFIER TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation loss:",
    f"{best_validation_loss:.4f}"
)

print(
    "Best classifier:",
    MODEL_PATH
)