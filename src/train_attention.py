import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from attention_model import RadiologyAttentionModel


# ============================================================
# SETTINGS
# ============================================================

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
IMAGE_DIR = "data/images"
VOCAB_PATH = "data/vocab.json"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_attention_model.pth"
)

BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

MAX_LENGTH = 40

EMBED_DIM = 256
DECODER_DIM = 512
ATTENTION_DIM = 256

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VOCABULARY
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx = vocab["word2idx"]

VOCAB_SIZE = len(word2idx)

PAD_IDX = word2idx["<pad>"]
START_IDX = word2idx["<start>"]
END_IDX = word2idx["<end>"]
UNK_IDX = word2idx["<unk>"]


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return str(text).lower().split()


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):

        self.df = pd.read_csv(csv_file)

        self.image_dir = image_dir
        self.transform = transform

        # Try common column names
        if "caption" in self.df.columns:
            self.caption_column = "caption"

        elif "impression" in self.df.columns:
            self.caption_column = "impression"

        elif "report" in self.df.columns:
            self.caption_column = "report"

        else:
            raise ValueError(
                "CSV must contain caption, "
                "impression, or report column."
            )


    def __len__(self):

        return len(self.df)


    def encode_caption(self, text):

        words = tokenize(text)

        tokens = [START_IDX]

        for word in words:

            tokens.append(
                word2idx.get(
                    word,
                    UNK_IDX
                )
            )

        tokens.append(END_IDX)

        tokens = tokens[:MAX_LENGTH]

        if tokens[-1] != END_IDX:
            tokens[-1] = END_IDX

        while len(tokens) < MAX_LENGTH:

            tokens.append(PAD_IDX)

        return torch.tensor(
            tokens,
            dtype=torch.long
        )


    def __getitem__(self, index):

        row = self.df.iloc[index]

        # --------------------------------------------
        # Find image
        # --------------------------------------------

        if "image_path" in self.df.columns:

            image_path = str(
                row["image_path"]
            )

        elif "image_id" in self.df.columns:

            image_path = os.path.join(
                self.image_dir,
                str(row["image_id"]) + ".png"
            )

        else:

            raise ValueError(
                "CSV needs image_path "
                "or image_id column."
            )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform is not None:

            image = self.transform(
                image
            )


        caption_text = row[
            self.caption_column
        ]


        if pd.isna(caption_text):
            caption_text = ""


        caption = self.encode_caption(
            caption_text
        )


        return image, caption


# ============================================================
# TRANSFORMS
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
# LOAD DATA
# ============================================================

print("=" * 60)
print("ATTENTION MODEL TRAINING")
print("=" * 60)

print("\nDevice:", device)
print("Vocabulary:", VOCAB_SIZE)


train_dataset = XRayDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    transform
)

val_dataset = XRayDataset(
    VAL_CSV,
    IMAGE_DIR,
    transform
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
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
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

model = RadiologyAttentionModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    decoder_dim=DECODER_DIM,
    attention_dim=ATTENTION_DIM
)

model = model.to(device)


trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"\nTotal parameters: {total:,}"
)

print(
    f"Trainable parameters: {trainable:,}"
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)


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

    total_loss = 0

    start_time = time.time()


    for batch_number, (
        images,
        captions
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        captions = captions.to(device)


        optimizer.zero_grad()


        outputs, _ = model(
            images,
            captions
        )


        # Model predicts tokens after <start>
        targets = captions[:, 1:]


        loss = criterion(
            outputs.reshape(
                -1,
                VOCAB_SIZE
            ),
            targets.reshape(-1)
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )


        optimizer.step()


        total_loss += loss.item()


        if batch_number % 50 == 0:

            elapsed = (
                time.time()
                -
                start_time
            ) / 60

            print(
                f"Batch "
                f"{batch_number}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f} "
                f"| Average: "
                f"{total_loss / batch_number:.4f} "
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

    total_loss = 0


    with torch.no_grad():

        for images, captions in val_loader:

            images = images.to(device)
            captions = captions.to(device)


            outputs, _ = model(
                images,
                captions
            )


            targets = captions[:, 1:]


            loss = criterion(
                outputs.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                targets.reshape(-1)
            )


            total_loss += loss.item()


    return (
        total_loss
        /
        len(val_loader)
    )


# ============================================================
# TRAINING LOOP
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_val_loss = float("inf")


for epoch in range(1, NUM_EPOCHS + 1):

    print()
    print("=" * 60)
    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )
    print("=" * 60)


    train_loss = train_epoch()

    val_loss = validate()


    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss


        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "vocab_size":
                VOCAB_SIZE,

            "embed_dim":
                EMBED_DIM,

            "decoder_dim":
                DECODER_DIM,

            "attention_dim":
                ATTENTION_DIM,

            "train_loss":
                train_loss,

            "validation_loss":
                val_loss
        }


        torch.save(
            checkpoint,
            MODEL_PATH
        )


        print(
            "\nBest attention model saved:"
        )

        print(
            MODEL_PATH
        )


print()
print("=" * 60)
print("ATTENTION TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Model:",
    MODEL_PATH
)