import os
import json
import time

import torch
import torch.nn as nn

from dataset_loader import get_dataloader
from model import RadiologyCaptionModel


# ============================================================
# SETTINGS
# ============================================================

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
VOCAB_FILE = "data/vocab.json"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Full training
EPOCHS = 5

# None = use complete dataset
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

EMBED_SIZE = 256
HIDDEN_SIZE = 512

# Print progress every N batches
PRINT_EVERY = 50


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("================================")
print("TRAINING CONFIGURATION")
print("================================")

print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx = vocab["word2idx"]

vocab_size = len(word2idx)

PAD_IDX = word2idx["<pad>"]

print("Vocabulary size:", vocab_size)


# ============================================================
# DATA LOADERS
# ============================================================

print("\nLoading datasets...")

train_loader = get_dataloader(
    TRAIN_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = get_dataloader(
    VAL_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = RadiologyCaptionModel(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE
)

model = model.to(device)


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel parameters:")
print("Total parameters:", f"{total_params:,}")
print("Trainable parameters:", f"{trainable_params:,}")
print(
    "Frozen parameters:",
    f"{total_params - trainable_params:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)


# ============================================================
# OPTIMIZER
# ============================================================

# Only optimize parameters that are trainable.
optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(epoch_number):

    model.train()

    total_loss = 0.0
    processed_batches = 0

    start_time = time.time()

    for batch_idx, (images, captions) in enumerate(train_loader):

        # Only used if we deliberately limit training
        if (
            MAX_TRAIN_BATCHES is not None
            and batch_idx >= MAX_TRAIN_BATCHES
        ):
            break

        images = images.to(device)
        captions = captions.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images,
            captions
        )

        # The model output length matches caption length
        targets = captions

        # Calculate loss
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        # Backpropagation
        loss.backward()

        # Prevent exploding LSTM gradients
        torch.nn.utils.clip_grad_norm_(
            filter(
                lambda parameter: parameter.requires_grad,
                model.parameters()
            ),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1

        # Progress
        if (batch_idx + 1) % PRINT_EVERY == 0:

            average_loss = (
                total_loss / processed_batches
            )

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch_number}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Batch Loss: {loss.item():.4f} | "
                f"Average Loss: {average_loss:.4f} | "
                f"Time: {elapsed / 60:.1f} min"
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    average_loss = total_loss / processed_batches

    return average_loss


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    processed_batches = 0

    with torch.no_grad():

        for batch_idx, (images, captions) in enumerate(val_loader):

            if (
                MAX_VAL_BATCHES is not None
                and batch_idx >= MAX_VAL_BATCHES
            ):
                break

            images = images.to(device)
            captions = captions.to(device)

            outputs = model(
                images,
                captions
            )

            targets = captions

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_loss += loss.item()
            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    average_loss = total_loss / processed_batches

    return average_loss


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

best_val_loss = float("inf")

print("\n================================")
print("STARTING TRAINING")
print("================================")

training_start = time.time()


for epoch in range(1, EPOCHS + 1):

    print("\n================================")
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("================================")

    epoch_start = time.time()

    # -------------------------
    # Training
    # -------------------------

    train_loss = train_one_epoch(epoch)

    # -------------------------
    # Validation
    # -------------------------

    print("\nRunning validation...")

    val_loss = validate()

    epoch_time = time.time() - epoch_start

    # -------------------------
    # Results
    # -------------------------

    print("\n--------------------------------")
    print(f"EPOCH {epoch} RESULTS")
    print("--------------------------------")

    print(
        "Train Loss:",
        round(train_loss, 4)
    )

    print(
        "Validation Loss:",
        round(val_loss, 4)
    )

    print(
        "Epoch Time:",
        round(epoch_time / 60, 2),
        "minutes"
    )

    # -------------------------
    # Save best model
    # -------------------------

    if val_loss < best_val_loss:

        print(
            f"Validation improved: "
            f"{best_val_loss:.4f} -> {val_loss:.4f}"
        )

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "vocab_size":
                    vocab_size,

                "embed_size":
                    EMBED_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "train_loss":
                    train_loss,

                "validation_loss":
                    val_loss
            },
            MODEL_PATH
        )

        print(
            "Best model saved:",
            MODEL_PATH
        )

    else:

        print(
            "Validation did not improve."
        )

        print(
            "Best validation loss remains:",
            round(best_val_loss, 4)
        )


# ============================================================
# FINISHED
# ============================================================

total_training_time = (
    time.time() - training_start
)

print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print(
    "Best validation loss:",
    round(best_val_loss, 4)
)

print(
    "Total training time:",
    round(total_training_time / 60, 2),
    "minutes"
)

print(
    "Best model:",
    MODEL_PATH
)