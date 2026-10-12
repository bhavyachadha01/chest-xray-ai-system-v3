import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from model import RadiologyCaptionModel
except ModuleNotFoundError:
    from src.model import RadiologyCaptionModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
VOCAB_PATH = PROJECT_ROOT / "data" / "vocab.json"

MAX_LENGTH = 40


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Caption model not found: {MODEL_PATH}"
    )

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"Vocabulary not found: {VOCAB_PATH}"
    )


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab = json.load(f)


word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]

print(
    "Vocabulary size:",
    len(word2idx)
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

if "<start>" in word2idx:
    START_TOKEN = "<start>"

elif "<START>" in word2idx:
    START_TOKEN = "<START>"

else:
    raise KeyError(
        "Start token not found in vocabulary."
    )


if "<end>" in word2idx:
    END_TOKEN = "<end>"

elif "<END>" in word2idx:
    END_TOKEN = "<END>"

else:
    raise KeyError(
        "End token not found in vocabulary."
    )


START_IDX = word2idx[START_TOKEN]
END_IDX = word2idx[END_TOKEN]

print(
    "START token:",
    START_TOKEN,
    START_IDX
)

print(
    "END token:",
    END_TOKEN,
    END_IDX
)


# ============================================================
# LOAD CAPTION CHECKPOINT
# ============================================================

print(
    "\nLoading trained caption model..."
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Training loss:",
    checkpoint.get(
        "train_loss",
        "unknown"
    )
)

print(
    "Validation loss:",
    checkpoint.get(
        "validation_loss",
        "unknown"
    )
)


# ============================================================
# CREATE CAPTION MODEL
# ============================================================

model = RadiologyCaptionModel(
    vocab_size=checkpoint["vocab_size"],
    embed_size=checkpoint["embed_size"],
    hidden_size=checkpoint["hidden_size"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Caption model loaded successfully."
)


# ============================================================
# IMAGE PREPROCESSING
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
# GENERATE CAPTION
# ============================================================

def generate_caption(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    generated_words = []


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        # Encode image
        image_features = model.encoder(
            image_tensor
        )


        # Image feature becomes first LSTM input
        lstm_input = (
            image_features
            .unsqueeze(1)
        )


        output, hidden = (
            model.decoder.lstm(
                lstm_input
            )
        )


        # Start token
        current_token = torch.tensor(
            [[START_IDX]],
            dtype=torch.long,
            device=device
        )


        # ----------------------------------------------------
        # GENERATE WORDS
        # ----------------------------------------------------

        for _ in range(MAX_LENGTH):

            embedding = (
                model.decoder.embedding(
                    current_token
                )
            )


            output, hidden = (
                model.decoder.lstm(
                    embedding,
                    hidden
                )
            )


            logits = model.decoder.fc(
                output[:, -1, :]
            )


            predicted_idx = (
                torch.argmax(
                    logits,
                    dim=1
                )
                .item()
            )


            # Stop at end token
            if predicted_idx == END_IDX:
                break


            word = idx2word.get(
                str(predicted_idx)
            )

            # Some vocabularies may use
            # integer dictionary keys.
            if word is None:
                word = idx2word.get(
                    predicted_idx,
                    "<unk>"
                )


            # Avoid displaying special tokens
            if word not in {
                START_TOKEN,
                END_TOKEN,
                "<pad>",
                "<PAD>"
            }:

                generated_words.append(
                    word
                )


            current_token = torch.tensor(
                [[predicted_idx]],
                dtype=torch.long,
                device=device
            )


    caption = " ".join(
        generated_words
    ).strip()


    if not caption:

        return (
            "The caption model did not generate "
            "a textual impression."
        )


    return caption


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Please provide an X-ray image."
        )

        print()

        print(
            "Example:"
        )

        print(
            'python src/generate_caption.py '
            '"data/images/CXR1044_IM-0036-1001.png"'
        )

        return


    image_path = sys.argv[1]


    print(
        "\nGenerating caption..."
    )


    generated_caption = (
        generate_caption(
            image_path
        )
    )


    print()

    print(
        "=" * 60
    )

    print(
        "RADIOLOGY CAPTION GENERATION"
    )

    print(
        "=" * 60
    )


    print(
        "\nImage:"
    )

    print(
        image_path
    )


    print(
        "\nGenerated impression:"
    )

    print(
        generated_caption
    )


    print()

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()