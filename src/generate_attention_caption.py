import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from attention_model import RadiologyAttentionModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_attention_model.pth"
VOCAB_PATH = "data/vocab.json"

MAX_LENGTH = 40

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD VOCABULARY
# ============================================================

def load_vocabulary():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    return word2idx, idx2word


# ============================================================
# INDEX -> WORD
# ============================================================

def index_to_word(index, idx2word):

    # JSON dictionary keys may be strings.
    if str(index) in idx2word:
        return idx2word[str(index)]

    if index in idx2word:
        return idx2word[index]

    return "<unk>"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nAttention checkpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            "Wait for src/train_attention.py to finish "
            "and create the checkpoint."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    vocab_size = checkpoint["vocab_size"]

    embed_dim = checkpoint.get(
        "embed_dim",
        256
    )

    decoder_dim = checkpoint.get(
        "decoder_dim",
        512
    )

    attention_dim = checkpoint.get(
        "attention_dim",
        256
    )

    model = RadiologyAttentionModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    return image_tensor.to(device)


# ============================================================
# GREEDY ATTENTION DECODING
# ============================================================

def generate_caption(
    model,
    image_tensor,
    word2idx,
    idx2word
):

    start_idx = word2idx["<start>"]
    end_idx = word2idx["<end>"]
    pad_idx = word2idx["<pad>"]

    generated_words = []
    generated_indices = []

    all_attention = []

    with torch.no_grad():

        # --------------------------------------------
        # DenseNet spatial features
        # [1, 49, 1024]
        # --------------------------------------------

        encoder_features = model.encoder(
            image_tensor
        )

        # --------------------------------------------
        # Initialize LSTM hidden/cell states
        # --------------------------------------------

        h, c = model.decoder.initialize_states(
            encoder_features
        )

        current_word = torch.tensor(
            [start_idx],
            dtype=torch.long,
            device=device
        )

        # --------------------------------------------
        # Autoregressive generation
        # --------------------------------------------

        for _ in range(MAX_LENGTH):

            embedding = model.decoder.embedding(
                current_word
            )

            context, alpha = (
                model.decoder.attention(
                    encoder_features,
                    h
                )
            )

            lstm_input = torch.cat(
                [
                    embedding,
                    context
                ],
                dim=1
            )

            h, c = model.decoder.lstm(
                lstm_input,
                (h, c)
            )

            logits = model.decoder.fc(h)

            predicted_idx = int(
                torch.argmax(
                    logits,
                    dim=1
                ).item()
            )

            # Store attention for visualization later
            all_attention.append(
                alpha.squeeze(0).cpu()
            )

            # Stop when <end> is predicted
            if predicted_idx == end_idx:
                break

            # Ignore unwanted special tokens
            if predicted_idx not in [
                start_idx,
                pad_idx
            ]:

                word = index_to_word(
                    predicted_idx,
                    idx2word
                )

                if word not in [
                    "<start>",
                    "<end>",
                    "<pad>"
                ]:

                    generated_words.append(
                        word
                    )

                    generated_indices.append(
                        predicted_idx
                    )

            current_word = torch.tensor(
                [predicted_idx],
                dtype=torch.long,
                device=device
            )

    caption = " ".join(
        generated_words
    )

    return (
        caption,
        all_attention,
        generated_indices
    )


# ============================================================
# PUBLIC FUNCTION FOR GRADIO
# ============================================================

def generate_report(image_path):

    word2idx, idx2word = (
        load_vocabulary()
    )

    model, checkpoint = (
        load_model()
    )

    image_tensor = prepare_image(
        image_path
    )

    caption, attention, _ = (
        generate_caption(
            model,
            image_tensor,
            word2idx,
            idx2word
        )
    )

    return caption


# ============================================================
# COMMAND-LINE PROGRAM
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            'python src/generate_attention_caption.py '
            '"path_to_image.png"'
        )

        return

    image_path = sys.argv[1]

    print("=" * 60)
    print("ATTENTION RADIOLOGY REPORT GENERATOR")
    print("=" * 60)

    print("\nDevice:", device)

    print(
        "\nLoading attention model..."
    )

    word2idx, idx2word = (
        load_vocabulary()
    )

    model, checkpoint = (
        load_model()
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
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

    print(
        "\nImage:",
        image_path
    )

    image_tensor = prepare_image(
        image_path
    )

    caption, attention, _ = (
        generate_caption(
            model,
            image_tensor,
            word2idx,
            idx2word
        )
    )

    print("\n" + "-" * 60)

    print(
        "GENERATED ATTENTION REPORT"
    )

    print("-" * 60)

    if caption.strip():

        print(caption)

    else:

        print(
            "[Model generated an empty report]"
        )

    print("-" * 60)

    print(
        "\nGenerated tokens:",
        len(attention)
    )


if __name__ == "__main__":
    main()