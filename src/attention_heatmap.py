import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from attention_model import RadiologyAttentionModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_attention_model.pth"
VOCAB_PATH = "data/vocab.json"

OUTPUT_DIR = "results/attention_maps"

MAX_LENGTH = 40

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# Must match attention-model training
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
# LOAD VOCABULARY
# ============================================================

def load_vocabulary():

    if not os.path.exists(VOCAB_PATH):

        raise FileNotFoundError(
            f"Vocabulary not found: {VOCAB_PATH}"
        )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        vocab = json.load(file)

    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    return word2idx, idx2word


# ============================================================
# INDEX -> WORD
# ============================================================

def index_to_word(
    index,
    idx2word
):

    # JSON dictionary keys are often stored as strings.

    if str(index) in idx2word:

        return idx2word[
            str(index)
        ]

    if index in idx2word:

        return idx2word[
            index
        ]

    return "<unk>"


# ============================================================
# LOAD ATTENTION MODEL
# ============================================================

def load_attention_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nAttention checkpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            "Finish attention training before "
            "generating heatmaps."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    vocab_size = checkpoint[
        "vocab_size"
    ]

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
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model, checkpoint


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image_path
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        original_image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    return (
        original_image,
        image_tensor
    )


# ============================================================
# GENERATE REPORT + ATTENTION
# ============================================================

def generate_with_attention(
    model,
    image_tensor,
    word2idx,
    idx2word
):

    start_idx = word2idx[
        "<start>"
    ]

    end_idx = word2idx[
        "<end>"
    ]

    pad_idx = word2idx[
        "<pad>"
    ]

    generated_words = []

    attention_weights = []

    with torch.no_grad():

        # ----------------------------------------------------
        # IMAGE ENCODER
        #
        # Output:
        # [1, 49, 1024]
        #
        # 49 = 7 x 7 spatial regions
        # ----------------------------------------------------

        encoder_features = (
            model.encoder(
                image_tensor
            )
        )

        # ----------------------------------------------------
        # INITIAL LSTM STATE
        # ----------------------------------------------------

        h, c = (
            model.decoder
            .initialize_states(
                encoder_features
            )
        )

        # Start generation with <start>

        current_word = torch.tensor(
            [start_idx],
            dtype=torch.long,
            device=device
        )

        # ----------------------------------------------------
        # AUTOREGRESSIVE GENERATION
        # ----------------------------------------------------

        for _ in range(
            MAX_LENGTH
        ):

            # Word embedding

            embedding = (
                model.decoder
                .embedding(
                    current_word
                )
            )

            # ----------------------------------------------
            # ATTENTION
            #
            # alpha shape:
            # [1, 49]
            # ----------------------------------------------

            context, alpha = (
                model.decoder
                .attention(
                    encoder_features,
                    h
                )
            )

            # LSTM input =
            # word embedding + attended image context

            lstm_input = torch.cat(
                [
                    embedding,
                    context
                ],
                dim=1
            )

            h, c = (
                model.decoder
                .lstm(
                    lstm_input,
                    (h, c)
                )
            )

            logits = (
                model.decoder
                .fc(h)
            )

            predicted_idx = int(
                torch.argmax(
                    logits,
                    dim=1
                ).item()
            )

            # Stop at <end>.
            # We don't need to visualize attention
            # for the end token.

            if predicted_idx == end_idx:

                break

            # Ignore special tokens

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

                    # Store attention associated
                    # with this generated word.

                    attention_weights.append(
                        alpha
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

            current_word = torch.tensor(
                [predicted_idx],
                dtype=torch.long,
                device=device
            )

    report = " ".join(
        generated_words
    )

    return (
        report,
        generated_words,
        attention_weights
    )


# ============================================================
# CREATE GLOBAL 7 x 7 ATTENTION MAP
# ============================================================

def create_average_attention_map(
    attention_weights
):

    if not attention_weights:

        return None

    # Shape:
    # [number_of_generated_words, 49]

    attention_array = np.stack(
        attention_weights,
        axis=0
    )

    # Average attention across all generated words.

    average_attention = np.mean(
        attention_array,
        axis=0
    )

    # 49 regions -> 7 x 7

    attention_map = (
        average_attention
        .reshape(
            7,
            7
        )
    )

    # Normalize to 0-1 for visualization.

    minimum = np.min(
        attention_map
    )

    maximum = np.max(
        attention_map
    )

    if maximum > minimum:

        attention_map = (
            attention_map
            -
            minimum
        ) / (
            maximum
            -
            minimum
        )

    return attention_map


# ============================================================
# RESIZE ATTENTION MAP
# ============================================================

def resize_attention_map(
    attention_map,
    image_size
):

    if attention_map is None:

        return None

    # Convert normalized 0-1 array
    # to grayscale PIL image.

    heatmap_image = Image.fromarray(
        np.uint8(
            attention_map
            *
            255
        )
    )

    # image_size is:
    # (width, height)

    heatmap_image = (
        heatmap_image.resize(
            image_size,
            resample=Image.Resampling.BILINEAR
        )
    )

    resized_map = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        )
        /
        255.0
    )

    return resized_map


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_attention_overlay(
    original_image,
    resized_attention,
    output_path,
    report=None
):

    if resized_attention is None:

        raise ValueError(
            "No attention weights were "
            "generated for this image."
        )

    # Convert image for matplotlib.

    image_array = np.asarray(
        original_image
    )

    # Create one figure containing
    # the original image + heatmap overlay.

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image_array,
        cmap="gray"
    )

    plt.imshow(
        resized_attention,
        cmap="jet",
        alpha=0.40
    )

    plt.axis(
        "off"
    )

    if report:

        # Keep title short enough for demo.
        title = report

        if len(title) > 100:

            title = (
                title[:100]
                +
                "..."
            )

        plt.title(
            title,
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1
    )

    plt.close()


# ============================================================
# MAIN HEATMAP FUNCTION
#
# This function can later be imported by app.py.
# ============================================================

def generate_attention_heatmap(
    image_path,
    output_path=None
):

    # --------------------------------------------------------
    # LOAD EVERYTHING
    # --------------------------------------------------------

    word2idx, idx2word = (
        load_vocabulary()
    )

    model, checkpoint = (
        load_attention_model()
    )

    (
        original_image,
        image_tensor
    ) = prepare_image(
        image_path
    )

    # --------------------------------------------------------
    # GENERATE REPORT + ATTENTION
    # --------------------------------------------------------

    (
        report,
        generated_words,
        attention_weights
    ) = generate_with_attention(
        model,
        image_tensor,
        word2idx,
        idx2word
    )

    # --------------------------------------------------------
    # BUILD 7 x 7 GLOBAL ATTENTION MAP
    # --------------------------------------------------------

    attention_map = (
        create_average_attention_map(
            attention_weights
        )
    )

    if attention_map is None:

        raise ValueError(
            "The model did not generate usable "
            "attention weights."
        )

    # --------------------------------------------------------
    # RESIZE TO ORIGINAL X-RAY
    # --------------------------------------------------------

    resized_attention = (
        resize_attention_map(
            attention_map,
            original_image.size
        )
    )

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if output_path is None:

        image_name = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{image_name}_attention.png"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_attention_overlay(
        original_image,
        resized_attention,
        output_path,
        report=report
    )

    return {
        "report":
            report,

        "words":
            generated_words,

        "attention_weights":
            attention_weights,

        "attention_map":
            attention_map,

        "output_path":
            output_path,

        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None
            ),

        "validation_loss":
            checkpoint.get(
                "validation_loss",
                None
            )
    }


# ============================================================
# WORD-SPECIFIC ATTENTION MAP
# ============================================================

def generate_word_attention_heatmap(
    image_path,
    word_index,
    output_path=None
):

    word2idx, idx2word = (
        load_vocabulary()
    )

    model, _ = (
        load_attention_model()
    )

    (
        original_image,
        image_tensor
    ) = prepare_image(
        image_path
    )

    (
        report,
        generated_words,
        attention_weights
    ) = generate_with_attention(
        model,
        image_tensor,
        word2idx,
        idx2word
    )

    if not generated_words:

        raise ValueError(
            "No words were generated."
        )

    if (
        word_index < 0
        or
        word_index >= len(
            generated_words
        )
    ):

        raise IndexError(
            f"word_index must be between "
            f"0 and "
            f"{len(generated_words) - 1}"
        )

    # Get attention for one generated word.

    attention_map = (
        attention_weights[
            word_index
        ]
        .reshape(
            7,
            7
        )
    )

    # Normalize.

    minimum = np.min(
        attention_map
    )

    maximum = np.max(
        attention_map
    )

    if maximum > minimum:

        attention_map = (
            attention_map
            -
            minimum
        ) / (
            maximum
            -
            minimum
        )

    resized_attention = (
        resize_attention_map(
            attention_map,
            original_image.size
        )
    )

    selected_word = (
        generated_words[
            word_index
        ]
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if output_path is None:

        image_name = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        safe_word = (
            selected_word
            .replace(
                "/",
                "_"
            )
            .replace(
                "\\",
                "_"
            )
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            (
                f"{image_name}_"
                f"{word_index}_"
                f"{safe_word}.png"
            )
        )

    save_attention_overlay(
        original_image,
        resized_attention,
        output_path,
        report=(
            f"Attention for word: "
            f"{selected_word}"
        )
    )

    return {
        "report":
            report,

        "word":
            selected_word,

        "word_index":
            word_index,

        "attention_map":
            attention_map,

        "output_path":
            output_path
    }


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            'python src/attention_heatmap.py '
            '"path_to_xray.png"\n'
        )

        return

    image_path = sys.argv[1]

    print("=" * 60)
    print("ATTENTION HEATMAP GENERATOR")
    print("=" * 60)

    print(
        "\nDevice:",
        device
    )

    print(
        "Image:",
        image_path
    )

    print(
        "\nLoading attention model..."
    )

    try:

        result = (
            generate_attention_heatmap(
                image_path
            )
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )

        return

    print(
        "\nCheckpoint epoch:",
        result[
            "checkpoint_epoch"
        ]
    )

    print(
        "Validation loss:",
        result[
            "validation_loss"
        ]
    )

    print()
    print("-" * 60)
    print("GENERATED REPORT")
    print("-" * 60)

    if result["report"]:

        print(
            result["report"]
        )

    else:

        print(
            "[Empty report]"
        )

    print()
    print(
        "Generated words:",
        len(
            result["words"]
        )
    )

    print()
    print(
        "7 x 7 attention map:"
    )

    print(
        result[
            "attention_map"
        ]
    )

    print()
    print(
        "Heatmap saved:"
    )

    print(
        result[
            "output_path"
        ]
    )

    print()
    print("=" * 60)
    print(
        "ATTENTION VISUALIZATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()