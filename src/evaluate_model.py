import json
import os
import re
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model import RadiologyCaptionModel


MODEL_PATH = "models/best_model.pth"
VOCAB_PATH = "data/vocab.json"
TEST_CSV = "processed/test.csv"
IMAGE_DIR = "data/images"

OUTPUT_FILE = "results/generated_reports/test_predictions.csv"

MAX_LENGTH = 40

# Keep this small initially so evaluation is quick on CPU.
# After it works, change to None for the entire test set.
TEST_LIMIT = 100


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VOCABULARY
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]

START_IDX = word2idx["<start>"]
END_IDX = word2idx["<end>"]


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

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
# TOKENIZER
# ============================================================

def tokenize(text):

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )


# ============================================================
# GENERATE CAPTION
# ============================================================

def generate_caption(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    generated_words = []

    with torch.no_grad():

        image_features = model.encoder(image)

        lstm_input = image_features.unsqueeze(1)

        _, hidden = model.decoder.lstm(
            lstm_input
        )

        current_token = torch.tensor(
            [[START_IDX]],
            dtype=torch.long,
            device=device
        )

        for _ in range(MAX_LENGTH):

            embedding = model.decoder.embedding(
                current_token
            )

            output, hidden = model.decoder.lstm(
                embedding,
                hidden
            )

            logits = model.decoder.fc(
                output[:, -1, :]
            )

            predicted_idx = torch.argmax(
                logits,
                dim=1
            ).item()

            if predicted_idx == END_IDX:
                break

            word = idx2word.get(
                str(predicted_idx),
                "<unk>"
            )

            generated_words.append(word)

            current_token = torch.tensor(
                [[predicted_idx]],
                dtype=torch.long,
                device=device
            )

    return " ".join(generated_words)


# ============================================================
# ROUGE-L
# ============================================================

def lcs_length(x, y):

    m = len(x)
    n = len(y)

    previous = [0] * (n + 1)

    for i in range(1, m + 1):

        current = [0] * (n + 1)

        for j in range(1, n + 1):

            if x[i - 1] == y[j - 1]:

                current[j] = previous[j - 1] + 1

            else:

                current[j] = max(
                    previous[j],
                    current[j - 1]
                )

        previous = current

    return previous[n]


def rouge_l(reference, prediction):

    if not reference or not prediction:
        return 0.0

    lcs = lcs_length(
        reference,
        prediction
    )

    precision = lcs / len(prediction)
    recall = lcs / len(reference)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        /
        (precision + recall)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    print("Device:", device)

    df = pd.read_csv(TEST_CSV)

    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)

    results = []

    smoothing = SmoothingFunction().method1

    generated_counter = Counter()

    for number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        image_id = str(row["image_id"])

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".png"
        )

        if not os.path.exists(image_path):

            print(
                f"[{number}/{len(df)}] "
                f"Missing: {image_id}"
            )

            continue

        # Use impression if available.
        # Otherwise fall back to report.
        reference_text = row.get(
            "impression",
            ""
        )

        if pd.isna(reference_text) or not str(reference_text).strip():

            reference_text = row.get(
                "report",
                ""
            )

        reference_text = str(reference_text)

        try:

            prediction = generate_caption(
                image_path
            )

        except Exception as error:

            print(
                f"[{number}/{len(df)}] "
                f"ERROR {image_id}: {error}"
            )

            continue

        reference_tokens = tokenize(
            reference_text
        )

        prediction_tokens = tokenize(
            prediction
        )

        bleu1 = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing
        )

        bleu2 = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        )

        bleu3 = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=(1/3, 1/3, 1/3, 0),
            smoothing_function=smoothing
        )

        bleu4 = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )

        rouge = rouge_l(
            reference_tokens,
            prediction_tokens
        )

        generated_counter[prediction] += 1

        results.append({
            "image_id": image_id,
            "actual_impression": reference_text,
            "generated_impression": prediction,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "rouge_l": rouge
        })

        print(
            f"[{number}/{len(df)}] "
            f"{image_id} -> {prediction}"
        )


    if not results:

        print("No images were evaluated.")
        return


    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        "Images evaluated:",
        len(results_df)
    )

    print(
        "BLEU-1:",
        round(results_df["bleu1"].mean(), 4)
    )

    print(
        "BLEU-2:",
        round(results_df["bleu2"].mean(), 4)
    )

    print(
        "BLEU-3:",
        round(results_df["bleu3"].mean(), 4)
    )

    print(
        "BLEU-4:",
        round(results_df["bleu4"].mean(), 4)
    )

    print(
        "ROUGE-L:",
        round(results_df["rouge_l"].mean(), 4)
    )


    print()
    print("Most common generated reports:")

    for caption, count in generated_counter.most_common(10):

        print(
            f"{count:4d}  {caption}"
        )


    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
    