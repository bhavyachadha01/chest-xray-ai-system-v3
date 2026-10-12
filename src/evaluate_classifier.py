
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prepare_classifier_data import (
    CLASSES,
    LABEL_PATTERNS,
    split_labels,
    contains_finding
)

from predict_disease import (
    load_classifier,
    prepare_image,
    device
)


TEST_CSV = ROOT / "processed" / "test.csv"
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "results"

THRESHOLD = 0.50


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_CSV)

    # Missing annotations are not verified negatives.
    df = df.dropna(subset=["labels", "image_id"])

    df = df.drop_duplicates(subset=["image_id"])

    print("Labelled test images:", len(df))

    model, classes, checkpoint = load_classifier()

    if list(classes) != list(CLASSES):
        raise ValueError("Classifier classes do not match.")

    y_true = []
    y_score = []
    image_ids = []

    for index, row in enumerate(df.itertuples(), start=1):

        image_id = str(row.image_id).strip()
        image_path = IMAGE_DIR / (image_id + ".png")

        if not image_path.exists():
            continue

        labels = split_labels(row.labels)

        target = [
            contains_finding(labels, LABEL_PATTERNS[c])
            for c in CLASSES
        ]

        try:
            image = prepare_image(str(image_path))

            with torch.no_grad():
                logits = model(image)
                scores = torch.sigmoid(logits)

            scores = scores.squeeze(0).cpu().numpy()

            y_true.append(target)
            y_score.append(scores)
            image_ids.append(image_id)

        except Exception as error:
            print("Skipped:", image_id, error)

        if index % 25 == 0:
            print(f"Processed {index}/{len(df)}")

    if not y_true:
        print("No usable test images found.")
        return

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    y_pred = (y_score >= THRESHOLD).astype(int)

    rows = []

    for j, disease in enumerate(CLASSES):

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true[:, j],
                y_pred[:, j],
                average="binary",
                zero_division=0
            )
        )

        if len(np.unique(y_true[:, j])) == 2:
            auc = roc_auc_score(
                y_true[:, j],
                y_score[:, j]
            )
        else:
            auc = np.nan

        rows.append({
            "disease": disease,
            "positive_samples": int(y_true[:, j].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auc,
            "mean_model_score": y_score[:, j].mean()
        })

    results = pd.DataFrame(rows)

    print("\nCLASSIFIER EVALUATION")
    print("=" * 80)

    print(
        results.round(4).to_string(index=False)
    )

    results.to_csv(
        OUTPUT_DIR / "classifier_metrics.csv",
        index=False
    )

    predictions = pd.DataFrame({
        "image_id": image_ids
    })

    for j, disease in enumerate(CLASSES):
        predictions[f"{disease}_actual"] = y_true[:, j]
        predictions[f"{disease}_score"] = y_score[:, j]

    predictions.to_csv(
        OUTPUT_DIR / "classifier_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print("results/classifier_metrics.csv")
    print("results/classifier_predictions.csv")

    print(
        "\nNOTE: These are preliminary metrics against "
        "incomplete Open-I annotations, not clinically "
        "validated disease-detection performance."
    )


if __name__ == "__main__":
    main()
