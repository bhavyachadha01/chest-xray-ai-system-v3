import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = "processed/reports.csv"
IMAGE_DIR = "data/images"
OUTPUT_CSV = "processed/disease_labels.csv"


# ============================================================
# CLASSES FOR OUR CLASSIFIER
# ============================================================

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


# ============================================================
# OPEN-I LABEL VARIATIONS
# ============================================================

LABEL_PATTERNS = {

    "atelectasis": [
        "atelectasis",
        "atelectases",
        "pulmonary atelectasis",
        "focal atelectasis",
        "discoid atelectasis",
    ],

    "cardiomegaly": [
        "cardiomegaly",
        "enlarged heart",
    ],

    "emphysema": [
        "emphysema",
    ],

    "opacity": [
        "opacity",
        "opacities",
    ],

    "pneumothorax": [
        "pneumothorax",
    ],

    "pleural_effusion": [
        "pleural effusion",
        "pleural fluid",
        "loculated effusion",
    ],

    "nodule": [
        "nodule",
        "nodules",
        "pulmonary nodule",
    ],

    "scarring": [
        "scarring",
        "scar",
    ],
}


# ============================================================
# SPLIT LABEL STRING
# ============================================================

def split_labels(value):

    if pd.isna(value):
        return []

    return [
        label.strip().lower()
        for label in str(value).split("|")
        if label.strip()
    ]


# ============================================================
# CHECK WHETHER FINDING EXISTS
# ============================================================

def contains_finding(labels, patterns):

    for label in labels:

        for pattern in patterns:

            if label == pattern:
                return 1

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PREPARING DISEASE CLASSIFIER DATA")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):

        raise FileNotFoundError(
            f"Cannot find: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("\nInput rows:", len(df))
    print("Columns:", list(df.columns))

    if "image_id" not in df.columns:

        raise ValueError(
            "image_id column not found."
        )

    if "labels" not in df.columns:

        raise ValueError(
            "labels column not found."
        )

    rows = []

    missing_images = 0

    for _, row in df.iterrows():

        image_id = str(
            row["image_id"]
        ).strip()

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".png"
        )

        if not os.path.exists(image_path):

            missing_images += 1
            continue

        labels = split_labels(
            row["labels"]
        )

        output_row = {

            "image_id":
                image_id,

            "image_path":
                image_path
        }

        for disease in CLASSES:

            output_row[disease] = (
                contains_finding(
                    labels,
                    LABEL_PATTERNS[disease]
                )
            )

        rows.append(
            output_row
        )

    output_df = pd.DataFrame(
        rows
    )

    # One row per image
    output_df = (
        output_df
        .drop_duplicates(
            subset=["image_id"]
        )
        .reset_index(drop=True)
    )

    os.makedirs(
        "processed",
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(
        "\nUsable X-rays:",
        len(output_df)
    )

    print(
        "Missing images:",
        missing_images
    )

    print("\nPositive examples:")
    print("-" * 40)

    for disease in CLASSES:

        count = int(
            output_df[disease].sum()
        )

        print(
            f"{disease:20s} {count}"
        )

    finding_count = (
        output_df[CLASSES]
        .sum(axis=1)
    )

    print(
        "\nImages containing at least "
        "one selected finding:",
        int((finding_count > 0).sum())
    )

    print(
        "Images with none of the "
        "selected findings:",
        int((finding_count == 0).sum())
    )

    print(
        "\nSaved:",
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()