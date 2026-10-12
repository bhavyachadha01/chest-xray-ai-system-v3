
from pathlib import Path
import pandas as pd

PROCESSED = Path("processed")

LABELS_FILE = PROCESSED / "disease_labels.csv"

SPLITS = {
    "train": PROCESSED / "train.csv",
    "val": PROCESSED / "val.csv",
    "test": PROCESSED / "test.csv"
}

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


def main():

    labels_df = pd.read_csv(LABELS_FILE)

    labels_df["image_id"] = (
        labels_df["image_id"].astype(str).str.strip()
    )

    all_studies = {}

    for split_name, split_file in SPLITS.items():

        split_df = pd.read_csv(split_file)

        split_df["image_id"] = (
            split_df["image_id"].astype(str).str.strip()
        )

        # Verify no study overlap
        all_studies[split_name] = set(
            split_df["report_file"].dropna()
        )

        # Preserve only images belonging to this split
        result = split_df[
            ["report_file", "image_id"]
        ].merge(
            labels_df[
                ["image_id", "image_path"] + CLASSES
            ],
            on="image_id",
            how="inner"
        )

        result = result.drop_duplicates(
            subset=["image_id"]
        )

        output_path = (
            PROCESSED / f"classifier_{split_name}.csv"
        )

        result.to_csv(
            output_path,
            index=False
        )

        print(f"\n{split_name.upper()}")
        print("Images:", len(result))
        print("Studies:", result["report_file"].nunique())

        print("Positive counts:")
        print(result[CLASSES].sum().to_string())

        print("Saved:", output_path)

    # Ensure studies never overlap
    assert all_studies["train"].isdisjoint(
        all_studies["val"]
    )

    assert all_studies["train"].isdisjoint(
        all_studies["test"]
    )

    assert all_studies["val"].isdisjoint(
        all_studies["test"]
    )

    print("\nNo study overlap detected.")
    print("Classifier splits created successfully.")


if __name__ == "__main__":
    main()
