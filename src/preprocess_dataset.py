import pandas as pd
import re
import os

INPUT_FILE = "data/dataset.csv"
OUTPUT_FILE = "data/processed_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# Keep rows that have an impression
df = df.dropna(subset=["impression"])

# Fill missing findings
df["findings"] = df["findings"].fillna("")

def clean_text(text):
    text = str(text).lower()

    # Replace de-identification token
    text = re.sub(r"\bxxxx\b", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["findings"] = df["findings"].apply(clean_text)
df["impression"] = df["impression"].apply(clean_text)

# Our initial caption target
df["caption"] = df["impression"]

# Remove empty captions
df = df[df["caption"].str.len() > 0]

# Remove duplicate image entries, if any
df = df.drop_duplicates(subset=["image_name"])

os.makedirs("data", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("--------------------------------")
print("PREPROCESSING COMPLETE")
print("--------------------------------")
print("Rows remaining:", len(df))
print("Unique reports:", df["report_id"].nunique())
print("Unique images:", df["image_name"].nunique())
print("Missing captions:", df["caption"].isna().sum())

print("\nExample captions:\n")

for caption in df["caption"].head(5):
    print(">", caption)
    print()

import pandas as pd
import re

INPUT_FILE = "data/dataset.csv"
OUTPUT_FILE = "data/processed_dataset.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))

# ---------------------------------
# 1. Missing-value information
# ---------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# We need an impression for caption training
df = df.dropna(subset=["impression"]).copy()

# Findings are useful but may be missing
df["findings"] = df["findings"].fillna("")


# ---------------------------------
# 2. Clean radiology text
# ---------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove IU X-Ray de-identification placeholder
    text = re.sub(r"\bxxxx\b", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["findings"] = df["findings"].apply(clean_text)
df["impression"] = df["impression"].apply(clean_text)


# ---------------------------------
# 3. Caption target
# ---------------------------------

# For our first model, predict the IMPRESSION
df["caption"] = df["impression"]


# ---------------------------------
# 4. Remove empty captions
# ---------------------------------

df = df[df["caption"].str.len() > 0]


# ---------------------------------
# 5. Remove duplicate image rows
# ---------------------------------

df = df.drop_duplicates(subset=["image_name"])


# ---------------------------------
# 6. Save
# ---------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------
# Results
# ---------------------------------

print("\n-------------------------------")
print("PREPROCESSING COMPLETE")
print("-------------------------------")

print("Rows remaining:", len(df))
print("Unique reports:", df["report_id"].nunique())
print("Unique images:", df["image_name"].nunique())
print("Missing captions:", df["caption"].isnull().sum())

print("\nExample captions:")

for caption in df["caption"].head(5):
    print("\n>", caption)

print("\nSaved:", OUTPUT_FILE)