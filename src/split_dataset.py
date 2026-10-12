import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed_dataset.csv"

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

RANDOM_STATE = 42

# Load processed dataset
df = pd.read_csv(INPUT_FILE)

print("Total images:", len(df))
print("Total reports:", df["report_id"].nunique())

# -----------------------------------
# Get unique report IDs
# -----------------------------------

report_ids = df["report_id"].unique()

# -----------------------------------
# 80% train, 20% temporary
# -----------------------------------

train_ids, temp_ids = train_test_split(
    report_ids,
    test_size=0.20,
    random_state=RANDOM_STATE
)

# -----------------------------------
# Split remaining 20% into
# 10% validation + 10% test
# -----------------------------------

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=RANDOM_STATE
)

# -----------------------------------
# Select image rows
# -----------------------------------

train_df = df[df["report_id"].isin(train_ids)].copy()
val_df = df[df["report_id"].isin(val_ids)].copy()
test_df = df[df["report_id"].isin(test_ids)].copy()

# -----------------------------------
# Save
# -----------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# -----------------------------------
# Check for leakage
# -----------------------------------

train_reports = set(train_df["report_id"])
val_reports = set(val_df["report_id"])
test_reports = set(test_df["report_id"])

assert train_reports.isdisjoint(val_reports)
assert train_reports.isdisjoint(test_reports)
assert val_reports.isdisjoint(test_reports)

print("\n-------------------------------")
print("DATASET SPLIT COMPLETE")
print("-------------------------------")

print("\nTRAIN")
print("Images:", len(train_df))
print("Reports:", train_df["report_id"].nunique())

print("\nVALIDATION")
print("Images:", len(val_df))
print("Reports:", val_df["report_id"].nunique())

print("\nTEST")
print("Images:", len(test_df))
print("Reports:", test_df["report_id"].nunique())

print("\nLeakage check: PASSED")

print("\nFiles created:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)