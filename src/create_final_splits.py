import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

TRAIN_CSV = "processed/classifier_v2_train.csv"
VAL_CSV = "processed/classifier_v2_val.csv"

OUT_TRAIN = "processed/classifier_v3_train.csv"
OUT_VAL = "processed/classifier_v3_val.csv"
OUT_TEST = "processed/classifier_v3_test.csv"

RANDOM_STATE = 42

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

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# LOAD ALL DATA
# ============================================================

train_old = pd.read_csv(TRAIN_CSV)
val_old = pd.read_csv(VAL_CSV)

df = pd.concat(
    [train_old, val_old],
    ignore_index=True
)

print("=" * 70)
print("CREATING RARE-CLASS-AWARE V3 SPLITS")
print("=" * 70)

print("Total images:", len(df))
print(
    "Unique reports:",
    df["report_file"].nunique()
)


# ============================================================
# CREATE REPORT-LEVEL TABLE
# ============================================================

# Multiple X-rays can belong to the same report.
# A report must stay entirely inside one split.

report_table = (
    df.groupby("report_file")[CLASSES]
    .max()
    .reset_index()
)

report_table["num_images"] = (
    report_table["report_file"]
    .map(
        df["report_file"].value_counts()
    )
)

print(
    "Report-level rows:",
    len(report_table)
)


# ============================================================
# SEPARATE PNEUMOTHORAX REPORTS
# ============================================================

pneumo_reports = report_table[
    report_table["pneumothorax"] == 1
].copy()

other_reports = report_table[
    report_table["pneumothorax"] == 0
].copy()

pneumo_reports = pneumo_reports.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

other_reports = other_reports.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print()
print(
    "Pneumothorax-positive reports:",
    len(pneumo_reports)
)


# ============================================================
# ALLOCATE RARE PNEUMOTHORAX REPORTS
# ============================================================

# Rough target:
# 80% train
# 10% validation
# 10% test
#
# But ensure validation AND test receive positive reports.

n_p = len(pneumo_reports)

if n_p < 3:

    raise ValueError(
        "Not enough pneumothorax-positive reports "
        "to place at least one in train, validation and test."
    )

n_p_test = max(
    1,
    round(n_p * 0.10)
)

n_p_val = max(
    1,
    round(n_p * 0.10)
)

# Ensure at least one remains for training.

while (
    n_p_test
    +
    n_p_val
    >=
    n_p
):

    if n_p_test > 1:
        n_p_test -= 1

    elif n_p_val > 1:
        n_p_val -= 1

    else:
        break


test_p = pneumo_reports.iloc[
    :n_p_test
]

val_p = pneumo_reports.iloc[
    n_p_test:
    n_p_test + n_p_val
]

train_p = pneumo_reports.iloc[
    n_p_test + n_p_val:
]


print()
print("Rare-class report allocation:")
print(
    "Train pneumothorax reports:",
    len(train_p)
)
print(
    "Validation pneumothorax reports:",
    len(val_p)
)
print(
    "Test pneumothorax reports:",
    len(test_p)
)


# ============================================================
# ALLOCATE ALL OTHER REPORTS
# ============================================================

n_other = len(other_reports)

n_other_test = round(
    n_other * 0.10
)

n_other_val = round(
    n_other * 0.10
)

test_other = other_reports.iloc[
    :n_other_test
]

val_other = other_reports.iloc[
    n_other_test:
    n_other_test + n_other_val
]

train_other = other_reports.iloc[
    n_other_test + n_other_val:
]


# ============================================================
# BUILD REPORT LISTS
# ============================================================

train_report_ids = set(
    pd.concat(
        [
            train_p["report_file"],
            train_other["report_file"]
        ]
    )
)

val_report_ids = set(
    pd.concat(
        [
            val_p["report_file"],
            val_other["report_file"]
        ]
    )
)

test_report_ids = set(
    pd.concat(
        [
            test_p["report_file"],
            test_other["report_file"]
        ]
    )
)


# ============================================================
# SAFETY CHECK
# ============================================================

assert train_report_ids.isdisjoint(
    val_report_ids
)

assert train_report_ids.isdisjoint(
    test_report_ids
)

assert val_report_ids.isdisjoint(
    test_report_ids
)


# ============================================================
# CREATE IMAGE-LEVEL DATASETS
# ============================================================

train_final = df[
    df["report_file"].isin(
        train_report_ids
    )
].copy()

val_final = df[
    df["report_file"].isin(
        val_report_ids
    )
].copy()

test_final = df[
    df["report_file"].isin(
        test_report_ids
    )
].copy()


# Shuffle rows inside each split.

train_final = train_final.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

val_final = val_final.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

test_final = test_final.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

assert (
    len(train_final)
    +
    len(val_final)
    +
    len(test_final)
    ==
    len(df)
)

assert set(
    train_final["report_file"]
).isdisjoint(
    set(val_final["report_file"])
)

assert set(
    train_final["report_file"]
).isdisjoint(
    set(test_final["report_file"])
)

assert set(
    val_final["report_file"]
).isdisjoint(
    set(test_final["report_file"])
)


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

def show_split(
    name,
    data
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Images:",
        len(data)
    )

    print(
        "Reports:",
        data["report_file"].nunique()
    )

    print()
    print(
        "Positive image labels:"
    )

    for disease in CLASSES:

        print(
            f"{disease:20s}: "
            f"{int(data[disease].sum())}"
        )

    print()
    print(
        "Positive reports:"
    )

    report_labels = (
        data.groupby("report_file")[CLASSES]
        .max()
    )

    for disease in CLASSES:

        print(
            f"{disease:20s}: "
            f"{int(report_labels[disease].sum())}"
        )


show_split(
    "TRAIN",
    train_final
)

show_split(
    "VALIDATION",
    val_final
)

show_split(
    "TEST",
    test_final
)


# ============================================================
# PNEUMOTHORAX SAFETY CHECK
# ============================================================

for name, data in [
    ("TRAIN", train_final),
    ("VALIDATION", val_final),
    ("TEST", test_final)
]:

    positives = int(
        data["pneumothorax"].sum()
    )

    if positives == 0:

        raise RuntimeError(
            f"{name} contains zero "
            "pneumothorax-positive images."
        )


# ============================================================
# SAVE
# ============================================================

train_final.to_csv(
    OUT_TRAIN,
    index=False
)

val_final.to_csv(
    OUT_VAL,
    index=False
)

test_final.to_csv(
    OUT_TEST,
    index=False
)


print()
print("=" * 70)
print("V3 SPLITS CREATED SUCCESSFULLY")
print("=" * 70)

print(OUT_TRAIN)
print(OUT_VAL)
print(OUT_TEST)

print()
print(
    "IMPORTANT: Do not evaluate the existing V2 checkpoint "
    "on the new V3 test split as an independent test result."
)

print(
    "V3 must be trained using classifier_v3_train.csv."
)