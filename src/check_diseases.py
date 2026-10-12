
import pandas as pd
from collections import Counter

# Load IU X-Ray reports
df = pd.read_csv("processed/reports.csv")

counter = Counter()

# Extract individual disease/finding labels
for value in df["labels"].dropna():
    labels = str(value).split("|")

    for label in set(labels):
        label = label.strip().lower()

        if label:
            counter[label] += 1

print("\nIU X-RAY DISEASE / FINDING LABELS")
print("=" * 50)
print("Total unique labels:", len(counter))
print()

for disease, count in counter.most_common():
    print(f"{disease:<40} {count}")

# Save to CSV
result = pd.DataFrame(
    counter.most_common(),
    columns=["disease_or_finding", "image_report_rows"]
)

result.to_csv(
    "processed/disease_counts.csv",
    index=False
)

print("\nSaved: processed/disease_counts.csv")
