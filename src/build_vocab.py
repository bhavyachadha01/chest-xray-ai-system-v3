import pandas as pd
import re
import json
from collections import Counter

TRAIN_FILE = "data/train.csv"
VOCAB_FILE = "data/vocab.json"

MIN_FREQ = 2

# Load ONLY training data
df = pd.read_csv(TRAIN_FILE)

counter = Counter()


def tokenize(text):
    text = str(text).lower()

    # Keep words and numbers
    tokens = re.findall(r"\b\w+\b", text)

    return tokens


# Count words
for caption in df["caption"]:
    tokens = tokenize(caption)
    counter.update(tokens)


# Special tokens
word2idx = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "<unk>": 3
}

# Add words appearing at least MIN_FREQ times
for word, count in counter.items():

    if count >= MIN_FREQ:
        word2idx[word] = len(word2idx)


# Reverse vocabulary
idx2word = {
    str(index): word
    for word, index in word2idx.items()
}


vocab = {
    "word2idx": word2idx,
    "idx2word": idx2word
}


# Save vocabulary
with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4)


print("--------------------------------")
print("VOCABULARY CREATED")
print("--------------------------------")

print("Training captions:", len(df))
print("Unique raw words:", len(counter))
print("Vocabulary size:", len(word2idx))
print("Minimum frequency:", MIN_FREQ)

print("\nMost common words:")

for word, count in counter.most_common(20):
    print(f"{word:20} {count}")

print("\nSaved:", VOCAB_FILE)
