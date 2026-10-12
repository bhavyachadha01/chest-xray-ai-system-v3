import json
import torch

from dataset_loader import get_dataloader
from model import RadiologyCaptionModel


# Load vocabulary
with open("data/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab["word2idx"])


# Check GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("--------------------------------")
print("MODEL TEST")
print("--------------------------------")

print("Device:", device)
print("Vocabulary size:", vocab_size)


# Load 2 examples
loader = get_dataloader(
    "data/train.csv",
    batch_size=2,
    shuffle=False
)

images, captions = next(iter(loader))

images = images.to(device)
captions = captions.to(device)

print("Image shape:", images.shape)
print("Caption shape:", captions.shape)


# Create model
model = RadiologyCaptionModel(
    vocab_size=vocab_size,
    embed_size=256,
    hidden_size=512
)

model = model.to(device)


# Test forward pass
model.eval()

with torch.no_grad():

    outputs = model(
        images,
        captions
    )


print("\n--------------------------------")
print("MODEL TEST SUCCESSFUL")
print("--------------------------------")

print("Output shape:", outputs.shape)

print("\nExpected:")
print("Batch size:", images.shape[0])
print("Sequence length:", captions.shape[1])
print("Vocabulary size:", vocab_size)