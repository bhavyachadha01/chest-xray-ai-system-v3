import json
import re

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RadiologyDataset(Dataset):

    def __init__(self, csv_file, vocab_file, transform=None):

        self.df = pd.read_csv(csv_file)

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.word2idx = vocab["word2idx"]
        self.transform = transform

        self.pad_idx = self.word2idx["<pad>"]
        self.start_idx = self.word2idx["<start>"]
        self.end_idx = self.word2idx["<end>"]
        self.unk_idx = self.word2idx["<unk>"]

    def __len__(self):
        return len(self.df)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", str(text).lower())

    def encode_caption(self, caption):

        tokens = self.tokenize(caption)

        encoded = [self.start_idx]

        for token in tokens:
            encoded.append(
                self.word2idx.get(token, self.unk_idx)
            )

        encoded.append(self.end_idx)

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]
        caption = row["caption"]

        # Chest X-rays are grayscale.
        # Convert to RGB because pretrained CNNs expect 3 channels.
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.encode_caption(caption)

        return image, caption


# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    # ImageNet normalization for pretrained CNN
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def collate_fn(batch):

    images, captions = zip(*batch)

    images = torch.stack(images)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


def get_dataloader(
    csv_file,
    vocab_file="data/vocab.json",
    batch_size=8,
    shuffle=True
):

    dataset = RadiologyDataset(
        csv_file=csv_file,
        vocab_file=vocab_file,
        transform=image_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )

    return loader


# Test the loader
if __name__ == "__main__":

    loader = get_dataloader(
        "data/train.csv",
        batch_size=4
    )

    images, captions = next(iter(loader))

    print("--------------------------------")
    print("DATA LOADER WORKING")
    print("--------------------------------")

    print("Image batch shape:", images.shape)
    print("Caption batch shape:", captions.shape)

    print("\nFirst encoded caption:")
    print(captions[0])

    print("\nNumber of training batches:")
    print(len(loader))