import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()

        # Pretrained DenseNet121
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        self.features = densenet.features

        # Freeze pretrained DenseNet convolutional layers
        # This makes training much faster on CPU
        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # This layer will still be trained
        self.fc = nn.Linear(
            1024,
            embed_size
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):

        features = self.features(images)

        features = self.relu(features)

        features = self.pool(features)

        features = torch.flatten(
            features,
            start_dim=1
        )

        features = self.fc(features)

        features = self.dropout(features)

        return features


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, image_features, captions):

        # Remove final caption token
        embeddings = self.embedding(
            captions[:, :-1]
        )

        # Image feature becomes first input
        image_features = image_features.unsqueeze(1)

        embeddings = torch.cat(
            (image_features, embeddings),
            dim=1
        )

        outputs, _ = self.lstm(embeddings)

        outputs = self.dropout(outputs)

        outputs = self.fc(outputs)

        return outputs


class RadiologyCaptionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512
    ):
        super().__init__()

        self.encoder = EncoderCNN(
            embed_size=embed_size
        )

        self.decoder = DecoderLSTM(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size
        )

    def forward(self, images, captions):

        image_features = self.encoder(images)

        outputs = self.decoder(
            image_features,
            captions
        )

        return outputs