import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# DENSENET SPATIAL ENCODER
# ============================================================

class EncoderCNN(nn.Module):

    def __init__(self, encoded_size=7):
        super().__init__()

        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        # Remove DenseNet classifier.
        # Keep convolutional feature extractor.
        self.features = densenet.features

        # Start frozen for CPU-friendly training.
        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(
            (encoded_size, encoded_size)
        )

        self.encoder_dim = 1024


    def forward(self, images):

        # images:
        # [batch, 3, 224, 224]

        features = self.features(images)

        features = torch.relu(features)

        # [batch, 1024, 7, 7]
        features = self.pool(features)

        # [batch, 7, 7, 1024]
        features = features.permute(
            0, 2, 3, 1
        )

        batch_size = features.size(0)

        # 7 x 7 = 49 image regions
        features = features.view(
            batch_size,
            -1,
            self.encoder_dim
        )

        # [batch, 49, 1024]

        return features


# ============================================================
# SOFT ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        attention_dim
    ):
        super().__init__()

        self.encoder_attention = nn.Linear(
            encoder_dim,
            attention_dim
        )

        self.decoder_attention = nn.Linear(
            decoder_dim,
            attention_dim
        )

        self.full_attention = nn.Linear(
            attention_dim,
            1
        )

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)


    def forward(
        self,
        encoder_features,
        hidden_state
    ):

        # encoder_features:
        # [batch, 49, 1024]

        encoder_projection = (
            self.encoder_attention(
                encoder_features
            )
        )

        # hidden_state:
        # [batch, decoder_dim]

        decoder_projection = (
            self.decoder_attention(
                hidden_state
            )
        ).unsqueeze(1)

        attention_scores = (
            self.full_attention(
                self.relu(
                    encoder_projection
                    +
                    decoder_projection
                )
            )
        ).squeeze(2)

        # [batch, 49]

        alpha = self.softmax(
            attention_scores
        )

        # Weighted image context
        context = (
            encoder_features
            *
            alpha.unsqueeze(2)
        ).sum(dim=1)

        # context:
        # [batch, 1024]

        return context, alpha


# ============================================================
# ATTENTION LSTM DECODER
# ============================================================

class DecoderWithAttention(nn.Module):

    def __init__(
        self,
        vocab_size,
        encoder_dim=1024,
        embed_dim=256,
        decoder_dim=512,
        attention_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim

        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # LSTM receives:
        # word embedding + attended image context
        self.lstm = nn.LSTMCell(
            embed_dim + encoder_dim,
            decoder_dim
        )

        # Initialize hidden/cell states from image
        self.init_h = nn.Linear(
            encoder_dim,
            decoder_dim
        )

        self.init_c = nn.Linear(
            encoder_dim,
            decoder_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            decoder_dim,
            vocab_size
        )


    def initialize_states(
        self,
        encoder_features
    ):

        mean_features = encoder_features.mean(
            dim=1
        )

        h = self.init_h(
            mean_features
        )

        c = self.init_c(
            mean_features
        )

        return h, c


    def forward(
        self,
        encoder_features,
        captions
    ):

        batch_size = encoder_features.size(0)

        sequence_length = captions.size(1)

        embeddings = self.embedding(
            captions
        )

        h, c = self.initialize_states(
            encoder_features
        )

        predictions = torch.zeros(
            batch_size,
            sequence_length - 1,
            self.vocab_size,
            device=encoder_features.device
        )

        attention_weights = torch.zeros(
            batch_size,
            sequence_length - 1,
            encoder_features.size(1),
            device=encoder_features.device
        )

        # Teacher forcing
        for t in range(sequence_length - 1):

            context, alpha = self.attention(
                encoder_features,
                h
            )

            lstm_input = torch.cat(
                [
                    embeddings[:, t, :],
                    context
                ],
                dim=1
            )

            h, c = self.lstm(
                lstm_input,
                (h, c)
            )

            logits = self.fc(
                self.dropout(h)
            )

            predictions[:, t, :] = logits

            attention_weights[:, t, :] = alpha

        return predictions, attention_weights


# ============================================================
# COMPLETE MODEL
# ============================================================

class RadiologyAttentionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        decoder_dim=512,
        attention_dim=256
    ):
        super().__init__()

        self.encoder = EncoderCNN()

        self.decoder = DecoderWithAttention(
            vocab_size=vocab_size,
            encoder_dim=1024,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )


    def forward(
        self,
        images,
        captions
    ):

        encoder_features = self.encoder(
            images
        )

        predictions, attention_weights = (
            self.decoder(
                encoder_features,
                captions
            )
        )

        return predictions, attention_weights