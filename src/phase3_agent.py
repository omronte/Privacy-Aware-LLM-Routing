
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum supported length {self.pe.size(1)}."
            )

        return x + self.pe[:, :seq_len, :]


class PrivacyR1Agent(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 384,
        n_heads: int = 8,
        n_layers: int = 2,
        hidden_dim: int = 512
    ):
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by n_heads."
            )

        self.embedding_dim = embedding_dim

        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.actor_head = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Linear(
            embedding_dim,
            1
        )


    def forward(
        self,
        embeddings: torch.Tensor
    ):

        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape (batch, sequence, features)."
            )

        if embeddings.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embeddings.size(-1)}."
            )

        x = self.pos_encoder(embeddings)

        contextualized_chunks = self.transformer_encoder(x)

        action_probs = self.actor_head(
            contextualized_chunks
        )

        state_values = self.critic_head(
            contextualized_chunks
        )

        return action_probs, state_values
