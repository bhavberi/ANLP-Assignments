# %%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (
            self.head_dim * num_heads == embedding_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.q = nn.Linear(self.head_dim, self.head_dim)
        self.k = nn.Linear(self.head_dim, self.head_dim)
        self.v = nn.Linear(self.head_dim, self.head_dim)
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, value, key, query, mask):
        n = query.size(0)
        query_len, key_len, value_len = query.size(1), key.size(1), value.size(1)

        value = self.v(value.reshape(n, value_len, self.num_heads, self.head_dim))
        query = self.q(query.reshape(n, query_len, self.num_heads, self.head_dim))
        key = self.k(key.reshape(n, key_len, self.num_heads, self.head_dim))

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float("inf"))
        attention = F.softmax(energy / np.sqrt(self.head_dim), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            n, query_len, self.embedding_dim
        )
        out = self.fc(out)

        return out


# %%
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
    ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.layer_norm1 = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.layer_norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.layer_norm2(forward + x)
        return out
