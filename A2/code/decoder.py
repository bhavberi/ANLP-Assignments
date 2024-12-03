# %%
import torch
import torch.nn as nn

from model_utils import TransformerBlock, MultiHeadAttention

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class DecoderBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, forward_expansion: int, dropout: float
    ):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, forward_expansion, dropout
        )
        self.layer_norm = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.layer_norm(attention + x)
        out = self.transformer_block(value, key, query, src_mask)
        return out


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_len: int,
    ):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        n, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(n, seq_len).to(device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc(x)
        return out
