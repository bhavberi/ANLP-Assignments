# %%
import torch
import torch.nn as nn

from model_utils import TransformerBlock
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_len: int,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(n, seq_len).to(device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
