# %%
import os
import random
import numpy as np
import pickle as pkl
from torchinfo import summary
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# %%
from transformer import Transformer
from utils import (
    read_data,
    word_tokenizer,
    MyDataset,
    flatten_concatenation,
)

# %%
en_train = "../ted-talks-corpus/train.en"
fr_train = "../ted-talks-corpus/train.fr"
en_val = "../ted-talks-corpus/dev.en"
fr_val = "../ted-talks-corpus/dev.fr"
en_test = "../ted-talks-corpus/test.en"
fr_test = "../ted-talks-corpus/test.fr"

# %%
padding_before = False
plot_losses = False

# %%
embedding_dim = 300
max_length = 64
lr = 1e-4

heads = 6
layers = 6

epochs = 15
batch_size = 32

# %%
os.makedirs("../models", exist_ok=True)

save_path = "../models/transformer"

save_path = save_path + f"_heads{heads}_layers{layers}"

save_path = save_path + ".pth"

print(f"Saving model to {save_path}")

# %%
random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print("Using Random Seed:", random_seed)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# %% [markdown]
# ## Main

# %%
train_en, train_fr = read_data(en_train, fr_train)
val_en, val_fr = read_data(en_val, fr_val)
test_en, test_fr = read_data(en_test, fr_test)

# %%
train_en_words = [word_tokenizer(s) for s in train_en]
train_fr_words = [word_tokenizer(s) for s in train_fr]
val_en_words = [word_tokenizer(s) for s in val_en]
val_fr_words = [word_tokenizer(s) for s in val_fr]
test_en_words = [word_tokenizer(s) for s in test_en]
test_fr_words = [word_tokenizer(s) for s in test_fr]

all_en_words = flatten_concatenation(train_en_words + val_en_words + test_en_words)
all_fr_words = flatten_concatenation(train_fr_words + val_fr_words + test_fr_words)

# %%
en_word_counts = Counter(all_en_words)
assert en_word_counts.total() == len(all_en_words)
fr_word_counts = Counter(all_fr_words)
assert fr_word_counts.total() == len(all_fr_words)

en_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
fr_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

for word, count in en_word_counts.items():
    # if count > 1:
    en_vocab[word] = len(en_vocab.keys())

for word, count in fr_word_counts.items():
    # if count > 1:
    fr_vocab[word] = len(fr_vocab.keys())

# %%

# Load vocab
with open("../vocab/en_vocab.pkl", "rb") as f:
    en_vocab = pkl.load(f)

with open("../vocab/fr_vocab.pkl", "rb") as f:
    fr_vocab = pkl.load(f)

# %%
train_dataset = MyDataset(
    train_en_words,
    train_fr_words,
    en_vocab,
    fr_vocab,
    padding_before,
)
val_dataset = MyDataset(
    val_en_words,
    val_fr_words,
    en_vocab,
    fr_vocab,
    padding_before,
)
test_dataset = MyDataset(
    test_en_words,
    test_fr_words,
    en_vocab,
    fr_vocab,
    padding_before,
)

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)
print("Length of train_loader:", len(train_loader))
print("Length of val_loader:", len(val_loader))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print("Length of test_loader:", len(test_loader))

# %%
model = Transformer(
    len(en_vocab),
    len(fr_vocab),
    embed_size=embedding_dim,
    num_layers=layers,
    heads=heads,
    forward_expansion=4,
    dropout=0.2,
    max_len=max_length,
    save_path=save_path,
).to(device)

# %%
print(model)
summary(
    model,
    input_data=[
        torch.randint(
            low=0, high=len(en_vocab), size=(batch_size, max_length), device=device
        ),
        torch.randint(
            low=0, high=len(fr_vocab), size=(batch_size, max_length), device=device
        ),
    ],
)

# %%
optimizer = optim.Adam(model.parameters(), lr=lr)  # type: ignore
criterion = nn.CrossEntropyLoss(ignore_index=0)

# %%
train_losses, val_losses = model.fit(
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=epochs,
)

if plot_losses:
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.show()

# %%
model.load()

# %%
model.evaluate(test_loader, criterion)
