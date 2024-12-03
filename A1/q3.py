# %% [markdown]
# ## Imports

# %%
import os
import random
from tqdm import tqdm
import numpy as np
from collections import Counter

# import matplotlib.pyplot as plt
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# %%
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize, sent_tokenize

# %% [markdown]
# ## Config

# %%
file_path = "Auguste_Maquet.txt"
random_seed = 42
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
max_length = 64

perplexity_folder = "perplexity"
perplexity_save_path = "2021111013-LM3-{}-perplexity.txt"

# %%
embedding_dim = 300
hidden_dim = 300
lr = 0.0001

batch_size = 64
epochs = 10  # 10 / 1
train_model: bool = True
tqdm_disable: bool = False

model_save_folder = "models"
model_save_path = "language_model_q3.pth"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# %%
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# %%
if not os.path.exists(perplexity_folder):
    os.makedirs(perplexity_folder)
perplexity_save_path = os.path.join(perplexity_folder, perplexity_save_path)

if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)
model_save_path = os.path.join(model_save_folder, model_save_path)

# %% [markdown]
# ## Utils


# %%
def flatten_concatenation(list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
        flat_list += sublist
    return flat_list


# %%
def clean_sentence(text):
    text = text.lower()
    text = " ".join(text.split())
    text = "".join([c for c in text if c.isalnum() or c == " "])

    return text


def sentence_tokenizer(text):
    text = text.strip()
    sentences = sent_tokenize(text)

    sentences = [clean_sentence(s) for s in sentences]
    sentences = [s for s in sentences if s and s != ""]  # remove empty strings

    return sentences


def word_tokenizer(sentence):
    words = word_tokenize(sentence)
    # words = [w for w in words if w.isalnum()]
    return words


# %%
def pad_sequence(sequence, max_len, before=True):
    if before:
        padded_sequence = ["<pad>"] * (max_len - len(sequence)) + sequence
    else:
        padded_sequence = sequence + ["<pad>"] * (max_len - len(sequence))
    return padded_sequence


# %%
class LMDataset(Dataset):
    def __init__(self, data, vocab, max_len, embedding_matrix):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len

        for i in tqdm(range(len(data))):
            d = data[i]
            d = d[: self.max_len]
            d = pad_sequence(d, self.max_len)
            d = [int(self.vocab.get(w, self.vocab["unk"])) for w in d]

            sentence_vectors = torch.zeros(
                (self.max_len - 1, embedding_dim), dtype=torch.float, device=device
            )
            for idx, word in enumerate(d[:-1]):
                sentence_vectors[idx] = embedding_matrix[word]

            self.data.append((sentence_vectors, torch.tensor(d[1:], device=device)))

        print(f"Number of samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# %%
class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix,
        num_layers=2,
        num_heads=4,
        dropout=0.25,
    ):
        super(TransformerLanguageModel, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)

        self.positional_encoding = self.create_positional_encoding(
            max_length, embedding_dim
        ).to(device)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=num_layers
        )
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_length).to(
            device
        )

        self.linear = nn.Linear(hidden_dim, vocab_size)  # Output layer

    def forward(self, x):
        seq_length = x.shape[1]
        positional_encoded = x + self.positional_encoding[:, :seq_length, :]
        padding_mask = (x == 0).all(dim=-1).float()
        tgt_mask = self.tgt_mask[:seq_length, :seq_length]

        output = self.transformer(
            positional_encoded,
            positional_encoded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )
        x = self.linear(output)

        return x

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# %%
def evaluate(model, test_loader, criterion, input_dim):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(test_loader, 0), total=len(test_loader), disable=tqdm_disable
        ):
            x, y = data

            outputs = model(x)

            outputs = outputs.view(-1, input_dim)
            y = y.reshape(-1)

            loss = criterion(outputs, y)

            val_loss += loss.item()

    return val_loss / len(test_loader)


# %%
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    input_dim,
    max_length,
):
    min_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        count = 0
        for i, data in tqdm(
            enumerate(train_loader, 0), total=len(train_loader), disable=tqdm_disable
        ):
            x, y = data

            optimizer.zero_grad()
            outputs = model(x)

            outputs = outputs.view(-1, input_dim)
            y = y.reshape(-1)

            loss = criterion(outputs, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            count += 1

        average_loss = running_loss / count

        print(f"Epoch: {epoch+1}, Train loss: {average_loss}")

        val_loss = evaluate(model, val_loader, criterion, input_dim)
        print(f"Epoch: {epoch+1}, Val loss: {val_loss}", end=" ")

        perplexity = np.exp(val_loss)
        print(f"Perplexity: {perplexity}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Model saved at Epoch: ", epoch + 1)


# %%
def get_perplexity(
    model, data, vocab, filename, criterion, sentences, embedding_matrix
):
    sentence_perplexities = []
    all_sentence_losses = []

    model.eval()
    with torch.no_grad():
        for sentence in tqdm(data, disable=tqdm_disable):
            inputs = sentence[:max_length]
            inputs = pad_sequence(inputs, max_length)
            inputs = [vocab.get(w, vocab["unk"]) for w in inputs]

            sentence_vectors = torch.zeros(
                (max_length - 1, embedding_dim), dtype=torch.float, device=device
            )
            for idx, word in enumerate(inputs[:-1]):
                sentence_vectors[idx] = embedding_matrix[word]

            x = sentence_vectors.unsqueeze(0)
            y = torch.tensor(inputs[1:], device=device).unsqueeze(0)

            outputs = model(x)

            outputs = outputs.view(-1, len(vocab))
            y = y.reshape(-1)

            loss = criterion(outputs, y)
            loss = loss.item()

            sentence_perplexity = np.exp(loss)
            all_sentence_losses.append(loss)
            sentence_perplexities.append(sentence_perplexity)

    with open(filename, "w") as fp:
        for index, p in enumerate(sentence_perplexities):
            fp.write(f"{sentences[index]} :- {p}\n")

        overall_perplexity = np.exp(np.mean(all_sentence_losses))
        fp.write(f"Overall Perplexity: {overall_perplexity}\n")
        print(f"Overall Perplexity: {overall_perplexity}")


# %% [markdown]
# ## Main Code

# %%
with open(file_path, "r") as file:
    text = file.read()

# %%
sentences = sentence_tokenizer(text)
print("Number of sentences: ", len(sentences))

random.shuffle(sentences)

words_sentences = [word_tokenizer(s) for s in sentences]
assert len(words_sentences) == len(sentences)

words = flatten_concatenation(words_sentences)
print("Number of words: ", len(words))
print("Number of unique words: ", len(set(words)))

# %%
word_counts = Counter(words)
assert word_counts.total() == len(words)
print("Most common words: ", word_counts.most_common(5))

# %%
vocab = {"unk": 1, "<pad>": 0}
for index, (word, count) in enumerate(word_counts.items()):
    vocab[word] = len(vocab.keys())

print(len(vocab))

# %%
embedding_matrix = torch.zeros(len(vocab), embedding_dim, dtype=torch.float)

embed_path = "fasttext-wiki-news-subwords-300.bin"
if not os.path.exists(embed_path):
    embedding_model = api.load("fasttext-wiki-news-subwords-300")
    embedding_model.save(embed_path)  # type: ignore
else:
    embedding_model = KeyedVectors.load(embed_path)
unk_embedding = embedding_model["unk"]  # type: ignore

for word, idx in vocab.items():
    if word == "<pad>":
        embedding_matrix[idx] = torch.zeros(embedding_dim, dtype=torch.float)
    elif word in embedding_model:
        embedding_matrix[idx] = torch.tensor(embedding_model[word], dtype=torch.float)  # type: ignore
    else:
        embedding_matrix[idx] = torch.tensor(unk_embedding, dtype=torch.float)

# %%
train_size = int(len(words_sentences) * train_ratio)
val_size = int(len(words_sentences) * val_ratio)

train_data = words_sentences[:train_size]
train_sentences = sentences[:train_size]
val_data = words_sentences[train_size : train_size + val_size]
val_sentences = sentences[train_size : train_size + val_size]
test_data = words_sentences[train_size + val_size :]
test_sentences = sentences[train_size + val_size :]

# %%
train_dataset = LMDataset(train_data, vocab, max_length, embedding_matrix)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Length of train_loader: ", len(train_loader))

val_dataset = LMDataset(val_data, vocab, max_length, embedding_matrix)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Length of val_loader: ", len(val_loader))

test_dataset = LMDataset(test_data, vocab, max_length, embedding_matrix)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Length of test_loader: ", len(test_loader))

# %%
vocab_size = len(vocab)
num_layers = 2

model = TransformerLanguageModel(
    vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_layers=num_layers
).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

# %%
if train_model:
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs,
        vocab_size,
        max_length,
    )

# %%
model = TransformerLanguageModel(
    vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_layers=num_layers
).to(device)
if os.path.exists(model_save_path):
    model.load_state_dict(
        torch.load(model_save_path, weights_only=True, map_location=device)
    )

# %%
# Test the model
test_loss = evaluate(model, test_loader, criterion, vocab_size)

print(f"Test loss: {test_loss}", end=", ")
print(f"Test Perplexity: {np.exp(test_loss)}")

# %%
get_perplexity(
    model,
    train_data,
    vocab,
    perplexity_save_path.format("train"),
    criterion,
    train_sentences,
    embedding_matrix,
)
get_perplexity(
    model,
    val_data,
    vocab,
    perplexity_save_path.format("val"),
    criterion,
    val_sentences,
    embedding_matrix,
)
get_perplexity(
    model,
    test_data,
    vocab,
    perplexity_save_path.format("test"),
    criterion,
    test_sentences,
    embedding_matrix,
)
