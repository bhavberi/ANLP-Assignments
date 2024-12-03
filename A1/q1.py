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

perplexity_folder = "perplexity"
perplexity_save_path = "2021111013-LM1-{}-perplexity.txt"

# %%
embedding_dim = 300
hidden_dim = 300
lr = 0.005 # 0.001
momentum = 0.9

batch_size = 64
epochs = 30  # 30/1
print_every = 250
train_model: bool = True
tqdm_disable: bool = False

model_save_folder = "models"
model_save_path = "language_model_q1.pth"

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
def five_gram_tokenizer(words_sentences):
    five_grams_sentences = []
    ignored_sentences = 0
    for sentence in tqdm(words_sentences, disable=tqdm_disable):
        five_grams = [
            (sentence[i : i + 5], sentence[i + 5]) for i in range(len(sentence) - 5)
        ]
        if len(five_grams) > 0:
            five_grams_sentences.append(five_grams)
        else:
            ignored_sentences += 1
    
    print(f"Ignored {ignored_sentences} sentences due to length")
    return five_grams_sentences

# %%
class LMDataset(Dataset):
    def __init__(self, words_sentences, vocab):
        data = five_gram_tokenizer(words_sentences)
        self.data = flatten_concatenation(data)
        self.vocab = vocab

        print(f"Number of samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = [self.vocab.get(w, self.vocab["unk"]) for w in x]
        y = self.vocab.get(y, self.vocab["unk"])
        return torch.tensor(x), torch.tensor(y)

# %%
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False

        self.fc1 = nn.Linear(
            embedding_dim * 5, hidden_dim, dtype=torch.float
        )  # First hidden layer
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(
            hidden_dim, hidden_dim, dtype=torch.float
        )  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, vocab_size, dtype=torch.float)  # Output layer

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply the first hidden layer
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  # Apply the second hidden layer
        x = self.fc3(x)  # Output
        # x = F.log_softmax(x, dim=1)  # Output with log soft max
        return x

# %%
def evaluate(model, test_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), disable=tqdm_disable):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    print_every: int | None = 200,
):
    min_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if print_every and i % print_every == print_every - 1:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss / print_every}")
                running_loss = 0.0

        val_loss = evaluate(model, val_loader, criterion)
        perplexity = np.exp(val_loss)

        if print_every:
            print(f"Epoch: {epoch+1}, Val loss: {val_loss}", end=" ")
            print(f"Perplexity: {perplexity}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            if print_every:
                print("Model saved at Epoch: ", epoch + 1)

# %%
def get_perplexity(model, data, vocab, filename, criterion, sentences):
    sentence_perplexities = []

    model.eval()
    with torch.no_grad():
        five_gram_sentences = five_gram_tokenizer(data)
        all_sentence_losses = []
        for five_gram_sentence in tqdm(five_gram_sentences, disable=tqdm_disable):
            running_loss = 0.0
            for five_gram in five_gram_sentence:
                inputs, labels = five_gram
                inputs = [vocab.get(w, vocab["unk"]) for w in inputs]
                labels = vocab.get(labels, vocab["unk"])

                inputs = torch.tensor(inputs).unsqueeze(0).to(device)
                labels = torch.tensor(labels).unsqueeze(0).to(device)
                # To bring batch size of 1

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            sentence_perplexity = np.exp(running_loss / len(five_gram_sentence))
            all_sentence_losses.append(running_loss / len(five_gram_sentence))
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
vocab = {"unk": 0}
for index, (word, count) in enumerate(word_counts.items()):
    vocab[word] = index + 1

print(len(vocab))

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
train_dataset = LMDataset(train_data, vocab)
val_dataset = LMDataset(val_data, vocab)
test_dataset = LMDataset(test_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
embedding_matrix = torch.zeros(len(vocab), embedding_dim, dtype=torch.float)

if not os.path.exists("fasttext-wiki-news-subwords-300.bin"):
    embedding_model = api.load("fasttext-wiki-news-subwords-300")
    embedding_model.save("fasttext-wiki-news-subwords-300.bin")  # type: ignore
else:
    embedding_model = KeyedVectors.load("fasttext-wiki-news-subwords-300.bin")

unk_embedding = embedding_model["unk"]  # type: ignore

for word, idx in vocab.items():
    if word in embedding_model:
        embedding_matrix[idx] = torch.tensor(embedding_model[word], dtype=torch.float)  # type: ignore
    else:
        embedding_matrix[idx] = torch.tensor(unk_embedding, dtype=torch.float)
    #     embedding_matrix.append([0]*300)

# %%
vocab_size = len(vocab)

model = LanguageModel(vocab_size, embedding_dim, hidden_dim, embedding_matrix).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss().to(device)

# criterion = nn.NLLLoss().to(device) # Use softmax in the model

# %%
if train_model:
    train(model, train_loader, val_loader, criterion, optimizer, epochs, print_every)

# %%
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, embedding_matrix).to(device)
if os.path.exists(model_save_path):
    model.load_state_dict(
    torch.load(model_save_path, weights_only=True, map_location=device)
)

# %%
# Test the model
test_loss = evaluate(model, test_loader, criterion)

print(f"Test loss: {test_loss}")
print(f"Test Perplexity: {np.exp(test_loss)}")

# %%
get_perplexity(model, train_data, vocab, perplexity_save_path.format("train"), criterion, train_sentences)
get_perplexity(model, val_data, vocab, perplexity_save_path.format("val"), criterion, val_sentences)
get_perplexity(model, test_data, vocab, perplexity_save_path.format("test"), criterion, test_sentences)


