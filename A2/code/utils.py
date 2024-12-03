# %%
import re
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import  Dataset

# %%
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

# %%

max_length = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Utils

# %%

def clean_text(text):
    text = str(text).lower().strip()
    text = text.rstrip('\n')
    # text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s.,;!?':()\[\]{}-]", " ", text)  # Keep selected punctuation marks, symbols and apostrophes
    text = re.sub(r"\s+", " ", text)

    text = text.encode("utf-8", errors="ignore").decode("utf-8")  # Corrected encoding

    return text

def clean_sentences(sentences):
    sentences = [clean_text(sentence) for sentence in sentences]
    sentences = [s for s in sentences if s and s != ""]  # remove empty strings
    return sentences

# %%
def read_data(en_path, fr_path):
    with open(en_path, "r") as f:
        en_data = f.readlines()
    with open(fr_path, "r") as f:
        fr_data = f.readlines()

    assert len(en_data) == len(fr_data), "Data mismatch"

    en_data = clean_sentences(en_data)
    fr_data = clean_sentences(fr_data)

    assert len(en_data) == len(fr_data), "Data mismatch in cleaned data"

    return en_data, fr_data

def word_tokenizer(sentence):
    words = word_tokenize(sentence)
    return words

# %%
def flatten_concatenation(list_of_lists, unique=False):
    # flat_list = []
    # for sublist in list_of_lists:
    #     flat_list += sublist

    # flat_list = list(set(flat_list))
    # return flat_list
    flat_array = np.concatenate(list_of_lists)
    if unique:
        flat_list = np.unique(flat_array).tolist()
    else:
        flat_list = flat_array.tolist()
    return flat_list

# %%
def reverse_vocab(vocab):
    return {v: k for k, v in vocab.items()}

# %%
def return_words_till_EOS(lst, eos=2):
    if eos not in lst:
        return lst
    return lst[:lst.index(eos)]

# %% [markdown]
# ### Dataset

# %%
def pad_sequence(sequence, max_len, before=True, pad_token=0):
    if len(sequence) > max_len:
        return sequence[:max_len]
    elif before:
        return [pad_token] * (max_len - len(sequence)) + sequence
    else:
        return sequence + [pad_token] * (max_len - len(sequence))

# %%
class MyDataset(Dataset):
    def __init__(
        self,
        en_data,
        fr_data,
        en_vocab,
        fr_vocab,
        pad_before=False,
    ):
        self.en_data = []
        self.fr_data = []
        self.labels = []
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

        assert len(en_data) == len(fr_data)
        self.length = len(en_data)

        en_pad = self.en_vocab["<pad>"]
        en_unk = self.en_vocab["<unk>"]
        en_sos = self.en_vocab["<sos>"]
        en_eos = self.en_vocab["<eos>"]
        fr_pad = self.fr_vocab["<pad>"]
        fr_unk = self.fr_vocab["<unk>"]
        fr_sos = self.fr_vocab["<sos>"]
        fr_eos = self.fr_vocab["<eos>"]

        tqdm_obj = tqdm(
            total=self.length, desc="Creating dataset"
        )
        for index, (en_sentence, fr_sentence) in enumerate(zip(en_data, fr_data)):
            en_indices = [int(self.en_vocab.get(w, en_unk)) for w in en_sentence]
            en_indices = [en_sos] + en_indices[: max_length - 2] + [en_eos]
            en_indices = pad_sequence(
                en_indices, max_length, before=pad_before, pad_token=en_pad
            )
            self.en_data.append(
                torch.tensor(en_indices, dtype=torch.int, device=device)
            )

            fr_indices1 = [int(self.fr_vocab.get(w, fr_unk)) for w in fr_sentence]
            fr_indices = [fr_sos] + fr_indices1
            fr_indices = pad_sequence(
                fr_indices, max_length, before=pad_before, pad_token=fr_pad
            )
            self.fr_data.append(
                torch.tensor(fr_indices, dtype=torch.int, device=device)
            )

            fr_indices = fr_indices1 + [fr_eos]
            fr_indices = pad_sequence(
                fr_indices, max_length, before=pad_before, pad_token=fr_pad
            )
            self.labels.append(torch.tensor(fr_indices, device=device))

            if index % 10 == 0:
                tqdm_obj.update(10)

        tqdm_obj.close()

        print(f"Dataset created with {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.en_data[idx], self.fr_data[idx], self.labels[idx]


# %% [markdown]
# ### Model

# %%
def create_positional_encoding(max_length, embedding_dim):
    pe = torch.zeros(max_length, embedding_dim)
    position = torch.arange(0, max_length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe.to(device)


def make_src_mask(src):
    src1 = src
    if len(src.shape) == 3:
        src1 = torch.sum(src, dim=-1)

    src_mask = (src1 != 0).unsqueeze(1).unsqueeze(2)
    return src_mask.to(device)


def make_trg_mask(trg):
    trg1 = trg
    if len(trg.shape) == 3:
        trg1 = torch.sum(trg, dim=-1)
    
    n, trg_len = trg1.size()
    trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
    return trg_mask.to(device)