# %%
import os
import re
import csv
import time
import random
import numpy as np
from tqdm import tqdm
from itertools import islice
from torchinfo import summary as modelinfo
# import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


try: 
    from rouge_score import rouge_scorer
except:
    # %pip install rouge-score==0.1.2
    # from rouge_score import rouge_scorer
    print("Rouge Score not available")
    exit(1)
    
try:
    from peft import get_peft_model, LoraConfig, TaskType
except:
    # %pip install peft==0.13.2
    # from peft import get_peft_model, LoraConfig, TaskType
    print("PEFT not available")

# %% [markdown]
# ## Config

# %%
train_filepath = "./cnn_dailymail/train.csv"
val_filepath = "./cnn_dailymail/validation.csv"
test_filepath = "./cnn_dailymail/test.csv"

# train_filepath = "/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv"
# val_filepath = "/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv"
# test_filepath = "/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv"

max_train_samples = 5000
max_val_samples = 1000
max_test_samples = 1000

# %%
to_train = False
model_name = "EleutherAI/gpt-neo-125m"
tuning_type = "lora"

lr = 6e-5
epochs = 5
mini_batch_size = 32

# For LORA
r = 16
alpha = 32

# %%
assert tuning_type in ["none", "last", "lora"], "Invalid tuning type"
if tuning_type == "none": to_train=False
os.makedirs("models", exist_ok=True)

# %%
random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

print("Using Random Seed:", random_seed)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# %% [markdown]
# ## Utils

# %%
dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"

# acceptable ways to end a sentence
END_TOKENS = [
    ".",
    "!",
    "?",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]

# %%
def remove_period(line):
    if line[-1] in END_TOKENS:
        return line[:-1]
    return line


def remove_punctuations(line):
    return re.sub(r"[^\w\s]", " ", line)


def clean_data(data):
    for i in range(len(data)):
        data[i]["article"] = remove_punctuations(data[i]["article"])
        data[i]["highlights"] = remove_punctuations(data[i]["highlights"])

        data[i]["article"] = re.sub(r"\s+", " ", data[i]["article"]).strip()
        data[i]["highlights"] = re.sub(r"\s+", " ", data[i]["highlights"]).strip()

    return data

# %%
def read_data(filepath, max_length=None):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        if max_length is not None:
            rows = islice(reader, max_length)
        else:
            rows = reader
        data = list(rows)
    
    return clean_data(data)

# %%
def freeze_last_layer(model):
    assert tuning_type == "last", "Only last layer fine-tuning is supported"

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.lm_head.parameters():
        param.requires_grad = True

def lora(model):
    assert tuning_type == "lora"

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        bias="none"
    )

    model = get_peft_model(model, config)

    return model

# %%
def make_tokenizer(model_name):
    """Returns GPT2 tokenizer after adding separator and padding tokens"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    special_tokens = {"pad_token": "<pad>", "sep_token": "<sep>"}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def make_model(model_name, len_tokenizer, tuning_type="last"):
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len_tokenizer)

    if tuning_type == "last":
        freeze_last_layer(model)
    elif tuning_type == "lora":
        model = lora(model)

    model.to(device)
    return model

# %%
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, article_max_length=512, summary_max_length=128, type="train"):
        assert type in ["train", "val", "test"], "Invalid dataset type"

        self.type = type
        if type == "test":
            summary_max_length = 0

        self.tokenizer = tokenizer
        self.article_max_length = article_max_length
        self.summary_max_length = summary_max_length
        self.instruction_tokens = self.tokenizer.encode("summarize: ")
        self.sep_token = self.tokenizer.encode(" " + self.tokenizer.sep_token + " ")
        self.max_length = article_max_length + summary_max_length + len(self.instruction_tokens) + len(self.sep_token)
        
        self.processed_data = self._process_all_data(data, 4)

    def _attention_mask(self, padding_length):
        return [1] * (self.max_length - padding_length) + [0] * padding_length

    def _process_data(self, data):
        article_ids = self.tokenizer.encode(
            data["article"], truncation=True, max_length=self.article_max_length
        )
        if self.type != "test":
            abstract_ids = self.tokenizer.encode(
                data["highlights"], truncation=True, max_length=self.summary_max_length
            )
        else:
            abstract_ids = []

        # Combine all components
        content = self.instruction_tokens + article_ids + self.sep_token + abstract_ids

        if self.type != "test":
            padding_length = self.max_length - len(content)
            padded_content = content + [self.tokenizer.pad_token_id] * padding_length
        else:
            padding_length = self.max_length - len(content)
            padded_content = [self.tokenizer.pad_token_id] * padding_length + content

        return {
            "text": padded_content,
            "sep_idx": len(article_ids),
            "article_len": len(article_ids),
            "summary_len": len(abstract_ids),
            "attention_mask": self._attention_mask(padding_length),
            "highlights": data["highlights"],
        }
    
    def _process_all_data(self, data, num_workers):
        processed_data = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the thread pool
            futures = [executor.submit(self._process_data, item) for item in data]
            
            # Use tqdm to track the progress
            for future in tqdm(as_completed(futures), total=len(data), desc="Processing Data"):
                processed_data.append(future.result())

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        processed_item = self.processed_data[idx]
        return {
            "article": torch.tensor(processed_item["text"]),
            "sep_idx": processed_item["sep_idx"],
            "article_len": processed_item["article_len"],
            "summary_len": processed_item["summary_len"],
            "attention_mask": torch.tensor(processed_item["attention_mask"]),
            "highlights": processed_item["highlights"],
        }

# %%
def evaluate(model, val_loader, loss_fn, summary_max_length=128):
    """
    Evaluate the model on validation/test data

    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        loss_fn: Loss function (typically CrossEntropyLoss with ignore_index set to pad_token_id)

    Returns:
        float: Average loss per batch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch["article"].to(device)
            sep_idx = batch["sep_idx"].squeeze()
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            # decoded_output = tokenizer.decode(outputs.logits[0].argmax(dim=-1))
            # decoded_input = tokenizer.decode(inputs[0])
            # print(decoded_input, decoded_output, sep="\n\n")

            # Get logits and labels for summary portion only
            shift_logits = outputs.logits[..., sep_idx:-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = inputs[..., sep_idx + 1 :].contiguous()
            shift_labels = shift_labels.view(-1)

            shift_logits = shift_logits[:summary_max_length]
            shift_labels = shift_labels[:summary_max_length]
            
            # print()
            # print(tokenizer.decode(shift_logits.argmax(dim=-1)))
            # print(shift_labels)

            loss = loss_fn(shift_logits, shift_labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / (num_batches if num_batches > 0 else 1)

# %%
def train(
    model,
    optimiser,
    loss_fn,
    train_loader,
    val_loader,
    num_epochs,
    max_grad_norm=1.0,
    mini_batch_size=4,
    summary_max_length=128,
    save_path="models/model.pt",
    print_every=5,
):
    global_step = 0
    best_val_loss = float("inf")
    tr_loss = 0.0
    last_tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(epoch_iterator):
            inputs = batch["article"].to(device)
            sep_idx = batch["sep_idx"].squeeze()
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, labels=inputs, attention_mask=attention_mask)

            shift_logits = outputs.logits[..., sep_idx:-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = inputs[..., sep_idx + 1 :].contiguous()
            shift_labels = shift_labels.view(-1)

            shift_logits = shift_logits[:summary_max_length]
            shift_labels = shift_labels[:summary_max_length]

            loss = loss_fn(shift_logits, shift_labels)
            loss /= mini_batch_size
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % mini_batch_size == 0:
                optimiser.step()
                model.zero_grad()
                global_step += 1

                if global_step % print_every == 0:
                    log_loss = (tr_loss - logging_loss) / mini_batch_size
                    logging_loss = tr_loss

                    print(f"Step: {global_step}, Mini-Batch Loss: {log_loss:.4f}")

        print(
            f"Epoch: {epoch+1}, Avg Training Loss: {(tr_loss - last_tr_loss) / len(train_loader):.4f}/sample",
            end=", ",
        )
        last_tr_loss = tr_loss

        val_loss = evaluate(model, val_loader, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Validation Loss: {val_loss:.4f}")

# %%
def predict(
    model,
    tokenizer,
    text,
    article_max_length=512,
    max_length=128,
    preprocess=True,
    sep_idx=None,
):
    model.eval()
    if preprocess:
        text_ids = (
            tokenizer.encode("summarize: ")
            + tokenizer.encode(text)[:article_max_length]
            + tokenizer.encode(tokenizer.sep_token)
        )
        sep_idx = len(text_ids) - 1
        inputs = torch.tensor(text_ids).unsqueeze(0).to(device)
    else:
        assert sep_idx is not None, "sep_idx must be provided if preprocess is False"
        assert isinstance(sep_idx, torch.Tensor), "sep_idx must be a list"
        inputs = text.to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
        )

    if preprocess:
        generated_summary = tokenizer.decode(
            outputs[0][sep_idx + 1 :], skip_special_tokens=True
        )
        return generated_summary
    else:
        generated_summaries = []
        for i in range(len(sep_idx)):
            sep_idx_i = sep_idx[i].item()
            generated_summary = tokenizer.decode(outputs[i][sep_idx_i + 1 :], skip_special_tokens=True)
            generated_summaries.append(generated_summary)

        return generated_summaries

# %%
def test_score(model, tokenizer, test_loader, article_max_length=512, max_length=128, print_every=False):
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i in tqdm(test_loader):
        text = i["article"]
        summary = i["highlights"]

        generated_summary = predict(
            model,
            tokenizer,
            text,
            article_max_length,
            max_length,
            preprocess=False,
            sep_idx=i["sep_idx"],
        )

        for j in range(len(summary)):
            scores = rouge_scorer_obj.score(summary[j], generated_summary[j])
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
            
            if print_every:
                print(scores)

    for key in rouge_scores:
        rouge_scores[key] = np.mean(rouge_scores[key])

    return rouge_scores

# %% [markdown]
# ## Main

# %%
train_data = read_data(train_filepath, max_train_samples)
val_data = read_data(val_filepath, max_val_samples)
test_data = read_data(test_filepath, max_test_samples)

# %%
tokenizer = make_tokenizer(model_name)
len_tokenizer = len(tokenizer)

model = make_model(model_name, len_tokenizer, tuning_type)

# %%
modelinfo(model)

# %%
if to_train:
    train_dataset = SummarizationDataset(train_data, tokenizer, type="train")
    val_dataset = SummarizationDataset(val_data, tokenizer, type="val")
test_dataset = SummarizationDataset(test_data, tokenizer, type="test")

if to_train:
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
optimiser = optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# %%
if to_train:
    start_time = time.time()
    train(
        model,
        optimiser,
        loss_fn,
        train_loader,
        val_loader,
        num_epochs=epochs,
        mini_batch_size=mini_batch_size,
        save_path=f"models/model_{tuning_type}.pt",
        print_every=40,
    )
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total Training Time: {total_training_time / 60:.2f} minutes\n")

if os.path.exists(f"models/model_{tuning_type}.pt"):
    model.load_state_dict(torch.load(f"models/model_{tuning_type}.pt", map_location=device, weights_only=True))
    print("Model loaded successfully")
else:
    print("No model checkpoint found")

# %%
# # Predict first 5 examples from test data

# for i in range(5):
#     text = test_data[i]["article"]
#     summary = test_data[i]["highlights"]
#     generated_summary = predict(model, tokenizer, text)

#     print(f"Example {i+1}")
#     print("Text:", text)
#     print("Actual Summary:", summary)
#     print("Generated Summary:", generated_summary)
#     print("\n")

# %%
print("Test Scores:", test_score(model, tokenizer, test_loader, print_every=False))

# %%
print(f"Allocated Memory: {torch.cuda.memory_allocated() / (1024 ** 3):.4f} GB")
print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.4f} GB")
print(f"Reserved Memory: {torch.cuda.memory_reserved() / (1024 ** 3):.4f} GB")
print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / (1024 ** 3):.4f} GB")
