# %%
import os
import json
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from encoder import Encoder
from decoder import Decoder
from utils import (
    make_src_mask,
    make_trg_mask,
    pad_sequence,
    return_words_till_EOS,
    reverse_vocab,
    create_positional_encoding,
    word_tokenizer,
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        embed_size: int = 512,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 50,
        save_path=None,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
        )
        self.best_val_loss = float("inf")
        self.save_path = save_path

    def forward(self, src, trg):
        src_mask = make_src_mask(src)
        trg_mask = make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

    def fit(self, train_loader, val_loader, criterion, optimizer, num_epochs: int = 10):
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for src, trg, label in tqdm(train_loader, total=len(train_loader)):
                optimizer.zero_grad()
                output = self(src, trg)
                output = output.reshape(-1, output.size(-1))
                label = label.reshape(-1)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")

            if device == "cuda":
                torch.cuda.empty_cache()

            val_loss = self.evaluate(val_loader, criterion, True)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss}")

            if self.save_path and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.state_dict(), self.save_path)

        return train_losses, val_losses

    def evaluate(self, val_loader, criterion, tqdm_disabled: bool = False):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg_input, trg_target in tqdm(
                val_loader, total=len(val_loader), disable=tqdm_disabled
            ):
                output = self(src, trg_input)
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                trg_target = trg_target.view(-1)
                loss = criterion(output, trg_target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        return val_loss

    def load(self, path=None):
        if path:
            self.load_state_dict(torch.load(path))
        elif self.save_path:
            self.load_state_dict(torch.load(self.save_path))
        else:
            raise ValueError("No model path provided")

    def predict(
        self,
        src,
        src_preprocessed=False,
        max_len=64,
        return_sentence=False,
        fr_vocab=None,
        en_vocab=None,
        start_token_idx=1,
        end_token_idx=2,
        padding_before=False,
    ):
        self.eval()  # Set the model to evaluation mode

        if not src_preprocessed:
            assert en_vocab is not None
            src = word_tokenizer(src)
            src = [en_vocab.get(w, en_vocab["<unk>"]) for w in src]
            src = [start_token_idx] + src[: max_len - 2] + [end_token_idx]
            src = pad_sequence(src, max_len, before=padding_before, pad_token=0)
            src = torch.tensor([src], dtype=torch.int, device=device)

        trg = torch.tensor([[start_token_idx]], dtype=torch.int, device=device)

        src_mask = make_src_mask(src)
        with torch.no_grad():
            enc_src = self.encoder(src, src_mask)

        for _ in range(max_len):
            trg_mask = make_trg_mask(trg)

            with torch.no_grad():
                output = self.decoder(trg, enc_src, src_mask, trg_mask)
                output = output[:, -1]

            next_token = output.argmax(-1).unsqueeze(0)
            trg = torch.cat((trg, next_token), dim=1)

            if next_token.item() == end_token_idx:
                break

        generated_sequence = trg.squeeze(0).tolist()[1:]
        if generated_sequence[-1] == 2:
            generated_sequence = generated_sequence[:-1]

        if return_sentence:
            assert fr_vocab is not None
            fr_vocab_rev = reverse_vocab(fr_vocab)
            generated_sequence = [fr_vocab_rev[idx] for idx in generated_sequence]

        return generated_sequence

    def test(self, test_loader, en_vocab, fr_vocab, test_en, test_fr, max_length=64, padding_before=False):
        self.eval()
        bleu_scores = []
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        predicts = []
        references = []
        candidates = []

        reverse_fr_vocab = reverse_vocab(fr_vocab)
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        with torch.no_grad():
            for src, _, label in tqdm(test_loader, total=len(test_loader)):
                for i in range(src.size(0)):
                    src_i = src[i].unsqueeze(0)
                    trg_target_i = label[i].unsqueeze(0)

                    candidate = self.predict(
                        src_i,
                        src_preprocessed=True,
                        max_len=max_length,
                        fr_vocab=fr_vocab,
                        start_token_idx=en_vocab["<sos>"],
                        end_token_idx=en_vocab["<eos>"],
                        padding_before=padding_before,
                    )
                    candidate = [reverse_fr_vocab[idx] for idx in candidate]

                    reference = return_words_till_EOS(
                        trg_target_i.squeeze(0).tolist(), eos=fr_vocab["<eos>"]
                    )
                    reference = [reverse_fr_vocab[idx] for idx in reference]

                    bleu_score = sentence_bleu(
                        [reference],
                        candidate,
                        smoothing_function=SmoothingFunction().method1,
                    )
                    bleu_scores.append(bleu_score)

                    candidate_str = " ".join(candidate)
                    reference_str = " ".join(reference)

                    predicts.append(candidate_str)
                    candidates.append(candidate)
                    references.append([reference])

                    rouge_score = scorer.score(reference_str, candidate_str)
                    rouge_scores["rouge1"].append(rouge_score["rouge1"].fmeasure)
                    rouge_scores["rouge2"].append(rouge_score["rouge2"].fmeasure)
                    rouge_scores["rougeL"].append(rouge_score["rougeL"].fmeasure)

        os.makedirs("../results", exist_ok=True)

        overall_results = []
        for i in range(len(test_loader)):
            overall_results.append(
                {
                    "src": test_en[i],
                    "expected": test_fr[i],
                    "predicted": predicts[i],
                    "bleu": bleu_scores[i],
                    "rouge1": rouge_scores["rouge1"][i],
                    "rouge2": rouge_scores["rouge2"][i],
                    "rougeL": rouge_scores["rougeL"][i],
                }
            )
        with open("../results/overall_results.json", "w") as f:
            json.dump(overall_results, f, indent=4)

        with open("../results/testbleu.txt", "w") as f:
            for i in range(len(test_loader)):
                f.write(f"{test_en[i]} {bleu_scores[i]}\n")

        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"Test BLEU Score: {np.mean(bleu_scores)}")
        print(f"Corpus BLEU Score: {corpus_bleu(references, candidates)}")
        print(f"Test ROUGE-1 Score: {np.mean(rouge_scores['rouge1'])}")
        print(f"Test ROUGE-2 Score: {np.mean(rouge_scores['rouge2'])}")
        print(f"Test ROUGE-L Score: {np.mean(rouge_scores['rougeL'])}")

        return (
            np.mean(bleu_scores),
            np.mean(rouge_scores["rouge1"]),
            np.mean(rouge_scores["rouge2"]),
            np.mean(rouge_scores["rougeL"]),
        )
