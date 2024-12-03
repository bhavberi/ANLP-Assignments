# %%
import os
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import profiler
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% [markdown]
# ## Config

# %%
test_dataset_path = "./ptb.test.txt"

# %%
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "openlm-research/open_llama_3b_v2"
# model_name = "EleutherAI/gpt-neo-125m"
model_name = "allenai/OLMo-1B-hf"

# %%
partial_no_layers_quantise = 12

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# %%
os.makedirs("models", exist_ok=True)

# %% [markdown]
# ## Utils


# %%
def read_data(filepath, limit=None):
    with open(filepath, "r") as f:
        data = f.readlines()
    data = [line.strip().replace("\n", "<eos>") for line in data]
    if limit:
        data = data[:limit]
    data = "\n".join(data)
    return data


# %%
def calculate_perplexity(model, encodings):
    max_length = model.config.max_position_embeddings
    stride = max_length // 2  # To avoid too much truncation
    nlls = []

    encodings = encodings.to(device)

    with profiler.profile(
        use_device=str(device), use_cpu=False, use_kineto=True
    ) as prof:
        for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
            begin_loc = i
            end_loc = min(i + max_length, encodings.input_ids.size(1))
            trg_len = end_loc - begin_loc  # Target length
            input_ids = encodings.input_ids[:, begin_loc:end_loc]

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

    perplexity = torch.exp(torch.stack(nlls).sum() / end_loc)
    profiler_obj = prof.total_average()

    return (
        perplexity.item(),
        profiler_obj,
        len(range(0, encodings.input_ids.size(1), stride)),
    )


# %%
class INT8Layer(nn.Module):
    def __init__(self, input_features, output_features, bias=True, dtype=torch.float32):
        super(INT8Layer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.dtype = dtype

        self.register_buffer(
            "weight",
            torch.randint(
                -128, 127, (output_features, input_features), dtype=torch.int8
            ),
        )
        self.register_buffer("scales", torch.randn((output_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, output_features), dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def forward(self, input):
        quantized_weights = self.weight.to(input.dtype)
        output = F.linear(input, quantized_weights) * self.scales

        if self.bias is not None:
            output += self.bias

        return output

    def quantize(self, weights, bias):
        if bias is not None:
            self.bias = bias.clone()

        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)
        self.weight = (weights / scales.unsqueeze(-1)).to(torch.int8)
        self.scales = scales


# %%
# def replace_linear_layer(model, layer):
#     torch.cuda.empty_cache()
#     for name, child in model.named_children():
#         if isinstance(child, nn.Linear):
#             og_bias = child.bias
#             og_weights = child.weight

#             with torch.no_grad():
#                 new_layer = layer(
#                     child.in_features,
#                     child.out_features,
#                     bias=og_bias is not None,
#                     dtype=og_weights.dtype,
#                 )

#             setattr(model, name, new_layer)
#             getattr(model, name).quantize(og_weights, og_bias)
#         else:
#             print("Calling", name)
#             replace_linear_layer(child, layer)


# %%
def replace_linear_layer(model, layer, partial=False, cont=False, quantise_n=8):
    torch.cuda.empty_cache()
    for name, child in model.named_children():
        cont1 = True
        if partial and not cont:
            cont1 = name.isdigit() and int(name) < quantise_n
        if cont1 and isinstance(child, nn.Linear):
            og_bias = child.bias
            og_weights = child.weight

            with torch.no_grad():
                new_layer = layer(
                    child.in_features,
                    child.out_features,
                    bias=og_bias is not None,
                    dtype=og_weights.dtype,
                )

            setattr(model, name, new_layer)
            getattr(model, name).quantize(og_weights, og_bias)
        else:
            replace_linear_layer(
                child, layer, partial=partial, cont=cont1, quantise_n=quantise_n
            )


# %%
def testing(model, test_encodings, desc=""):
    # Run the perplexity calculation and profiling
    perplexity, profiler_obj, n_items = calculate_perplexity(model, test_encodings)
    memory_footprint_before_quantization = model.get_memory_footprint() / 1e6

    cuda_time_ms = profiler_obj.device_time / 1e3
    inference_latency_ms = cuda_time_ms / n_items

    print(desc)
    print(f"Perplexity: {perplexity}")
    print(f"Cuda Time: {cuda_time_ms:.4f} ms")
    print(f"Inference Latency: {inference_latency_ms:.4f} ms per inference")
    print(f"Memory Footprint: {memory_footprint_before_quantization:.2f} MB")


# %% [markdown]
# ## Main

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# %%
test_data = read_data(test_dataset_path)
test_encodings = tokenizer(test_data, return_tensors="pt")

# %%
desc = "Before Quantization:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained("models/model_before_quantization")

# %%
del model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# %%
replace_linear_layer(model, INT8Layer)

# %%
desc = "After Full Quantization:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained("models/model_after_full_quantization")

# %%
del model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# %%
replace_linear_layer(
    model, INT8Layer, partial=True, quantise_n=partial_no_layers_quantise
)

# %%
desc = f"After Partial Quantization on first {partial_no_layers_quantise} Layers:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained("models/model_after_partial_quantization")
