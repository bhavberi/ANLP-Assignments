# %%
import os
from tqdm import tqdm

import torch
from torch.autograd import profiler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%
# from huggingface_hub import notebook_login
# notebook_login()

# %% [markdown]
# ## Config

# %%
test_dataset_path = "./ptb.test.txt"

# %%
model_name = "allenai/OLMo-1B-hf"

# %%
bit4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

bit8_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

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
del model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bit8_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# %%
desc = "After 8-bit Quantization:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained("models/model_after_8-bit_quantization")

# %%
del model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bit4_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# %%
desc = "After 4-bit Quantization:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained("models/model_after_4-bit_quantization")

# %%
del model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# %%
desc = "After NF4-bit Quantization:"
testing(model, test_encodings, desc)

# %%
model.save_pretrained(
    "models/model_after_nf-4_quantization"
)  # , push_to_hub=True, repo_id="bhavberi/OLMo-1B-NF4")

# %%
model = AutoModelForCausalLM.from_pretrained(
    "bhavberi/OLMo-1B-NF4",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# %%
desc = "After NF4-bit Quantization HF:"
testing(model, test_encodings, desc)
