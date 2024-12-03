# ANLP A-4

> **Name**: Bhav Beri
>
> **Roll Number**: 2021111013

----

This repository contains a report for the Assignment 4 (Quantisation) of the Advanced Natural Language Processing (ANLP) course. 

1. Install the required packages. Make sure to install the following version of the packages specifically:
    - transformers==4.46.2 
    - bitsandbytes>0.37.0
    - torch==2.4.0
> Note: The code has been tested only for these package versions. In case of any issues, please install the exact versions of the packages mentioned above.

2. For Task-1, run the `task-1.py` file. Similarly, for Task-2, run the `task-2.py` file.
3. To run the GGUF model, follow the following steps:
    1. Install llama.cpp following [https://huggingface.co/docs/hub/en/gguf-llamacpp](https://huggingface.co/docs/hub/en/gguf-llamacpp) \
        OR 
        ```bash
        git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && LLAMA_CURL=1 make
        ```
    
        Note: Remember to build llama.cpp with LLAMA_CURL=1 :)

    2. Download the model from the below mentioned model repository and use the argument `-m <path to model>` to specify the model\
    OR\
    Directly load the model from huggingface using argument `--hf-repo bhavberi/ANLP-A4 --hf-file OLMo-1B-hf.gguf`.
    3. ***`BONUS`***: Run the GGUF model using command `./llama-cli -m <path> -p <prompt>`. Few arguments:
        - `-m` : Path to the model. If directly loading from huggingface, use as mentioned in point 2.
        - `-p` : The prompt to be used for the model.
        - `-n <number>` : Number of tokens to generate. Better to use this argument to avoid infinite generation.
        - `-cnv`: You can remove this to run the CLI in chat completion mode. Else it runs in interactive mode.
    
        Example: `./llama-cli -m ../OLMo-1B-hf.gguf -p "You are a good AI model" -n 50`.\
        To run the model in server mode, use the command `./llama-server -m ../OLMo-1B-hf.gguf`.


Link to models repository: [https://huggingface.co/bhavberi/ANLP-A4/tree/main](https://huggingface.co/bhavberi/ANLP-A4/tree/main)

For testing purposes, you can also directly load NF4 (Task 2) quantized model from huggingface using standard methods. The model URL is `https://huggingface.co/bhavberi/OLMo-1B-NF4` with name `bhavberi/OLMo-1B-NF4`.
```py
model = AutoModelForCausalLM.from_pretrained(
    "bhavberi/OLMo-1B-NF4",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
```