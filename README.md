# Memory Efficient LM Compression using Fisher Information from Low-Rank Representations

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-%3E%3D4.30-yellow)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-%3E%3D0.5-green)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/%F0%9F%A4%97%20TRL-%3E%3D0.7-red)](https://github.com/huggingface/trl)

**"Memory Efficient LM Compression using Fisher Information from Low-Rank Representations"**
*Daniil Moskovskiy, Sergey Pletenev, Sergey Zagoruyko, Alexander Panchenko*

The code allows reproducing experiments on compressing Large Language Models (LLMs) like Llama-2 and Llama-3.1 using Fisher-Weighted Singular Value Decomposition (FWSVD). The Fisher information is efficiently estimated using gradients obtained during Low-Rank Adaptation (LoRA) fine-tuning.

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Step 1: Generate Fisher Information](#step-1-generate-fisher-information)
  - [Step 2: Apply Compression](#step-2-apply-compression)
- [Citation](#-citation)
- [License](#-license)

## 💡 Introduction

Large Language Models (LLMs) have shown remarkable capabilities but often come with significant computational and memory costs. Model compression techniques are crucial for deploying these models in resource-constrained environments. This work explores Singular Value Decomposition (SVD) for compressing LLM weight matrices, enhanced by incorporating Fisher information.

We propose using Fisher Information, estimated from the gradients during LoRA fine-tuning, as a weighting mechanism for SVD (FWSVD). This approach aims to preserve the most critical parts of the weight matrices, leading to better performance retention after compression compared to standard SVD. This repository provides the necessary tools to:

1.  Estimate Fisher Information for specific layers of Llama models using LoRA.
2.  Apply both standard SVD and Fisher-Weighted SVD (FWSVD) to compress these models.

## ✨ Features

*   Implementation of Fisher-Weighted SVD (FWSVD) for LLM compression.
*   Efficient Fisher Information estimation using LoRA gradients via `trl` and `peft`.
*   Script (`generate_fisher_llm.py`) to compute and save Fisher Information estimates.
*   Script (`compress_llama.py`) to apply SVD or FWSVD compression to Llama models (e.g., Llama-2, Llama-3.1).
*   Customizable compression ranks per layer.
*   Integration with Hugging Face `transformers`, `peft`, and `trl` libraries.

## 📂 Repository Structure

```
.
├── LICENSE
├── README.md
├── compress_llama.py
├── generate_fisher_llm.py
└── llama_comp/
├── init.py
├── compression_utils.py
└── utils_svd.py
```


## ✅ Prerequisites

*   Python 3.8+
*   PyTorch (>= 1.13)
*   CUDA-enabled GPU (recommended for reasonable performance)
*   Hugging Face Hub account and token (if using gated models like Llama-2/3)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/s-nlp/lora_fwsvd.git
    cd lora_fwsvd
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets peft trl accelerate bitsandbytes numpy scikit-learnd
    ```
    *Note: `bitsandbytes` might be needed if using quantization features from `transformers`/`peft`, although not explicitly shown in the provided snippets.*

4.  **Log in to Hugging Face Hub (if needed):**
    ```bash
    huggingface-cli login
    ```

## ▶️ Usage

The process involves two main steps: generating the Fisher Information weights and then applying the compression using these weights.

### Step 1: Generate Fisher Information

This step uses LoRA fine-tuning on a dataset to estimate the Fisher Information for the target model's weights.

1.  **Configure `generate_fisher_llm.py`:**
    *   Modify the `dataset` loading part if you want to use a different dataset (e.g., `wikitext`, `c4`, etc.). The current example uses `"robbiegwaldd/dclm-micro"`.
    *   Adjust `peft_config` (e.g., `r`, `lora_alpha`, `target_modules`) and `training_args` (e.g., `max_seq_length`, `per_device_train_batch_size`, `num_train_epochs`, `output_dir`) as needed.
    *   Set the base model name in the `CustomTrainer` initialization (e.g., `"meta-llama/Llama-2-7b-hf"`, `"meta-llama/Llama-3.1-8B"`).

2.  **Run the script:**
    ```bash
    python generate_fisher_llm.py
    ```

3.  **Output:** This will generate a pickle file (e.g., `./_tmp6_llama2_7b_all-linear/fisher_XXXX.pkl`) containing a dictionary where keys are layer names and values are the squared gradients (Fisher estimates) accumulated during training.

### Step 2: Apply Compression

This step takes the original model and the generated Fisher Information to perform SVD or FWSVD compression.

1.  **Configure `compress_llama.py`:**
    *   Set the `pretrained` variable to the Hugging Face model identifier you want to compress (must match the one used for Fisher generation or be compatible).
    *   **Load the Fisher Information:** Add code to load the `.pkl` file generated in Step 1.
      ```python
      import pickle
      import torch 

      fisher_path = './_tmp6_llama2_7b_all-linear/fisher_XXXX.pkl'
      with open(fisher_path, 'rb') as fp:
          fisher_raw = pickle.load(fp)
      fisher = {k: v.cpu().float() for k, v in fisher_raw.items()}
      model_.to_compression(compress=True,
                          weight=fisher if fisher else None,
                          )
      ```
    *   Define the compression rank (`rank`) for each layer. The example uses a dictionary `trunc_raw`. You can set a global rank or specify per layer. `rank=0` skips compression for that layer.
    *   Specify which layers to compress using `layer_mask` (a regex pattern).
    *   Choose the `compression_type`: `'svd'` (uses `w_svd` from `utils_svd.py`, which implements standard SVD if `weight=None` and FWSVD if `weight` is provided).
    *   Set the `save_pretrained` path to where you want to save the compressed model and tokenizer.

2.  **Run the script:**
    ```bash
    python compress_llama.py
    ```

3.  **Output:** The script will print the compression ratio and save the compressed model artifacts (config, weights, tokenizer) to the specified directory (e.g., `./compressed_llamas/llama-Llama-3-8b-hf-09-svd`). This compressed model can then be loaded using `AutoModelForCausalLM.from_pretrained(...)`.

## 🎓 Citation

```bibtex
@inproceedings{moskovskiy2024memory, % TODO: Update with publication details when available
  title={Memory Efficient LM Compression using Fisher Information from Low-Rank Representations},
  author={Moskovskiy, Daniil and Pletenev, Sergey and Zagoruyko, Sergey and Panchenko, Alexander},
  booktitle={Proceedings of the Conference Name}, % TODO: Add Conference/Journal
  year={2024}, % TODO: Update Year
  url={https://arxiv.org/abs/xxxx.xxxxx} % TODO: Add arXiv or publication URL
}