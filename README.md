# Babymodel 🧠📖

A tiny GPT-style language model trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Built using PyTorch with inspiration from the minimalist and effective design philosophy of [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 🚀 Overview

Babymodel is a custom-built, modular small language model project developed from scratch to:

- Tokenize real-world text (TinyStories)
- Train a transformer-based model
- Generate coherent story-style completions
- Support full checkpoint saving/resuming
- Train entirely on consumer GPU (RTX 4060)
- Be easily extensible and lightweight

This project marks a full-stack journey through LLM design, training, and generation — wrapped in a clean, scalable codebase.

---

## 📚 Dataset: TinyStories

We used the open-source `TinyStories` dataset curated by Microsoft Research:
- Simple English narratives for toddlers (~2–4 years)
- Clean and syntactically simple
- Perfect for training small models to generate story-style completions

Tokenization is done using [tiktoken](https://github.com/openai/tiktoken), leveraging the GPT-2 tokenizer.

---

## 🏗️ Architecture

- **Inspired by:** `nanoGPT`
- **Model:** GPT-style transformer, ~10–15M parameters
- **Tokenizer:** `tiktoken` (GPT-2 compatible)
- **Training:** Manual PyTorch training loop
- **Optimizer:** `AdamW` with warmup and cosine decay
- **Precision:** Supports `float32`, `bfloat16`, or `float16`
- **Checkpointing:** Full training state saved (model, optimizer, scheduler, scaler, iteration)

---

## 🧰 Project Structure

```
Babymodel/
├── data/                      # tokenized .bin files (not tracked)
├── src/
│   ├── model.py              # Transformer model
│   ├── train.py              # Training loop
│   ├── tokenizer.py          # Preprocessing and tokenization
│   ├── generate.py           # Inference script
├── config/
│   └── config.py             # Hyperparameters and training config
├── requirements.txt
├── .gitignore
├── README.md
```

---

## 🏃 Getting Started

### 🔧 Setup
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 🧪 Tokenize Dataset
```bash
python -m src.tokenizer
```

### 🎓 Train Model
```bash
python -m src.train
```
Supports resume from checkpoint and custom iteration length via `config.py`

### 🗣️ Generate
```bash
python -m src.generate
```

---

## ✨ Sample Output

After 50k+ iterations, Babymodel starts generating story-style sequences:

**Prompt:**
```text
Once upon a time there was a tiny robot who
```

**Output:**
```text
Once upon a time there was a tiny robot who lived in a far away home. One day, a little girl named Lily decided to go on a special trip in the forest. She walked and walked until she got there but suddenly she started to go. Her mom called her and said, "Don't worry, Lily. We will get you my special ride. And let's keep her dry."

Lily ran inside to touch the pilot and it was so bright. She ran around and saw that it was so thick.
```

> ⚠️ Still slightly incoherent — but improves with longer training, better prompts, or sampling settings.

Try tuning:
```python
temperature = 0.7
max_new_tokens = 100
top_k = 20
```

---

## 💡 Inspiration

This project is inspired by:
- [nanoGPT](https://github.com/karpathy/nanoGPT) for clean architecture
- TinyStories research paper by Microsoft
- The joy of training models that speak 🧠💬



## 🤗 Uploading to Hugging Face Hub for `transformers` Compatibility

This project includes an updated script (`src/upload_to_hub.py`) to upload your trained model to the Hugging Face Hub in a way that is compatible with the `transformers` library, allowing usage with `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.

### Prerequisites

1.  **Install Dependencies**:
    Ensure all necessary libraries, including `transformers`, `tokenizers`, and `huggingface_hub`, are installed. The `requirements.txt` file has been updated:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Login to Hugging Face CLI**:
    You need to be authenticated with the Hugging Face Hub. If you haven't done so already, run the following command and enter your Hugging Face API token (with write permissions):
    ```bash
    huggingface-cli login
    ```

### 🚀 Upload Script (`src/upload_to_hub.py`)

Once your model has been trained (i.e., `final_model.pt` or `best_model_params.pt` exists), you can use the enhanced upload script:

```bash
python -m src.upload_to_hub
```

This script will perform the following steps:

1.  **Prompt for Credentials**: Ask for your Hugging Face username and the desired model repository name on the Hub, if these are not already set as environment variables (`HF_USERNAME`, `HF_MODEL_NAME`).
2.  **Load Model and Config**:
    *   Instantiate the modified `GPTConfig` (from `config.config.py`), which is `transformers.PretrainedConfig` compatible.
    *   Instantiate the modified `GPT` model (from `src/model.py`), which now inherits from `transformers.PreTrainedModel`.
3.  **Load Weights**: Load the weights from your saved `final_model.pt` (or `best_model_params.pt`) into the `transformers`-compatible `GPT` model. The script includes basic state dictionary key mapping, which might need refinement if layer names significantly diverged.
4.  **Prepare Temporary Directory**: Create a temporary local directory (e.g., `hf_upload_temp`) to stage files for uploading.
5.  **Save Model**: Use `model.save_pretrained(temp_dir)` to save the model weights as `pytorch_model.bin` and the model configuration as `config.json` in the temporary directory.
6.  **Save Tokenizer**:
    *   Initialize a `GPT2TokenizerFast.from_pretrained("gpt2")` (as the original model uses `tiktoken` with `gpt2` settings).
    *   Use `tokenizer.save_pretrained(temp_dir)` to save the tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.) in the temporary directory.
7.  **Generate Model Card**: Create a detailed `README.md` (model card) specifically for the Hugging Face Hub. This card will include:
    *   Model description, architecture details.
    *   Instructions on how to load the model using `AutoModelForCausalLM.from_pretrained(YOUR_REPO_ID, trust_remote_code=True)` and `AutoTokenizer.from_pretrained(YOUR_REPO_ID)`.
    *   The `trust_remote_code=True` flag is necessary because this model uses custom Python code defined in `src/model.py`.
8.  **Upload to Hub**:
    *   Upload the entire contents of the temporary directory (which now includes `pytorch_model.bin`, `config.json`, tokenizer files, and the Hub `README.md`) to the specified Hugging Face model repository.
    *   Crucially, it will also upload the `src/model.py` file to the root of your Hugging Face model repository. This makes the custom model class available to the `transformers` library when `trust_remote_code=True` is used.
9.  **Cleanup**: Remove the temporary local directory.

After these steps, your model will be accessible on the Hugging Face Hub (e.g., `https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME`) and can be loaded by anyone using the `transformers` library, provided they use `trust_remote_code=True`.

### Example Usage from Hugging Face Hub

Once uploaded, the model can be loaded and used like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your actual Hugging Face repository ID
repo_id = "YOUR_USERNAME/YOUR_MODEL_NAME"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Load the model
# trust_remote_code=True is essential as this model uses custom code (model.py)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

# Now you can use the model for generation
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

## 🧑‍💻 Author
**Exquisique** — GenAI explorer, language enthusiast, and poetic dreamer.

---

## 📜 License
MIT — open source, train your own baby language model! 🚀

