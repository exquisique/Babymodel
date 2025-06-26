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



## 🧑‍💻 Author
**Exquisique** — GenAI explorer, language enthusiast, and poetic dreamer.

---

## 📜 License
MIT — open source, train your own baby language model! 🚀

