import os
import torch
import tiktoken
from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub.utils import HfFolder

from src.model import GPT, GPTConfig
from config.config import TrainConfig # Assuming this has best_model_params_path

def main():
    # Get Hugging Face credentials and model name
    hf_username = os.environ.get("HF_USERNAME")
    hf_model_name = os.environ.get("HF_MODEL_NAME")

    if not hf_username:
        hf_username = input("Enter your Hugging Face username: ")
    if not hf_model_name:
        hf_model_name = input("Enter the desired model name on Hugging Face Hub: ")

    repo_id = f"{hf_username}/{hf_model_name}"

    print(f"Preparing to upload model to: {repo_id}")

    # --- 1. Load Model ---
    print("Loading model...")
    gpt_config = GPTConfig()
    model = GPT(gpt_config)

    # Load the trained weights
    # Assuming the final model is saved as 'final_model.pt' in the root directory as per train.py
    model_weights_path = "final_model.pt"
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights not found at {model_weights_path}")
        print("Please ensure you have trained the model and 'final_model.pt' exists.")
        # Attempt to load from best_model_params.pt if final_model.pt is not found
        train_config = TrainConfig()
        best_model_path = train_config.best_model_params_path
        if os.path.exists(best_model_path):
            print(f"Attempting to load weights from checkpoint: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu') # Load to CPU
            model.load_state_dict(checkpoint['model_state_dict'])
            # Save it as final_model.pt for consistency in hub upload
            torch.save(model.state_dict(), model_weights_path)
            print(f"Loaded and saved weights from {best_model_path} to {model_weights_path}")
        else:
            print(f"Error: Neither {model_weights_path} nor {best_model_path} found.")
            return

    else:
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))

    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Prepare Tokenizer ---
    # The project uses tiktoken with "gpt2" encoding.
    # For Hugging Face, we typically save the tokenizer's configuration.
    # Since tiktoken is a direct wrapper, we can note its usage.
    # For full HF compatibility, one might convert to a `transformers` tokenizer,
    # but for now, we'll upload the model and specify tokenizer type.
    print("Tokenizer: tiktoken (gpt2)")

    # --- 3. Create Model Card ---
    print("Creating model card...")
    model_card_content = f"""
---
license: mit
tags:
- pytorch
- gpt
- tinystories
- babymodel
---

# {hf_model_name}

This is a GPT-style language model based on the Babymodel architecture, trained on the TinyStories dataset.

## Model Description

- **Architecture:** GPT-like transformer ({gpt_config.n_layer} layers, {gpt_config.n_head} heads, {gpt_config.n_embd} embedding dimensions)
- **Vocabulary Size:** {gpt_config.vocab_size}
- **Context Length:** {gpt_config.block_size}
- **Tokenizer:** `tiktoken` with `gpt2` encoding.

## Training Data

The model was trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

## How to Use (Conceptual)

```python
import torch
import tiktoken
# from your_model_loading_script import GPT, GPTConfig # Ensure these are accessible

# Load config and model (example)
# config = GPTConfig()
# model = GPT(config)
# model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu")) # Assuming 'pytorch_model.bin' is the HF name
# model.eval()

# enc = tiktoken.get_encoding("gpt2")
# prompt = "Once upon a time"
# tokens = enc.encode(prompt)
# input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

# with torch.no_grad():
#    output_ids = model.generate(input_ids, max_new_tokens=50) # Using the model's generate method

# generated_text = enc.decode(output_ids[0].tolist())
# print(generated_text)
```

**Note:** This model is uploaded as a `.pt` file containing the state dictionary. You'll need the model class definition (`GPT` from `src/model.py`) and `GPTConfig` from `config/config.py` to load it. The tokenizer is `tiktoken` using `gpt2` pre-trained tokenizer.

For direct use with Hugging Face `transformers` library, further conversion might be needed.
"""
    card = ModelCard(model_card_content)

    # --- 4. Upload to Hugging Face Hub ---
    api = HfApi()

    # Check login status
    token = HfFolder.get_token()
    if not token:
        print("You need to login to Hugging Face Hub first. Run 'huggingface-cli login'")
        return

    print(f"Creating repository {repo_id} on Hugging Face Hub...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    print(f"Uploading model weights ({model_weights_path}) to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=model_weights_path,
            path_in_repo="pytorch_model.bin", # Standard name for PyTorch models on Hub
            repo_id=repo_id,
            repo_type="model",
        )
        print("Model weights uploaded successfully.")
    except Exception as e:
        print(f"Error uploading model weights: {e}")
        return

    print(f"Uploading model card (README.md) to {repo_id}...")
    try:
        # Create a temporary file for the model card if upload_file needs a path
        with open("temp_README.md", "w", encoding="utf-8") as f:
            f.write(str(card))

        api.upload_file(
            path_or_fileobj="temp_README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        os.remove("temp_README.md") # Clean up temporary file
        print("Model card uploaded successfully.")
    except Exception as e:
        print(f"Error uploading model card: {e}")
        if os.path.exists("temp_README.md"):
            os.remove("temp_README.md")
        return

    # Upload config files for reproducibility
    config_files_to_upload = {
        "config/config.py": "config.py", # Store it as config.py in the repo
        "src/model.py": "model.py"      # Store the model definition
    }

    for local_path, repo_path in config_files_to_upload.items():
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path} in {repo_id}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"{local_path} uploaded successfully.")
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")
        else:
            print(f"Warning: Config file {local_path} not found. Skipping.")


    print("\n--- Upload Complete! ---")
    print(f"Model available at: https://huggingface.co/{repo_id}")
    print("Please ensure you have the necessary files (`model.py`, `config.py`) and `tiktoken` to use this model.")

if __name__ == "__main__":
    main()
