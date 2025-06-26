# src/upload_to_hub.py
import os
import torch
import shutil # For directory operations
from huggingface_hub import HfApi, ModelCard, HfFolder
from transformers import GPT2TokenizerFast # For saving a HF compatible tokenizer

# Import the *modified* GPTConfig and GPT model
from config.config import GPTConfig, TrainConfig
from src.model import GPT # This is now the PreTrainedModel version

TEMP_UPLOAD_DIR = "hf_upload_temp"

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

    # Create a temporary directory for storing files to be uploaded
    if os.path.exists(TEMP_UPLOAD_DIR):
        shutil.rmtree(TEMP_UPLOAD_DIR)
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    # --- 1. Load Configuration & Instantiate Model ---
    print("Loading configuration and instantiating model...")
    # Use the modified GPTConfig that inherits from PretrainedConfig
    gpt_config = GPTConfig()
    # Instantiate the modified GPT model (which is a PreTrainedModel)
    model = GPT(gpt_config)

    # --- 2. Load Trained Weights ---
    # Load the original state_dict from final_model.pt or best_model_params.pt
    model_weights_path = "final_model.pt"
    train_config_obj = TrainConfig() # To get best_model_params_path
    best_model_path = train_config_obj.best_model_params_path

    loaded_state_dict = None
    if os.path.exists(model_weights_path):
        print(f"Loading weights from {model_weights_path}...")
        loaded_state_dict = torch.load(model_weights_path, map_location='cpu')
    elif os.path.exists(best_model_path):
        print(f"Attempting to load weights from checkpoint: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        loaded_state_dict = checkpoint['model_state_dict']
        # Optionally save these as final_model.pt for future non-HF use
        # torch.save(loaded_state_dict, model_weights_path)
        # print(f"Loaded and saved weights from {best_model_path} to {model_weights_path}")
    else:
        print(f"Error: Neither {model_weights_path} nor {best_model_path} found. Cannot proceed.")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    # --- State Dict Key Mapping (Important for custom models) ---
    # The keys in your saved state_dict might not perfectly match the names
    # in the Hugging Face PreTrainedModel structure. You may need to map them.
    # Example: if original was 'transformer.layer.0...' and HF is 'transformer.h.0...'
    # This is a placeholder, adjust based on your actual model structure vs. HF expectations.
    # For this specific model, names were kept fairly consistent.
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        new_key = k
        # Example mapping (if needed):
        # if k.startswith("transformer.layers"):
        #    new_key = k.replace("transformer.layers", "transformer.h")
        # elif k == "output_head.weight":
        #    new_key = "lm_head.weight"
        new_state_dict[new_key] = v

    # If lm_head was tied to wte, ensure lm_head.weight is present if wte.weight is
    if 'transformer.wte.weight' in new_state_dict and 'lm_head.weight' not in new_state_dict:
        print("INFO: lm_head.weight not found in state_dict, attempting to use transformer.wte.weight (tie_weights).")
        new_state_dict['lm_head.weight'] = new_state_dict['transformer.wte.weight']

    try:
        model.load_state_dict(new_state_dict, strict=False) # Use strict=False initially for debugging mismatches
        print("Model weights loaded into Hugging Face compatible model structure.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to mismatched keys between your saved model and the HF model structure.")
        print("Please check the state_dict_key_mapping section in the script.")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- 3. Save Model & Configuration using save_pretrained ---
    print(f"Saving model and configuration to {TEMP_UPLOAD_DIR}...")
    model.save_pretrained(TEMP_UPLOAD_DIR) # Saves pytorch_model.bin and config.json
    print("Model and config.json saved.")

    # --- 4. Prepare and Save Tokenizer ---
    # The original project uses tiktoken with "gpt2".
    # We'll save a GPT2TokenizerFast for compatibility.
    print("Preparing and saving tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.save_pretrained(TEMP_UPLOAD_DIR)
        print(f"GPT2TokenizerFast (based on 'gpt2') saved to {TEMP_UPLOAD_DIR}.")
    except Exception as e:
        print(f"Could not save tokenizer: {e}")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    # --- 5. Create Model Card ---
    print("Creating model card...")
    # Use GPTConfig attributes that are now HF compatible
    model_card_content = f"""
---
license: mit
tags:
- pytorch
- gpt
- tinystories
- babymodel
- causal-lm
language: en
pipeline_tag: text-generation
# Important for models with custom code
auto_model_class: AutoModelForCausalLM
---

# {hf_model_name}

This is a GPT-style language model (`babymodel` architecture) trained on the TinyStories dataset.
This repository contains the model weights, configuration, and tokenizer files compatible with the Hugging Face `transformers` library.

## Model Description

- **Model Type:** `{gpt_config.model_type}`
- **Architecture:** GPT-like transformer
  - Layers: `{gpt_config.num_hidden_layers}`
  - Heads: `{gpt_config.num_attention_heads}`
  - Embedding Dimension: `{gpt_config.hidden_size}`
- **Vocabulary Size:** `{gpt_config.vocab_size}`
- **Context Length:** `{gpt_config.max_position_embeddings}`
- **Tokenizer:** `GPT2TokenizerFast` (standard `gpt2` vocabulary and merges).

## Training Data

The model was trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

## How to Use with Hugging Face `transformers`

**Important:** This model uses custom code. You will need to pass `trust_remote_code=True` when loading the model. The necessary `model.py` file is included in this repository.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}" # Example: "your_username/your_model_name"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Load model
# trust_remote_code=True is required because we use the custom model.py
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

# Example prompt
prompt = "Once upon a time, in a land far away,"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
# You can use various generation parameters, e.g., max_length, num_beams, do_sample, top_k, top_p
outputs = model.generate(**inputs, max_length=100, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
```

## Model Files

- `pytorch_model.bin`: The model weights.
- `config.json`: Model configuration file.
- `tokenizer.json`, `vocab.json`, `merges.txt`: Tokenizer files.
- `model.py`: The custom model definition (required for `trust_remote_code=True`).

"""
    model_card = ModelCard(model_card_content)
    model_card.save(os.path.join(TEMP_UPLOAD_DIR, 'README.md')) # Save as README.md in the temp dir
    print("Model card created and saved to temp directory.")

    # --- 6. Upload to Hugging Face Hub ---
    api = HfApi()

    token = HfFolder.get_token()
    if not token:
        print("You need to login to Hugging Face Hub first. Run 'huggingface-cli login'")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    print(f"Creating or updating repository {repo_id} on Hugging Face Hub...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print(f"Repository {repo_id} ensured.")
    except Exception as e:
        print(f"Error creating/accessing repository: {e}")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    print(f"Uploading contents of {TEMP_UPLOAD_DIR} to {repo_id}...")
    try:
        # Upload the entire folder (model weights, config, tokenizer files, model card)
        api.upload_folder(
            folder_path=TEMP_UPLOAD_DIR,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {hf_model_name} model files",
        )
        print("Folder uploaded successfully.")

        # Additionally, upload the model.py script to the root of the repo
        # This is crucial for trust_remote_code=True
        model_script_path = "src/model.py"
        if os.path.exists(model_script_path):
            print(f"Uploading {model_script_path} to the root of {repo_id} for trust_remote_code=True...")
            api.upload_file(
                path_or_fileobj=model_script_path,
                path_in_repo="model.py", # Save as model.py in the repo root
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add custom model.py for trust_remote_code"
            )
            print(f"{model_script_path} uploaded successfully.")
        else:
            print(f"WARNING: {model_script_path} not found. trust_remote_code=True might not work without it in the repo.")

    except Exception as e:
        print(f"Error uploading files: {e}")
        shutil.rmtree(TEMP_UPLOAD_DIR)
        return

    # --- 7. Cleanup ---
    try:
        shutil.rmtree(TEMP_UPLOAD_DIR)
        print(f"Cleaned up temporary directory: {TEMP_UPLOAD_DIR}")
    except Exception as e:
        print(f"Error cleaning up temporary directory: {e}")

    print("\n--- Upload Complete! ---")
    print(f"Model should be available at: https://huggingface.co/{repo_id}")
    print(f"Remember to use `trust_remote_code=True` when loading with AutoModelForCausalLM.")

if __name__ == "__main__":
    main()
