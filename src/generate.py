# src/generate.py

import torch
import tiktoken
from config.config import GPTConfig
from src.model import GPT

# -------------------- Load Tokenizer --------------------
enc = tiktoken.get_encoding("gpt2")

# -------------------- Load Model ------------------------
config = GPTConfig()
model = GPT(config)
checkpoint = torch.load("best_model_params.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------- Generate Text ---------------------

def generate_text(prompt, max_new_tokens=100, temperature=1.0, top_k=40):
    print(f"\n🧠 Prompt: {prompt}")
    context = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    output_text = enc.decode(output_ids[0].tolist())
    print(f"\n📝 Generated:\n{output_text}\n")


if __name__ == "__main__":
    # Try different prompts!
    generate_text("Once upon a time there was a tiny robot who")
    generate_text("The little bird said")
