# streamlit_app.py

import streamlit as st
import torch
from src.model import GPT
from config.config import GPTConfig
import tiktoken

# -------------------- Load Model --------------------
gpt_config = GPTConfig()
model = GPT(gpt_config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=torch.device("cpu"))['model_state_dict'])
model.eval()

# -------------------- Tokenizer ---------------------
enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

# -------------------- Generation Function ---------------------
def generate_text(prompt, max_new_tokens=100, temperature=0.7, top_k=20):
    model_input = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)
    generated = model.generate(model_input, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(generated[0].tolist())

# -------------------- Streamlit UI ---------------------
st.title("🤖 Babymodel Story Generator")
st.markdown("Tiny GPT-style model trained on TinyStories ✨")

prompt = st.text_area("Enter a story prompt:", "Once upon a time there was a tiny robot who")
max_tokens = st.slider("Max new tokens", 20, 200, 100)
temp = st.slider("Temperature", 0.1, 1.5, 0.7)
top_k = st.slider("Top-k sampling", 0, 100, 20)

if st.button("Generate Story"):
    with st.spinner("Babymodel is thinking..."):
        output = generate_text(prompt, max_new_tokens=max_tokens, temperature=temp, top_k=top_k)
        st.markdown("### 📖 Generated Story")
        st.write(output)
