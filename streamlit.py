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

# Initialize session state for temperature and top_k if not already set
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7  # Default balanced temperature
if 'top_k' not in st.session_state:
    st.session_state.top_k = 40  # Default balanced top_k

PRESET_SETTINGS = {
    "Precise": {"temperature": 0.3, "top_k": 10},
    "Balanced": {"temperature": 0.7, "top_k": 40},
    "Creative": {"temperature": 1.0, "top_k": 80},
    "Custom": {"temperature": st.session_state.temperature, "top_k": st.session_state.top_k} # Placeholder for custom
}

def update_sliders_from_preset(preset_name):
    if preset_name in PRESET_SETTINGS and preset_name != "Custom":
        st.session_state.temperature = PRESET_SETTINGS[preset_name]["temperature"]
        st.session_state.top_k = PRESET_SETTINGS[preset_name]["top_k"]

def get_current_preset_name():
    for name, settings in PRESET_SETTINGS.items():
        if name == "Custom": continue
        if st.session_state.temperature == settings["temperature"] and \
           st.session_state.top_k == settings["top_k"]:
            return name
    return "Custom"


prompt = st.text_area("Enter a story prompt:", "Once upon a time there was a tiny robot who")
max_tokens = st.slider("Max new tokens", 20, 200, 100)

st.sidebar.title("Generation Parameters")

# Determine the initial index for the selectbox
current_preset_name = get_current_preset_name()
preset_options = list(PRESET_SETTINGS.keys())
try:
    initial_preset_index = preset_options.index(current_preset_name)
except ValueError:
    initial_preset_index = preset_options.index("Custom")


selected_preset = st.sidebar.selectbox(
    "Creativity Level:",
    options=preset_options,
    index=initial_preset_index,
    on_change=lambda: update_sliders_from_preset(st.session_state.selected_preset_key), # Use a key to get current selection
    key="selected_preset_key" # Assign a key to the selectbox widget
)

# If a preset is selected (and it's not "Custom" from initialization), update sliders
if selected_preset != "Custom" and \
    (st.session_state.temperature != PRESET_SETTINGS[selected_preset]["temperature"] or \
     st.session_state.top_k != PRESET_SETTINGS[selected_preset]["top_k"]):
    update_sliders_from_preset(selected_preset)


# Sliders for temperature and top_k, now controlled by session state
# Use a callback to update the "Custom" preset if sliders are changed manually
def on_slider_change():
    st.session_state.selected_preset_key = "Custom" # Set preset to custom if sliders are moved

temp_slider_val = st.sidebar.slider(
    "Temperature", 0.1, 1.5,
    st.session_state.temperature,
    step=0.01,
    on_change=on_slider_change,
    key="temp_slider" # Add key for slider
)
top_k_slider_val = st.sidebar.slider(
    "Top-k sampling", 0, 100,
    st.session_state.top_k,
    step=1,
    on_change=on_slider_change,
    key="top_k_slider" # Add key for slider
)

# Update session state if sliders were changed by the user
# This ensures that even if on_change isn't perfectly robust in all st versions for programmatic changes,
# we capture the final slider values.
if temp_slider_val != st.session_state.temperature:
    st.session_state.temperature = temp_slider_val
    # No need to set to custom here, on_slider_change handles the selectbox key

if top_k_slider_val != st.session_state.top_k:
    st.session_state.top_k = top_k_slider_val
    # No need to set to custom here, on_slider_change handles the selectbox key

# Ensure "Custom" preset reflects current slider values if it's selected
if st.session_state.selected_preset_key == "Custom": # check the key that on_slider_change sets
    PRESET_SETTINGS["Custom"]["temperature"] = st.session_state.temperature
    PRESET_SETTINGS["Custom"]["top_k"] = st.session_state.top_k


if st.button("Generate Story"):
    with st.spinner("Babymodel is thinking..."):
        # Use temperature and top_k from session state, which are updated by presets or manual slider changes
        output = generate_text(prompt, max_new_tokens=max_tokens, temperature=st.session_state.temperature, top_k=st.session_state.top_k)
        st.markdown("### 📖 Generated Story")
        st.write(output)
