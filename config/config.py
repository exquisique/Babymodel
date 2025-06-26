# config/config.py
from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    model_type = "babymodel" # Required for custom models in Hugging Face

    def __init__(
        self,
        vocab_size=50257,
        block_size=128, # Renamed to max_position_embeddings for HF conventions
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True,
        # Standard Hugging Face PretrainedConfig arguments below, add as needed
        pad_token_id=None, # GPT-2 tokenizer might not have a pad token explicitly set this way
        bos_token_id=50256, # Standard for GPT-2 like tokenizers
        eos_token_id=50256, # Standard for GPT-2 like tokenizers
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = block_size # HF uses this name
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.hidden_size = n_embd # HF uses this name for embedding dimension
        self.num_attention_heads = n_head # HF uses this name
        self.num_hidden_layers = n_layer # HF uses this name
        self.dropout = dropout
        self.bias = bias
        # For CausalLM, intermediate_size is often 4 * hidden_size for MLP
        self.intermediate_size = 4 * self.hidden_size
        # activation_function is also common, defaults to 'gelu' often
        self.activation_function = "gelu"


class TrainConfig:
    batch_size = 32
    max_iters = 80000
    learning_rate = 1e-4
    warmup_steps = 1000
    min_lr = 5e-4
    eval_iters = 1000
    gradient_accumulation_steps = 32
    dtype = 'bfloat16'
    best_model_params_path = "best_model_params.pt"
    seed = 42

class DatasetConfig:
    dataset_name = "roneneldan/TinyStories"
