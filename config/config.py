# config/config.py

class GPTConfig:
    vocab_size = 50257
    block_size = 128
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1
    bias = True

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
