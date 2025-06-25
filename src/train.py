# src/train.py

import torch
import numpy as np
import os
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from contextlib import nullcontext
from tqdm.auto import tqdm

from config.config import GPTConfig, TrainConfig
from src.model import GPT

# ---------------------- Configs --------------------------
gpt_config = GPTConfig()
train_config = TrainConfig()

# ---------------------- Hyperparams ----------------------
block_size = gpt_config.block_size
batch_size = train_config.batch_size
eval_iters = train_config.eval_iters
max_iters = train_config.max_iters
learning_rate = train_config.learning_rate
min_lr = train_config.min_lr
warmup_steps = train_config.warmup_steps
grad_accum_steps = train_config.gradient_accumulation_steps

# ---------------------- Device Setup ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"
torch.set_default_device(device)
torch.manual_seed(train_config.seed)

# Precision handling
dtype = train_config.dtype
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---------------------- Batching -------------------------
def get_batch(split):
    data = np.memmap(f'data/{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ---------------------- Model Init -----------------------
model = GPT(gpt_config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-9
)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))


# ---------------------- Loss Estimation ------------------
def estimate_loss():
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'validation']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------- Resume Logic ---------------------
best_val_loss = float('inf')
train_losses, val_losses = [], []
start_iter = 0

if os.path.exists(train_config.best_model_params_path):
    print("🔁 Resuming from previous checkpoint...")
    checkpoint = torch.load(train_config.best_model_params_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_iter = checkpoint['iter'] + 1
    best_val_loss = checkpoint['best_val_loss']
else:
    print("🚂 Starting training from scratch...")

# ---------------------- Training Loop --------------------
for iter in tqdm(range(start_iter, max_iters), desc="Training"):
    if iter % eval_iters == 0 and iter != 0:
        losses = estimate_loss()
        train_loss = losses["train"]
        val_loss = losses["validation"]
        print(f"Step {iter}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'iter': iter,
                'best_val_loss': best_val_loss
            }, train_config.best_model_params_path)
            print("💾 Saved new best model!")

    X, Y = get_batch("train")

    with ctx:
        logits, loss = model(X, Y)
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

    if (iter + 1) % grad_accum_steps == 0 or (iter + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

# ---------------------- Save Final Model -----------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'iter': iter,
    'best_val_loss': best_val_loss
}, train_config.best_model_params_path)

torch.save(model.state_dict(), "final_model.pt")
print("💾 Saved final model to final_model.pt ✅")
