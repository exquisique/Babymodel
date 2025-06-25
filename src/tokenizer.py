# src/tokenizer.py

import os
import numpy as np
from tqdm.auto import tqdm
import tiktoken
from datasets import load_dataset

def download_dataset():
    print("Downloading TinyStories dataset...")
    return load_dataset("roneneldan/TinyStories")

def encode_dataset(ds, encoder, filename_prefix="train", num_proc=8):
    def process(example):
        ids = encoder.encode_ordinary(example['text'])  # GPT-2 tokenizer
        return {'ids': ids, 'len': len(ids)}

    print(f"Tokenizing split: {filename_prefix}")
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the split",
        num_proc=num_proc,
    )

    arr_len = np.sum(tokenized['len'], dtype=np.uint64)
    filename = filename = f"data/{filename_prefix}.bin"

    dtype = np.uint16  # GPT-2 vocab fits in uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    idx = 0
    total_batches = 1024

    for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
        batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    
    arr.flush()
    print(f"Saved {filename} to disk.")
    return filename

def prepare_tokenized_files():
    enc = tiktoken.get_encoding("gpt2")
    ds = download_dataset()
    if not os.path.exists("train.bin"):
        encode_dataset(ds["train"], enc, filename_prefix="train")
    if not os.path.exists("validation.bin"):
        encode_dataset(ds["validation"], enc, filename_prefix="validation")

if __name__ == "__main__":
    prepare_tokenized_files()
