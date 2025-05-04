# streaming_finetune.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering, AdamW
import math
import pickle

# 1) Config
CSV_PATH       = "output.csv"
CHUNK_SIZE     = 50          # rows per mini‑dataset
GLOBAL_EPOCHS  = 50
LR             = 5e-5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 2) Model + Processor
model_name = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_name)
model     = BlipForQuestionAnswering.from_pretrained(model_name).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# 3) Custom small Dataset
class SmallVQADataset(Dataset):
    def __init__(self, df_chunk, processor):
        self.rows      = df_chunk.to_dict(orient="records")
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(
            image,
            row["question"],
            caption=row["caption"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # tokenize one‑word answer
        with self.processor.as_target_processor():
            labels = self.processor(
                text=row["answer"],
                padding="max_length",
                truncation=True,
                max_length=8,
                return_tensors="pt",
            ).input_ids
        inputs["labels"] = labels
        # squeeze batch dim
        for k,v in inputs.items():
            inputs[k] = v.squeeze(0)
        return inputs

# 4) Training over chunks
# Calculate how many total chunks per CSV pass
total_rows = sum(1 for _ in open(CSV_PATH)) - 1  # minus header
chunks_per_epoch = math.ceil(total_rows / CHUNK_SIZE)

for global_epoch in range(1, GLOBAL_EPOCHS + 1):
    print(f"\n=== Global Epoch {global_epoch}/{GLOBAL_EPOCHS} ===")
    # iterate CSV in chunks
    for chunk_idx, df_chunk in enumerate(
        pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
    ):
        print(f"-- Chunk {chunk_idx+1}/{chunks_per_epoch} ({len(df_chunk)} rows)")
        
        # load last checkpoint if exists
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch{global_epoch-1}_chunk{chunk_idx}.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        # prepare DataLoader
        ds = SmallVQADataset(df_chunk, processor)
        dl = DataLoader(ds, batch_size=8, shuffle=True, pin_memory=True)
        
        # train 1 mini‑epoch on this chunk
        model.train()
        running_loss = 0.0
        for batch in dl:
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dl)
        print(f"   → chunk loss: {avg_loss:.4f}")
        
        # save checkpoint for this chunk
        save_path = os.path.join(CHECKPOINT_DIR, f"epoch{global_epoch}_chunk{chunk_idx}.pt")
        torch.save(model.state_dict(), save_path)
    
    # at the end of each global epoch, you could also save a full model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch{global_epoch}_full.pt"))

print("\nTraining complete!")
