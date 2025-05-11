import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Blip2Processor,
    Blip2ForQuestionAnswering,
    Seq2SeqTrainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from huggingface_hub import HfApi

# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_latest_adapter(api: HfApi, repo_id: str, device: torch.device):
    files = api.list_repo_files(repo_id, token=HF_TOKEN)
    epochs = sorted(
        int(f.split("-",1)[1]) for f in files
        if f.startswith("epoch-") and f.split("-",1)[1].isdigit()
    )
    if not epochs:
        print("No existing LoRA checkpoint; starting fresh.")
        return None, 0

    latest = epochs[-1]
    print(f"Loading LoRA adapter epoch-{latest}…")
    # instantiate a fresh BLIP-2 base + LoRA wrapper
    adapter = get_peft_model(
        Blip2ForQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-large"),
        peft_config=None
    )
    adapter.load_adapter(f"{repo_id}/epoch-{latest}", device_map="auto")
    print(f"Loaded LoRA adapter epoch-{latest}")
    return adapter, latest

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
TRAIN_CSV    = "./train.csv"
VAL_CSV      = "./val.csv"

# CSV must have columns: [image_id, image_path, question, answer]
# we'll ignore `image_id` and read each image from `image_path`

REPO_ID      = "HLGS/RASPBERRY_BLIP2_VQA_LORA"
HF_TOKEN     = os.environ["HF_TOKEN"]

CHUNK_SIZE   = 5_000
TOTAL_EPOCHS = 60
BATCH_SIZE   = 8
ACCUM_STEPS  = 4
MAX_ON_HUB   = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(42)
api = HfApi()

# ----------------------------------------------------
# Load processor & base BLIP-2 (2 B parameter FLAN-T5 Large)
# ----------------------------------------------------
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-flan-t5-large",
    cache_dir="./proc_cache"
)
base_model = Blip2ForQuestionAnswering.from_pretrained(
    "Salesforce/blip2-flan-t5-large",
    cache_dir="./model_cache"
).to(DEVICE)

# ----------------------------------------------------
# Wrap with LoRA adapter
# ----------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.VISION_QUESTION_ANSWERING,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(base_model, lora_config)

# optionally resume from last epoch on the Hub
adapter, last_epoch = load_latest_adapter(api, REPO_ID, DEVICE)
if adapter is not None:
    model = adapter
    start_epoch = last_epoch + 1
else:
    start_epoch = 1

# ----------------------------------------------------
# Dataset & collator
# ----------------------------------------------------
class VQADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return {
            "image":    img,
            "question": row["question"],
            "answer":   row["answer"]
        }

def collate_fn(batch):
    imgs      = [item["image"]    for item in batch]
    questions = [item["question"] for item in batch]
    answers   = [item["answer"]   for item in batch]

    enc = processor(
        images=imgs,
        text=questions,
        return_tensors="pt",
        padding=True
    )
    enc["labels"] = processor.tokenizer(
        answers,
        padding=True,
        return_tensors="pt"
    ).input_ids

    # move everything to GPU/CPU
    return {k: v.to(DEVICE) for k, v in enc.items()}

# ----------------------------------------------------
# Metrics (no accuracy)
# ----------------------------------------------------
bertscore = evaluate.load("bertscore")
bartscore = evaluate.load("bart_score")
rouge     = evaluate.load("rouge")

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    decoded_preds  = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    decoded_labels = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )
    decoded_labels = [lbl.strip() for lbl in decoded_labels]

    # BERTScore
    bs = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="en"
    )
    bs_p = sum(bs["precision"]) / len(bs["precision"])
    bs_r = sum(bs["recall"])    / len(bs["recall"])
    bs_f1= sum(bs["f1"])        / len(bs["f1"])

    # BARTScore
    bt = bartscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        batch_size=len(decoded_preds)
    )
    bt_avg = sum(bt["score"]) / len(bt["score"])

    # ROUGE
    rg = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "bertscore_precision": bs_p,
        "bertscore_recall":    bs_r,
        "bertscore_f1":        bs_f1,
        "bartscore":           bt_avg,
        "rouge1":              rg["rouge1"],
        "rouge2":              rg["rouge2"],
        "rougeL":              rg["rougeL"],
    }

# ----------------------------------------------------
# Load dataframes
# ----------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
n        = len(train_df)

# ----------------------------------------------------
# Training & Hub upload loop
# ----------------------------------------------------
for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{TOTAL_EPOCHS} ===")

    # chunking to avoid OOM
    for start in range(0, n, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, n)
        tr_ds = VQADataset(train_df.iloc[start:end])
        vl_ds = VQADataset(val_df)

        args = TrainingArguments(
            output_dir="./lora_ckpt",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=ACCUM_STEPS,
            fp16=True,
            num_train_epochs=1,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_steps=1,
            save_total_limit=1,
            learning_rate=5e-5,
            remove_unused_columns=False,
            report_to="none",
            push_to_hub=True,
            hub_model_id=REPO_ID,
            hub_token=HF_TOKEN,
            hub_strategy="every_save",
            save_total_limit=MAX_ON_HUB,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tr_ds,
            eval_dataset=vl_ds,
            data_collator=collate_fn,
            tokenizer=processor.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train(resume_from_checkpoint=False)
        metrics = trainer.evaluate()
        print({k: f"{v:.4f}" for k, v in metrics.items()})

    print(f"Finished epoch {epoch}; LoRA adapter pushed to {REPO_ID}/epoch-{epoch}")

print(" All done!")
