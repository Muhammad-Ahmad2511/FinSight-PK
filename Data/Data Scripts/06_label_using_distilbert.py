# =============================================================================
# Label the full Dawn Business 2015-2026 corpus with the fine-tuned DistilBERT
# (finsight_distilbert/best_model). Paste into ONE Colab cell and run.
# Runtime > Change runtime type > GPU
# =============================================================================

get_ipython().system('pip install -q transformers accelerate')

from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU -- Runtime > Change runtime type > GPU")

# --- Config -- EDIT THESE THREE LINES IF YOUR FILE/COLUMNS DIFFER ---
DRIVE_DIR = "/content/drive/MyDrive/finsight_distilbert"
BEST_MODEL_DIR = f"{DRIVE_DIR}/best_model"
INPUT_PATH = f"{DRIVE_DIR}/dawn_business_2015_to_2026.csv"   # <- change if it's elsewhere / named differently
TITLE_COL = "title"
TEXT_COL = "raw_text"
OUTPUT_PATH = f"{DRIVE_DIR}/dawn_business_2015_to_2026_labeled.csv"

MAX_LEN = 256
BATCH_SIZE = 64   # inference only, no gradients -- can go bigger than training's 16
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- Load model + tokenizer from your saved best checkpoint ---
if not os.path.exists(BEST_MODEL_DIR):
    raise FileNotFoundError(f"best_model not found at {BEST_MODEL_DIR} -- check DRIVE_DIR / training finished.")

model = DistilBertForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(BEST_MODEL_DIR)
model.eval()

# --- Load the corpus to label ---
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Couldn't find {INPUT_PATH}. Put dawn_business_2015_to_2026.csv in {DRIVE_DIR}, "
        f"or edit INPUT_PATH above to point at its actual location."
    )

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

if TITLE_COL not in df.columns or TEXT_COL not in df.columns:
    raise KeyError(
        f"Expected columns '{TITLE_COL}' and '{TEXT_COL}' not both found in the file. "
        f"Edit TITLE_COL/TEXT_COL above to match your actual column names: {list(df.columns)}"
    )

df["text"] = (df[TITLE_COL].fillna("") + " " + df[TEXT_COL].fillna("")).str.strip()

# --- Dataset for batched inference ---
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

infer_ds = InferenceDataset(df["text"], tokenizer)
infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Run inference over the whole corpus ---
all_preds, all_confs = [], []
all_probs = {"Negative": [], "Neutral": [], "Positive": []}

with torch.no_grad():
    for step, batch in enumerate(infer_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_preds.extend(preds)
        all_confs.extend(probs.max(axis=1))
        all_probs["Negative"].extend(probs[:, 0])
        all_probs["Neutral"].extend(probs[:, 1])
        all_probs["Positive"].extend(probs[:, 2])

        if step % 50 == 0:
            print(f"  batch {step}/{len(infer_loader)} ({(step+1)*BATCH_SIZE} rows so far)")

df["distilbert_label"] = [ID2LABEL[p] for p in all_preds]
df["distilbert_confidence"] = all_confs
df["prob_negative"] = all_probs["Negative"]
df["prob_neutral"] = all_probs["Neutral"]
df["prob_positive"] = all_probs["Positive"]

# --- Save ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Labeled {len(df)} rows.")
print(df["distilbert_label"].value_counts())
print(f"Saved to: {OUTPUT_PATH}")
