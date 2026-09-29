# =============================================================================
# DistilBERT fine-tuning on KSE-100 news sentiment
# Run each "CELL" block as a separate cell in Google Colab (Runtime > GPU).
# =============================================================================

# ============ CELL 1: setup ============
get_ipython().system('pip install -q transformers accelerate scikit-learn')

from google.colab import drive, files
drive.mount('/content/drive')

import os, json, time, zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU -- Runtime > Change runtime type > GPU")

# ============ CELL 2: config ============
# Everything persists under Drive, so a disconnect never loses progress --
# resume logic in CELL 7 picks this back up automatically.
DRIVE_DIR = "/content/drive/MyDrive/finsight_distilbert"
os.makedirs(DRIVE_DIR, exist_ok=True)
CKPT_DIR = f"{DRIVE_DIR}/checkpoints"
BEST_DIR = f"{DRIVE_DIR}/best_model"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
CKPT_PATH = f"{CKPT_DIR}/last_checkpoint.pt"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
DOWNLOAD_EVERY = 5          # also push a .zip to your browser's Downloads every N epochs
DROP_AMBIGUOUS = True       # drop the 362 M056 "neither upstream label" rows from training
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ============ CELL 3: load data from Drive ============
# One-time setup: in Google Drive, create the folder DRIVE_DIR (see CELL 2)
# and drop kse100_full_row_labels.csv into it. After that this cell just
# reads it straight from Drive -- no upload prompt, ever again.
DATA_PATH = f"{DRIVE_DIR}/kse100_full_row_labels.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Put kse100_full_row_labels.csv in {DRIVE_DIR} (via the Drive web UI or "
        f"Colab's file browser on the left) and re-run this cell."
    )

full_df = pd.read_csv(DATA_PATH)
full_df["text_combined"] = (full_df["title"].fillna("") + " " + full_df["text"].fillna("")).str.strip()
full_df["label_id"] = full_df["Final_Label"].map(LABEL2ID)

train_df = full_df[full_df["Split"] == "train"].reset_index(drop=True)
val_df = full_df[full_df["Split"] == "val"].reset_index(drop=True)
test_df = full_df[full_df["Split"] == "test"].reset_index(drop=True)

if DROP_AMBIGUOUS:
    before = len(train_df)
    train_df = train_df[train_df["Confidence"] != "AMBIGUOUS"].reset_index(drop=True)
    print(f"Dropped {before - len(train_df)} AMBIGUOUS (M056) rows from training ({before} -> {len(train_df)})")

# unify the column name each split's dataset will read from
for df in (train_df, val_df, test_df):
    df["text"] = df["text_combined"]

print("train/val/test sizes:", len(train_df), len(val_df), len(test_df))

# ============ CELL 4: class weights (handles the 37.5/34.0/28.6 imbalance) ============
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["label_id"].values,
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights (Negative, Neutral, Positive):", class_weights)

# ============ CELL 5: tokenizer + datasets ============
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = NewsDataset(train_df["text"], train_df["label_id"], tokenizer)
val_ds = NewsDataset(val_df["text"], val_df["label_id"], tokenizer)
test_ds = NewsDataset(test_df["text"], test_df["label_id"], tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

# ============ CELL 6: model / optimizer ============
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ============ CELL 7: resume from Drive if a checkpoint exists ============
start_epoch = 0
best_val_f1 = 0.0
if os.path.exists(CKPT_PATH):
    print("Found existing checkpoint on Drive -- resuming...")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_f1 = ckpt.get("best_val_f1", 0.0)
    print(f"Resumed from epoch {start_epoch} (best_val_f1 so far = {best_val_f1:.4f})")
else:
    print("No checkpoint found -- starting fresh.")

# ============ CELL 8: eval helper ============
def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            p = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds.extend(p)
            trues.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    return acc, f1, preds, trues

def download_safety_copy(epoch):
    zip_path = f"/content/distilbert_epoch{epoch + 1}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(CKPT_PATH, arcname="last_checkpoint.pt")
        if os.path.isdir(BEST_DIR):
            for fname in os.listdir(BEST_DIR):
                zf.write(f"{BEST_DIR}/{fname}", arcname=f"best_model/{fname}")
    files.download(zip_path)

# ============ CELL 9: training loop ============
for epoch in range(start_epoch, EPOCHS):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(out.logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if step % 100 == 0:
            print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss {loss.item():.4f}")

    val_acc, val_f1, _, _ = evaluate(val_loader)
    elapsed = time.time() - t0
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | {elapsed / 60:.1f} min | "
        f"train_loss {total_loss / len(train_loader):.4f} | "
        f"val_acc {val_acc:.4f} | val_macro_f1 {val_f1:.4f}"
    )

    # Saved to Drive EVERY epoch -- this is what actually survives a disconnect.
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "val_acc": val_acc,
            "val_f1": val_f1,
        },
        CKPT_PATH,
    )

    # Best-weights snapshot, kept separately, overwritten only on improvement.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(BEST_DIR)
        tokenizer.save_pretrained(BEST_DIR)
        with open(f"{BEST_DIR}/metrics.json", "w") as f:
            json.dump({"epoch": epoch + 1, "val_acc": val_acc, "val_f1": val_f1}, f)
        print(f"  -> new best model saved (val_macro_f1={val_f1:.4f})")

    # Extra local safety copy pushed to your browser's Downloads every 5 epochs.
    if (epoch + 1) % DOWNLOAD_EVERY == 0 or (epoch + 1) == EPOCHS:
        download_safety_copy(epoch)

print("Training complete. Best val macro-F1:", best_val_f1)

# ============ CELL 10: final test evaluation (loads the BEST checkpoint) ============
best_model = DistilBertForSequenceClassification.from_pretrained(BEST_DIR).to(device)
model = best_model
test_acc, test_f1, preds, trues = evaluate(test_loader)
print(f"TEST accuracy: {test_acc:.4f}  |  TEST macro-F1: {test_f1:.4f}")
print(classification_report(trues, preds, target_names=["Negative", "Neutral", "Positive"]))
