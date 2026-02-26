"""
NSFW Classifier — EfficientNet-V2-S Training Script
====================================================
Colab-ready. Upload unified_training_dataset.zip to Google Drive,
then run this top to bottom. Saves best model back to Drive.

Dataset structure inside the zip:
  raw_data/{class}/IMAGES/*.jpg   (drawings, hentai, neutral, porn, sexy)
"""

import os
import sys
import json
import zipfile
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================
# CONFIG  (edit these)
# ==========================
DRIVE_ZIP     = "/content/drive/MyDrive/unified_training_dataset.zip"
EXTRACT_DIR   = "/content/unified_data"
SAVE_DIR      = "/content/drive/MyDrive/nsfw_v2s"

MODEL_NAME    = "tf_efficientnetv2_s"
CLASSES       = ["drawings", "hentai", "neutral", "porn", "sexy"]

BATCH_SIZE    = 64       # reduce to 32 if you hit OOM
NUM_WORKERS   = 2
EPOCHS        = 25
LR            = 3e-4
LABEL_SMOOTH  = 0.1
SEED          = 42


# ==========================
# REPRODUCIBILITY
# ==========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==========================
# DATASET SETUP
# ==========================
def unzip_dataset():
    """Unzip from Drive if not already done."""
    extract_path = Path(EXTRACT_DIR)
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"Dataset already extracted at {EXTRACT_DIR}. Skipping.")
        return

    print(f"Unzipping {DRIVE_ZIP}  →  {EXTRACT_DIR} ...")
    with zipfile.ZipFile(DRIVE_ZIP, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print("Unzip complete.")


def discover_raw_data_root(base: Path) -> Path:
    """
    Walk until we find a dir that contains our class folders.
    Handles nested zips like data/raw_data/drawings/...
    """
    for root, dirs, _ in os.walk(base):
        if len(set(dirs) & set(CLASSES)) >= 2:
            p = Path(root)
            print(f"Found dataset root: {p}")
            return p
    return base


def collect_samples(raw_root: Path):
    """
    Collect all (image_path, class_idx) pairs.
    Structure: raw_root/{class}/IMAGES/*.{jpg,jpeg,png,...}
    Falls back to raw_root/{class}/*.* if no IMAGES subdir exists.
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    samples = []

    for idx, cls in enumerate(CLASSES):
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            print(f"  [WARN] Missing class folder: {cls_dir}")
            continue

        # Prefer IMAGES/ sub-folder, otherwise use cls_dir directly
        images_dir = cls_dir / "IMAGES"
        search_dir = images_dir if images_dir.exists() else cls_dir

        found = [
            p for p in search_dir.iterdir()
            if p.suffix.lower() in VALID_EXTS and p.is_file()
        ]
        print(f"  {cls:<10} → {len(found):>6} images")
        samples.extend([(str(p), idx) for p in found])

    return samples


class NSFWDataset(Dataset):
    """Lightweight image dataset with per-item transform."""

    def __init__(self, samples, transform=None):
        self.samples   = samples    # list of (path_str, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a black image on corrupt file so training doesn't crash
            img = Image.new("RGB", (224, 224), 0)

        if self.transform:
            img = self.transform(img)
        return img, label


# ==========================
# TRANSFORMS
# ==========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # applied after ToTensor
    # ↑ inserted after Normalize below via Compose order
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Note: RandomErasing works on tensors, so we do it last
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ==========================
# TRAIN / VAL HELPERS
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

        pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                         acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)

        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

        pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                         acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


# ==========================
# MAIN
# ==========================
def main():
    # ── 0. Mount Drive (Colab only) ────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass  # not in Colab, skip

    # ── 1. Data ────────────────────────────────────────────────────
    unzip_dataset()
    raw_root = discover_raw_data_root(Path(EXTRACT_DIR))

    print("\n📂 Collecting samples...")
    all_samples = collect_samples(raw_root)
    print(f"\nTotal images found: {len(all_samples)}")

    labels = [s[1] for s in all_samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(all_samples)),
        test_size=0.15,
        stratify=labels,
        random_state=SEED,
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

    print(f"\nSplit  →  train: {len(train_samples)}  |  val: {len(val_samples)}")
    print(f"Class balance (train): "
          f"{ {CLASSES[k]:v for k,v in Counter(s[1] for s in train_samples).items()} }")

    train_ds = NSFWDataset(train_samples, train_tfms)
    val_ds   = NSFWDataset(val_samples,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── 2. Model ───────────────────────────────────────────────────
    print(f"\n🔧 Loading {MODEL_NAME} ...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Training Loop ───────────────────────────────────────────
    best_acc = 0.0
    history  = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.4f}", end="")

        history.append({"epoch": epoch,
                        "train_acc": tr_acc, "train_loss": tr_loss,
                        "val_acc":   vl_acc, "val_loss":   vl_loss})

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("  ← 💾 saved", end="")
        print()

        # Always keep latest checkpoint too
        torch.save(model.state_dict(), save_dir / "last_model.pth")

    print(f"\n✅ Training complete.  Best val acc: {best_acc:.4f}")

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 4. Post-Training Dynamic Quantization ──────────────────────
    print("\n⚙️  Quantizing model (INT8 dynamic) ...")
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location="cpu"))
    model.eval().cpu()

    model_q = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},            # Linear layers quantized; Conv2d not supported in dynamic
        dtype=torch.qint8
    )

    q_path = save_dir / "best_model_int8.pth"
    torch.save(model_q.state_dict(), q_path)

    # Print size comparison
    fp32_mb = os.path.getsize(save_dir / "best_model.pth") / 1e6
    int8_mb = os.path.getsize(q_path) / 1e6
    print(f"  FP32 model : {fp32_mb:.1f} MB")
    print(f"  INT8 model : {int8_mb:.1f} MB  ({100*(1-int8_mb/fp32_mb):.0f}% smaller)")
    print(f"\nAll outputs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
