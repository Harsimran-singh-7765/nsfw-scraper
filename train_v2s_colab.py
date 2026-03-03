"""
NSFW Classifier — EfficientNet-V2-S Training Script (Refined)
============================================================
Colab-ready. Optimized for 'dataset_latest.zip'.
Saves best model and metadata back to Google Drive.

Usage:
1. Open in Google Colab.
2. Ensure you have the 'dataset_latest.zip' in your Drive or a share link.
3. Run all cells.
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
# If you have the file in your own Drive:
DRIVE_ZIP     = "/content/drive/MyDrive/NSFW_Dataset/dataset_latest.zip"
# If you are using a SHARED LINK from another account, paste the ID here:
# NOTE: Paste ONLY the ID (the long string of letters/numbers), NOT the whole URL.
# Example: if link is https://drive.google.com/file/d/1A2B3C.../view, ID is '1A2B3C...'
SHARED_FILE_ID = "18nXsCJyccJU3ZBksp02Depyd4vNESZ1u" 

EXTRACT_DIR   = "/content/dataset_latest"
SAVE_DIR      = "/content/drive/MyDrive/nsfw_v2s_model"

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
def download_from_link(file_id, output_path):
    """Download from a public Google Drive link using gdown."""
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


def unzip_dataset(zip_path):
    """Unzip to local Colab storage."""
    extract_path = Path(EXTRACT_DIR)
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"Dataset already extracted at {EXTRACT_DIR}. Skipping.")
        return

    print(f"Unzipping {zip_path}  →  {EXTRACT_DIR} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print("Unzip complete.")


def discover_raw_data_root(base: Path) -> Path:
    """Find the directory containing class folders."""
    for root, dirs, _ in os.walk(base):
        # Check if at least 2 class folders exist here
        if len(set(dirs) & set(CLASSES)) >= 2:
            return Path(root)
    return base


def collect_samples(raw_root: Path):
    """Collect (image_path, class_idx) pairs."""
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    samples = []

    for idx, cls in enumerate(CLASSES):
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            print(f"  [WARN] Missing class folder: {cls_dir}")
            continue

        # Look in IMAGES/ sub-folder first
        images_dir = cls_dir / "IMAGES"
        search_dir = images_dir if images_dir.exists() else cls_dir

        found = []
        for p in search_dir.iterdir():
            if not p.is_file():
                continue
            
            # Use lower-case filename for check
            fname = p.name.lower()
            # Check if any valid extension is in the filename (to handle .jpg?width=...)
            if any(ext in fname for ext in VALID_EXTS):
                found.append(p)

        print(f"  {cls:<10} → {len(found):>6} images")
        samples.extend([(str(p), idx) for p in found])

    return samples


class NSFWDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
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
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

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

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, correct / total


# ==========================
# MAIN
# ==========================
def main():
    # ── 0. Drive Setup ─────────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass

    # Handle download path
    zip_path = Path(DRIVE_ZIP)
    if SHARED_FILE_ID:
        local_zip = "/content/dataset_latest.zip"
        print(f"Downloading dataset from shared link (ID: {SHARED_FILE_ID})...")
        download_from_link(SHARED_FILE_ID, local_zip)
        zip_path = Path(local_zip)
    
    if not zip_path.exists():
        print(f"❌ Error: Dataset zip not found at {zip_path}")
        return

    # ── 1. Data Processing ─────────────────────────────────────────
    unzip_dataset(zip_path)
    raw_root = discover_raw_data_root(Path(EXTRACT_DIR))

    print("\n📂 Collecting samples...")
    all_samples = collect_samples(raw_root)
    print(f"Total images found: {len(all_samples)}")

    if not all_samples:
        print("❌ Error: No images found!")
        return

    labels = [s[1] for s in all_samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(all_samples)),
        test_size=0.15,
        stratify=labels,
        random_state=SEED,
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

    print(f"Split  →  train: {len(train_samples)}  |  val: {len(val_samples)}")

    train_ds = NSFWDataset(train_samples, train_tfms)
    val_ds   = NSFWDataset(val_samples,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── 2. Model Initialization ────────────────────────────────────
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

        history.append({"epoch": epoch, "train_acc": tr_acc, "train_loss": tr_loss,
                        "val_acc": vl_acc, "val_loss": vl_loss})

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("  ← 💾 saved", end="")
        print()
        torch.save(model.state_dict(), save_dir / "last_model.pth")

    print(f"\n✅ Training complete. Best val acc: {best_acc:.4f}")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 4. Post-Training Dynamic Quantization ──────────────────────
    print("\n⚙️  Quantizing model (INT8 dynamic) ...")
    model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location="cpu"))
    model.eval().cpu()

    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    q_path = save_dir / "best_model_int8.pth"
    torch.save(model_q.state_dict(), q_path)

    fp32_mb = os.path.getsize(save_dir / "best_model.pth") / 1e6
    int8_mb = os.path.getsize(q_path) / 1e6
    print(f"  FP32 model : {fp32_mb:.1f} MB")
    print(f"  INT8 model : {int8_mb:.1f} MB  ({100*(1-int8_mb/fp32_mb):.0f}% smaller)")
    print(f"\nAll outputs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
