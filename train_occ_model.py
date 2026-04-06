#!/usr/bin/env python3
"""
Train a binary occupancy classifier: tile (1) vs empty (0).

Uses the same 48x48 grayscale preprocessing as the tile letter model.
Crops come from training_data/<A-Z>/ (tiles) and training_data/_empty/ (empties).

Usage:
  python train_occ_model.py [--epochs 40] [--max-empty 60000]
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Preprocessing (must match board.cpp preprocess_for_cnn)
# ---------------------------------------------------------------------------

def preprocess(img):
    """48x48 grayscale, polarity-normalized, histogram-equalized, float [0,1]."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    if img.mean() < 128:
        img = 255 - img
    img = cv2.equalizeHist(img)
    return img.astype(np.float32) / 255.0

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OccDataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples  # list of (path, label)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((48, 48), dtype=np.uint8)

        if self.augment:
            # Random rotation ±5°
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT_101)
            # Random JPEG quality
            if random.random() < 0.3:
                q = random.randint(15, 70)
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            # Random brightness
            if random.random() < 0.3:
                delta = random.randint(-20, 20)
                img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        processed = preprocess(img)
        tensor = torch.from_numpy(processed).unsqueeze(0)  # [1, 48, 48]
        return tensor, label

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OccCNN(nn.Module):
    """Binary occupancy classifier. Same input size as TileCNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # 48->24
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # 24->12
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # 12->6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-empty", type=int, default=60000,
                        help="Cap empty samples to balance classes")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--onnx", default="models/occ_model.onnx")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect samples
    data_dir = Path(args.data)
    tiles = []
    for letter_dir in sorted(data_dir.iterdir()):
        if letter_dir.name.startswith("_") or not letter_dir.is_dir():
            continue
        if not letter_dir.name[0].isupper():
            continue
        for f in letter_dir.glob("*.png"):
            tiles.append((str(f), 1))

    empty_dir = data_dir / "_empty"
    empties = []
    if empty_dir.exists():
        for f in empty_dir.glob("*.png"):
            empties.append((str(f), 0))

    # Also include blanks as tiles
    blank_dir = data_dir / "_blank"
    if blank_dir.exists():
        for sub in blank_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.png"):
                    tiles.append((str(f), 1))

    print(f"Tiles: {len(tiles)}, Empties: {len(empties)}")

    # Balance: cap empties
    random.seed(42)
    if len(empties) > args.max_empty:
        random.shuffle(empties)
        empties = empties[:args.max_empty]
        print(f"Capped empties to {len(empties)}")

    all_samples = tiles + empties
    random.shuffle(all_samples)

    val_n = int(len(all_samples) * args.val_split)
    val_samples = all_samples[:val_n]
    train_samples = all_samples[val_n:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = OccDataset(train_samples, augment=True)
    val_ds = OccDataset(val_samples, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)

    model = OccCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = "models/occ_model_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        saved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            saved = f"  -> Saved best (val_acc={val_acc:.4f})"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss/train_total:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss/val_total:.4f} val_acc={val_acc:.4f}  "
              f"P={precision:.4f} R={recall:.4f} FP={fp} FN={fn}{saved}")

    # Load best and export ONNX
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, 1, 48, 48)
    torch.onnx.export(model, dummy, args.onnx,
                      opset_version=11,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      dynamo=False)
    print(f"\nExported ONNX: {args.onnx}")
    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
