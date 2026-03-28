#!/usr/bin/env python3
"""Train or fine-tune the tile CNN model (A-Z classifier).

Preprocessing must exactly match board.cpp preprocess_for_cnn():
  1. Resize to 48x48 (INTER_AREA)
  2. Convert to grayscale
  3. Polarity normalize: invert if mean < 128 (ensure light background)
  4. Histogram equalize
  5. Convert to float [0, 1]

Usage:
  # Train from scratch on board + rack data:
  python train_tile_model.py --data training_data rack_training_data --epochs 60

  # Fine-tune existing model on rack data only:
  python train_tile_model.py --data rack_training_data --resume models/tile_model_best.pt --epochs 20 --lr 0.0003

  # Train with board data, augmented with rack data:
  python train_tile_model.py --data training_data --aux-data rack_training_data --aux-weight 3 --epochs 60
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


# Must match board.cpp preprocess_for_cnn()
CNN_INPUT_SIZE = 48
NUM_CLASSES = 26


def preprocess(img_bgr):
    """Replicate C++ preprocess_for_cnn exactly."""
    resized = cv2.resize(img_bgr, (CNN_INPUT_SIZE, CNN_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    # Polarity normalize: ensure light background
    if gray.mean() < 128:
        gray = 255 - gray

    # Histogram equalization
    gray = cv2.equalizeHist(gray)

    return gray.astype(np.float32) / 255.0


class TileDataset(Dataset):
    """ImageFolder-style dataset for tile crops. Expects dirs A/ through Z/."""

    def __init__(self, root_dirs, augment=False):
        self.samples = []  # (path, label_idx)
        self.augment = augment

        if isinstance(root_dirs, (str, Path)):
            root_dirs = [root_dirs]

        for root in root_dirs:
            root = Path(root)
            if not root.exists():
                print(f"Warning: {root} does not exist, skipping")
                continue
            for letter_idx in range(26):
                letter = chr(ord('A') + letter_idx)
                letter_dir = root / letter
                if not letter_dir.exists():
                    continue
                for img_path in sorted(letter_dir.glob("*.png")):
                    self.samples.append((str(img_path), letter_idx))

        print(f"  Loaded {len(self.samples)} samples from {root_dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback: return zeros
            return torch.zeros(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE), label

        if self.augment:
            img = self._augment(img)

        gray = preprocess(img)
        tensor = torch.from_numpy(gray).unsqueeze(0)  # 1xHxW
        return tensor, label

    def _augment(self, img):
        h, w = img.shape[:2]

        # Random small rotation (-5 to +5 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random small translation (up to 8% of size)
        if random.random() < 0.3:
            dx = random.randint(-max(1, w // 12), max(1, w // 12))
            dy = random.randint(-max(1, h // 12), max(1, h // 12))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random scale (90%-110%)
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            # Crop/pad back to original size
            if new_w > w:
                x0 = (new_w - w) // 2
                img = img[:, x0:x0 + w]
            elif new_w < w:
                pad = w - new_w
                img = cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2,
                                         cv2.BORDER_REPLICATE)
            if new_h > h:
                y0 = (new_h - h) // 2
                img = img[y0:y0 + h, :]
            elif new_h < h:
                pad = h - new_h
                img = cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0,
                                         cv2.BORDER_REPLICATE)

        # JPEG compression simulation
        if random.random() < 0.3:
            quality = random.randint(15, 70)
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # Random brightness/contrast
        if random.random() < 0.2:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)     # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img


class TileCNN(nn.Module):
    """Must match the architecture in board.cpp / existing ONNX model."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 0
            nn.ReLU(),                          # 1
            nn.MaxPool2d(2, 2),                 # 2: 48->24
            nn.Conv2d(16, 32, 3, padding=1),    # 3
            nn.ReLU(),                          # 4
            nn.MaxPool2d(2, 2),                 # 5: 24->12
            nn.Conv2d(32, 64, 3, padding=1),    # 6
            nn.ReLU(),                          # 7
            nn.MaxPool2d(2, 2),                 # 8: 12->6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 0
            nn.Linear(64 * 6 * 6, 128),         # 1
            nn.ReLU(),                          # 2
            nn.Dropout(0.3),                    # 3
            nn.Linear(128, num_classes),         # 4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            for i in range(labels.size(0)):
                li = labels[i].item()
                per_class_total[li] += 1
                if predicted[i].item() == li:
                    per_class_correct[li] += 1
    acc = correct / max(1, total)
    loss = total_loss / max(1, total)
    return loss, acc, per_class_correct, per_class_total


def export_onnx(model, path, device):
    model.eval()
    dummy = torch.randn(1, 1, CNN_INPUT_SIZE, CNN_INPUT_SIZE).to(device)
    torch.onnx.export(model, dummy, path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                      opset_version=11)
    print(f"Exported ONNX model to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train tile CNN model')
    parser.add_argument('--data', nargs='+', required=True,
                        help='Training data directories (ImageFolder format)')
    parser.add_argument('--aux-data', nargs='*', default=[],
                        help='Auxiliary data to oversample (e.g., rack crops)')
    parser.add_argument('--aux-weight', type=int, default=3,
                        help='Oversampling weight for auxiliary data')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (.pt file)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='models/tile_model_best.pt')
    parser.add_argument('--onnx', type=str, default='models/tile_model.onnx')
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    print("Loading primary data...")
    primary = TileDataset(args.data, augment=not args.no_augment)

    datasets = [primary]
    if args.aux_data:
        print(f"Loading auxiliary data (weight={args.aux_weight})...")
        for _ in range(args.aux_weight):
            aux = TileDataset(args.aux_data, augment=not args.no_augment)
            datasets.append(aux)

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else primary

    # Train/val split
    n = len(full_dataset)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    # Create non-augmented version for validation
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = TileCNN().to(device)
    if args.resume:
        print(f"Loading weights from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device)
        val_loss, val_acc, pc_correct, pc_total = evaluate(model, val_loader,
                                                           criterion, device)
        scheduler.step()

        # Find worst classes
        worst = []
        for i in range(NUM_CLASSES):
            if pc_total[i] > 0:
                acc = pc_correct[i] / pc_total[i]
                if acc < 1.0:
                    worst.append((acc, chr(ord('A') + i), pc_correct[i], pc_total[i]))
        worst.sort()

        worst_str = ""
        if worst:
            worst_str = " worst: " + ", ".join(
                f"{l}={c}/{t}" for a, l, c, t in worst[:5])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{worst_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # Load best and export ONNX
    model.load_state_dict(torch.load(args.output, map_location=device,
                                     weights_only=True))
    export_onnx(model, args.onnx, device)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    _, final_acc, pc_correct, pc_total = evaluate(model, val_loader,
                                                  criterion, device)
    print(f"Val accuracy: {final_acc:.4f}")
    print("\nPer-letter accuracy:")
    for i in range(NUM_CLASSES):
        if pc_total[i] > 0:
            acc = pc_correct[i] / pc_total[i]
            mark = " ***" if acc < 1.0 else ""
            print(f"  {chr(ord('A') + i)}: {pc_correct[i]}/{pc_total[i]} = {acc:.1%}{mark}")


if __name__ == '__main__':
    main()
