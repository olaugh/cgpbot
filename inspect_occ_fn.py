#!/usr/bin/env python3
"""Show the false negative crops from the occupancy model (tiles predicted as empty)."""

import cv2
import numpy as np
import torch
from pathlib import Path
from train_occ_model import OccCNN, preprocess
import random
import html
import base64

def main():
    model = OccCNN()
    model.load_state_dict(torch.load("models/occ_model_best.pt", map_location="cpu", weights_only=True))
    model.eval()

    data_dir = Path("training_data")

    # Collect all tile samples (same split as training with seed=42)
    tiles = []
    for letter_dir in sorted(data_dir.iterdir()):
        if letter_dir.name.startswith("_") or not letter_dir.is_dir():
            continue
        if not letter_dir.name[0].isupper():
            continue
        for f in letter_dir.glob("*.png"):
            tiles.append((str(f), 1))

    blank_dir = data_dir / "_blank"
    if blank_dir.exists():
        for sub in blank_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.png"):
                    tiles.append((str(f), 1))

    empties = []
    empty_dir = data_dir / "_empty"
    if empty_dir.exists():
        for f in empty_dir.glob("*.png"):
            empties.append((str(f), 0))

    random.seed(42)
    if len(empties) > 60000:
        random.shuffle(empties)
        empties = empties[:60000]

    all_samples = tiles + empties
    random.shuffle(all_samples)

    val_n = int(len(all_samples) * 0.1)
    val_samples = all_samples[:val_n]

    # Find FN and FP
    fn_list = []
    fp_list = []
    for path, label in val_samples:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        processed = preprocess(img)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(1).item()
            conf = torch.softmax(out, dim=1)[0]

        if pred == 0 and label == 1:  # FN: tile predicted as empty
            fn_list.append((path, conf[1].item(), conf[0].item()))
        elif pred == 1 and label == 0:  # FP: empty predicted as tile
            fp_list.append((path, conf[1].item(), conf[0].item()))

    print(f"False Negatives (tiles missed): {len(fn_list)}")
    print(f"False Positives (empties called tile): {len(fp_list)}")

    # Generate HTML
    out_html = "/tmp/occ_fn_debug.html"
    with open(out_html, "w") as f:
        f.write("<!DOCTYPE html><html><head><style>")
        f.write("body{background:#222;color:#eee;font-family:monospace}")
        f.write(".grid{display:flex;flex-wrap:wrap;gap:8px}")
        f.write(".card{text-align:center;background:#333;padding:6px;border-radius:4px}")
        f.write(".card img{width:96px;height:96px;image-rendering:pixelated}")
        f.write(".fn{border:2px solid #f88} .fp{border:2px solid #8f8}")
        f.write("</style></head><body>")

        f.write(f"<h2>False Negatives: {len(fn_list)} tiles predicted as empty</h2>")
        f.write("<div class='grid'>")
        for path, tile_conf, empty_conf in sorted(fn_list, key=lambda x: x[1]):
            img = cv2.imread(path)
            if img is None:
                continue
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode()
            name = Path(path).stem
            letter = Path(path).parent.name
            f.write(f"<div class='card fn'>")
            f.write(f"<img src='data:image/png;base64,{b64}'>")
            f.write(f"<div>{letter} tile_conf={tile_conf:.3f}</div>")
            f.write(f"<div style='font-size:0.7em;color:#888'>{name[:40]}</div>")
            f.write(f"</div>")
        f.write("</div>")

        if fp_list:
            f.write(f"<h2>False Positives: {len(fp_list)} empties predicted as tile</h2>")
            f.write("<div class='grid'>")
            for path, tile_conf, empty_conf in sorted(fp_list, key=lambda x: -x[1]):
                img = cv2.imread(path)
                if img is None:
                    continue
                _, buf = cv2.imencode(".png", img)
                b64 = base64.b64encode(buf).decode()
                name = Path(path).stem
                f.write(f"<div class='card fp'>")
                f.write(f"<img src='data:image/png;base64,{b64}'>")
                f.write(f"<div>tile_conf={tile_conf:.3f}</div>")
                f.write(f"<div style='font-size:0.7em;color:#888'>{name[:40]}</div>")
                f.write(f"</div>")
            f.write("</div>")

        f.write("</body></html>")

    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
