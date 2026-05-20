#!/usr/bin/env python3
"""Create JPEG-compressed variants of PNG test screenshots.

For each .png in testdata/, creates:
  - <name>_jpeg.jpg  (quality 50)
  - <name>_lowjpeg.jpg (quality 20)
with a symlinked/copied .cgp file.

Usage:
  python3 testgen/scripts/jpeg_augment.py
"""
import os
import shutil
from pathlib import Path

import cv2

TESTDATA = Path(__file__).resolve().parent.parent.parent / "testdata"
QUALITIES = [
    ("jpeg", 50),
    ("lowjpeg", 20),
]


def main():
    pngs = sorted(TESTDATA.glob("*.png"))
    created = 0
    skipped = 0

    for png_path in pngs:
        stem = png_path.stem
        cgp_path = TESTDATA / f"{stem}.cgp"
        if not cgp_path.exists():
            continue

        img = cv2.imread(str(png_path))
        if img is None:
            print(f"  SKIP {stem} (unreadable)")
            skipped += 1
            continue

        for suffix, quality in QUALITIES:
            jpg_name = f"{stem}_{suffix}.jpg"
            jpg_path = TESTDATA / jpg_name
            cgp_dst = TESTDATA / f"{stem}_{suffix}.cgp"

            if jpg_path.exists():
                skipped += 1
                continue

            cv2.imwrite(str(jpg_path), img,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            shutil.copy2(str(cgp_path), str(cgp_dst))
            created += 1

    print(f"Created {created} JPEG variants, skipped {skipped}")
    total = len(list(TESTDATA.glob("*.cgp")))
    print(f"Total test cases: {total}")


if __name__ == "__main__":
    main()
