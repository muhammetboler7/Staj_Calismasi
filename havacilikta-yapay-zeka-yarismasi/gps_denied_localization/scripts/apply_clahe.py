#!/usr/bin/env python3
"""Bir görüntü klasörüne CLAHE (kontrast sınırlı adaptif histogram eşitleme)
uygulayıp PNG olarak yeni bir klasöre yazar. Düşük kontrastlı (termal gibi)
görüntülerde ORB özellik tespitini iyileştirmek için kullanılır.

Kullanım:
    python3 apply_clahe.py src_dir dst_dir --n 1450 --clip 3.0 --tile 8
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (verilmezse hepsi)")
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--tile", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    frames = sorted(f for f in os.listdir(args.src_dir) if f.lower().endswith(exts))
    if args.n:
        frames = frames[: args.n]

    clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(args.tile, args.tile))

    for i, f in enumerate(frames):
        im = cv2.imread(os.path.join(args.src_dir, f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"UYARI: okunamadi: {f}")
            continue
        out = clahe.apply(im)
        name = os.path.splitext(f)[0] + ".png"
        cv2.imwrite(os.path.join(args.dst_dir, name), out)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(frames)}")

    print(f"Bitti: {len(frames)} kare -> {args.dst_dir}")


if __name__ == "__main__":
    main()
