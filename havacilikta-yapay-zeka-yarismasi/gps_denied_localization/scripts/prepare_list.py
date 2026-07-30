#!/usr/bin/env python3
"""ORB-SLAM3 mono_tuyz için görüntü listesi üretir.

Frame'ler videodan 4'er atlanarak çıkarılmış (frame_000000, 000004, ...).
Orijinal video 30 fps varsayımıyla efektif dt = 4/30 s.

Kullanım:
    python3 prepare_list.py <media_klasoru> <cikti_list.txt> [--n 400]
"""
import argparse
import os

FRAME_STEP = 4
VIDEO_FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media_dir")
    ap.add_argument("out_list")
    ap.add_argument("--n", type=int, default=400, help="kullanılacak frame sayısı")
    ap.add_argument("--dt", type=float, default=None,
                    help="frame'ler arası süre (s); verilmezse 4/30 (2024 formatı) kullanılır")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".webp")
    frames = sorted(f for f in os.listdir(args.media_dir) if f.lower().endswith(exts))
    frames = frames[: args.n]

    dt = args.dt if args.dt is not None else FRAME_STEP / VIDEO_FPS
    with open(args.out_list, "w") as f:
        f.write("# timestamp image_path\n")
        for i, name in enumerate(frames):
            f.write(f"{i * dt:.6f} {os.path.join(os.path.abspath(args.media_dir), name)}\n")

    print(f"{len(frames)} frame -> {args.out_list}")


if __name__ == "__main__":
    main()
