#!/usr/bin/env python3
"""ORB-SLAM3 yörüngesini vo_xy_sapma.png ile aynı düzende çizer:
üst-sol geniş XY yörünge (kalibrasyon penceresi vurgulu), sağda X/Y/Z eksen
alt grafikleri, altta tam genişlik sapma grafiği.

align_and_evaluate.py'deki hizalama fonksiyonlarını (umeyama, plane2d,
kuaterniyon->düzlem projeksiyonu) aynen kullanır; sadece çizimi değiştirir.

Kullanım:
    python3 plot_vo_style.py traj.txt translations.csv \
        --align-start 650 --align-n 350 --predict-n 1250 \
        --dt 0.033333 --mode plane2d --out results/plot.png
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_and_evaluate import (  # noqa: E402
    load_gt, load_traj, umeyama, umeyama_2d, make_plane_projector,
)


def build_predictor(traj, quats, align_ids, dst, mode, has_z):
    """dst: (N,2) x,y veya (N,3) x,y,z. has_z=False ise predict(i) sadece
    (x,y) döner (2-vektör) - z hiç kestirilmez/uydurulmaz."""
    if mode == "plane2d":
        P = make_plane_projector(quats, align_ids)
        n = np.cross(P[:, 0], P[:, 1])
        src2 = np.array([traj[i] for i in align_ids]) @ P
        s, A, t2 = umeyama_2d(src2, dst[:, :2])
        if has_z:
            h = np.array([traj[i] for i in align_ids]) @ n
            a_z, c = np.polyfit(h, dst[:, 2], 1)

            def predict(i):
                p2 = A @ (P.T @ traj[i]) + t2
                return np.array([p2[0], p2[1], a_z * float(n @ traj[i]) + c])
        else:
            def predict(i):
                return A @ (P.T @ traj[i]) + t2
        return predict, s
    else:
        src = np.array([traj[i] for i in align_ids])
        if has_z:
            s, R, t = umeyama(src, dst)

            def predict(i):
                return s * R @ traj[i] + t
        else:
            # 3B mod ama z GT yok: yine de 3B benzerlik coz (z=0 hedefiyle,
            # align_and_evaluate.py'nin eski davranisi), ama sonuc raporunda
            # z hic kullanilmaz - has_z=False durumunda plane2d onerilir.
            dst3 = np.hstack([dst, np.zeros((len(dst), 1))])
            s, R, t = umeyama(src, dst3)

            def predict(i):
                p = s * R @ traj[i] + t
                return p[:2]
        return predict, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("gt_csv")
    ap.add_argument("--align-start", type=int, required=True)
    ap.add_argument("--align-n", type=int, required=True)
    ap.add_argument("--predict-n", type=int, required=True,
                    help="hizalama penceresinden SONRA kaç frame tahmin edilecek")
    ap.add_argument("--mode", choices=["3d", "plane2d"], default="plane2d")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--title", default="Nadir ORB-SLAM3 — X/Y/Z konum kestirimi")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt, has_z = load_gt(args.gt_csv)
    traj, quats = load_traj(args.traj, args.dt)

    a0, a1 = args.align_start, args.align_start + args.align_n
    p1 = a1 + args.predict_n
    align_ids = [i for i in range(a0, a1) if i in traj and i in gt]
    pred_ids = [i for i in range(a1, p1) if i in traj]
    if len(align_ids) < 10:
        raise SystemExit(f"Hizalama için yeterli izlenen frame yok! ({len(align_ids)})")

    if has_z:
        dst = np.array([[gt[i][0], gt[i][1], gt[i][2]] for i in align_ids])
    else:
        dst = np.array([[gt[i][0], gt[i][1]] for i in align_ids])
    predict, scale = build_predictor(traj, quats, align_ids, dst, args.mode, has_z)

    all_ids = sorted(set(align_ids) | set(pred_ids))
    frames, px, py, pz = [], [], [], []
    for i in all_ids:
        p = predict(i)
        frames.append(i); px.append(p[0]); py.append(p[1])
        pz.append(p[2] if has_z else np.nan)
    frames = np.array(frames); px = np.array(px); py = np.array(py); pz = np.array(pz)

    gt_ids = sorted(i for i in range(a0, p1) if i in gt)
    gx = np.array([gt[i][0] for i in gt_ids])
    gy = np.array([gt[i][1] for i in gt_ids])
    gz = np.array([gt[i][2] if has_z else np.nan for i in gt_ids])
    gt_ids = np.array(gt_ids)

    pred_common = [i for i in pred_ids if i in gt]
    if pred_common:
        pdx = np.array([predict(i)[0] for i in pred_common]) - np.array([gt[i][0] for i in pred_common])
        pdy = np.array([predict(i)[1] for i in pred_common]) - np.array([gt[i][1] for i in pred_common])
        if has_z:
            pdz = (np.array([predict(i)[2] for i in pred_common])
                   - np.array([gt[i][2] for i in pred_common]))
            rmse = float(np.sqrt((pdx**2 + pdy**2 + pdz**2).mean()))
        else:
            rmse = float(np.sqrt((pdx**2 + pdy**2).mean()))
    else:
        rmse = float("nan")

    plot(args, frames, px, py, pz, gt_ids, gx, gy, gz, align_ids, pred_ids, rmse, has_z)
    dims = "3B" if has_z else "2B (x,y)"
    print(f"Umeyama olcegi: {scale:.4f}  [mod: {args.mode}]")
    print(f"Hizalama [{a0},{a1}): {len(align_ids)}/{args.align_n} izlendi")
    print(f"Tahmin   [{a1},{p1}): {len(pred_ids)}/{args.predict_n} izlendi")
    print(f"Tahmin bolgesi {dims} RMSE: {rmse:.3f} m  ({len(pred_common)} ortak frame)")


def plot(args, frames, px, py, pz, gt_ids, gx, gy, gz, align_ids, pred_ids, rmse, has_z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BLACK, ORANGE, LBLUE, GREEN, RED, PURPLE = (
        "#1a1a19", "#eb6834", "#86b6ef", "#008300", "#e34948", "#4a3aa7")

    a0 = args.align_start
    a1 = a0 + args.align_n
    p1 = a1 + args.predict_n

    nrows = 4 if has_z else 3
    xy_span = 3 if has_z else 2
    fig = plt.figure(figsize=(18, 13 if has_z else 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(nrows, 2, height_ratios=[1] * nrows, hspace=0.45, wspace=0.22)

    def style(ax):
        ax.grid(True, color="#eeeeea", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def shade(ax):
        ax.axvspan(a0, a1, color=LBLUE, alpha=0.18, lw=0)
        ax.axvline(a1, color="#888880", lw=1, ls=":")

    # ---- ust-sol: XY yorunge (buyuk, has_z'ye gore 2-3 satir kapliyor) ----
    ax1 = fig.add_subplot(gs[0:xy_span, 0])
    ax1.plot(gx, gy, color=BLACK, lw=2.2, label="Gerçek (GT)", zorder=3)
    ax1.plot(px, py, color=ORANGE, lw=2.2, ls="--", label="Tahmin (ORB-SLAM3)", zorder=3)

    align_mask = np.isin(frames, align_ids)
    ax1.scatter(px[align_mask], py[align_mask], color=LBLUE, s=14, zorder=4,
                label=f"kalibrasyon ({a0}-{a1-1})")
    if len(gx):
        ax1.scatter([gx[0]], [gy[0]], color=GREEN, s=140, zorder=5, label="başlangıç")
    if len(gx):
        ax1.scatter([gx[-1]], [gy[-1]], color=RED, s=140, marker="X", zorder=5,
                    label=f"bitiş ({gt_ids[-1]})")

    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title("Üst görünüm yörünge (X-Y)", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, loc="best")
    ax1.set_aspect("equal", adjustable="datalim")
    style(ax1)

    # ---- sag: X, Y, Z eksenleri ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gt_ids, gx, color=BLACK, lw=1.8, label="gerçek")
    ax2.plot(frames, px, color=ORANGE, lw=1.8, ls="--", label="tahmin")
    shade(ax2); ax2.legend(frameon=False, fontsize=8)
    ax2.set_ylabel("X (m)"); ax2.set_title("X ekseni: gerçek vs tahmin", fontsize=12, fontweight="bold")
    style(ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(gt_ids, gy, color=BLACK, lw=1.8, label="gerçek")
    ax3.plot(frames, py, color=ORANGE, lw=1.8, ls="--", label="tahmin")
    shade(ax3); ax3.legend(frameon=False, fontsize=8)
    ax3.set_ylabel("Y (m)"); ax3.set_title("Y ekseni: gerçek vs tahmin", fontsize=12, fontweight="bold")
    style(ax3)

    if has_z:
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(gt_ids, gz, color=BLACK, lw=1.8, label="gerçek")
        ax4.plot(frames, pz, color=ORANGE, lw=1.8, ls="--", label="tahmin")
        shade(ax4); ax4.legend(frameon=False, fontsize=8)
        ax4.set_ylabel("Z (m)"); ax4.set_title("Z ekseni (irtifa): gerçek vs tahmin", fontsize=12, fontweight="bold")
        style(ax4)

    # ---- alt: tam genislik sapma ----
    ax5 = fig.add_subplot(gs[xy_span, :])
    common = np.array(sorted(set(frames.tolist()) & set(gt_ids.tolist())))
    gxd = dict(zip(gt_ids.tolist(), gx)); gyd = dict(zip(gt_ids.tolist(), gy))
    pxd = dict(zip(frames.tolist(), px)); pyd = dict(zip(frames.tolist(), py))
    ex = np.array([abs(pxd[i] - gxd[i]) for i in common])
    ey = np.array([abs(pyd[i] - gyd[i]) for i in common])

    ax5.axvspan(a0, a1, color=LBLUE, alpha=0.18, lw=0, label="kalibrasyon")
    ax5.axvline(a1, color="#888880", lw=1, ls=":")
    ax5.plot(common, ex, color="#2a78d6", lw=1.6, label="|x sapma|")
    ax5.plot(common, ey, color=GREEN, lw=1.6, label="|y sapma|")

    if has_z:
        gzd = dict(zip(gt_ids.tolist(), gz)); pzd = dict(zip(frames.tolist(), pz))
        ez = np.array([abs(pzd[i] - gzd[i]) for i in common])
        etot = np.sqrt(ex**2 + ey**2 + ez**2)
        ax5.plot(common, ez, color=PURPLE, lw=1.6, label="|z sapma|")
        ax5.plot(common, etot, color=BLACK, lw=2.2, label="toplam 3B sapma")
        dims = "3B"
    else:
        etot = np.sqrt(ex**2 + ey**2)
        ax5.plot(common, etot, color=BLACK, lw=2.2, label="toplam 2B sapma")
        dims = "2B (x,y)"

    ax5.set_xlabel("kare no"); ax5.set_ylabel("sapma (m)")
    ax5.set_title("Sapma (|tahmin - gerçek|)", fontsize=12, fontweight="bold")
    ax5.legend(frameon=False, fontsize=9, ncol=5, loc="upper left")
    style(ax5)

    fig.suptitle(
        f"{args.title}  (kalib {a0}-{a1-1}, test {a1}-{p1-1} | "
        f"test {dims} RMSE={rmse:.1f} m)",
        fontsize=15, fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
