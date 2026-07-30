#!/usr/bin/env python3
"""GPS kopması senaryosu: ORB-SLAM3 mono yörüngesini bilinen frame'lerin
translation verisiyle hizalar, sonraki frame'lerin konumunu tahmin eder ve
ground truth ile karşılaştırır.

Monoküler SLAM ölçeği belirsizdir; hizalama penceresinde GT ile Umeyama
benzerlik dönüşümü (ölçek + rotasyon + öteleme) kestirilir, GPS'siz bölüme
uygulanır. GT'de translation_z varsa z ekseni de hizalanır ve değerlendirilir.

Kullanım:
    python3 align_and_evaluate.py traj.txt translations.csv \
        --align-start 600 --align-n 200 --predict-n 400 \
        --mode plane2d --dt 0.033333 --out-dir results/
"""
import argparse
import csv
import os

import numpy as np

DEFAULT_DT = 4 / 30.0  # 2024 formatı: her 4. frame, 30 fps video


def load_gt(csv_path):
    """frame_index (satır sırası) -> (x, y) veya (x, y, z). Ayraç , veya ; olabilir."""
    gt = {}
    has_z = False
    with open(csv_path) as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        for i, row in enumerate(csv.DictReader(f, delimiter=delim)):
            if "translation_z" in row and row["translation_z"] not in (None, ""):
                gt[i] = (float(row["translation_x"]), float(row["translation_y"]),
                         float(row["translation_z"]))
                has_z = True
            else:
                gt[i] = (float(row["translation_x"]), float(row["translation_y"]))
    return gt, has_z


def load_traj(traj_path, dt):
    """TUM formatı -> frame_index -> pos(3,), quats: frame_index -> (qx,qy,qz,qw)"""
    traj, quats = {}, {}
    with open(traj_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            idx = int(round(float(parts[0]) / dt))
            traj[idx] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            quats[idx] = np.array([float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7])])
    return traj, quats


def quat_to_optical_axis(q):
    """Kuaterniyon (x,y,z,w) -> kameranın optik ekseninin (kamera +z) dünya
    koordinatındaki yönü = R_wc'nin 3. sütunu."""
    x, y, z, w = q
    return np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


def umeyama(src, dst):
    """src (N,3) -> dst (N,3) benzerlik dönüşümü: s, R, t  (dst ≈ s R src + t)"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def make_plane_projector(quats, align_ids):
    """Hizalama penceresindeki ortalama optik ekseni 'aşağı' (yer düzlemi
    normali) kabul edip, 3B SLAM konumlarını bu düzleme izdüşüren 3x2'lik
    taban döndürür. Nadir bakışta optik eksen ≈ yerçekimi yönü."""
    n = np.mean([quat_to_optical_axis(quats[i]) for i in align_ids], axis=0)
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2], axis=1)  # (3,2)


def umeyama_2d(src, dst):
    """src (N,2) -> dst (N,2), yansıma da denenir (izdüşüm sonrası el
    yönü/chirality belirsiz). En düşük fit hatasını veren (s, A, t) döner."""
    best = None
    for flip in (1.0, -1.0):
        s_f = src * np.array([1.0, flip])
        mu_s, mu_d = s_f.mean(0), dst.mean(0)
        xs, xd = s_f - mu_s, dst - mu_d
        cov = xd.T @ xs / len(s_f)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[1, 1] = -1
        R = U @ S @ Vt
        var_s = (xs ** 2).sum() / len(s_f)
        s = np.trace(np.diag(D) @ S) / var_s
        t = mu_d - s * R @ mu_s
        A = s * R @ np.diag([1.0, flip])
        err = np.linalg.norm((src @ A.T + t) - dst, axis=1)
        rmse = np.sqrt((err ** 2).mean())
        if best is None or rmse < best[0]:
            best = (rmse, s, A, t)
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("gt_csv")
    ap.add_argument("--align-start", type=int, default=0,
                    help="hizalama penceresinin başladığı frame indeksi")
    ap.add_argument("--align-n", type=int, default=200)
    ap.add_argument("--predict-n", type=int, default=200)
    ap.add_argument("--align-last", type=int, default=0,
                    help="0=tüm pencere; K>0 ise hizalamada pencerenin sadece son K frame'i kullanılır")
    ap.add_argument("--mode", choices=["3d", "plane2d"], default="plane2d",
                    help="3d: 3B Umeyama. plane2d: kamera yöneliminden yer düzlemi "
                         "kestirilip 2B hizalama (varsayılan; düz-hat pencerelerinde kararlı)")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT,
                    help="frame'ler arası süre (s); traj timestamp -> frame index eşlemesi için")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gt, has_z = load_gt(args.gt_csv)
    traj, quats = load_traj(args.traj, args.dt)
    a0, a1 = args.align_start, args.align_start + args.align_n
    p1 = a1 + args.predict_n

    align_ids = [i for i in range(a0, a1) if i in traj and i in gt]
    if args.align_last > 0:
        align_ids = align_ids[-args.align_last:]
    pred_ids = [i for i in range(a1, p1) if i in traj]
    print(f"GT z sütunu: {'VAR' if has_z else 'YOK'}")
    print(f"Hizalama penceresi [{a0},{a1}): {len(align_ids)}/{args.align_n} frame izlendi")
    print(f"Tahmin penceresi   [{a1},{p1}): {len(pred_ids)}/{args.predict_n} frame izlendi")
    if len(align_ids) < 10:
        raise SystemExit("Hizalama için yeterli izlenen frame yok!")

    src = np.array([traj[i] for i in align_ids])
    dst = np.array([[gt[i][0], gt[i][1], gt[i][2] if has_z else 0.0]
                    for i in align_ids])

    if args.mode == "plane2d":
        P = make_plane_projector(quats, align_ids)           # (3,2)
        n = np.cross(P[:, 0], P[:, 1])                       # düzlem normali
        s, A, t2 = umeyama_2d(src @ P, dst[:, :2])
        # z: düzlem-dışı bileşen GT irtifaya serbest eğimli doğrusal regresyonla
        # oturtulur (eğim işaret dahil; GT z yoksa xy ölçeği kullanılır)
        h = src @ n
        if has_z:
            a_z, c = np.polyfit(h, dst[:, 2], 1)
        else:
            a_z, c = s, 0.0
        def predict(i):
            p2 = A @ (P.T @ traj[i]) + t2
            return np.array([p2[0], p2[1], a_z * float(n @ traj[i]) + c])
    else:
        s, R, t = umeyama(src, dst)
        def predict(i):
            return s * R @ traj[i] + t
    print(f"Umeyama ölçeği: {s:.4f} (SLAM birimi -> metre)  [mod: {args.mode}]")

    fit = np.array([predict(i) for i in align_ids])
    fit_err = np.linalg.norm(fit[:, :2] - dst[:, :2], axis=1)
    print(f"Hizalama fit RMSE (xy): {np.sqrt((fit_err**2).mean()):.3f} m")
    if has_z:
        fz = fit[:, 2] - dst[:, 2]
        print(f"Hizalama fit RMSE (z):  {np.sqrt((fz**2).mean()):.3f} m")

    rows = []
    for i in pred_ids:
        p = predict(i)
        row = {"frame_idx": i, "pred_x": p[0], "pred_y": p[1], "pred_z": p[2]}
        if i in gt:
            row["gt_x"], row["gt_y"] = gt[i][0], gt[i][1]
            row["err_x"] = float(p[0] - gt[i][0])
            row["err_y"] = float(p[1] - gt[i][1])
            row["err"] = float(np.hypot(row["err_x"], row["err_y"]))
            if has_z:
                row["gt_z"] = gt[i][2]
                row["err_z"] = float(p[2] - gt[i][2])
        rows.append(row)

    out_csv = os.path.join(args.out_dir, "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame_idx", "pred_x", "pred_y", "pred_z",
            "gt_x", "gt_y", "gt_z", "err_x", "err_y", "err_z", "err"])
        w.writeheader()
        w.writerows(rows)

    errs = np.array([r["err"] for r in rows if "err" in r])
    errs_x = np.array([r["err_x"] for r in rows if "err_x" in r])
    errs_y = np.array([r["err_y"] for r in rows if "err_y" in r])
    errs_z = np.array([r["err_z"] for r in rows if "err_z" in r])
    if len(errs):
        print(f"\n--- GPS'siz bölüm ({len(errs)} frame) ---")
        print(f"Yatay (xy) RMSE: {np.sqrt((errs**2).mean()):.3f} m   "
              f"Ortalama: {errs.mean():.3f} m   Medyan: {np.median(errs):.3f} m   Maks: {errs.max():.3f} m")
        print(f"X ekseni RMSE: {np.sqrt((errs_x**2).mean()):.3f} m   "
              f"bias: {errs_x.mean():+.3f} m   Maks|x|: {np.abs(errs_x).max():.3f} m")
        print(f"Y ekseni RMSE: {np.sqrt((errs_y**2).mean()):.3f} m   "
              f"bias: {errs_y.mean():+.3f} m   Maks|y|: {np.abs(errs_y).max():.3f} m")
        if len(errs_z):
            print(f"Z ekseni RMSE: {np.sqrt((errs_z**2).mean()):.3f} m   "
                  f"bias: {errs_z.mean():+.3f} m   Maks|z|: {np.abs(errs_z).max():.3f} m")
        else:
            print("Z ekseni: GT'de translation_z yok, doğrulanamıyor.")

    plot(gt, has_z, predict, align_ids, rows, args, a1, p1)
    print(f"\nÇıktılar: {out_csv}, {os.path.join(args.out_dir, 'trajectory.png')}")


def plot(gt, has_z, predict, align_ids, rows, args, a1, p1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, AQUA, GRAY, RED, YELLOW = "#2a78d6", "#1baf7a", "#9a9a92", "#e34948", "#eda100"

    nrow = 3 if has_z else 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 5.25 * nrow))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    def style(ax):
        ax.grid(True, color="#eeeeea", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    gt_xy = np.array([gt[i][:2] for i in range(args.align_start, p1) if i in gt])
    ax1.plot(gt_xy[:, 0], gt_xy[:, 1], color=BLUE, lw=2, label="Gerçek yol (GT)")

    fit = np.array([predict(i) for i in align_ids])
    ax1.plot(fit[:, 0], fit[:, 1], color=GRAY, lw=2, ls="--",
             label="SLAM (hizalama penceresi, GPS var)")

    px = [r["pred_x"] for r in rows]
    py = [r["pred_y"] for r in rows]
    ax1.plot(px, py, color=AQUA, lw=2, label="SLAM tahmini (GPS yok)")
    if rows:
        ax1.scatter([px[0]], [py[0]], color=AQUA, s=50, zorder=5)
        ax1.annotate("GPS koptu", (px[0], py[0]), textcoords="offset points",
                     xytext=(8, 8), fontsize=9, color="#555550")

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Drone yörüngesi (üstten): gerçek ve tahmin")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_aspect("equal", adjustable="datalim")
    style(ax1)

    err_rows = [(r["frame_idx"], r["err"]) for r in rows if "err" in r]
    if err_rows:
        fi, fe = zip(*err_rows)
        ax2.plot(fi, fe, color=BLUE, lw=2)
        ax2.set_xlabel("Frame indeksi")
        ax2.set_ylabel("Yatay (xy) konum hatası (m)")
        ax2.set_title("GPS'siz bölümde frame başına yatay hata")
        style(ax2)

    for ax, key, color, name in ((ax3, "err_x", RED, "X"), (ax4, "err_y", AQUA, "Y")):
        er = [(r["frame_idx"], r[key]) for r in rows if key in r]
        if er:
            fi, fe = zip(*er)
            ax.plot(fi, fe, color=color, lw=2)
            ax.axhline(0, color="#cccbc0", lw=1)
            ax.set_xlabel("Frame indeksi")
            ax.set_ylabel(f"{name} hatası (m, pred - GT)")
            ax.set_title(f"{name} ekseninde işaretli hata")
            style(ax)

    if has_z:
        ax5, ax6 = axes[2]
        er = [(r["frame_idx"], r["err_z"]) for r in rows if "err_z" in r]
        if er:
            fi, fe = zip(*er)
            ax5.plot(fi, fe, color=YELLOW, lw=2)
            ax5.axhline(0, color="#cccbc0", lw=1)
            ax5.set_xlabel("Frame indeksi")
            ax5.set_ylabel("Z hatası (m, pred - GT)")
            ax5.set_title("Z (irtifa) ekseninde işaretli hata")
            style(ax5)
        zi = [(i, gt[i][2]) for i in range(args.align_start, p1) if i in gt]
        fi_g, z_g = zip(*zi)
        ax6.plot(fi_g, z_g, color=BLUE, lw=2, label="GT irtifa")
        fit_ids = align_ids
        ax6.plot(fit_ids, [predict(i)[2] for i in fit_ids], color=GRAY, lw=2, ls="--",
                 label="SLAM (hizalama)")
        pz = [(r["frame_idx"], r["pred_z"]) for r in rows]
        fi_p, z_p = zip(*pz)
        ax6.plot(fi_p, z_p, color=AQUA, lw=2, label="SLAM tahmini")
        ax6.set_xlabel("Frame indeksi")
        ax6.set_ylabel("z (m)")
        ax6.set_title("İrtifa profili: gerçek ve tahmin")
        ax6.legend(frameon=False, fontsize=9)
        style(ax6)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "trajectory.png"), dpi=140)


if __name__ == "__main__":
    main()
