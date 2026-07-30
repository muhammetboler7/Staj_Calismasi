#!/usr/bin/env python
"""
GOREV 2 — Sadece ORB-SLAM3 (+ istege bagli OnlineAligner), ana pipeline'dan
(main.py) BAGIMSIZ, tek basina calisir. Sunucuya baglanmaz, Gorev 1/3 hic
calismaz.

ONEMLI - ORB-SLAM3 vs OnlineAligner farki:
  ORB-SLAM3'un tek basina urettigi poz, KENDI KEYFI OLCEGINDEDIR (metre
  DEGIL, rotasyonu/olcegi de bilinmez) - GT_CSV verilmezse cikti budur.
  Gercek dunya (metre) cinsinden konum istiyorsan, OnlineAligner'i da
  calistirmak GEREKIR - o da GT (gercek konum) icerern bir kalibrasyon
  verisine ihtiyac duyar: ilk CALIB_FRAMES kadar karenin GERCEK x/y/z'sini
  (translation_gt.csv gibi bir dosyadan) "kalibrasyon noktasi" olarak
  aligner'a verip, geri kalan karelerde bu kalibrasyona dayanarak tahmin
  ureteceksin. Yani: EVET, en basta (GT_CSV verirsen) bir miktar gercek GT
  yollaman gerekiyor - onsuz sadece ham/olceksiz SLAM cikisi elde edersin.

Kullanim: asagidaki SABITLER blogunu duzenleyip calistir:
    conda activate teknofest_yarisma
    cd havacilikta-yapay-zeka-yarismasi/tek_kullanimlik
    python gorev2_orbslam.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TAKIM_BAGLANTI_ARAYUZU"))

# ============================== SABITLER ==============================
# Karelerin SIRAYLA (dosya adina gore artan) bulundugu klasor.
FRAMES_DIR = "/home/pc_5053/GSS/seyma2/havacilikta-yapay-zeka-Sunucu-Gelistirmeleri/media/THYZ_2026_Online_Yarisma/"

# Sonuc CSV'sinin yazilacagi yol.
OUTPUT_CSV = "./gorev2_ciktisi.csv"

# None birakilirsa src/orbslam_stream.py'deki DEFAULT_BINARY/DEFAULT_VOCAB/
# DEFAULT_SETTINGS kullanilir (gps_denied_localization/ altindaki hazir
# dosyalar, artik havacilikta-yapay-zeka-yarismasi icinde). Farkli bir
# video/kamera icin buraya tam yol ver.
BINARY_PATH = None
VOCAB_PATH = None
SETTINGS_PATH = None    # ornek: farkli cozunurluk/kamera icin yeni bir .yaml

# --- OnlineAligner (istege bagli - gercek metre cinsinden konum icin) ---
# None birakilirsa OnlineAligner HIC calismaz, cikti sadece ham SLAM pozudur.
# Bir GT CSV'si verirsen (sutunlar: frame_numbers,translation_x,translation_y,
# translation_z - translation_gt.csv ile ayni format), ilk CALIB_FRAMES kare
# "kalibrasyon" (GT biliniyormus gibi) sayilir, sonrasi OnlineAligner ile
# tahmin edilir.
GT_CSV = "/home/pc_5053/GSS/seyma2/havacilikta-yapay-zeka-Sunucu-Gelistirmeleri/translations/THYZ_2026_Online_Yarisma.csv"
CALIB_FRAMES = 450   # GT_CSV verildiyse, ilk bu kadar kare kalibrasyon sayilir

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# ========================================================================

from src.orbslam_stream import OrbSlamStream

if GT_CSV:
    from src.online_alignment import OnlineAligner


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["frame_numbers"]] = (
                float(row["translation_x"]), float(row["translation_y"]), float(row["translation_z"]))
    return gt


def main():
    frame_files = sorted(
        f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(IMG_EXTS)
    )
    if not frame_files:
        raise SystemExit(f"Kare bulunamadi: {FRAMES_DIR}")
    print(f"{len(frame_files)} kare bulundu, ORB-SLAM3 baslatiliyor...")

    orbslam = OrbSlamStream(
        binary_path=BINARY_PATH, vocab_path=VOCAB_PATH, settings_path=SETTINGS_PATH,
        stderr_log_path="./gorev2_orbslam_stderr.log",
    )

    gt = load_gt(GT_CSV) if GT_CSV else None
    aligner = OnlineAligner() if GT_CSV else None

    rows = []
    n_ok = n_fail = 0
    try:
        for i, fname in enumerate(frame_files):
            fpath = os.path.join(FRAMES_DIR, fname)
            pose = orbslam.track(fpath)
            if pose is None:
                n_fail += 1
                rows.append([fname, "FAIL", "", "", "", "", "", "", "", "", "", "", ""])
                print(f"{fname}: TAKIP KAYBEDILDI")
                continue

            n_ok += 1
            x, y, z, qx, qy, qz, qw, med_depth, n_tracked = pose
            wx = wy = wz = ""

            if aligner is not None:
                stem = os.path.splitext(fname)[0]
                g = gt.get(stem)
                if g is not None and i < CALIB_FRAMES:
                    aligner.add_calibration_point((x, y, z), (qx, qy, qz, qw), g,
                                                  med_depth=med_depth, n_tracked=n_tracked)
                    wx, wy, wz = g   # kalibrasyon karesinde GT'nin kendisi
                elif aligner.ready():
                    pred = aligner.predict((x, y, z), med_depth=med_depth)
                    wx, wy, wz = pred

            rows.append([fname, "OK", x, y, z, qx, qy, qz, med_depth, n_tracked, wx, wy, wz])
            if (i + 1) % 50 == 0:
                msg = f"{i+1}/{len(frame_files)}  ham_slam=({x:.2f},{y:.2f},{z:.2f})"
                if aligner is not None:
                    msg += f"  dunya=({wx},{wy},{wz})" if wx != "" else "  dunya=henuz hazir degil"
                print(msg)
    finally:
        orbslam.close()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "durum", "slam_x", "slam_y", "slam_z", "qx", "qy", "qz",
                    "med_depth", "n_tracked", "dunya_x", "dunya_y", "dunya_z"])
        w.writerows(rows)

    print(f"\nTAMAMLANDI. Basarili={n_ok}  Kayip={n_fail}  -> {OUTPUT_CSV}")
    if GT_CSV is None:
        print("Not: GT_CSV verilmedigi icin cikti ham/olceksiz SLAM pozudur (dunya_x/y/z sutunlari bos).")


if __name__ == "__main__":
    main()
