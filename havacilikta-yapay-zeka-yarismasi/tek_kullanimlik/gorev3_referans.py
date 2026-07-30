#!/usr/bin/env python
"""
GOREV 3 — Sadece referans nesne esleme (ReferenceMatcher), ana pipeline'dan
(main.py) BAGIMSIZ, tek basina calisir. Sunucuya baglanmaz, Gorev 1/2 hic
calismaz.

Bir YA DA BIRDEN FAZLA referans goruntusunu bir kez ekler, sonra bir
klasordeki (ya da tek bir) karede/karelerde HEPSINI TEK SEFERDE arar (kare
tek islenir, tum referanslarla karsilastirilir - ReferenceMatcher.match()
zaten boyle tasarlanmis); bulunan kutulari kareye cizip kaydeder.

Kullanim: asagidaki SABITLER blogunu duzenleyip calistir:
    conda activate teknofest_yarisma
    cd havacilikta-yapay-zeka-yarismasi/tek_kullanimlik
    python gorev3_referans.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TAKIM_BAGLANTI_ARAYUZU"))

# ============================== SABITLER ==============================
# Tek bir referans goruntusu YA DA bir klasor. Klasorse icindeki HER goruntu
# AYRI bir referans olarak eklenir (kimlik = dosya adi, uzantisiz) ve hepsi
# ayni kare taramasinda TEK SEFERDE denenir.
REFERENCE_PATH = "referanslar/"

# Tek bir kare YA DA bir klasor (klasorse icindeki tum goruntulerde aranir).
SEARCH_PATH = "bir_kare.jpg"

# Bulunan kutularin cizilip kaydedilecegi klasor.
OUTPUT_DIR = "./gorev3_ciktilari/"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# ========================================================================

import cv2
from PIL import Image, ImageDraw

from src.reference_matcher import ReferenceMatcher

# Farkli referanslari gorsel olarak ayirt etmek icin donusen renk paleti.
_PALETTE = [(255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 0),
            (255, 0, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255)]


def list_inputs(path):
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(IMG_EXTS)
        )
    return [path]


def main():
    print("ReferenceMatcher yukleniyor (YOLOE + DINOv2-L, ilk calistirmada model indirebilir)...")
    rm = ReferenceMatcher()

    ref_files = list_inputs(REFERENCE_PATH)
    if not ref_files:
        raise SystemExit(f"Referans bulunamadi: {REFERENCE_PATH}")

    ref_ids = []
    for rf in ref_files:
        ref_bgr = cv2.imread(rf, cv2.IMREAD_COLOR)
        if ref_bgr is None:
            print(f"UYARI: referans okunamadi, atlaniyor: {rf}")
            continue
        rid = os.path.splitext(os.path.basename(rf))[0]
        rm.add_reference(rid, ref_bgr)
        ref_ids.append(rid)
        print(f"Referans eklendi: {rid} <- {rf}")
    if not ref_ids:
        raise SystemExit("Hicbir referans eklenemedi.")
    colors = {rid: _PALETTE[i % len(_PALETTE)] for i, rid in enumerate(ref_ids)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    search_files = list_inputs(SEARCH_PATH)
    if not search_files:
        raise SystemExit(f"Aranacak kare bulunamadi: {SEARCH_PATH}")

    for fpath in search_files:
        frame_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"{fpath}: okunamadi, atlaniyor.")
            continue

        # TEK cagri - kare bir kez islenir, TUM referanslarla karsilastirilir.
        hits = rm.match(frame_bgr, ref_ids=ref_ids)
        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)

        print(f"\n=== {fpath} ===")
        for rid in ref_ids:
            if rid in hits:
                x1, y1, x2, y2, score, branch = hits[rid]
                color = colors[rid]
                print(f"  {rid}: BULUNDU kutu=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                      f"skor={score:.3f} kol={branch}")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, max(0, y1 - 14)), f"{rid} {score:.2f} ({branch})", fill=color)
            else:
                print(f"  {rid}: bulunamadi.")

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(fpath))
        img.save(out_path)

    print(f"\nTAMAMLANDI. Ciktilar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
