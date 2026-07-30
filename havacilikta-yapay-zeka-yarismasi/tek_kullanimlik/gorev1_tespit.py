#!/usr/bin/env python
"""
GOREV 1 — Nesne tespiti (YOLO) + istege bagli UAP/UAI inis karari (MobileNetV3),
ana pipeline'dan (main.py) BAGIMSIZ, tek basina calisir. Sunucuya baglanmaz.

Kullanim: asagidaki SABITLER blogunu duzenleyip calistir:
    conda activate teknofest_yarisma
    cd havacilikta-yapay-zeka-yarismasi/tek_kullanimlik
    python gorev1_tespit.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TAKIM_BAGLANTI_ARAYUZU"))

# ============================== SABITLER ==============================
# Tek bir goruntu dosyasi YA DA bir klasor yolu (klasorse icindeki tum
# .jpg/.jpeg/.png/.webp dosyalari islenir).
INPUT_PATH = "bir_kare.jpg"

# None birakilirsa src/object_detection_model.py'deki varsayilan MODEL_PATH
# kullanilir (yarismada gercekten kullanilan agirlik). Baska bir agirlik
# denemek istersen buraya tam yolunu yaz (string).
MODEL_PATH_OVERRIDE = None

CONF_THRESHOLD = 0.25          # YOLO tespit guven esigi

# UAP/UAI icin MobileNetV3 "bos/dolu" inis kararini da calistir mi?
# (Insan/tasit kesisimi + pist-kenar kontrolu de dahil, bkz. README_LAST.md §4.2)
RUN_PAD_DECISION = True

# Etiketli (kutulu) ciktilarin kaydedilecegi klasor.
OUTPUT_DIR = "./gorev1_ciktilari/"
# ========================================================================

import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.object_detection_model import MODEL_PATH as DEFAULT_MODEL_PATH
from src.pad_pipeline import PadLandingClassifier, decide_pad, CLF_W, DOLU_THRESH

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_inputs(path):
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(IMG_EXTS)
        )
    return [path]


def main():
    model_path = MODEL_PATH_OVERRIDE or DEFAULT_MODEL_PATH
    print(f"YOLO agirligi yukleniyor: {model_path}")
    model = YOLO(model_path)

    pad_clf = None
    if RUN_PAD_DECISION:
        try:
            pad_clf = PadLandingClassifier(CLF_W)
            print(f"Pad siniflandirici yuklendi: {CLF_W}")
        except Exception as e:
            print(f"UYARI: pad siniflandirici yuklenemedi ({e}); sadece YOLO kararina dusulecek.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list_inputs(INPUT_PATH)
    if not files:
        raise SystemExit(f"Girdi bulunamadi: {INPUT_PATH}")

    for fpath in files:
        print(f"\n=== {fpath} ===")
        results = model(fpath, conf=CONF_THRESHOLD, verbose=False)

        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)

        person_boxes, vehicle_boxes, pad_boxes = [], [], []
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            print(f"  {cls_name:<14} conf={conf:.2f}  kutu=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), f"{cls_name} {conf:.2f}", fill=(255, 0, 0))

            low = cls_name.lower()
            if "insan" in low:
                person_boxes.append((x1, y1, x2, y2))
            elif "tasit" in low or "taşıt" in low:
                vehicle_boxes.append((x1, y1, x2, y2))
            elif "uap" in low or "uai" in low or "uaİ" in low:
                pad_boxes.append((x1, y1, x2, y2))

        if RUN_PAD_DECISION and pad_clf is not None and pad_boxes:
            frame_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
            obstacles = person_boxes + vehicle_boxes
            for pbox in pad_boxes:
                r = decide_pad(frame_bgr, pbox, obstacles, pad_clf, thresh=DOLU_THRESH)
                karar = "INILEBILIR" if r["inilebilir"] else "INILEMEZ"
                print(f"  -> Pist {pbox}: {karar}")

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(fpath))
        img.save(out_path)
        print(f"  Etiketli kare kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
