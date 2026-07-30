#!/usr/bin/env python
"""
UAP/UAI INIS KARAR PIPELINE'I — MobileNetV3 tabanli iki-asamali karar.

Akis (main.py -> ObjectDetectionModel.detect icinden cagrilir):
  1) YOLO kareyi tespit eder (nesne_detection_model.py kendi calistirir).
  2) Her UAP/UAI pisti icin bu modul karari verir:
       a) Pist tamami kare ICINDE degil mi (kenardan tasiyor)? -> INILEMEZ (0)
          (Sartname 2.1.2 / Sekil 9: "uygun" olabilmesi icin alanin tamami
           kare icinde olmali.)
       b) Pist uzerinde bir Insan/Tasit bbox merkezi var mi?    -> INILEMEZ (0)
          (Sartname Sekil 10: alan uzerinde nesne varsa inise uygun degil;
           insan VE tasit acikca sayilmistir. Bu nesneler ayrica normal
           tespit olarak da gonderilir - o is object_detection_model'de.)
       c) Aksi halde pist kirilir -> MobileNetV3 'bos/dolu' siniflandiricisi
          calisir; dolu olasiligi esigi asarsa                  -> INILEMEZ (0)
          degilse                                               -> INILEBILIR (1)

Sartname iniş durumu degerleri (Tablo 5): 1=Uygun (inilebilir),
0=Uygun Degil (inilemez), -1=Inis Alani Degil (yalniz tasit/insan).

KOD ICINDE KULLANIM (ObjectDetectionModel):
    from src.pad_pipeline import PadLandingClassifier, decide_pad
    clf = PadLandingClassifier(CLF_W)
    r = decide_pad(frame_bgr, (x1,y1,x2,y2), obstacle_boxes, clf)  # insan+tasit kutulari
    # r["inilebilir"] -> True/False

BAGIMSIZ CLI (gorsel dogrulama; kendi YOLO'sunu yukler):
    python -m src.pad_pipeline <girdi> [cikti]
      <girdi> : resim (.jpg/.png) | KLASOR | video (.mp4)
"""
import os
import sys
import glob
import logging

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    _TORCH_OK = True
except Exception as _e:  # torch/torchvision yoksa siniflandirici devre disi kalir
    _TORCH_OK = False
    _TORCH_IMPORT_ERR = _e

# ================== AYARLAR ==================
# Gorev 1 YOLO agirliklari (yalniz BAGIMSIZ CLI icin; ana akista YOLO'yu
# object_detection_model.py yukler ve tespitleri hazir verir).
YOLO_W = ("/media/CezeriDrive_12TB/StajyerCalismalari/2026_Yaz/Muhammet_Boler/"
          "Muhammet_Boler/son_hazirlik/egitim_sonucu/weights/"
          "yeni_final_freeze10_uai_33ep.pt")
# MobileNetV3-Small 'bos/dolu' siniflandirici agirliklari.
CLF_W = ("/media/CezeriDrive_12TB/StajyerCalismalari/2026_Yaz/Muhammet_Boler/"
         "8_temmuz_egitim/pad_classifier/best_v2.pt")

DOLU_THRESH = 0.35     # dolu olasiligi bunu asarsa INILEMEZ
MARGIN = 0.20          # pisti kirparken eklenen bosluk orani
EDGE_TOL_PX = 2        # pist bu kadar pikselden fazla kenara degerse "tam icinde degil"
YOLO_IMGSZ = 1280      # yalniz CLI
YOLO_CONF = 0.25       # yalniz CLI

# YOLO sinif numaralari (son_hazirlik/dataset/data.yaml):
#   0 Insan, 1 Tasit, 2 UAP, 3 UAP_Inemez, 4 UAI, 5 UAI_Inemez
PERSON_CLASS = 0
VEHICLE_CLASS = 1
# YOLO'nun kendi "_Inemez" etiketi yok sayilir; inis karari BU sistem verir.
PAD_NAME = {2: "UAP", 3: "UAP", 4: "UAI", 5: "UAI"}
# =============================================

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


class PadLandingClassifier:
    """UAP/UAI pistinin 'bos/dolu' oldugunu kestiren MobileNetV3-Small.

    Checkpoint bicimi (best_v2.pt): {"model": state_dict, "classes": [...],
    "img": int}. classes 'dolu' etiketini icermelidir.
    """

    def __init__(self, ckpt=CLF_W, device=None):
        if not _TORCH_OK:
            raise RuntimeError(f"PyTorch/torchvision yuklenemedi: {_TORCH_IMPORT_ERR}")
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.dev)
        self.classes = ck["classes"]
        self.img = int(ck["img"])
        self.dolu_idx = self.classes.index("dolu")
        self._norm = transforms.Normalize(_NORM_MEAN, _NORM_STD)
        net = models.mobilenet_v3_small()
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(self.classes))
        net.load_state_dict(ck["model"])
        net.eval().to(self.dev)
        self.net = net

    @torch.no_grad() if _TORCH_OK else (lambda f: f)
    def dolu_prob(self, crop_bgr):
        """Kirpilmis pist goruntusu (BGR) icin 'dolu' olasiligini [0,1] dondurur."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img, self.img))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        t = self._norm(t).unsqueeze(0).to(self.dev)
        return torch.softmax(self.net(t), 1)[0, self.dolu_idx].item()


def _center_in(box, pad):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return pad[0] <= cx <= pad[2] and pad[1] <= cy <= pad[3]


def _touches_edge(box, w, h, tol=EDGE_TOL_PX):
    """Pistin tamami kare icinde DEGIL mi (kenardan tasiyor mu)?"""
    x1, y1, x2, y2 = box
    return x1 <= tol or y1 <= tol or x2 >= w - tol or y2 >= h - tol


def decide_pad(frame_bgr, pad_box, obstacle_boxes, classifier,
               thresh=DOLU_THRESH, margin=MARGIN, require_full_in_frame=True):
    """Tek bir UAP/UAI pisti icin inis karari.

    frame_bgr      : tam kare (BGR np.array)
    pad_box        : (x1, y1, x2, y2) pist kutusu (piksel)
    obstacle_boxes : [(x1,y1,x2,y2), ...] tespit edilen insan VE tasit
                     kutulari (Sartname Sekil 10: alanin uzerinde bu
                     nesnelerden biri varsa inise uygun degildir)
    classifier     : PadLandingClassifier ornegi (None ise siniflandirici atlanir)

    Doner: dict(inilebilir: bool, sebep: str, dolu_prob: float|None)
    """
    x1, y1, x2, y2 = (int(v) for v in pad_box)
    h, w = frame_bgr.shape[:2]

    # a) Alanin tamami kare icinde degilse "uygun" olamaz (Sartname 2.1.2).
    if require_full_in_frame and _touches_edge((x1, y1, x2, y2), w, h):
        return dict(inilebilir=False, sebep="alan_kismen_disarida", dolu_prob=None)

    # b) Pist uzerinde insan/tasit varsa inise uygun degil (Sekil 10).
    if any(_center_in(p, (x1, y1, x2, y2)) for p in obstacle_boxes):
        return dict(inilebilir=False, sebep="nesne_pistte", dolu_prob=None)

    # c) Siniflandirici: pisti (pay ile) kirp, dolu olasiligini olc.
    if classifier is None:
        # Siniflandirici yoksa temkinli davranmak yerine "temiz" varsayalim;
        # cagiran taraf YOLO sinifindan gelen ilk karari koruyabilir.
        return dict(inilebilir=True, sebep="siniflandirici_yok", dolu_prob=None)

    mx = int((x2 - x1) * margin / 2)
    my = int((y2 - y1) * margin / 2)
    crop = frame_bgr[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
    if crop.size == 0:
        return dict(inilebilir=False, sebep="bos_kirpma", dolu_prob=None)
    p = classifier.dolu_prob(crop)
    dolu = p >= thresh
    return dict(inilebilir=(not dolu),
                sebep=("dolu_nesne" if dolu else "temiz"),
                dolu_prob=round(p, 3))


# ============================================================================
# BAGIMSIZ CLI — kendi YOLO'sunu yukler, gorsel dogrulama uretir.
# Ana yarisma akisi bu sinifi KULLANMAZ (YOLO'yu tek sefer object_detection_
# model.py calistirir); burasi yalniz elle test/gorsellestirme icindir.
# ============================================================================
class PadLandingPipeline:
    def __init__(self, yolo_w=YOLO_W, clf_w=CLF_W, thresh=DOLU_THRESH):
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_w)
        self.clf = PadLandingClassifier(clf_w)
        self.thresh = thresh

    def process(self, frame):
        """frame: BGR np.array -> (results, dets). results her pist icin karar."""
        use_cuda = _TORCH_OK and torch.cuda.is_available()
        res = self.yolo.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False,
                                device=0 if use_cuda else "cpu")[0]
        dets = [(int(b.cls[0]), *map(int, b.xyxy[0].tolist())) for b in res.boxes]
        # Pist ustu engel kontrolu: insan VE tasit kutulari (Sartname Sekil 10).
        obstacles = [d[1:5] for d in dets if d[0] in (PERSON_CLASS, VEHICLE_CLASS)]
        out = []
        for c, x1, y1, x2, y2 in dets:
            if c not in PAD_NAME:
                continue
            r = decide_pad(frame, (x1, y1, x2, y2), obstacles, self.clf, self.thresh)
            r["cls"] = PAD_NAME[c]
            r["box"] = (x1, y1, x2, y2)
            out.append(r)
        return out, dets

    def annotate(self, frame, results, dets=None):
        vis = frame.copy()
        if dets:
            for c, x1, y1, x2, y2 in dets:
                if c == PERSON_CLASS:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(vis, "insan", (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
                elif c == 1:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)
        for r in results:
            x1, y1, x2, y2 = r["box"]
            good = r["inilebilir"]
            col = (0, 190, 0) if good else (0, 0, 255)
            if r["dolu_prob"] is None:
                tag = "INILEBILIR" if good else f"INILEMEZ ({r['sebep']})"
            else:
                tag = (f"INILEBILIR (p={r['dolu_prob']:.2f})" if good
                       else f"INILEMEZ (dolu p={r['dolu_prob']:.2f})")
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
            yt = y1 - 8 if y1 > 24 else y2 + 22
            cv2.putText(vis, f"{r['cls']}: {tag}", (max(0, x1), yt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return vis


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".PNG", ".JPG")


def _run_image(pipe, inp, outp):
    frame = cv2.imread(inp)
    if frame is None:
        print("okunamadi:", inp)
        return
    results, dets = pipe.process(frame)
    cv2.imwrite(outp, pipe.annotate(frame, results, dets))
    print(f"{os.path.basename(inp)} -> {len(results)} pist: " +
          ", ".join(f"{r['cls']}:{'INILEBILIR' if r['inilebilir'] else 'INILEMEZ'}"
                    f"({r['sebep']})" for r in results))


def _run_video(pipe, inp, outp, fps=15):
    cap = cv2.VideoCapture(inp)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results, dets = pipe.process(frame)
        wr.write(pipe.annotate(frame, results, dets))
        i += 1
        if i % 100 == 0:
            print(f"{i} kare islendi", flush=True)
    cap.release()
    wr.release()
    print(f"video -> {outp} ({i} kare)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    pipe = PadLandingPipeline()
    if os.path.isdir(inp):
        outdir = out or (inp.rstrip("/") + "_pred")
        os.makedirs(outdir, exist_ok=True)
        files = [f for f in sorted(glob.glob(os.path.join(inp, "*"))) if f.endswith(IMG_EXT)]
        for f in files:
            _run_image(pipe, f, os.path.join(outdir, os.path.basename(f)))
        print(f"\nbitti -> {outdir} ({len(files)} resim)")
    elif inp.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        _run_video(pipe, inp, out or (os.path.splitext(inp)[0] + "_pred.mp4"))
    else:
        base, ext = os.path.splitext(inp)
        _run_image(pipe, inp, out or (base + "_pred" + ext))
