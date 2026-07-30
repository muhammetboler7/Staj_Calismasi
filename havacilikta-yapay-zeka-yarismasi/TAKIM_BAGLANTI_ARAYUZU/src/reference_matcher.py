#!/usr/bin/env python
"""
GOREV 3 — Referans Nesne Esleme (TAM OTOMATIK iki-kollu hibrit, TEK MODEL: DINOv2-L).

Kullanici referans basina HICBIR sey secmez; tum parametreler sabittir ve kol
secimi OTOMATIK yapilir. Kanit: bench_g3*/bench_dense*/bench_conf/diag_router/
test_v2dense deneyleri (bkz. gorev_3_her_sey.md, gorev3_deneme_ciktilari/).

Iki kol (her referans icin her karede ikisi de calisir), IKISI DE DINOv2-L kullanir:
  1) DISCRETE (ayrik nesne): YOLOE prompt-free -> confidence ile top-K(30) buda -> her
     aday baglam payiyla kirpilip 336'ya olceklenir (olcek-normalize) -> DINOv2-L
     GLOBAL embedding -> referansin TTA setiyle max-kosinus -> en iyi aday.
     KRITIK: confidence budama sart (budamadan 1/8, top-10 ile 4/8; distraktor eler).
     Ek suzgecler (28-29 Temmuz 437-cift gercek-yarisma benchmarki ile dogrulandi,
     coarse %41.88 -> final %65.68 hit@0.5):
       a) MEDYAN BOYUT FILTRESI: bu referansin su ana kadar gozlemlenmis (bu oturumda,
          bu referans icin) secilen kutu boylarinin CANLI/akan medyani hesaplanir; aday
          kutulardan bu medyanin [0.35x, 2.8x] araligi disinda kalanlar elenir (DINOv2'nin
          yanlislikla sectigi saçma boyuttaki kutulari onler). Not: cevrimdisi benchmarkta
          TUM pencere onceden bilinip medyan tek seferde hesaplanmisti (hindsight); canli
          sistemde bu imkansiz oldugu icin burada CANLI/akan medyan kullanilir (ilk birkaç
          karede az ornekle guvenilirligi dusuktur, pencere ilerledikce yakinsar).
       b) UYARLAMALI HAVUZ GENISLETME: secilen adayin benzerligi ESCALATE_SIM_THRESH
          altindaysa (dogru nesne muhtemelen ilk 30 adaya girmedi), YOLOE 120 adaya
          kadar genisletilmis ikinci bir turla tekrar denenir.
       c) ROMA INCELTME: secilen kaba kutunun etrafi (RoMa varsa) referansla yogun
          piksel-piksel eslestirilir; eslesme "akil sagligi" kontrolunu (merkez kaymasi +
          alan orani) geçerse kutu hassaslastirilir, gecmezse kaba kutu korunur. RoMa
          (vismatch paketi) yuklu degilse bu adim sessizce atlanir.
  2) PATTERN (buyuk dokulu bolge/arazi): DINOv2-L DENSE correspondence (proposal YOK)
     -> cok-olcekli patch-token pencere-havuzlama -> referansin dense betimleyicisiyle
     kosinus -> en iyi pencere. saha gibi homojen bolgeleri bulur.

Otomatik kol secimi:
  pattern_score (dense oz-benzerlik) "bu homojen bir bolge mi" olcer (homojen alan ->
  yuksek dense oz-benzerlik). pattern_score >= PATTERN_ROUTE_THRESH ise pattern kutusu,
  degilse discrete kutusu. Esik SABIT; kullanici mudahalesi yok. 29 Temmuz'da gercek
  437-cift benchmarkta esik taramasi yapildi: 0.68 -> %50.11, 0.80 -> %65.68 (0.80-0.98
  arasi kararli plato, ayni sonuc) - esik 0.80'e guncellendi.

Kullanim (ObjectDetectionModel.detect icinden):
    from src.reference_matcher import ReferenceMatcher
    rm = ReferenceMatcher()
    rm.add_reference(ref_url, ref_image_bgr)             # oturum basi
    hits = rm.match(frame_bgr, ref_ids=aktif_referanslar)
    # hits: {ref_url: (x1,y1,x2,y2, score, "discrete"|"pattern")}
"""
import collections
import logging

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import timm
    from ultralytics import YOLOE
    _DEPS_OK = True
except Exception as _e:
    _DEPS_OK = False
    _DEP_ERR = _e

# RoMa (vismatch) opsiyonel: yuklu degilse discrete kol kaba kutuda kalir, sistem
# cokmez (Gorev 3'un geri kalani gibi ayni "sessizce atla" felsefesi).
try:
    import vismatch
    _ROMA_OK = True
except Exception as _roma_e:
    _ROMA_OK = False
    _ROMA_ERR = _roma_e

# ---- Sabit ayarlar (kullanici degistirmez) ----
YOLOE_PF_W = "yoloe-11l-seg-pf.pt"
DINO_MODEL = "vit_large_patch14_dinov2.lvd142m"   # DINOv2-L (hem discrete hem pattern)
PATCH = 14                                        # DINOv2-L patch boyutu
DET_IMGSZ = 1920
DET_CONF = 0.001
# Sadece en yuksek CONF_TOPK aday kullanildigi icin max_det dusuk; YOLOE-PF (seg)
# maske son-islemesini 300 yerine ~30 tespitle sinirlar (hiz+bellek).
DET_MAX = 30
CONF_TOPK = 30          # confidence ile bu kadar adaya buda (distraktor eleme)
CROP = 336              # DINOv2-L crop matching girdi boyutu (336=24*14)
CTX = 1.3
# Pattern olcekleri 14'un kati (DINOv2-L patch14). En buyuk (1920) olcek gereksiz.
PATTERN_SCALES = [(1022, 574), (1526, 868)]
PATTERN_KS = [2, 3, 4, 6, 8, 12, 16, 24]
PATTERN_ROUTE_THRESH = 0.80   # pattern_score bunun ustundeyse pattern kolu
TTA_ROT = (0, 45, 90, 135, 180, 225, 270, 315)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# ---- Discrete kol ek suzgecleri (28-29 Temmuz benchmarki, bkz. yukaridaki docstring) ----
WIDE_DET_MAX = 120            # dusuk benzerlikte genisletilen aday havuzu
ESCALATE_SIM_THRESH = 0.40    # bu benzerligin altinda genis havuza gecilir
SIZE_MIN_RATIO = 0.35         # canli medyan boyutuna gore kabul araligi (alt)
SIZE_MAX_RATIO = 2.8          # canli medyan boyutuna gore kabul araligi (ust)
SIZE_MIN_SAMPLES = 3          # bu kadar ornek birikmeden boyut filtresi devreye girmez
ROMA_PAD_FACTOR = 0.8         # RoMa icin kaba kutu etrafinda birakilan pay
ROMA_MIN_INLIERS = 8          # bu sayidan az inlier varsa RoMa sonucu reddedilir
ROMA_MAX_CENTER_SHIFT_RATIO = 1.0   # RoMa kutusu kaba kutunun kosegeninden fazla kaymissa reddedilir
ROMA_MIN_AREA_RATIO = 0.6     # RoMa kutusu kaba kutunun alaninin bu oranindan kucuk olamaz
ROMA_MAX_AREA_RATIO = 6.25    # ... bu oranindan buyuk olamaz (2.5^2)


class ReferenceMatcher:
    def __init__(self, device=None):
        if not _DEPS_OK:
            raise RuntimeError(f"ReferenceMatcher bagimliliklari yuklenemedi: {_DEP_ERR}")
        self.dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.det = YOLOE(YOLOE_PF_W)
        self.dino = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0,
                                      dynamic_img_size=True).eval().to(self.dev).half()
        self._nprefix = self.dino.num_prefix_tokens
        self._mean = torch.tensor(_MEAN).view(1, 3, 1, 1).to(self.dev).half()
        self._std = torch.tensor(_STD).view(1, 3, 1, 1).to(self.dev).half()
        self.ref_disc = {}   # ref_id -> [K,D] DINOv2-L TTA global embedding (discrete)
        self.ref_pat = {}    # ref_id -> [K,D] DINOv2-L dense region-mean TTA (pattern)
        self.ref_img_bgr = {}          # ref_id -> orijinal referans goruntusu (RoMa inceltme icin)
        self._ref_sizes = collections.defaultdict(list)   # ref_id -> canli/akan kutu-boyu gecmisi
        self.roma = None
        if _ROMA_OK:
            try:
                self.roma = vismatch.get_matcher("roma", device=self.dev)
            except Exception as e:
                logging.warning(f"RoMa yuklenemedi, discrete kol inceltmesiz calisacak: {e}")
        elif not _ROMA_OK:
            logging.info(f"vismatch/RoMa yuklu degil, discrete kol inceltmesiz calisacak: {_ROMA_ERR}")
        logging.info(f"ReferenceMatcher hazir (YOLOE-PF + DINOv2-L{'+ RoMa' if self.roma else ''}) dev={self.dev}")

    def _norm(self, bchw):
        return (bchw.div_(255) - self._mean) / self._std

    @torch.no_grad()
    def _embed(self, crops_rgb):
        """list[HxWx3 RGB] -> [N,D] normalize GLOBAL embedding (discrete)."""
        if not crops_rgb:
            return None
        b = self._norm(torch.from_numpy(np.stack(crops_rgb)).permute(0, 3, 1, 2).to(self.dev).half())
        fs = []
        for i in range(0, len(b), 64):
            fs.append(self.dino(b[i:i + 64]).float())
        return F.normalize(torch.cat(fs), dim=1)

    @torch.no_grad()
    def _grid(self, rgb):
        """HxWx3 RGB -> [1,D,Hp,Wp] normalize PATCH-token gridi (pattern/dense)."""
        H, W = rgb.shape[:2]
        t = self._norm(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.dev).half())
        f = self.dino.forward_features(t)[:, self._nprefix:, :]
        D = f.shape[-1]
        return F.normalize(f.reshape(1, H // PATCH, W // PATCH, D).permute(0, 3, 1, 2), dim=1)

    def _tta(self, rgb, sz):
        sq = cv2.resize(rgb, (sz, sz)); out = []
        for a in TTA_ROT:
            M = cv2.getRotationMatrix2D((sz / 2, sz / 2), a, 1.0)
            r = cv2.warpAffine(sq, M, (sz, sz), borderValue=(114, 114, 114))
            out += [r, r[:, ::-1].copy()]
        return out

    def _crop_box(self, rgb, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max(x2 - x1, y2 - y1) * CTX
        M = cv2.getRotationMatrix2D((cx, cy), 0, CROP / s)
        M[0, 2] += CROP / 2 - cx; M[1, 2] += CROP / 2 - cy
        return cv2.warpAffine(rgb, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # ---------- discrete kol ek suzgecleri (boyut filtresi / genis havuz / RoMa) ----------
    @staticmethod
    def _box_size(box):
        return max(box[2] - box[0], box[3] - box[1])

    @staticmethod
    def _box_area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @classmethod
    def _pick_best_idx(cls, boxes, sims, med_size):
        """En yuksek benzerlikli aday indeksini doner; med_size verilmisse (yeterli
        gecmis ornegi varsa) [SIZE_MIN_RATIO, SIZE_MAX_RATIO] disindaki adaylari eler."""
        if med_size is not None:
            sizes = np.array([cls._box_size(b) for b in boxes])
            ratio = sizes / med_size
            ok = (ratio >= SIZE_MIN_RATIO) & (ratio <= SIZE_MAX_RATIO)
            if ok.any():
                sims_np = sims.cpu().numpy()
                return int(np.argmax(np.where(ok, sims_np, -1e9)))
        return int(sims.argmax())

    @staticmethod
    def _pad_roi(box, w, h, pad_factor=ROMA_PAD_FACTOR):
        x1, y1, x2, y2 = box
        pad = max(x2 - x1, y2 - y1) * pad_factor
        rx1 = max(0, int(x1 - pad)); ry1 = max(0, int(y1 - pad))
        rx2 = min(w, int(x2 + pad)); ry2 = min(h, int(y2 + pad))
        return rx1, ry1, rx2, ry2

    @staticmethod
    def _prep_rgb_chw(img_bgr):
        """RoMa/vismatch girdisi: kanal-once float32 [0,1] RGB."""
        rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    @classmethod
    def _sanity_check(cls, cand_box, base_box):
        """RoMa/genis-havuz sonucunun kaba kutudan cok sapmadigini dogrular."""
        bx1, by1, bx2, by2 = base_box
        b_cx, b_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        b_w, b_h = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
        b_diag = (b_w ** 2 + b_h ** 2) ** 0.5
        cx1, cy1, cx2, cy2 = cand_box
        c_cx, c_cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        center_shift = ((c_cx - b_cx) ** 2 + (c_cy - b_cy) ** 2) ** 0.5
        if center_shift > ROMA_MAX_CENTER_SHIFT_RATIO * b_diag:
            return False
        area_ratio = cls._box_area(cand_box) / max(b_w * b_h, 1.0)
        return ROMA_MIN_AREA_RATIO <= area_ratio <= ROMA_MAX_AREA_RATIO

    def _roma_refine(self, ref_bgr, frame_bgr, box):
        """Kaba kutuyu RoMa yogun eslesmesiyle inceltir; basarisiz/supheliyse None doner."""
        if self.roma is None:
            return None
        H, W = frame_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = self._pad_roi(box, W, H)
        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return None
        try:
            out = self.roma(self._prep_rgb_chw(ref_bgr), self._prep_rgb_chw(crop))
        except Exception as e:
            logging.warning(f"RoMa eslesmesi basarisiz: {e}")
            return None
        inlier_kpts1 = out.get("inlier_kpts1")
        if inlier_kpts1 is None or int(out.get("num_inliers", 0)) < ROMA_MIN_INLIERS:
            return None
        ix1, iy1 = float(inlier_kpts1[:, 0].min()), float(inlier_kpts1[:, 1].min())
        ix2, iy2 = float(inlier_kpts1[:, 0].max()), float(inlier_kpts1[:, 1].max())
        cand = (ix1 + rx1, iy1 + ry1, ix2 + rx1, iy2 + ry1)
        return cand if self._sanity_check(cand, box) else None

    @torch.no_grad()
    def _wide_candidates(self, frame_bgr, frame_rgb):
        """Dusuk-benzerlik durumunda genisletilmis (WIDE_DET_MAX) YOLOE aday havuzu."""
        dev0 = 0 if self.dev.startswith("cuda") else "cpu"
        res = self.det.predict(frame_bgr, imgsz=DET_IMGSZ, conf=DET_CONF, max_det=WIDE_DET_MAX,
                               verbose=False, device=dev0)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None
        boxes = res.boxes.xyxy.cpu().numpy()
        feats = self._embed([self._crop_box(frame_rgb, b) for b in boxes])
        return boxes, feats

    # ---------- oturum basi ----------
    @torch.no_grad()
    def add_reference(self, ref_id, ref_bgr):
        rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        self.ref_img_bgr[ref_id] = ref_bgr.copy()   # RoMa inceltmesi icin ham goruntu
        # discrete: DINOv2-L global TTA (336)
        self.ref_disc[ref_id] = self._embed(self._tta(rgb, CROP))
        # pattern: DINOv2-L dense region-mean TTA (224 = 16*14)
        vecs = []
        for v in self._tta(rgb, 224):
            g = self._grid(v)
            vecs.append(F.normalize(g.mean(dim=(2, 3)), dim=1))
        self.ref_pat[ref_id] = torch.cat(vecs, 0)

    # ---------- kare on-hazirlik (referanstan BAGIMSIZ, kare basi BIR KEZ) ----------
    @torch.no_grad()
    def _prep_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dev0 = 0 if self.dev.startswith("cuda") else "cpu"
        res = self.det.predict(frame_bgr, imgsz=DET_IMGSZ, conf=DET_CONF, max_det=DET_MAX,
                               verbose=False, device=dev0)[0]
        cand_feats = None; cand_boxes = None
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
            idx = np.argsort(-confs)[:CONF_TOPK]
            cand_boxes = boxes[idx]
            cand_feats = self._embed([self._crop_box(rgb, b) for b in cand_boxes])

        # Pattern: tum (olcek,k) pooled bolge betimleyicileri TEK tensorde (tek matmul).
        oH, oW = frame_bgr.shape[:2]; descs = []; bxs = []
        for (W, H) in PATTERN_SCALES:
            g = self._grid(cv2.cvtColor(cv2.resize(frame_bgr, (W, H)), cv2.COLOR_BGR2RGB))
            D, Hp, Wp = g.shape[1], g.shape[2], g.shape[3]; sx, sy = oW / W, oH / H
            for k in PATTERN_KS:
                if k >= Hp or k >= Wp:
                    continue
                pooled = F.normalize(F.avg_pool2d(g, k, 1), dim=1)
                oh, ow = pooled.shape[2], pooled.shape[3]
                descs.append(pooled.reshape(D, -1).T)
                rr, cc = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
                bxs.append(np.stack([cc * PATCH * sx, rr * PATCH * sy,
                                     (cc + k) * PATCH * sx, (rr + k) * PATCH * sy], -1).reshape(-1, 4))
        pat_desc = torch.cat(descs, 0) if descs else None
        pat_box = np.concatenate(bxs, 0) if bxs else None
        return dict(cand_feats=cand_feats, cand_boxes=cand_boxes,
                    pat_desc=pat_desc, pat_box=pat_box)

    @torch.no_grad()
    def match(self, frame_bgr, ref_ids=None):
        """Kare icin {ref_id: (x1,y1,x2,y2, score, branch)} dondurur (otomatik kol secimi).

        Kare BIR KEZ islenir (_prep_frame); her referans icin yalniz ucuz kosinus.
        Discrete kolda ek olarak: canli medyan-boyut filtresi, dusuk benzerlikte
        genis-havuz tirmanma ve (varsa) RoMa inceltmesi uygulanir (bkz. modul basi
        docstring)."""
        ids = [r for r in (ref_ids or self.ref_disc.keys()) if r in self.ref_disc]
        if not ids:
            return {}
        prep = self._prep_frame(frame_bgr)
        have_det = prep["cand_feats"] is not None
        pat_desc, pat_box = prep["pat_desc"], prep["pat_box"]
        frame_rgb = None   # yalniz genis-havuz tirmanmasi gerekirse hesaplanir (lazy)
        wide_cache = None  # bu karede birden fazla referans tirmanirsa YOLOE'yi tekrar cagirmasin
        out = {}
        for rid in ids:
            # Pattern kolu: tum bolge betimleyicileriyle TEK matmul.
            pscore = -1.0; pbox = None
            if pat_desc is not None:
                s = (pat_desc @ self.ref_pat[rid].T).max(dim=1).values
                i = int(s.argmax()); pscore = float(s[i])
                pbox = tuple(float(v) for v in pat_box[i])
            if pscore >= PATTERN_ROUTE_THRESH or not have_det:
                if pbox is not None:
                    out[rid] = (*pbox, pscore, "pattern")
                continue

            # Discrete kolu: onceden hesaplanmis aday embedding'leriyle kosinus.
            sims = (prep["cand_feats"] @ self.ref_disc[rid].T).max(dim=1).values
            j_raw = int(sims.argmax())
            raw_box = tuple(float(v) for v in prep["cand_boxes"][j_raw])
            # Boyut gecmisi HER ZAMAN budamasiz argmax'tan guncellenir (filtrenin
            # kendi secimini kendine referans alip kilitlenmesini onlemek icin).
            size_hist = self._ref_sizes[rid]
            size_hist.append(self._box_size(raw_box))
            med_size = float(np.median(size_hist)) if len(size_hist) >= SIZE_MIN_SAMPLES else None

            j = self._pick_best_idx(prep["cand_boxes"], sims, med_size)
            sim = float(sims[j])
            coarse_box = tuple(float(v) for v in prep["cand_boxes"][j])

            if sim < ESCALATE_SIM_THRESH:
                if wide_cache is None:
                    if frame_rgb is None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    wide_cache = self._wide_candidates(frame_bgr, frame_rgb)
                boxes_w, feats_w = wide_cache
                if boxes_w is not None:
                    sims_w = (feats_w @ self.ref_disc[rid].T).max(dim=1).values
                    jw = self._pick_best_idx(boxes_w, sims_w, med_size)
                    simw = float(sims_w[jw])
                    if simw > sim:
                        sim = simw
                        coarse_box = tuple(float(v) for v in boxes_w[jw])

            final_box = coarse_box
            if rid in self.ref_img_bgr:
                refined = self._roma_refine(self.ref_img_bgr[rid], frame_bgr, coarse_box)
                if refined is not None:
                    final_box = refined

            out[rid] = (*final_box, sim, "discrete")
        return out
