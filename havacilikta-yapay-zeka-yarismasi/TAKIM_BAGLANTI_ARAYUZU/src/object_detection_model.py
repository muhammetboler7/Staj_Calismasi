import logging
import time
import os
from collections import deque

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

from ultralytics import YOLO

from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction
from .orbslam_stream import OrbSlamStream, OrbSlamStreamError
from .online_alignment import OnlineAligner
from .translation_logger import TranslationComparisonLogger
from .pad_pipeline import PadLandingClassifier, decide_pad, CLF_W, DOLU_THRESH
from .reference_matcher import ReferenceMatcher


# ------------------------------------------------------------------------------
# GOREV 1 (Nesne Tespiti) modeli
# ------------------------------------------------------------------------------
# Egitilmis YOLO agirliklari - 4 SINIF (Insan, Tasit, UAP, UAI). UAP_Inemez/
# UAI_Inemez ayrimi YOLO seviyesinde YOK: inis karari zaten HER ZAMAN asagidaki
# MobileNetV3 siniflandiricisi (+ insan/tasit/kenar kontrolu) tarafindan verilir
# (decide_pad() - bkz. detect() icindeki "UAP/UAI inis karari" blogu), yani
# YOLO'nun 6-sinifli bir "ilk karar"i vermesi gereksiz ve fazladan karisiklik.
# Fine-tune karsilastirmasinda (8 model, gercek yarisma videosu uzerinde mAP)
# bu agirlik en iyi cikan oldu (mAP50=0.840 vs production'daki eski 6-sinifli
# agirlikta 0.742) - bkz. 8_temmuz_egitim/runs_finetune_model1/.../args.yaml.
MODEL_PATH = ("/media/CezeriDrive_12TB/StajyerCalismalari/2026_Yaz/Muhammet_Boler/"
              "8_temmuz_egitim/runs_finetune_model1/"
              "model1_4class_finetune_800_stratified/weights/best.pt")

# Tespit guven esigi (bu degerin altindaki kutular atilir).
CONF_THRESHOLD = 0.25

# Etiketlenmis (uzerine kutu cizilmis) karelerin kaydedilecegi klasor.
ANNOTATED_DIR = "./etiketlenmis_veriler/"

# Gorev 1 bonus: Tasit hareket durumu (Hareketli/Sabit) - PENCERELI YER
# DEGISTIRME. Onceki surum tek kare ciftindeki optical-flow sapmasina
# bakiyordu; 7.5 fps'te 3 m/s'lik arac karede ~8px kayarken akis gurultusu +
# kutu kenarindan zemine kacan keypoint'ler birkac px hata uretiyordu (SNR
# kotu, testte yetersiz kaldi). Yeni yontem karari ZAMANA yayar:
#   1) Arka plan hareketi (drone'un kendi hareketi): tasit kutulari DISINDAKI
#      zemin noktalari LK optical flow ile izlenir, RANSAC ile HOMOGRAFI fit
#      edilir. Zemin duzlem oldugu icin bu, donme dahil kamera hareketinin
#      goruntudeki TAM modelidir - "tasit hizi = gorunen hiz - drone hizi"
#      cikarmasini homografi kendisi yapar.
#   2) Her tasit kutusu kareler arasi (ongorulmus kutuyla) IoU + mesafe-kapisi
#      ile takip edilir. Her karede izin onceki kutusu homografiyle bu kareye
#      tasinir (wbox = "statikse nerede olurdu") ve simdiki kutuyla arasindaki
#      ADIM olculur. Adim, cerceve kenarina YAPISIK OLMAYAN kenardan alinir
#      (_edge_step): araba KARE'ye girerken kutu buyur (giris kenari pinli) -
#      arka kenar sabit; KARE'den cikarken kutu kuculur (cikis kenari pinli) -
#      yine arka kenar sabit. Merkez kullanmak, giris/cikista kutu buyuyup
#      kuculdugu icin sahte hareket uretiyordu; kenar-secimi bunu KOKTEN cozer
#      (giren hareketli arac hemen yakalanir, cikan duran arac Sabit kalir).
#   3) Karar, penceredeki adim vektorlerinin MEDYAN hizina gore verilir:
#      duran aracin adimlari sifir etrafinda titrer (medyan ~0), hareketli
#      aracin butun adimlari ayni yondedir (medyan = gercek hiz) - tek-cift
#      karara gore SNR kokten iyidir; tek tuk bozuk adim medyani kaydiramaz.
#      Kutu ici keypoint/akis gerekmez. Kutu boyu bir karede
#      BOX_SIZE_STAB_TOL'den fazla degisirse (iz komsu araca atladi) adim
#      gecmisi sifirlanir.
#   4) Metrik esik: piksel yer degistirme GSD (m/px) ile gercek hiza cevrilir.
#      GSD kaynagi oncelik sirasiyla:
#      (a) SLAM irtifasi: aligner.scale * med_depth / fx. Dogrulugu bagimsiz
#          sinandi: THYZ_2026 segmentinde medyan arac kutusu 142px, SLAM
#          GSD'siyle (0.0357) 5.0m arac boyu = gercekci; akis tabanli
#          tahminler ayni yerde 0.027-0.054 arasinda savruldu (medyan akis,
#          donme+parallaks altinda kotu bir vekildir; p10-p90 orani 3.6x).
#          Ilk denemedeki sahte "Hareketli" vakalari akis-GSD'nin sisme
#          doneminden geliyordu (0.048 vs dogru ~0.036).
#      (b) Akis kalibrasyonu (yalniz SLAM irtifasi YOKKEN): kayan pencerede
#          oran-toplamlari sum(kamera m)/sum(arka plan px) - tekil oranlarin
#          EMA'sindan cok daha kararli - ve >=GSD_MIN_SAMPLES cift olmadan
#          kullanilmaz.
#      (c) Hicbiri yoksa KARAR YOK - olceksiz piksel esigi kaldirildi.
VEHICLE_SPEED_THRESH_MS = 2.5   # bu hizin ustu HAREKETLI (m/s). 3.0'dan indirildi:
                                # medyan-adim hiziyla parkli arac gurultu tabani
                                # 0.4-1.8 m/s'e dustugu icin 2.5 guvenli; yavas
                                # suruklenen araclari da yakalar.
EFFECTIVE_FPS = 7.5             # kareler 30fps videodan 4'er atlanmis -> 7.5 kare/sn
MOVING_WINDOW_FRAMES = 8        # karar penceresi (~1.07 s) - adim gecmisi uzunlugu
VEHICLE_FLOW_DIFF_RATIO = 0.08  # parallaks payi: adim basina arka plan kaymasinin
                                # bu orani kadar yer degistirme, tasit zeminden
                                # yuksek oldugu icin sabitken de olusabilir
FX_1080P = 1389.7               # thyz2026.yaml Camera1.fx (1920 px genislik icin);
                                # farkli cozunurlukte fx genislikle oranlanir
BOX_SIZE_STAB_TOL = 0.35        # kutu eni/boyu bir karede bu orandan fazla degistiyse
                                # gecmis SIFIRLANIR - artik yalniz KIMLIK KARISMASI
                                # (iz komsu araca atladi) icin; kareye giris/cikis
                                # kaynakli buyume/kuculme artik ADIM (_edge_step)
                                # kenar secimiyle ele aliniyor, o yuzden esik gevsek.
EDGE_PIN_FRAC = 0.04            # kutu kenari cerceveye bu oran kadar yakinsa "pinli"
                                # sayilir ve adim olcumunde KARSI (uzak) kenar
                                # kullanilir. Genis tutmak guvenli: pinli olmayan
                                # rijit kutuda uzak-kenar adimi = merkez adimi.
MIN_DECIDE_STEPS = 3            # tam karar (Sabit VEYA Hareketli) icin gereken adim
EARLY_DECIDE_FACTOR = 1.2       # <MIN_DECIDE_STEPS adimda bile hiz esigin bu kati ustundeyse
                                # HEMEN Hareketli de (yalniz Hareketli - erken Sabit yok):
                                # otoyol araci ~11 m/s, esik 2.5 -> 1. adimda gecer, boylece
                                # yeni beliren hizli arac 3-4 kare beklemeden yakalanir;
                                # sinirdaki (2.5-4.5 m/s) araclar yine 3 adim bekler (guvenli)
MIN_BG_MATCH_POINTS = 12        # arka plan icin en az izlenen nokta
BG_FIT_MAX_ERR_PX = 2.5         # homografi arka plani bu hatadan kotu acikliyorsa karar verme
BOX_MASK_MARGIN_PX = 12         # arka plan maskesinde kutularin etrafina eklenen pay
GSD_MIN_CAM_MOVE_M = 0.15       # akis-GSD: kamera bundan az kaydiysa cift alinmaz
GSD_MIN_BG_FLOW_PX = 4.0        # akis-GSD: arka plan akisi bundan azsa cift alinmaz
GSD_FLOW_WINDOW = 60            # akis-GSD kayan pencere (cift sayisi, ~8 s)
GSD_MIN_SAMPLES = 5             # akis-GSD bu kadar cift birikmeden KULLANILMAZ
MOVING_VOTE_WINDOW = 3          # nihai karar son N pencere kararinin cogunluk oyu (histerezis)
TRACK_MATCH_IOU = 0.3           # kutulari kareler arasi eslestirme icin IoU esigi
TRACK_GATE_FACTOR = 1.6         # IoU tutmazsa mesafe-kapili eslestirme: merkez uzakligi
                                # bu kadar x (kutu buyuk kenari) icindeyse eslesir.
                                # Otoyol araci kareler arasi ~1 kutu boyu kayiyor -
                                # saf IoU koptugu icin izi bootstrap eden bu kapidir.
TRACK_GATE_SIZE_RATIO = 2.2     # mesafe-kapili eslestirmede kutu boyu bu orandan fazla
                                # farkliysa eslestirme (kucuk kutuyu buyuge baglama)
MOVING_LOCK_ON = 2              # ardarda bu kadar HAREKETLI olcumu izi KILITLER: iz
                                # kareden cikana kadar Hareketli raporlanir (kenarda
                                # yarim gorunen/olculemeyen hareketli arac Sabit'e
                                # dusmesin). Ardarda MOVING_LOCK_OFF net SABIT olcumu
                                # kilidi acar (duran araca takili kalmasin).
MOVING_LOCK_OFF = 2

# Gorev 1 bonus: Insan izleme (recall takviyesi). YOLO tek karede insani
# kacirdiginda (kismi kapanma/hareket bulanikligi/kucuk boy), bir onceki
# karede tespit edilmisse kutuyu optical-flow ile bu kareye TASIYIP yine
# raporlariz - boylece tek karelik "flicker" recall'i dusurmez. Uzun sureli
# gercek kayiplarda (kisi gercekten kareyi terk etti) PERSON_TRACK_MAX_MISS
# kare sonra iz dusurulur.
# Degerler, gercek Annotations/ GT'sine karsi (THYZ_2026_Online_Yarisma, ilk
# 400 kare) yapilan bir parametre taramasiyla secildi: max_miss>1 recall'i
# biraz daha artirsa da (drift biriken/hayalet kutular yuzunden) precision'i
# hizla dusuruyor (orn. max_miss=5 -> precision ~%49); max_miss=1 en iyi
# recall/precision dengesini (F1) veriyor. Baseline (yalniz YOLO) Recall@IoU0.5
# %56.52, Precision %96.30 -> bu ayarlarla Recall %67.39, Precision %87.32.
PERSON_TRACK_MATCH_IOU = 0.10    # bu karenin YOLO tespiti <-> onceki iz eslestirme esigi
PERSON_TRACK_MAX_MISS = 2        # ardarda bu kadar karede YOLO bulamazsa iz dusurulur
PERSON_TRACK_MIN_POINTS = 3      # kutu ici optical-flow bu kadar noktadan azsa tasima guvenilmez
PERSON_TRACK_MIN_HITS = 1        # bir iz, ardarda bu kadar GERCEK YOLO tespitiyle
                                 # "dogrulanmadan" optical-flow ile tasinamaz (1 =
                                 # devre disi: tek tespitten sonra bile 2 kare izlenir)
PERSON_TRACK_SUPPRESS_IOU = 0.3  # izlenen (coasted) bir kutu, bu karede YOLO'nun
                                 # bulduğu GERCEK bir kutuyla bu IoU'dan fazla
                                 # cakisiyorsa hayalet/cift kutu sayilip atilir -
                                 # optical-flow driftinin gercek tespitle ayni kisi
                                 # icin IKI kutu (biri kaymis-turuncu, biri yeni-
                                 # yesil) uretmesini onler

# Model sinif ISMI -> (yarisma sinifi, ilk inis durumu, ilk hareket durumu).
# ISIM bazli (index/pozisyon bazli DEGIL): boylece 6-sinifli (UAP/UAP_Inemez/
# UAI/UAI_Inemez ayrimini YOLO seviyesinde yapan production agirligi) VE
# 4-sinifli (bu ayrimi tamamen MobileNetV3 siniflandiriciya birakan, orn.
# fine-tune modelleri) agirliklar AYNI kodla, self.model.names sozlugune gore
# dogru calisir. Index pozisyonuna guvenmek modeller arasi sinif sirasi/sayisi
# farkli oldugunda yanlis sinifa (orn. UAI kutusunu UAP_Inemez olarak) esler.
# Ilk inis durumu UAP/UAI icin ONEMSIZ: decide_pad() (asagida) HER ZAMAN ezer;
# yalniz _Inemez ayrimi OLMAYAN modellerde bir placeholder olarak kalir.
NAME_TO_SEMANTICS = {
    "İnsan":      (classes["Insan"], landing_statuses["Inis Alani Degil"], moving_statuses["Tasit Degil"]),
    "Taşıt":      (classes["Tasit"], landing_statuses["Inis Alani Degil"], moving_statuses["Sabit"]),
    "UAP":        (classes["UAP"],   landing_statuses["Inilebilir"],       moving_statuses["Tasit Degil"]),
    "UAP_İnemez": (classes["UAP"],   landing_statuses["Inilemez"],         moving_statuses["Tasit Degil"]),
    "UAİ":        (classes["UAI"],   landing_statuses["Inilebilir"],       moving_statuses["Tasit Degil"]),
    "UAİ_İnemez": (classes["UAI"],   landing_statuses["Inilemez"],         moving_statuses["Tasit Degil"]),
}

# Gorev 1 (UAP/UAI inis karari): pist ailesi isimleri (6-sinifli VEYA 4-sinifli
# modelde ayni sekilde gecerli) ve insan sinifi ismi. YOLO'nun kendi
# UAP_Inemez/UAI_Inemez ayrimi (varsa) ILK karar olarak kalir; UAP/UAI
# kutulari icin nihai inis durumunu MobileNetV3 siniflandirici (pad_pipeline)
# verir: pist bos -> Inilebilir (1), dolu/insan/kismen-disarida -> Inilemez (0).
PAD_NAMES = {"UAP", "UAP_İnemez", "UAİ", "UAİ_İnemez"}
PERSON_NAME = "İnsan"


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url, model_path=None, enable_slam=True):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Gorev 1 modeli bir kere yuklenir; her karede yeniden yuklenmez.
        # model_path: varsayilan MODEL_PATH yerine baska bir agirlik denemek icin
        # (ornegin karsilastirma/video render script'lerinden).
        model_path = model_path or MODEL_PATH
        logging.info(f"Loading detection model: {model_path}")
        self.model = YOLO(model_path)

        # self.model.names (ultralytics checkpoint'inden) -> index bazli
        # calisma-zamani tablolari. Bkz. NAME_TO_SEMANTICS/PAD_NAMES/PERSON_NAME
        # yorumu: bu, 6-sinifli VE 4-sinifli agirliklarin ikisiyle de dogru
        # calismasini saglayan model-agnostik esleme.
        self._model_class_map = {}
        self._pad_model_classes = set()
        self._person_model_class = None
        for idx, name in self.model.names.items():
            sem = NAME_TO_SEMANTICS.get(name)
            if sem is None:
                logging.warning(f"Bilinmeyen model sinifi '{name}' (idx={idx}); bu sinif atlanacak.")
                continue
            self._model_class_map[idx] = sem
            if name in PAD_NAMES:
                self._pad_model_classes.add(idx)
            if name == PERSON_NAME:
                self._person_model_class = idx

        # Gorev 1: UAP/UAI inis karari icin MobileNetV3 'bos/dolu' siniflandirici.
        # Yuklenemezse (agirlik yok/torch yok) pist inis durumu YOLO sinifindan
        # gelen ilk karara duser; sistem yine calisir.
        try:
            self.pad_clf = PadLandingClassifier(CLF_W)
            logging.info(f"Loaded pad landing classifier: {CLF_W}")
        except Exception as e:
            self.pad_clf = None
            logging.error(f"Pad inis siniflandirici yuklenemedi ({CLF_W}): {e}; "
                          "UAP/UAI inis durumu YOLO sinifindan alinacak.")

        # Gorev 3: tam otomatik referans esleme (bkz. reference_matcher.py).
        # Yuklenemezse (agirlik/torch/timm yok) Gorev 3 atlanir; sistem calisir.
        try:
            self.reference_matcher = ReferenceMatcher()
        except Exception as e:
            self.reference_matcher = None
            logging.error(f"ReferenceMatcher yuklenemedi: {e}; Gorev 3 atlanacak.")
        self._added_refs = set()   # oturum-ici eklenen referans url'leri

        # Gorev 1 bonus: onceki karenin gri tonlamali hali - tasit hareket
        # tespiti (akis karsilastirma) icin kare-arasi optical flow'da kullanilir.
        self._prev_gray = None
        # Metrik olcek durumu: onceki karenin kamera x/y konumu (m), EMA ile
        # yumusatilmis GSD (m/px) ve son karardaki tasit hizlari (gorsellestirme).
        self._prev_cam_xy = None
        # Akis-GSD kayan penceresi: (kamera_m, akis_px) ciftleri; GSD =
        # sum(m)/sum(px). Yalniz SLAM irtifasi yokken kullanilir.
        self._gsd_moves = deque(maxlen=GSD_FLOW_WINDOW)
        self._gsd_flows = deque(maxlen=GSD_FLOW_WINDOW)
        self._last_speeds = []
        # Tasit iz gecmisi: [{box, wbox, steps, bg, votes, ...}] - steps: zemine
        # gore kare-arasi kutu adimlari (kenar-secimli, penceresel hareket
        # karari), bg: karsilik gelen arka plan adim buyuklukleri, votes:
        # histerezis, hlock: yapiskan Hareketli kilidi.
        self._move_tracks = []
        # Gorev 1 bonus: Insan izleri: [{box, missed}] - bkz. _track_persons.
        self._person_tracks = []

        # Gorev 2: canli ORB-SLAM3 sureci (bkz. orbslam_stream.py) oturum
        # boyunca ACIK kalir; her karede tek TrackMonocular cagrisiyla
        # beslenir. health_status==1 karelerde biriken (SLAM pozu, GT)
        # ciftleriyle OnlineAligner surekli yeniden hizalanir; health_status
        # ==0 karelerde bu hizalama SLAM pozunu dunya koordinatina tasir.
        # enable_slam=False: Gorev 2 tamamen devre disi (orn. yalniz Gorev 1/3'u
        # denemek/render etmek icin) - ORB-SLAM3 sureci hic baslatilmaz.
        if enable_slam:
            self.orbslam = OrbSlamStream(stderr_log_path="./_logs/orbslam_stream.log")
            self.aligner = OnlineAligner()
            self.translation_log = TranslationComparisonLogger()
        else:
            self.orbslam = None
            self.aligner = None
            self.translation_log = None

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Indirmek istedigimiz frame frames.json dosyasinda mevcut mu kontrol edelim
        image_name = img_url.split("/")[-1]
        # Eger indirecegimiz frame'i daha once indirmediysek indirme islemine gecelim
        if image_name not in images_files:
            headers = {'Authorization': f'Token {auth_token}'} if auth_token else {}
            for attempt in range(retries):
                    try:
                        response = requests.get(img_url, headers=headers, timeout=60)
                        response.raise_for_status()

                        img_bytes = response.content
                        with open(images_folder + image_name, 'wb') as img_file:
                            img_file.write(img_bytes)

                        t2 = time.perf_counter()
                        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                        return

                    except requests.exceptions.RequestException as e:
                        logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                        logging.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        # Eger indirecegimiz frame'i daha once indirdiysek indirme yapmadan devam edebiliriz
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    @staticmethod
    def _track_points(src_gray, dst_gray, pts):
        """Verilen noktalari LK optical flow ile src->dst izler; ileri-geri
        (forward-backward) kontrolden gecen (src, dst) ciftlerini dondurur.
        Geri-izleme, yanlis eslesmeleri (ornegin kutu kenarindan zemine
        'kacan' noktalari) eler."""
        if pts is None or len(pts) == 0:
            return None, None
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        dst_pts, status, _ = cv2.calcOpticalFlowPyrLK(src_gray, dst_gray, pts, None, **lk)
        if dst_pts is None:
            return None, None
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(dst_gray, src_gray, dst_pts, None, **lk)
        if back_pts is None:
            return None, None
        fb_err = np.linalg.norm((back_pts - pts).reshape(-1, 2), axis=1)
        valid = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (fb_err < 1.0)
        if valid.sum() == 0:
            return None, None
        return pts.reshape(-1, 2)[valid], dst_pts.reshape(-1, 2)[valid]

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _warp_pts(pts, H):
        """(N,2) noktalari 3x3 homografi ile tasir, (N,2) dondurur."""
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, H).reshape(-1, 2)

    @staticmethod
    def _edge_step(cur, prev_w, w, h):
        """Iki kare arasi tasit KUTU adimini (dx, dy), zemine-gore, hesaplar.
        prev_w: onceki kutunun homografiyle bu kareye tasinmis hali (statikse
        cur ile ortusur). Her eksende cerceve kenarina YAPISIK OLMAYAN kenar
        kullanilir: giren arac (ust pinli) alt kenardan, cikan arac (alt pinli)
        ust kenardan olculur - iki durumda da secilen kenar aracin gorunen
        fiziksel kismidir, kutu buyume/kuculmesinden etkilenmez. Kenar pinli
        degilse merkez (iki kenar ortalamasi, daha az gurultu) kullanilir."""
        m = EDGE_PIN_FRAC
        nl, nr = cur[0] < m * w, cur[2] > w - m * w
        nt, nb = cur[1] < m * h, cur[3] > h - m * h
        if nl and not nr:
            dx = cur[2] - prev_w[2]                    # sol pinli -> sag kenar
        elif nr and not nl:
            dx = cur[0] - prev_w[0]                    # sag pinli -> sol kenar
        else:
            dx = (cur[0] + cur[2]) / 2.0 - (prev_w[0] + prev_w[2]) / 2.0
        if nt and not nb:
            dy = cur[3] - prev_w[3]                    # ust pinli -> alt kenar
        elif nb and not nt:
            dy = cur[1] - prev_w[1]                    # alt pinli -> ust kenar
        else:
            dy = (cur[1] + cur[3]) / 2.0 - (prev_w[1] + prev_w[3]) / 2.0
        return (dx, dy)

    def _track_persons(self, curr_gray, person_boxes):
        """Insan recall takviyesi: bir onceki karede izlenen ama bu karede
        YOLO'nun eslemedigi insanlari kutu-ici optical-flow ile tasiyarak
        "hala orada" sayar (bkz. PERSON_TRACK_* sabitleri, dosya basi yorum).

        Donen deger: (this_frame_boxes, coasted_boxes)
          this_frame_boxes: bu karede gecerli TUM insan kutulari (YOLO + izlenen)
          coasted_boxes: bunlardan SADECE izlemeyle var sayilanlar - YOLO bu
            karede bulamadi, cagiran taraf bunlar icin AYRICA DetectedObject
            eklemeli (YOLO'nun bulduklari zaten ana dongude eklenmis olur).
        """
        prev_gray = self._prev_gray
        h, w = curr_gray.shape

        # Bu karenin YOLO tespitleri <-> onceki izler acgozlu IoU eslestirmesi.
        pairs = []
        for ti, t in enumerate(self._person_tracks):
            for di, box in enumerate(person_boxes):
                iou = self._box_iou(t["box"], box)
                if iou > 0:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        track_to_det = {}
        used_ti, used_di = set(), set()
        for iou, ti, di in pairs:
            if ti in used_ti or di in used_di or iou < PERSON_TRACK_MATCH_IOU:
                continue
            used_ti.add(ti); used_di.add(di)
            track_to_det[ti] = di

        new_tracks = []
        coasted_boxes = []
        for ti, t in enumerate(self._person_tracks):
            if ti in track_to_det:
                # Gercek YOLO tespiti: bir onceki karede izleniyorduysa (coast
                # sonrasi donus) dogrulama sayaci SIFIRDAN baslar - "arka arkaya
                # N gercek tespit" sarti, coast sonrasi tekrar saglanmali.
                t["hit_streak"] = 1 if t["missed"] > 0 else t.get("hit_streak", 1) + 1
                t["box"] = person_boxes[track_to_det[ti]]
                t["missed"] = 0
                new_tracks.append(t)
                continue
            # Bu karede YOLO eslesmedi: SADECE ardarda PERSON_TRACK_MIN_HITS
            # gercek tespitle "dogrulanmis" izler optical-flow ile tasinir -
            # tek karelik yanlis-pozitif bir YOLO tespiti hemen izlemeye
            # baslamaz (dogrulanmamis iz, ilk kacirmada direkt dusurulur).
            t["missed"] += 1
            if (t.get("hit_streak", 1) < PERSON_TRACK_MIN_HITS
                    or t["missed"] > PERSON_TRACK_MAX_MISS or prev_gray is None
                    or prev_gray.shape != curr_gray.shape):
                continue  # dogrulanmamis / cok uzun sure kayip / referans kare yok
            x1, y1, x2, y2 = t["box"]
            mx1, my1 = max(0, int(x1)), max(0, int(y1))
            mx2, my2 = min(w, int(x2)), min(h, int(y2))
            if mx2 - mx1 < 4 or my2 - my1 < 4:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[my1:my2, mx1:mx2] = 255
            pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=50,
                                          qualityLevel=0.01, minDistance=3, blockSize=7)
            src, dst = self._track_points(prev_gray, curr_gray, pts)
            if src is None or len(src) < PERSON_TRACK_MIN_POINTS:
                continue
            disp = np.median(dst - src, axis=0)
            new_box = (x1 + disp[0], y1 + disp[1], x2 + disp[0], y2 + disp[1])
            # Hayalet-kutu bastirma: tasinan kutu bu karede YOLO'nun buldugu
            # GERCEK bir kutuyla zaten cakisiyorsa (ornegin drift yuzunden bu
            # izin gercek tespitle eslesmesi kacmis olsa bile), gercek tespit
            # zaten o bolgeyi kapsiyor demektir - hayalet kutuyu ekleme, izi dusur.
            if any(self._box_iou(new_box, rb) >= PERSON_TRACK_SUPPRESS_IOU for rb in person_boxes):
                continue
            t["box"] = new_box
            new_tracks.append(t)
            coasted_boxes.append(t["box"])

        for di, box in enumerate(person_boxes):
            if di not in used_di:
                new_tracks.append(dict(box=box, missed=0, hit_streak=1))

        self._person_tracks = new_tracks
        return [t["box"] for t in self._person_tracks], coasted_boxes

    def _estimate_vehicle_moving(self, curr_gray, vehicle_boxes, cam_xy=None, alt_m=None):
        """Her tasit kutusu (x1,y1,x2,y2) icin Hareketli->True, Sabit->False,
        karar verilemedi->None dondurur (None durumunda cagiran taraf
        varsayilan durumu, "Sabit", korur).

        Yontem (penceresel yer degistirme - dosya basindaki blok yoruma bak):
          1) Arka plan homografisi (prev->curr) drone hareketini modeller.
          2) Her tasit izinin GECMIS merkezleri bu homografiyle simdiki kare
             koordinatina tasinir; boylece merkez gecmisi "zemin sabit"
             referansinda birikir.
          3) Karar, penceredeki NET yer degistirmeden: |son - ilk| * GSD *
             fps / adim_sayisi = hiz (m/s); esigi asarsa HAREKETLI.
          4) GSD birincil olarak SLAM irtifasindan (alt_m / fx); yoksa akis
             kalibrasyonu (EMA); o da yoksa piksel esigi.
        cam_xy: kameranin metre cinsinden x/y konumu (GT veya hizalanmis SLAM),
        alt_m: yerden yukseklik (aligner.scale * med_depth), metre.
        """
        self._last_speeds = [None] * len(vehicle_boxes)
        prev_gray = self._prev_gray
        prev_cam_xy = self._prev_cam_xy
        self._prev_gray = curr_gray
        self._prev_cam_xy = cam_xy

        h, w = curr_gray.shape

        # 1) Arka plan homografisi (prev -> curr). Bulunamazsa zincir kopar:
        # izlerin merkez gecmisi artik ortak koordinatta degildir, sifirlanir.
        H_bg = None
        if prev_gray is not None and prev_gray.shape == curr_gray.shape:
            bg_mask = np.full((h, w), 255, dtype=np.uint8)
            for (x1, y1, x2, y2) in vehicle_boxes:
                mx1 = max(0, int(x1) - BOX_MASK_MARGIN_PX)
                my1 = max(0, int(y1) - BOX_MASK_MARGIN_PX)
                mx2 = min(w, int(x2) + BOX_MASK_MARGIN_PX)
                my2 = min(h, int(y2) + BOX_MASK_MARGIN_PX)
                bg_mask[my1:my2, mx1:mx2] = 0
            bg_seed = cv2.goodFeaturesToTrack(
                curr_gray, mask=bg_mask, maxCorners=300, qualityLevel=0.01,
                minDistance=15, blockSize=7)
            bg_curr, bg_prev = self._track_points(curr_gray, prev_gray, bg_seed)
            if bg_curr is not None and len(bg_curr) >= MIN_BG_MATCH_POINTS:
                Hh, inlier_mask = cv2.findHomography(bg_prev, bg_curr, cv2.RANSAC, 2.0)
                if Hh is not None:
                    inl = inlier_mask.reshape(-1).astype(bool)
                    if inl.sum() >= MIN_BG_MATCH_POINTS:
                        pred_bg = self._warp_pts(bg_prev, Hh)
                        bg_fit_err = float(np.median(
                            np.linalg.norm(pred_bg[inl] - bg_curr[inl], axis=1)))
                        if bg_fit_err <= BG_FIT_MAX_ERR_PX:
                            H_bg = Hh
                # Akis-GSD ciftlerini biriktir (yalniz SLAM irtifasi yokken
                # kullanilacak yedek; oran-toplamlari tekil oranlardan kararli).
                if cam_xy is not None and prev_cam_xy is not None:
                    cam_move = float(np.linalg.norm(np.asarray(cam_xy, dtype=float)
                                                     - np.asarray(prev_cam_xy, dtype=float)))
                    bg_flow_mag = float(np.linalg.norm(np.median(bg_curr - bg_prev, axis=0)))
                    if cam_move >= GSD_MIN_CAM_MOVE_M and bg_flow_mag >= GSD_MIN_BG_FLOW_PX:
                        self._gsd_moves.append(cam_move)
                        self._gsd_flows.append(bg_flow_mag)

        # 2) Her izin onceki kutusunu bu kareye tasi (wbox) + bu karenin arka
        # plan adimini (bg_step, parallaks payi icin) hesapla. Homografi yoksa
        # zincir kopar: adim gecmisi sifirlanir.
        if H_bg is not None:
            for t in self._move_tracks:
                b = t["box"]
                wc = self._warp_pts([(b[0], b[1]), (b[2], b[3])], H_bg)
                t["wbox"] = (float(min(wc[0, 0], wc[1, 0])), float(min(wc[0, 1], wc[1, 1])),
                             float(max(wc[0, 0], wc[1, 0])), float(max(wc[0, 1], wc[1, 1])))
                ocx, ocy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                ncx, ncy = (t["wbox"][0] + t["wbox"][2]) / 2.0, (t["wbox"][1] + t["wbox"][3]) / 2.0
                t["bg_step"] = float(np.hypot(ncx - ocx, ncy - ocy))
        else:
            for t in self._move_tracks:
                t["steps"].clear()
                t["bg"].clear()
                t["wbox"] = None

        # GSD (m/px) secimi (gerekce dosya basindaki blok yorumda):
        #   1) SLAM irtifasi - arac-boyu sinamasiyla dogrulanan kaynak.
        #   2) Akis oran-toplamlari (yalniz SLAM yokken, >=GSD_MIN_SAMPLES cift).
        #   3) Ikisi de yoksa None -> KARAR VERILMEZ (varsayilan "Sabit" kalir).
        if alt_m is not None and alt_m > 0:
            gsd = alt_m / (FX_1080P * w / 1920.0)
        elif len(self._gsd_moves) >= GSD_MIN_SAMPLES:
            gsd = float(sum(self._gsd_moves) / sum(self._gsd_flows))
        else:
            gsd = None

        # 3) Kutulari izlere eslestir, adimi ekle, pencere karari ver.
        # Eslestirme, izin KENDI hiziyla ongorulmus kutusuna gore yapilir:
        # otoyol hizindaki arac kareler arasi kendi boyu kadar kayar, ongorusuz
        # IoU tutmaz ve iz her karede sifirdan baslardi (otoyolda surekli "?"
        # vakasi). Sabit izde hiz ~0 oldugundan davranis degismez.
        results = []
        remaining = list(self._move_tracks)
        pred_box = {}
        for t in remaining:
            base = t["wbox"] if t.get("wbox") is not None else t["box"]
            vx, vy = t.get("vel") or (0.0, 0.0)
            pred_box[id(t)] = (base[0] + vx, base[1] + vy, base[2] + vx, base[3] + vy)
        new_tracks = []
        for i, box in enumerate(vehicle_boxes):
            bcx, bcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            bsz = max(box[2] - box[0], box[3] - box[1])
            # 1) Once IoU (ongorulmus kutu ile): sabit/yavas araclar burada eslesir.
            best_iou, best_t = 0.0, None
            for t in remaining:
                iou = self._box_iou(box, pred_box[id(t)])
                if iou > best_iou:
                    best_iou, best_t = iou, t
            # 2) IoU tutmazsa mesafe-kapisi: hizli arac (~1 kutu boyu kayma)
            #    boyle bootstrap edilir. Boyut-uyumu ile yanlis eslesme onlenir.
            if best_t is None or best_iou < TRACK_MATCH_IOU:
                best_d, cand = float("inf"), None
                for t in remaining:
                    pb = pred_box[id(t)]
                    pcx, pcy = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
                    psz = max(pb[2] - pb[0], pb[3] - pb[1])
                    d = ((bcx - pcx) ** 2 + (bcy - pcy) ** 2) ** 0.5
                    sr = max(bsz, psz) / max(min(bsz, psz), 1.0)
                    if (d < TRACK_GATE_FACTOR * max(bsz, psz)
                            and sr <= TRACK_GATE_SIZE_RATIO and d < best_d):
                        best_d, cand = d, t
                best_t = cand if cand is not None else None
                best_iou = TRACK_MATCH_IOU if cand is not None else 0.0
            if best_t is not None and best_iou >= TRACK_MATCH_IOU:
                t = best_t
                remaining.remove(best_t)
                # Bu karenin adimi: onceki kutunun tasinmis hali (wbox) ile
                # simdiki kutu arasi, sinira yapismayan kenardan (bkz. _edge_step).
                if t.get("wbox") is not None:
                    t["steps"].append(self._edge_step(box, t["wbox"], w, h))
                    t["bg"].append(t.get("bg_step", 0.0))
            else:
                t = dict(steps=deque(maxlen=MOVING_WINDOW_FRAMES),
                         bg=deque(maxlen=MOVING_WINDOW_FRAMES),
                         votes=deque(maxlen=MOVING_VOTE_WINDOW), wh=None,
                         hlock=False, run_true=0, run_false=0, vel=None,
                         wbox=None, bg_step=0.0)
            # Kutu boyu kararlilik bekcisi: yalniz BUYUK degisim (kimlik
            # karismasi/iz atlamasi) icin - adim gecmisini sifirlar.
            bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
            if t["wh"] is not None:
                pw, ph = t["wh"]
                if (abs(bw - pw) > BOX_SIZE_STAB_TOL * max(pw, 1.0)
                        or abs(bh - ph) > BOX_SIZE_STAB_TOL * max(ph, 1.0)):
                    t["steps"].clear()
                    t["bg"].clear()
                    t["votes"].clear()
            t["wh"] = (bw, bh)
            t["box"] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            new_tracks.append(t)

            raw = None
            # Hiz = adim vektorlerinin bilesen-bazinda MEDYANI: tek tuk bozuk
            # adim (kimlik karismasi/tespit sicramasi) medyani kaydiramaz;
            # gercek harekette adimlar ayni yonde oldugu icin medyan = gercek hiz.
            ns = len(t["steps"])
            if ns >= 1 and gsd is not None:
                med_v = np.median(np.asarray(t["steps"], dtype=float), axis=0)
                t["vel"] = (float(med_v[0]), float(med_v[1]))  # eslestirme ongorusu
                bg_med = float(np.median(t["bg"])) if t["bg"] else 0.0
                speed = float(np.linalg.norm(med_v)) * gsd * EFFECTIVE_FPS
                self._last_speeds[i] = speed
                # Parallaks payi: tasit zeminden yuksek oldugu icin kamera
                # kayarken sabit tasit da zemine gore bir miktar "kayar";
                # bu, adim basina arka plan kaymasiyla oranli.
                thr = max(VEHICLE_SPEED_THRESH_MS,
                          VEHICLE_FLOW_DIFF_RATIO * bg_med * gsd * EFFECTIVE_FPS)
                if ns >= MIN_DECIDE_STEPS:
                    raw = bool(speed > thr)             # tam karar (Sabit veya Hareketli)
                elif speed > EARLY_DECIDE_FACTOR * thr:
                    raw = True                          # erken: yalniz ACIK Hareketli
                # aksi halde raw=None: henuz emin degiliz -> varsayilan Sabit

            # Yapiskan HAREKETLI kilidi: ardarda MOVING_LOCK_ON kez Hareketli
            # olculen iz kilitlenir; olcum uretilemeyen karelerde (kenarda
            # yarim gorunen/gecmisi sifirlanan cikis aninda) Hareketli
            # raporlanmaya devam eder. Ardarda MOVING_LOCK_OFF net Sabit
            # olcumu kilidi acar (arac durursa Sabit'e doner).
            if raw is True:
                t["run_true"] += 1
                t["run_false"] = 0
                if t["run_true"] >= MOVING_LOCK_ON:
                    t["hlock"] = True
            elif raw is False:
                t["run_false"] += 1
                t["run_true"] = 0
                if t["run_false"] >= MOVING_LOCK_OFF:
                    t["hlock"] = False

            # Histerezis: son MOVING_VOTE_WINDOW pencere kararinin cogunlugu;
            # oy yoksa (olcum uretilemiyor) kilit devredeyse Hareketli.
            if raw is not None:
                t["votes"].append(raw)
            v = t["votes"]
            if v:
                results.append(bool(sum(v) * 2 > len(v)) or bool(t["hlock"]))
            else:
                results.append(True if t["hlock"] else None)

        self._move_tracks = new_tracks
        return results

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files,
                active_refs=None, ref_image_paths=None, auth_token=None):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Ornek)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files, auth_token=auth_token)
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti (Gorev 1), pozisyon kestirim (Gorev 2) ve referans nesne tespiti (Gorev 3)
        # modellerinin tumu self.detect() icinde sira ile calistirilir.
        frame_image_path = images_folder + prediction.image_url.split("/")[-1]
        frame_results = self.detect(prediction, health_status,
                                    active_refs=active_refs or [],
                                    ref_image_paths=ref_image_paths or {},
                                    frame_image_path=frame_image_path)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction, health_status, active_refs=None, ref_image_paths=None, frame_image_path=None):
        # ======================================================================
        # GOREV 1 — Nesne Tespiti (egitilmis YOLO modeli)
        # ======================================================================
        # Model bu kare uzerinde calistirilir; her kutu yarisma formatina
        # (sinif + inis durumu + hareket durumu + koordinat) cevrilir.
        drawn_boxes = []  # etiketli kareyi cizmek icin (model_cls, x1, y1, x2, y2, conf)
        vehicle_detections = []  # (DetectedObject, (x1,y1,x2,y2)) - bonus hareket tespiti icin
        pad_detections = []      # (DetectedObject, (x1,y1,x2,y2)) - UAP/UAI inis karari icin
        person_boxes = []        # [(x1,y1,x2,y2)] - pist uzeri insan kontrolu icin
        try:
            results = self.model(frame_image_path, conf=CONF_THRESHOLD, verbose=False)
            for box in results[0].boxes:
                model_cls = int(box.cls[0])
                mapping = self._model_class_map.get(model_cls)
                if mapping is None:
                    continue
                comp_cls, landing_status, moving_status = mapping

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # cls TUPLE olmali — DetectedObject.create_payload icinde cls[0] kullanilir.
                cls = (comp_cls,)
                detected_obj = DetectedObject(cls, landing_status, moving_status, x1, y1, x2, y2)
                prediction.add_detected_object(detected_obj)
                drawn_boxes.append((model_cls, x1, y1, x2, y2, conf))
                if comp_cls == classes["Tasit"]:
                    vehicle_detections.append((detected_obj, (x1, y1, x2, y2)))
                if model_cls == self._person_model_class:
                    person_boxes.append((x1, y1, x2, y2))
                if model_cls in self._pad_model_classes:
                    pad_detections.append((detected_obj, (x1, y1, x2, y2)))
        except Exception as e:
            logging.error(f"Gorev 1 tespiti basarisiz ({frame_image_path}): {e}")

        # --- UAP/UAI inis karari (iki asamali: YOLO -> MobileNetV3) ---
        # Her tespit edilen UAP/UAI pisti kirilip 'bos/dolu' siniflandiriciya
        # verilir; pist uzerinde insan/tasit olmasi veya pistin tamaminin kare
        # icinde olmamasi da (Sartname 2.1.2 / Sekil 9-10) inise engeldir.
        # Sekil 10 insan VE tasiti acikca sayar: ikisi de deterministik kontrol
        # edilir (siniflandiriciya birakilmaz), MobileNetV3 ise tespit modelinin
        # taniyamadigi diger cisimler icin son kontroldur. Nihai karar
        # DetectedObject.landing_status uzerine yazilir:
        #   Inilebilir -> "1", Inilemez -> "0" (Sartname Tablo 5).
        if pad_detections:
            try:
                frame_bgr = cv2.imread(frame_image_path, cv2.IMREAD_COLOR)
            except Exception as e:
                frame_bgr = None
                logging.warning(f"Pist karari icin kare okunamadi ({frame_image_path}): {e}")
            if frame_bgr is not None:
                obstacle_boxes = person_boxes + [box for _, box in vehicle_detections]
                for detected_obj, box in pad_detections:
                    try:
                        r = decide_pad(frame_bgr, box, obstacle_boxes, self.pad_clf,
                                       thresh=DOLU_THRESH)
                        detected_obj.landing_status = (
                            landing_statuses["Inilebilir"] if r["inilebilir"]
                            else landing_statuses["Inilemez"])
                    except Exception as e:
                        logging.warning(f"Pist inis karari basarisiz ({frame_image_path}, "
                                        f"box={box}): {e}")

        # --- Etiketlenmis kareyi kaydet ---
        # Modelin bulduğu kutular orijinal kare uzerine cizilip ./etiketlenmis_veriler/
        # klasorune ayni dosya adiyla kaydedilir (ham kare _images altinda degismeden kalir).
        if frame_image_path:
            try:
                os.makedirs(ANNOTATED_DIR, exist_ok=True)
                img = Image.open(frame_image_path).convert("RGB")
                draw = ImageDraw.Draw(img)
                for (model_cls, x1, y1, x2, y2, conf) in drawn_boxes:
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    label = f"{self.model.names.get(model_cls, model_cls)} {conf:.2f}"
                    draw.text((x1, max(0, y1 - 10)), label, fill=(255, 0, 0))
                img.save(os.path.join(ANNOTATED_DIR, os.path.basename(frame_image_path)))
            except Exception as e:
                logging.warning(f"Etiketli kare kaydedilemedi ({frame_image_path}): {e}")

        # ======================================================================
        # GOREV 2 — Konum Kestirimi (canli ORB-SLAM3 + online plane2d hizalama)
        # ======================================================================
        # Health Status biti hava aracinin uydu haberlesmesinin saglikli olup olmadigini gosterir.
        # Health Status 0 ise sistem calismali 1 ise gelen verinin aynisini gonderilebilir.
        # SLAM her karede (health_status ne olursa olsun) beslenir: ORB-SLAM3'un
        # haritasi/izleme sureci kesintisiz olmali. Kalibrasyon noktasi ekleme
        # (health_status==1) ve dunya-koordinatinda tahmin uretme (health_status
        # ==0) ise asagida ayriliyor.
        slam_pose = None
        curr_cam_xy = None
        if self.orbslam is None:
            # enable_slam=False: Gorev 2 tamamen atlanir (orn. yalniz Gorev 1/3
            # render/test script'leri). Asagidaki Gorev 1 bonus hareket tespiti
            # bu durumda metrik olcegi akis-kalibrasyonuna (SLAM'siz yedek yol)
            # dusurur - bkz. _estimate_vehicle_moving.
            pass
        else:
            if frame_image_path:
                try:
                    slam_pose = self.orbslam.track(frame_image_path)
                except OrbSlamStreamError as e:
                    logging.error(f"ORB-SLAM3 stream basarisiz ({frame_image_path}): {e}")

            # curr_cam_xy: kameranin bu karedeki metre cinsinden x/y dunya konumu
            # (saglikli karede GT, degilse hizalanmis SLAM tahmini). Asagidaki
            # Gorev 1 bonus hareket tespitinde metrik olcek (GSD) icin kullanilir.
            if health_status is None:
                logging.info("No translation/health_status for this frame; skipping Mission 2 output.")
            elif health_status == '0':
                world_xyz = None
                if slam_pose is not None and self.aligner.ready():
                    world_xyz = self.aligner.predict(slam_pose[:3], med_depth=slam_pose[7])
                if world_xyz is not None:
                    curr_cam_xy = (world_xyz[0], world_xyz[1])
                    pred_translation_x, pred_translation_y, pred_translation_z = world_xyz
                    prediction.add_translation_object(
                        DetectedTranslation(pred_translation_x, pred_translation_y, pred_translation_z))
                    self.translation_log.log_our_prediction(
                        prediction.image_url, pred_translation_x, pred_translation_y, pred_translation_z)
                else:
                    logging.info("Gorev 2: SLAM tahmini hazir degil (takip kaybi veya yetersiz kalibrasyon); "
                                  "bu kare icin cikti uretilmedi.")
            else:
                # Saglikli (health_status '1'): GT konum mevcut, oldugu gibi gonderilir. GT degerleri
                # null ise float'a cevrilemez (str(None) -> "None" sunucuda reddedilir); bu durumda atla.
                if None not in (prediction.gt_translation_x, prediction.gt_translation_y, prediction.gt_translation_z):
                    curr_cam_xy = (prediction.gt_translation_x, prediction.gt_translation_y)
                    prediction.add_translation_object(DetectedTranslation(
                        prediction.gt_translation_x, prediction.gt_translation_y, prediction.gt_translation_z))
                    self.translation_log.log_server_gt(
                        prediction.image_url, prediction.gt_translation_x,
                        prediction.gt_translation_y, prediction.gt_translation_z)
                    if slam_pose is not None:
                        self.aligner.add_calibration_point(
                            slam_pose[:3], slam_pose[3:7],
                            (prediction.gt_translation_x, prediction.gt_translation_y, prediction.gt_translation_z),
                            med_depth=slam_pose[7], n_tracked=slam_pose[8])
                else:
                    logging.info("Healthy frame but GT translation is null; skipping Mission 2 output.")

        # --- Gorev 1 bonus: Tasit hareket durumu (akis karsilastirma) ---
        # NAME_TO_SEMANTICS'teki varsayilan "Sabit" degeri, hareket karar
        # verilemeyen (ornegin ilk kare, kutuda yeterli keypoint yok)
        # durumlarda oldugu gibi kalir. Onceki-kare durumu (self._prev_gray)
        # metodun icinde HER karede guncellenir (tasit olsun olmasin) - aksi
        # halde bir sonraki tasitli karede "onceki kare" cok eskide kalir ve
        # kare-arasi kucuk-hareket varsayimi (LK optical flow) bozulur.
        if frame_image_path:
            try:
                curr_gray = cv2.imread(frame_image_path, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                curr_gray = None
                logging.warning(f"Gri kare okunamadi ({frame_image_path}): {e}")

            if curr_gray is not None:
                # Insan recall takviyesi: bu kareye kadar guncel _prev_gray
                # burada hala ONCEKI karenin gri hali (asagidaki
                # _estimate_vehicle_moving cagirilana kadar degismez) -
                # dolayisiyla bu blok ONDAN ONCE calismali.
                try:
                    all_person_boxes, coasted_person_boxes = self._track_persons(
                        curr_gray, person_boxes)
                    for (x1, y1, x2, y2) in coasted_person_boxes:
                        coasted_obj = DetectedObject(
                            (classes["Insan"],), landing_statuses["Inis Alani Degil"],
                            moving_statuses["Tasit Degil"], x1, y1, x2, y2)
                        # Gorsellestirme/analiz icin isaret (API payload'ini etkilemez -
                        # create_payload bu alani okumaz): bu kutu YOLO'dan degil,
                        # optical-flow izlemesinden geldi.
                        coasted_obj.is_tracked = True
                        prediction.add_detected_object(coasted_obj)
                    if coasted_person_boxes:
                        logging.info(f"Insan izleme: {len(coasted_person_boxes)} kutu "
                                    f"YOLO'nun kacirdigi, optical-flow ile tasindi.")
                except Exception as e:
                    logging.warning(f"Insan izleme basarisiz ({frame_image_path}): {e}")

                try:
                    # SLAM irtifasi (m): sogukta/hover'da GSD kaynagi.
                    # ensure_scale() kalibrasyon (health=1) doneminde de fit'i
                    # tetikler - scale, predict cagrilana kadar bos kalmasin
                    # (oturum basindaki sahte "Hareketli" vakasinin nedenlerinden).
                    alt_m = None
                    if slam_pose is not None and slam_pose[7] > 0:
                        s = self.aligner.ensure_scale()
                        if s:
                            alt_m = s * slam_pose[7]
                    moving_flags = self._estimate_vehicle_moving(
                        curr_gray, [box for _, box in vehicle_detections],
                        cam_xy=curr_cam_xy, alt_m=alt_m)
                    for (detected_obj, _), moving in zip(vehicle_detections, moving_flags):
                        if moving is not None:
                            detected_obj.moving_status = (
                                moving_statuses["Hareketli"] if moving else moving_statuses["Sabit"])
                except Exception as e:
                    logging.warning(f"Tasit hareket tespiti basarisiz ({frame_image_path}): {e}")

        # ======================================================================
        # GOREV 3 — Referans Nesne Esleme (tam otomatik, src/reference_matcher.py)
        # ======================================================================
        # `active_refs` ana donguden gelen pencere-ici aktif referanslardir (main.py
        # zaten [frame_start, frame_end] araligina gore filtreler). Referanslar oturum
        # icinde bir kez eklenir (indirilmis goruntuden), sonra her aktif referans icin
        # kare aranir. Kol secimi (ayrik nesne / desen) OTOMATIK yapilir; elle secim yok.
        active_refs = active_refs or []
        if self.reference_matcher is not None and active_refs and frame_image_path:
            # 1) Yeni referanslari bir kez ekle (indirilmis kirpim goruntusunden).
            for ref in active_refs:
                rid = ref['url']
                if rid in self._added_refs:
                    continue
                ref_path = (ref_image_paths or {}).get(rid)
                if not (ref_path and os.path.exists(ref_path)):
                    continue
                try:
                    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                    if ref_img is not None:
                        self.reference_matcher.add_reference(rid, ref_img)
                        self._added_refs.add(rid)
                except Exception as e:
                    logging.warning(f"Gorev 3 referans eklenemedi ({rid}): {e}")

            # 2) Eklenmis aktif referanslar icin kareyi ara.
            win_ids = [r['url'] for r in active_refs if r['url'] in self._added_refs]
            if win_ids:
                try:
                    frame_bgr_g3 = cv2.imread(frame_image_path, cv2.IMREAD_COLOR)
                    if frame_bgr_g3 is not None:
                        hits = self.reference_matcher.match(frame_bgr_g3, ref_ids=win_ids)
                        for rid, (x1, y1, x2, y2, _score, _branch) in hits.items():
                            prediction.add_reference_prediction(
                                ReferencePrediction(rid, prediction.frame_url, x1, y1, x2, y2))
                except Exception as e:
                    logging.warning(f"Gorev 3 esleme basarisiz ({frame_image_path}): {e}")

        return prediction

    def close(self):
        """Oturum sonunda (veya hata sonrasi) ORB-SLAM3 surecini ve
        karsilastirma logunu duzgunce kapatir."""
        if self.orbslam is not None:
            try:
                self.orbslam.close()
            except Exception as e:
                logging.warning(f"OrbSlamStream close failed: {e}")
        if self.translation_log is not None:
            try:
                self.translation_log.close()
            except Exception as e:
                logging.warning(f"TranslationComparisonLogger close failed: {e}")
