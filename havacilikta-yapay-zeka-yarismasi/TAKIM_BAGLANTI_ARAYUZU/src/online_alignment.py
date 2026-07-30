"""
Gorev 2 icin online (canli) hizalama: ORB-SLAM3'un olcek/rotasyon belirsiz
mono pozunu, health_status==1 anlarinda sunucudan gelen GT translation ile
"plane2d" Umeyama benzerlik donusumune surekli yeniden fit ederek dunya
koordinatina tasir.

gps_denied_localization/scripts/align_and_evaluate.py'deki plane2d modunun
(bu projede en iyi sonucu veren yontem - bkz. proje notlari) OFFLINE/batch
versiyonunun matematigi aynen buradan tasindi; farki: burada sabit bir
[align_start, align_end) penceresi yerine, health_status==1 oldugu HER
karede biriken butun (SLAM pozu, quaternion, GT) ucluleri kullanilarak
fit surekli guncellenir (kullanicinin tercihi: surekli online refit).
"""
import numpy as np


def _quat_to_optical_axis(q):
    """Kuaterniyon (x,y,z,w) -> kameranin optik ekseninin (kamera +z)
    dunya koordinatindaki yonu = R_wc'nin 3. sutunu."""
    x, y, z, w = q
    return np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


def _make_plane_projector(quats):
    """Biriken kalibrasyon karelerindeki ortalama optik ekseni 'asagi' (yer
    duzlemi normali) kabul edip, 3B SLAM konumlarini bu duzleme izdusuren
    3x2'lik taban dondurur. Nadir bakista optik eksen ~ yercekimi yonu."""
    n = np.mean([_quat_to_optical_axis(q) for q in quats], axis=0)
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2], axis=1)  # (3,2)


def _local_vertical_weights(h, window=5):
    """h: kalibrasyon karelerinin duzlem-normaline izdusumu, EKLENME SIRASINDA
    (yani kare sirasinda). Her nokta icin, +-window komsulugundaki toplam
    degisimi (yerel dikey 'hiz' buyuklugu) agirlik olarak dondurur - drone
    duz ucarken (h neredeyse sabit) kucuk, tirmanirken/alcalirken (paralaks
    guclu) buyuk agirlik verir. Tek-adimlik farklar yerine pencereli fark
    kullanilmasi, ham gurultunun agirliga direkt sizmasini onler."""
    n = len(h)
    w = np.empty(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        span = max(1, hi - lo)
        w[i] = abs(h[hi] - h[lo]) / span
    floor = 0.05 * (w.max() if w.max() > 0 else 1.0)
    return np.maximum(w, floor)


def _umeyama_2d(src, dst):
    """src (N,2) -> dst (N,2), yansima da denenir (izdusum sonrasi el
    yonu/chirality belirsiz). En dusuk fit hatasini veren (s, A, t) doner."""
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


class OnlineAligner:
    """SLAM(x,y,z,quat) -> dunya(x,y,z) surekli (online) plane2d hizalama.

    Kullanim:
      - health_status==1 ve GT mevcut her karede add_calibration_point() cagir.
      - health_status==0 her karede, ready() True ise predict() ile dunya
        koordinatinda tahmin al.
    """

    # depth_blend harmani: 0.75 * derinlik-kanali + 0.25 * h-regresyonu.
    # 2026 test oturumunda secildi (Z RMSE 6.31 -> 4.81m) ve 3 oturum x 2
    # pencere dogrulamasinin 5/6'sinda kazandi; tek kaybi SLAM'in zaten
    # coktugu (xy 21m) termal kosuldu. Ayrinti:
    # TAKIM_BAGLANTI_ARAYUZU/z_ekseni_deneyleri_aciklama.md
    W_DEPTH = 0.75

    def __init__(self, min_points=15, z_mode="depth_blend"):
        """z_mode:
          - "depth_blend" (varsayilan, canli kosumda kullanilan): Z iki
            kanalin harmanidir: z = 0.75*z_derinlik + 0.25*z_h.
              z_derinlik = (+-s)*med_depth + c_d : med_depth, o karede takip
                edilen harita noktalarinin kamera-cercevesi derinlik medyani
                (mono_tuyz_stream'den gelir). Egim FIT EDILMEZ: derinlik 3B
                bir mesafe oldugu icin metrik cevrimi X/Y Umeyama olcegi s
                ile aynidir (isaret kalibrasyon korelasyonundan). Bu sayede
                kalibrasyon penceresi duz ucussa bile (egim ogrenilemezken)
                kanal calisir; sadece kesisim c_d ogrenilir (n_tracked
                agirlikli ortalama - cok nokta takip edilen kare daha
                guvenilir). Gerekce: SLAM z-pozisyonu birikimli drift tasir
                ve BA bunu DUZELTEMEZ (olculdu: online 5.91 vs global-BA
                5.96m); medyan derinlik her karede haritadan taze olculur,
                tahmin bolgesinde GT z ile korelasyonu 0.97 (h: 0.91),
                oracle tavani ~3.1m (h: ~5.1m).
              z_h = a_z*h + c : asagidaki "regression" ile ayni.
            Derinlik verisi yoksa (eski binary / med_depth<=0) kendiliginden
            "regression"a duser.
          - "regression": h (SLAM'in
            duzlem normaline izdusumu) ile GT z arasina BAGIMSIZ bir dogrusal
            regresyon (egim+kesisim) oturtulur.
          - "reuse_xy_scale" (denendi, REDDEDILDI): egim olarak X/Y hizala-
            masindan gelen AYNI olcek (s) kullanilir, sadece kesisim (c) ayri
            fit edilir. Gerekce boyleydi: SLAM haritasi gercek bir 3B benzer-
            likse (tek global olcek), z de ayni s ile olceklenmeli. THYZ_2026
            _Online_Yarisma_Test_Oturumu canli koşumunun offline yeniden-
            oynatilmasinda (440 kalibrasyon + 1800 tahmin karesi) test edildi:
            Z RMSE 7.07m (regression) -> 12.54m (reuse_xy_scale) - X/Y AYNI
            kaldi (0.996/1.313m, cunku bu degisiklik sadece z'yi etkiler).
            Yani SLAM haritasi izotropik degil: optik eksen (derinlik) boyun-
            daki efektif olcek, yanal (x/y) duzlemdekinden GERCEKTEN farkli -
            monokuler derinlik kestiriminin daha zayif/gurultulu olmasi bekle-
            nen bir sonuc. Bu yuzden z icin BAGIMSIZ regresyon (varsayilan)
            daha iyi; bu mod sadece referans/karsilastirma icin tutuluyor.
          - "weighted_regression" (deneysel): "regression" ile ayni bagimsiz
            egim+kesisim fit'i, ama kalibrasyon noktalari esit agirlikli
            degil - yerel dikey hareketi (paralaks) guclu olan noktalar
            (drone tirmanirken/alcalirken) fit'te daha cok, duz ucus anlari
            daha az agirlik tasir (bkz. _local_vertical_weights). Gerekce:
            derinlik yonundeki paralaks ne kadar buyukse, o bolgedeki h<->z
            iliskisi o kadar guvenilir - esit agirlik, bilgi tasimayan duz
            ucus noktalarini bilgi tasiyan noktalarla ayni degerde sayiyordu.
        """
        self.min_points = min_points
        self.z_mode = z_mode
        self._poses = []   # SLAM (x,y,z) - kalibrasyon karelerinde
        self._quats = []   # SLAM (qx,qy,qz,qw) - ayni kareler
        self._gts = []      # sunucu GT (x,y,z) - ayni kareler
        self._depths = []   # med_depth (yoksa/gecersizse nan) - ayni kareler
        self._ntracks = []  # n_tracked (yoksa 0) - ayni kareler
        self._fit = None    # (P, n, A, t2, a_z, c, a_d, c_d) - lazy, add ile invalide olur
        self.last_fit_rmse_xy = None
        self.scale = None   # X/Y Umeyama olcegi s (SLAM birimi -> metre); _refit doldurur.
                            # Tuketici: Gorev 1 hareket tespiti, GSD = (s*med_depth)/fx.

    def add_calibration_point(self, slam_xyz, slam_quat, gt_xyz,
                              med_depth=None, n_tracked=0):
        self._poses.append(np.asarray(slam_xyz, dtype=float))
        self._quats.append(np.asarray(slam_quat, dtype=float))
        self._gts.append(np.asarray(gt_xyz, dtype=float))
        self._depths.append(float(med_depth) if med_depth is not None and med_depth > 0
                            else np.nan)
        self._ntracks.append(int(n_tracked))
        self._fit = None

    def ready(self):
        return len(self._poses) >= self.min_points

    def ensure_scale(self):
        """Olcek s'i dondurur; gerekiyorsa fit'i hesaplar. predict()'ten farki:
        health=1 (kalibrasyon) doneminde de cagrilabilir - Gorev 1 hareket
        tespitinin GSD'si oturumun BASINDAN itibaren SLAM olcegini kullanabilsin
        diye (scale, predict cagrilana kadar bos kalmasin)."""
        if self.ready() and self._fit is None:
            self._refit()
        return self.scale

    def _refit(self):
        src = np.array(self._poses)
        dst = np.array(self._gts)
        P = _make_plane_projector(self._quats)
        n = np.cross(P[:, 0], P[:, 1])
        s, A, t2 = _umeyama_2d(src @ P, dst[:, :2])
        self.scale = float(abs(s))
        h = src @ n
        has_variation = len(set(np.round(h, 9))) >= 2
        if self.z_mode == "reuse_xy_scale":
            a_z = s
            c = float((dst[:, 2] - a_z * h).mean())
        elif self.z_mode == "weighted_regression" and has_variation:
            weights = _local_vertical_weights(h)
            a_z, c = np.polyfit(h, dst[:, 2], 1, w=weights)
        elif has_variation:
            a_z, c = np.polyfit(h, dst[:, 2], 1)
        else:
            a_z, c = 0.0, float(dst[:, 2].mean())

        # Derinlik kanali (depth_blend): egim = isaret * s (fit yok), kesisim
        # n_tracked agirlikli ortalama. Isaret, kalibrasyondaki korelasyondan;
        # derinlik varyasyonu yoksa korelasyon tanimsiz kalir -> kanal devre
        # disi (a_d None), predict kendiliginden saf h-regresyonuna duser.
        a_d, c_d = None, None
        if self.z_mode == "depth_blend":
            d = np.array(self._depths)
            w = np.array(self._ntracks, dtype=float)
            ok = np.isfinite(d)
            if ok.sum() >= self.min_points and np.std(d[ok]) > 0 and np.std(dst[ok, 2]) > 0:
                corr = np.corrcoef(d[ok], dst[ok, 2])[0, 1]
                if np.isfinite(corr) and corr != 0:
                    a_d = float(np.sign(corr)) * s
                    wk = np.maximum(w[ok], 1.0)
                    c_d = float(((dst[ok, 2] - a_d * d[ok]) * wk).sum() / wk.sum())
                    a_d = float(a_d)

        fit_xy = (src @ P) @ A.T + t2
        self.last_fit_rmse_xy = float(np.sqrt(np.mean(np.sum((fit_xy - dst[:, :2]) ** 2, axis=1))))
        self._fit = (P, n, A, t2, a_z, c, a_d, c_d)

    def predict(self, slam_xyz, med_depth=None):
        """SLAM (x,y,z) [+ istege bagli med_depth] -> tahmini dunya (x,y,z),
        veya None (henuz hazir degil). med_depth verilmez/gecersizse veya
        derinlik kanali kurulamadiysa Z saf h-regresyonundan gelir."""
        if not self.ready():
            return None
        if self._fit is None:
            self._refit()
        P, n, A, t2, a_z, c, a_d, c_d = self._fit
        xyz = np.asarray(slam_xyz, dtype=float)
        p2 = (xyz @ P) @ A.T + t2
        z = a_z * float(n @ xyz) + c
        if (self.z_mode == "depth_blend" and a_d is not None
                and med_depth is not None and med_depth > 0):
            z_depth = a_d * float(med_depth) + c_d
            z = self.W_DEPTH * z_depth + (1.0 - self.W_DEPTH) * z
        return (float(p2[0]), float(p2[1]), z)
