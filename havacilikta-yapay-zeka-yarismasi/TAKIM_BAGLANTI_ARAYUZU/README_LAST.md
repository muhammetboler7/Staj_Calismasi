# TAKIM_BAGLANTI_ARAYUZU — Sistem Rehberi

TEKNOFEST "Havacılıkta Yapay Zeka" yarışmasının 3 görevini (nesne tespiti,
GPS'siz konum kestirimi, referans nesne eşleme) tek pipeline'da çözen
**üretim** (ana yarışma) kodunun rehberi. Sadece `main.py`'nin gerçekten
çalıştırdığı dosyalar/mantık anlatılmıştır.

---

## 1) Kurulum ve Çalıştırma

```bash
conda activate teknofest_yarisma
cd TAKIM_BAGLANTI_ARAYUZU
pip install -r requirements.txt
```

`config/.env`:
```
TEAM_NAME=<takim_adi>
PASSWORD=<sifre>
EVALUATION_SERVER_URL="http://<sunucu_ip>:<port>/"
SESSION_NAME=<oturum_adi>
```

Çalıştırma:
```bash
python main.py
```

`main.py` sırasıyla: sunucudan ilerlemeyi (`get_progress`) okuyup kaldığı
yerden devam eder → referans nesneleri ve görsellerini baştan indirir →
`tqdm` ilerleme çubuğuyla tek tek kare ister (`get_current_frame` /
`get_current_translation`) → her kare için `ObjectDetectionModel.process()`'i
çağırır → sonucu `send_prediction()` ile sunucuya yollar. `Ctrl+C` veya hata
durumunda `finally` bloğu `detection_model.close()`'u çağırıp ORB-SLAM3
sürecini ve log dosyasını düzgün kapatır.

Ağırlıklar (`MODEL_PATH`, `CLF_W`, `YOLOE_PF_W`, `DINO_MODEL`) ilgili
modüllerin başındaki sabit yollardan/isimlerden otomatik yüklenir; DINOv2/
YOLOE ilk çalıştırmada `~/.cache`'e (ya da taşınmışsa proje-özel cache'e)
indirilir. ORB-SLAM3 ikili dosyası ve ayar dosyası `gps_denied_localization/`
altındadır (bkz. §5).

---

## 2) Bir Karede Sırayla Ne Çalışır? (`ObjectDetectionModel.detect()`)

1. **Görüntü indirilir/okunur** (`process()` içinde `download_image`).
2. **GÖREV 1 — Nesne tespiti (YOLO)**: eğitilmiş 4 sınıflı YOLO modeli
   (`self.model(frame_image_path)`) çalışır; her kutu `Taşıt/İnsan/UAP/UAİ`
   sınıfına ve varsayılan iniş/hareket durumuna çevrilir.
3. **UAP/UAİ iniş kararı** (`src/pad_pipeline.py`, bkz. §7): pist tamamen
   kare içinde mi, üzerinde insan/taşıt var mı, yoksa MobileNetV3 "boş/dolu"
   kararı.
4. **Etiketli kare kaydı**: YOLO kutuları çizilip `etiketlenmis_veriler/`
   klasörüne kaydedilir.
5. **GÖREV 2 — Konum kestirimi** (`orbslam_stream.py` + `online_alignment.py`,
   bkz. §5): kare her zaman ORB-SLAM3'e beslenir; `health_status=='1'` ise GT
   gönderilip kalibrasyon noktası eklenir, `'0'` ise `OnlineAligner.predict()`
   ile dünya koordinatına taşınmış tahmin gönderilir. Sonuç
   `TranslationComparisonLogger` ile loglanır.
6. **Görev 1 bonus — İnsan izleme** (bkz. §4.4): YOLO'nun bu karede kaçırdığı
   ama yakın zamanda tespit edilmiş insanlar optical-flow ile taşınıp yine
   raporlanır (recall takviyesi).
7. **Görev 1 bonus — taşıt hareket durumu** (bkz. §4.3): homografi tabanlı
   pencereli hareket kararı.
8. **GÖREV 3 — Referans nesne eşleme** (`reference_matcher.py`, bkz. §6):
   aktif referanslar bir kez eklenir, sonra her karede aranır.
9. Tamamlanan `prediction`, `main.py`'de `send_prediction()` ile gönderilir.

---

## 3) `src/` Modülleri

| Dosya | Rol |
|---|---|
| `object_detection_model.py` | Orkestratör: yukarıdaki adımların hepsini `detect()` içinde sırayla çağırır. |
| `constants.py` | Yarışma sınıf/durum kodları (`classes`, `landing_statuses`, `moving_statuses`) — değiştirilmez. |
| `connection_handler.py` | Sunucu HTTP istemcisi: login, ilerleme, kare/translation çekme, tahmin gönderme, referans indirme. |
| `pad_pipeline.py` | UAP/UAİ iniş kararı (`PadLandingClassifier` + `decide_pad`). |
| `orbslam_stream.py` | `OrbSlamStream`: ORB-SLAM3 alt-sürecini yönetir (bkz. §5). |
| `online_alignment.py` | `OnlineAligner`: SLAM pozunu GT dünya koordinatına taşıyan online hizalama (bkz. §5). |
| `translation_logger.py` | Her karedeki GT/tahmin konumunu `_logs/translation_karsilastirma_*.csv`'ye yazar. |
| `reference_matcher.py` | Görev 3'ün tamamı (bkz. §6). |
| `detected_object.py`, `detected_translation.py`, `translation.py`, `reference_prediction.py`, `frame_predictions.py` | Sunucuya gönderilecek JSON payload'larını üreten veri sınıfları. |

---

## 4) GÖREV 1 — Detay

### 4.1 Tespit
Eğitilmiş 4 sınıflı (`Taşıt/İnsan/UAP/UAİ`) YOLO ağırlığı `ultralytics.YOLO`
ile yüklenir ve her karede `self.model(frame_image_path, conf=0.25)` çağrılır.
Model checkpoint'inin kendi sınıf isimleri (`self.model.names`), koddaki
`NAME_TO_SEMANTICS` eşlemesiyle yarışma sınıflarına çevrilir — bu sayede hem
4 hem 6 sınıflı eski ağırlıklar aynı kodla çalışabilir.

### 4.2 UAP/UAİ iniş kararı (`pad_pipeline.py`)
İki aşamalı, deterministik-önce yaklaşım (Şartname Tablo 5/Şekil 9-10):
1. Pist kutusu kare sınırına yapışıyorsa (tamamı görünmüyorsa) → **İnilemez**.
2. Pist üzerinde bir İnsan/Taşıt kutusunun merkezi varsa → **İnilemez**
   (bu kontrol deterministiktir, sınıflandırıcıya bırakılmaz).
3. Aksi halde pist kırpılıp **MobileNetV3-Small** "boş/dolu" sınıflandırıcısına
   verilir; dolu olasılığı `DOLU_THRESH=0.35`'i geçerse **İnilemez**, geçmezse
   **İnilebilir**.

### 4.3 Taşıt hareket durumu (Hareketli/Sabit)
Tek kare çifti optical-flow karşılaştırması yetersiz kaldığı için (düşük
FPS'te gürültü/SNR sorunu), karar zamana yayılmış pencereli bir yöntemle
verilir:
1. Taşıt kutuları DIŞINDAKI zemin noktaları LK optical flow ile izlenir,
   RANSAC ile aralarına bir **homografi** oturtulur — bu, drone'un kendi
   hareketinin (dönme dahil) görüntüdeki tam modelidir.
2. Her taşıt izinin önceki kutusu bu homografiyle şimdiki kareye taşınır
   ("statik olsaydı burada olurdu" kutusu); gerçek kutuyla arasındaki **adım**,
   kare kenarına yapışık OLMAYAN kenardan ölçülür (giren araç alt kenardan,
   çıkan araç üst kenardan — kutu büyüme/küçülmesinden etkilenmeyen taraf).
3. Adımlar birkaç karelik pencerede birikir; hız = adımların **medyanı**
   (tek bozuk adım medyanı kaydırmaz) × GSD (m/piksel) × FPS. GSD öncelikle
   SLAM irtifasından (`aligner.scale * med_depth`), yoksa akış-kalibrasyonu
   yedeğinden gelir.
4. Histerezisli oylama + "yapışkan Hareketli kilidi" (ardışık N kez Hareketli
   ölçülünce kilitlenir, ardışık M kez net Sabit ölçülünce açılır) son kararı
   gürültüye karşı kararlı kılar.

### 4.4 İnsan izleme — recall takviyesi (`_track_persons`)
YOLO'nun İnsan sınıfında tek karede kişiyi kaçırması (kısmi kapanma, hareket
bulanıklığı, küçük boy) recall'i düşürüyordu. `_track_persons`, taşıt hareket
tespitinden BAĞIMSIZ, daha basit bir mekanizmayla bu kayıpları telafi eder:

1. **Eşleştirme**: bu karenin YOLO tespitleri, önceki karenin izleriyle IoU'ya
   göre açgözlü eşleştirilir (`PERSON_TRACK_MATCH_IOU=0.10`, kasıtlı gevşek).
2. **Eşleşen izler**: kutu doğrudan YOLO'nun yeni tespitiyle güncellenir —
   optical-flow bu durumda HİÇ çalışmaz (tespit varken ona güvenilir, üstüne
   ekstra bir "izleme" yapılmaz).
3. **Eşleşmeyen izler (YOLO kaçırdı)**: kutunun İÇİNDEKİ noktalar önceki
   kareden bu kareye LK optical flow ile izlenir (ileri-geri tutarlılık
   kontrolüyle); başarılı noktaların **medyan yer değiştirmesi** kutunun
   tamamına uygulanır (boyut sabit kalır, sadece kayar). Bu "izlenen" durum
   en fazla `PERSON_TRACK_MAX_MISS=2` kare üst üste sürebilir; sonra iz
   tamamen düşürülür. `PERSON_TRACK_MIN_HITS=1`: iz, daha önce en az 1 gerçek
   YOLO tespitiyle açılmış olması yeterli (yani tek görüşten sonra bile 2
   kare izlenebilir).
4. **Hayalet-kutu bastırma**: optical-flow'un tahmin ettiği kayma yanlış
   çıkıp izlenen kutu, o karede YOLO'nun bulduğu GERÇEK bir kutuyla zaten
   `PERSON_TRACK_SUPPRESS_IOU=0.3` üzerinde çakışıyorsa (ör. kişi yeniden
   tespit edildi ama drift yüzünden eski izle eşleşmedi, aynı kişi için hem
   eski hem yeni kutu görünürdü), izlenen kutu atılır.
5. Sonuçta üretilen "izlenen" (YOLO'nun değil, optical-flow'un bulduğu)
   kutular normal bir `DetectedObject` olarak `prediction`'a eklenir; iç
   kullanım için `is_tracked=True` ile işaretlenir (API payload'ını etkilemez).

**Doğrulanmış etki** (gerçek `Annotations/` GT'sine karşı, tüm 2250 kare):
İnsan Recall@IoU0.5 %33.82 → %40.60, Precision %85.90 → %69.78 (`max_miss=1`
ile ilk ayar). İlk 600 karede (hayalet-kutu bastırma + `min_hits=1` sonrası
son ayar): Recall %49.82 → %56.89, Precision %90.97 → %83.85, F1 %64.38 →
%67.79. Ayarlar, gerçek GT'ye karşı yapılan bir parametre taramasıyla
seçildi — `max_miss` arttıkça recall biraz daha artsa da drift kaynaklı
hayalet kutular precision'ı hızla düşürüyor.

---

## 5) GÖREV 2 — Detay (ORB-SLAM3 + Online Hizalama)

### 5.1 ORB-SLAM3 C++ tarafı (`mono_tuyz_stream.cc`)
Yarışma kareleri sunucudan tek tek geldiği için, ORB-SLAM3'ün normal
kullanım şekli (sabit bir görüntü listesini baştan sona işleyip sonda tek
seferde yazan `mono_tuyz` batch programı) işe yaramaz — süreç ve haritası
**oturum boyunca ayakta kalmalı**, her yeni kare geldiğinde tek bir
`TrackMonocular()` çağrısıyla beslenip pozun hemen geri okunabilmesi
gerekiyor. Bunun için yazılan `mono_tuyz_stream.cc`, ORB-SLAM3'ü headless
(viewer'sız) başlatıp stdin üzerinden satır-satır bir protokolle çalıştırır:

- **stdin (komut)**: `"<frame_id> <mutlak_goruntu_yolu>"` ya da `"QUIT"`.
- **proto-fd (ayrı bir pipe, sonuç)**: `"<frame_id> OK <tx> <ty> <tz> <qx> <qy> <qz> <qw> <med_depth> <n_tracked>"`,
  ya da izleme kaybedilirse `"<frame_id> FAIL"`, görüntü okunamazsa
  `"<frame_id> ERR <mesaj>"`. Süreç hazır olduğunda ilk satır `"READY"`.
- Sonuç neden ayrı bir pipe'da (stdout'ta değil)? ORB-SLAM3'ün kendi iç kodu
  ("New Map created with N points" gibi) durum mesajlarını rastgele anlarda
  doğrudan `std::cout`'a basıyor; bu, protokol satırlarıyla aynı kanalda
  karışırdı. Bu yüzden `--proto-fd=<N>` argümanıyla verilen, ebeveyn sürecin
  miras bıraktığı ayrı bir dosya tanımlayıcısına yazılır; stdout ORB-SLAM3'ün
  kendi gürültülü loglarına bırakılır.
- Her karede `TrackMonocular(im, timestamp)` çağrılır (`timestamp` gerçek
  zaman değil, tekdüze artan `frameCounter/30.0` — IMU kullanılmadığı için
  sadece sıralamayı garanti etmesi yeterli). Takip durumu `2` (OK) ise
  kamera pozu (`Tcw.inverse()` → dünya←kamera) `t` (translation) ve `q`
  (quaternion) olarak yazılır.
- **`med_depth`**: o karede izlenen harita noktalarının (`GetTrackedMapPoints`)
  kamera çerçevesindeki derinliğinin (Z bileşeni) **medyanı** (`n_tracked`:
  medyana giren nokta sayısı). Bu, Görev 2 Z ekseni için eklendi: SLAM'in
  kendi Z pozisyonu birikimli drift taşıyor ve bundle adjustment bunu
  düzeltemiyor (ölçüldü: online 5.91m vs global-BA 5.96m Z RMSE); medyan
  derinlik ise her karede haritadan taze ölçülüyor ve Z için çok daha güçlü
  bir sinyal.

### 5.2 Python tarafı (`orbslam_stream.py`)
`OrbSlamStream`, `mono_tuyz_stream` sürecini `subprocess.Popen` ile başlatır:
`os.pipe()` ile bir okuma/yazma ucu çifti açılır, yazma ucu `pass_fds` ile
çocuk sürece miras bırakılır ve `--proto-fd=<fd>` olarak iletilir; stdin
normal `PIPE`, stdout/stderr ise (istenirse) bir log dosyasına yönlendirilir
— ORB-SLAM3'ün gürültüsü hiçbir zaman ayrıştırılmaya çalışılmaz. Başlatmada
`select()` ile `"READY"` satırı beklenir (kelime dağarcığı yükleme — text
format `ORBvoc.txt` — yavaş olabileceği için 180 saniye zaman aşımı vardır).
`track(image_path)`: artan bir `frame_id` üretip stdin'e yazar, proto-fd'den
bir satır okur, `frame_id` eşleşmesini doğrular (protokol senkron kaymasını
yakalamak için) ve `OK` ise `(x,y,z,qx,qy,qz,qw,med_depth,n_tracked)`
tuple'ını, `FAIL/ERR` ise `None` döner. `close()`, `"QUIT"` yazıp süreci
düzgün kapatır (yanıt vermezse `kill()`).

### 5.3 Online hizalama matematiği (`online_alignment.py`)
ORB-SLAM3'ün mono pozu ölçek/rotasyon belirsizdir; `OnlineAligner` bunu
`health_status=='1'` anlarında biriken `(SLAM pozu, quaternion, GT)` üçlüleriyle
sürekli yeniden fit ederek dünya koordinatına taşır (sabit bir kalibrasyon
penceresi yerine, kalibrasyon noktası her geldiğinde `_fit` geçersiz kılınıp
bir sonraki `predict()` çağrısında yeniden hesaplanır):

- **X/Y (yatay düzlem)**: Kalibrasyon karelerindeki kamera optik eksenlerinin
  ortalaması "aşağı" (yer düzlemi normali) kabul edilip, 3B SLAM konumları bu
  düzleme izdüşürülür (`_make_plane_projector`). İzdüşen 2B noktalarla GT
  X/Y arasına, yansımayı da deneyen bir **2B Umeyama benzerlik dönüşümü**
  (ölçek `s` + rotasyon `R` + öteleme `t`, `_umeyama_2d`) oturtulur.
- **Z (yükseklik)**: `z_mode="depth_blend"` (varsayılan, üretimde kullanılan):
  iki bağımsız kanalın ağırlıklı harmanı, `z = 0.75·z_derinlik + 0.25·z_h`:
  - `z_derinlik = ±s·med_depth + c_d`: eğim **fit edilmez**, doğrudan X/Y
    hizalamasından gelen aynı ölçek `s` kullanılır (derinlik de 3B bir
    mesafe olduğu için metrik çevrimi aynıdır); işareti kalibrasyondaki
    korelasyondan gelir. Sadece kesişim `c_d` öğrenilir (`n_tracked`
    ağırlıklı ortalama — çok nokta takip edilen kare daha güvenilir). Bu
    sayede kalibrasyon penceresi düz uçuşsa bile (eğim öğrenilemese de)
    kanal çalışır.
  - `z_h = a_z·h + c`: SLAM konumunun düzlem-normaline izdüşümü (`h`) ile GT
    Z arasına oturtulan bağımsız doğrusal regresyon.
  - Derinlik verisi yoksa/geçersizse sistem kendiliğinden saf `z_h`
    regresyonuna düşer.
- `ensure_scale()`: `predict()` çağrılmasa bile (yani daha kalibrasyon
  aşamasındayken) ölçek `s`'in hazır olmasını sağlar — Görev 1'deki hareket
  tespitinin GSD hesabı oturumun başından itibaren bu ölçeği kullanabilsin
  diye.

---

## 6) GÖREV 3 — Detay (`reference_matcher.py`)

Tek model (**DINOv2-L**) iki farklı çalışma şeklinde kullanılır; kol seçimi
tamamen otomatiktir, referans başına elle ayar yoktur.

### 6.1 Referans hazırlığı (oturum başı, bir kez — `add_reference`)
Referans görseli iki ayrı temsile çevrilip önbelleğe alınır:
- **Discrete temsil**: görsel çoklu döndürmelerle (TTA) 336×336'ya
  kırpılıp DINOv2-L'den geçirilir → her rotasyon için **tek bir global
  embedding vektörü**.
- **Pattern temsil**: görsel 224×224'e (16×16 yama) getirilir, DINOv2-L'in
  ham 14×14'lük yama-token gridine (`forward_features`) bakılır ve bu
  yamaların ortalaması alınarak yine **tek bir vektöre** indirgenir (yine
  TTA'lı).

### 6.2 Kare başı hazırlık (`_prep_frame`, referanstan bağımsız, kare başı bir kez)
- **YOLOE** (prompt-free, "her nesneyi bul" modu) karede aday kutular üretir;
  confidence'a göre en olası **30** aday alınır (`CONF_TOPK`), her biri
  kırpılıp DINOv2-L ile embed edilir (discrete kol adayları).
- Kare ayrıca çok ölçekte (`PATTERN_SCALES`) DINOv2-L yama-gridine çevrilir;
  komşu yama grupları (`PATTERN_KS`: 2,3,4,6,8,12,16,24 boyutlarında,
  kayan pencere) ortalanıp her biri **tek bir pencere-vektörüne** indirgenir
  — kare üzerinde çok sayıda örtüşen aday pencere (pattern kolu adayları).

### 6.3 Kol seçimi ve eşleme (`match`, her referans için)
1. Karenin tüm pattern pencere-vektörleri, o referansın pattern
   vektörüyle kosinüs benzerliğine sokulur; en yüksek skoru veren pencere
   ve skoru (`pattern_score`) bulunur.
2. `pattern_score >= 0.80` ise (bölge referansa dokusal olarak çok
   benziyor demektir) doğrudan bu pencere sonuç olarak kullanılır —
   **discrete kol hiç çalıştırılmaz**. Eşik, gerçek 437 çiftlik yarışma
   verisiyle yapılan taramada belirlendi (0.68→%50.11, 0.80→%65.68,
   0.80-0.98 arası kararlı plato).
3. Aksi halde **discrete kol** işletilir: 30 YOLOE adayı referansla kosinüs
   benzerliğine göre sıralanır, ama önce **canlı medyan boyut filtresi**
   uygulanır — bu referans için o oturumda o ana kadar seçilmiş kutuların
   medyan boyu hesaplanıp, bunun `[0.35x, 2.8x]` aralığı dışındaki adaylar
   elenir (DINOv2'nin yanlışlıkla seçtiği saçma boyuttaki kutuları önler).
4. Seçilen adayın benzerliği düşükse (`<0.40`; doğru nesne muhtemelen ilk 30
   adaya girmedi), YOLOE'ye **120 adaya** genişletilmiş bir aday havuzuyla
   ikinci bir tur verilir.
5. Son olarak seçilen kaba kutunun etrafı, varsa **RoMa** (`vismatch`
   paketi) ile referansla piksel-piksel yoğun eşleştirilir; yeterli
   inlier sayısı ve makul merkez-kayması/alan-oranı testini geçerse kutu
   hassaslaştırılır, geçmezse kaba kutu korunur.

### 6.4 Doğrulanmış performans
Gerçek yarışma verisiyle yapılan 437 çift karşılaştırmada: sadece kaba
YOLOE+DINOv2 seçimi %41.88 isabet (IoU≥0.5); medyan filtresi + havuz
genişletme + RoMa inceltmesi eklenince %65.68'e çıktı. Tam canlı/akan
pipeline'da (önceden hiçbir şey görmeden, kare kare) bu oran %64.76 olarak
doğrulandı.

---

## 7) Özet

- **Görev 1**: YOLO tespiti → MobileNetV3 ile UAP/UAİ boş/dolu kararı →
  homografi + pencereli-adım tabanlı taşıt hareket kararı → optical-flow
  tabanlı İnsan izleme ile recall takviyesi (hayalet-kutu bastırmalı).
- **Görev 2**: Kesintisiz ORB-SLAM3 izleme (ayrı süreç, satır-protokolü) +
  `health_status==1` anlarında sürekli yeniden fit edilen 2B Umeyama (X/Y) +
  derinlik-harmanlı regresyon (Z) ile dünya koordinatına taşıma.
- **Görev 3**: DINOv2-L tabanlı iki kollu (ayrık nesne / dokulu-desen)
  otomatik yönlendirmeli eşleme; medyan-boyut filtresi, uyarlamalı aday
  havuzu ve RoMa inceltmesiyle güçlendirilmiş.
