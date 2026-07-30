# gps_denied_localization — Baştan Sona Ne Yaptık, Neden Böyle Yaptık

Bu belge `gps_denied_localization` klasöründe yapılan her şeyi; ORB-SLAM3'ün ne
olduğundan, kamera kalibrasyonuna, arkadaki matematiğe, aldığımız sonuçlara ve
neyin iyileştirilebileceğine kadar basit dille anlatır. `visualodometry.py`
için yazılan açıklama belgesiyle (`havacilikta-yapay-zeka-yarismasi/
TAKIM_BAGLANTI_ARAYUZU/visualodometry_aciklama.md`) aynı problemi çözüyoruz
(GPS koptuğunda konum tahmini) ama burada **hazır, çok daha güçlü bir kütüphane**
olan ORB-SLAM3'ü kullandık — kendi VO kodumuzu yazmak yerine.

---

## 1. Amaç Ne?

70-90 derece açıyla (neredeyse tam dikey, **nadir**) yere bakan bir drone
kamerası düşünelim. İlk 200 karede GPS sağlıklı ve gerçek konum (x, y —
metre cinsinden) biliniyor. Sonraki 200-400 karede GPS **kopuyor** ve elimizde
sadece kamera görüntüsü kalıyor. Amaç: bu görüntülerden, drone'un GPS'siz
geçirdiği süre boyunca nereye gittiğini tahmin etmek — yani GPS'in yerini
geçici olarak görüntü ile doldurmak.

IMU (ivme/dönüş sensörü) verimiz yok — sadece ham video kareleri var.

---

## 2. ORB-SLAM3 Nedir? (Basit Anlatım)

**SLAM** = *Simultaneous Localization and Mapping* — "Eş Zamanlı Konum
Belirleme ve Haritalama". Yani sistem hem "ben neredeyim?" (localization) hem
de "etrafımda ne var?" (mapping, bir 3B nokta bulutu) sorularını **aynı anda**
cevaplıyor. `visualodometry.py`'deki gibi sadece "bir önceki kareye göre ne
kadar hareket ettim?" değil, sistem geçmişte gördüğü tüm sahneyi bir **harita**
olarak biriktiriyor ve yeni kareleri bu haritaya göre konumluyor.

ORB-SLAM3, bunun açık kaynaklı, akademik camiada çok kullanılan, olgun bir
uygulamasıdır (Zaragoza Üniversitesi, 2020-2021). "ORB" kısmı, kullandığı
özellik noktası türünden geliyor (**O**riented **F**ast **R**otated **B**RIEF —
hem konum hem de o noktanın çevresini tanımlayan bir "parmak izi" içeren, çok
hızlı hesaplanan bir özellik noktası türü).

### Üç paralel iş parçacığı (thread)

ORB-SLAM3 arka planda üç şeyi aynı anda yapar:

1. **Tracking (Takip):** Her yeni karede ORB noktalarını bulur, bunları
   halihazırda bilinen 3B harita noktalarıyla eşleştirir, eşleşmelerden
   kameranın o anki pozunu (konum + yönelim) hesaplar. `visualodometry.py`'deki
   "iki kare arasındaki hareket" fikrine benzer ama burada karşılaştırma bir
   önceki kareyle değil, **birikmiş 3B harita** ile yapılıyor — bu çok daha
   kararlı, çünkü tek bir karenin gürültüsüne bağımlı değil.
2. **Local Mapping (Yerel Haritalama):** Belirli aralıklarla seçilen
   "anahtar kareler" (keyframe) haritaya eklenir, yeni 3B noktalar
   üçgenlenir (triangulation — iki farklı açıdan görülen aynı nokta, basit
   üçgen geometrisiyle 3B'de nerede olduğu hesaplanabilir), ve yakın
   çevredeki kareler/noktalar birlikte ince ayar yapılır (**bundle
   adjustment** — "hepsini birlikte en küçük hataya oturt" anlamında bir
   optimizasyon).
3. **Loop Closing (Döngü Kapama):** Sistem daha önce gördüğü bir yeri tekrar
   görürse (drone eski rotasına dönerse), bunu **DBoW2** adlı bir "görsel
   kelime dağarcığı" (bag-of-words — görüntüyü kaba bir "hangi görsel
   kelimeler var" özetine indirgeyip hızlı karşılaştırma yapan yöntem) ile
   tanır ve o ana kadar birikmiş tüm hatayı geriye doğru dağıtarak düzeltir.
   Bu, uzun uçuşlarda driftin (birikmiş hatanın) sıfırlanmasını sağlayan en
   güçlü mekanizmadır.

**Bizim senaryomuzda loop closing hiç tetiklenmedi** çünkü test videosu
düz/tek yönlü bir uçuş — drone eski rotasının üzerinden tekrar geçmiyor.
Bu önemli, aşağıda (Bölüm 8) tekrar değineceğiz.

### Neden `visualodometry.py`'deki basit yöntem değil de ORB-SLAM3?

`visualodometry.py` bilinçli olarak **çok basit ve hafif** tutulmuş (sadece
ardışık kare karşılaştırması, 2B benzerlik dönüşümü) — çünkü nadir kamera
varsayımına özel, hızlı, bağımlılığı az bir çözüm. ORB-SLAM3 ise **genel
amaçlı, tam 3B** bir SLAM sistemi: harita tutar, loop closing yapar, bundle
adjustment ile sürekli kendini düzeltir. Bedeli: çok daha ağır (C++ derlemesi
gerekir, saniyeler süren kurulum, karmaşık parametre seti). Bu projede amaç
"hazır, endüstri/akademi standardı bir SLAM kütüphanesi bu problemde ne kadar
iyi çalışır?" sorusuna cevap bulmaktı — `visualodometry.py`'nin özel-amaçlı
çözümüne bir **alternatif/karşılaştırma noktası**.

---

## 3. Veri

- **Video:** `TUYZ_Video_1_mx1vu` (Teknofest HYZ-2024 veri seti).
- **Kareler:** `8_temmuz_egitim/_extracted/hyz-2024-final-verileri/media/
  TUYZ_Video_1_mx1vu/` — 2378 adet JPEG, `frame_000000.jpg, frame_000004.jpg,
  ...` şeklinde adlandırılmış (orijinal videodan her 4. kare alınarak
  çıkarılmış, yani orijinal 30 fps video → efektif 7.5 fps).
- **Ground truth (GT, gerçek konum):**
  `data/translations/TUYZ_Video_1_mx1vu.csv` — her satırda `translation_x`,
  `translation_y` (metre) ve karşılık gelen kare adı. CSV satır sırası,
  kare dosyalarının sırasıyla birebir eşleşiyor. **Önemli:** bu CSV'de
  **z (irtifa) yok** — sadece yatay düzlemdeki (x,y) konum var.
- **Kamera:** Görüntü boyutu 1920×1080 (Full HD RGB).

**Kural:** proje genelinde kamera iç parametrelerini (intrinsics) **EXIF/
metadata'dan almak yasak** — sadece piksellerden veya resmi/bağımsız
kalibrasyon kaynaklarından elde edilmesi gerekiyor (bkz. Bölüm 5).

---

## 4. Adım Adım Ne Yaptık

### Adım 1 — ORB-SLAM3'ü Kaynaktan Derleme

`tools/ORB_SLAM3` altına resmi depoyu (`UZ-SLAMLab/ORB_SLAM3`) klonladık.
Kütüphane 2020'de C++11/14 ve eski bir Pangolin (görselleştirme kütüphanesi)
sürümüne göre yazılmış; bizim sistemimizde daha yeni bir derleyici (gcc 11) ve
daha yeni Pangolin (0.9.3) kurulu olduğu için **iki küçük yama** gerekti:

1. **C++ standardını 11/14'ten 17'ye yükseltmek** (`CMakeLists.txt` ve alt
   kütüphaneler `DBoW2`, `g2o`, `Sophus` içinde) — yeni Pangolin C++17
   istiyor.
2. **Bir derleme hatasını düzeltmek:** `LoopClosing.h` içinde
   `mnFullBAIdx` adlı bir sayaç yanlışlıkla `bool` (doğru/yanlış) tipinde
   tanımlanmış ama kodun başka yerinde `mnFullBAIdx++` ile **arttırılmaya**
   çalışılıyordu. C++11/14'te `bool++` izinliydi (kullanımdan kaldırılmış
   uyarısıyla), ama C++17'de bu tamamen **yasak** hale geldi (derleme
   hatası veriyor). Çözüm basit: tipini `int`'e çevirmek.

Bunlar kütüphanenin **mantığını hiç değiştirmeyen**, sadece modern derleyiciyle
uyumluluk için gereken yamalardı.

### Adım 2 — Özel Bir "Sürücü" (Driver) Programı Yazmak: `mono_tuyz`

ORB-SLAM3 örnek programları (`mono_tum`, `mono_kitti`, ...) genelde iş
bitince `SLAM.SaveTrajectoryTUM(...)` fonksiyonuyla **tüm yörüngeyi sona
kadar** kaydeder. Ama bu fonksiyon monoküler modda **çalışmıyor**
(`ERROR: SaveTrajectoryTUM cannot be used for monocular` — kütüphanenin
kendi kısıtlaması, çünkü monoküler moddaki poz geçmişi farklı şekilde
saklanıyor).

Daha da önemlisi: **bizim senaryomuz için tüm yörüngeyi sonunda kaydetmek
zaten yanlış olurdu.** Neden? Çünkü ORB-SLAM3 arka planda sürekli loop
closing ve bundle adjustment ile **geçmiş pozları da düzeltiyor**
(gelecekte gördüğü bir bilgiyle geçmişi iyileştiriyor). Gerçek GPS-kopması
senaryosunda ise, kopma anında sistemin sadece **o ana kadar bildiği**
bilgiyle tahmin yapması lazım — gelecekten "kopya çekmesi" mümkün değil.

Bu yüzden `Examples/Monocular/mono_tuyz.cc` adında özel bir program yazdık:

- Her kare işlendikten **hemen sonra**, `SLAM.TrackMonocular(...)`'ın o an
  döndürdüğü pozu (`Tcw` — kameradan dünyaya dönüşüm) alıp **anında** dosyaya
  yazıyor. Gelecekteki hiçbir düzeltmeyi beklemiyor.
- Sadece takip durumu "OK" (`GetTrackingState() == 2`) olan karelerin pozunu
  kaydediyor — takip kaybolduysa o kare atlanıyor.
- Çıktı formatı **TUM formatı**: her satır
  `zaman_damgası tx ty tz qx qy qz qw` (konum + kuaterniyon olarak
  yönelim/rotasyon — 4 sayıyla 3B dönüşü ifade eden standart bir gösterim).

`Tcw` "kameradan dünyaya" değil "dünyadan kameraya" dönüşümdür; biz kameranın
dünyadaki konumunu istediğimiz için `Twc = Tcw.inverse()` ile tersini
alıyoruz, sonra `Twc.translation()` ile konumu, `Twc.unit_quaternion()` ile
yönelimi çıkarıyoruz.

### Adım 3 — Kamera Kalibrasyonu

ORB-SLAM3'ün çalışması için kameranın iç parametrelerine (odak uzaklığı
`fx, fy`, optik merkez `cx, cy`, lens bozulma katsayıları `k1, k2, p1, p2`)
ihtiyaç var. Bunları **EXIF'ten almak yasak** olduğu için üç farklı kaynak
denedik:

**A) COLMAP ile kendinden-kalibrasyon (self-calibration), serbest model.**
COLMAP, sadece görüntülerden (hiçbir önceden bilinen ölçü olmadan) hem
kamera parametrelerini hem de sahnenin 3B yapısını **aynı anda** kestirebilen
bir *Structure-from-Motion* (SfM) programıdır. 400 karenin her 7'sinden
birini alıp (58 kare) COLMAP'e verdik; kendi feature noktalarını bulup
eşleştirdi, kameranın tüm parametrelerini serbest bırakarak kestirdi:
`fx=1743.1, fy=1817.8`, ortalama **reprojection error** (yeniden izdüşüm
hatası — 3B'den tekrar 2B'ye izdüşürülen noktaların gerçek piksel
konumundan ne kadar saptığı, kalibrasyon kalitesinin standart ölçüsü)
**0.51 piksel** — kulağa iyi geliyor.

**B) COLMAP ile kendinden-kalibrasyon, `fx=fy` kısıtlı model.** Aynı 58
kareyle, bu sefer `fx` ve `fy`'nin eşit olmasını zorunlu kıldık (gerçek
kameralarda genelde piksel kare olduğu için bu makul bir varsayım). Sonuç:
`f=1533.7`, reprojection error **0.53 piksel** — A ile neredeyse aynı
kalitede bir uyum, ama odak uzaklığı **%14 farklı** (1534 vs 1743)!

**Bu bir problem işareti.** İki farklı model, görüntüleri neredeyse aynı
kalitede açıklıyor ama birbirinden çok farklı odak uzaklığı söylüyor. Sebep:
kamera neredeyse tam dikey aşağı bakıyor ve altındaki zemin (yaklaşık) düz
bir yüzey. **Düz bir yüzeye bakan kamerada, "kamerayı biraz uzaklaştırıp
odak uzaklığını orantılı artırmak" ile "kamerayı olduğu yerde bırakmak"
neredeyse aynı görüntüyü üretir** — matematiksel olarak buna **düzlem
dejenerasyonu** (planar degeneracy) denir. SfM'in odak uzaklığını
zemin+ölçek birbirinden ayırt etmesi için sahnede düzlem-dışı (farklı
derinlikte) noktalar da görmesi gerekir; nadir bakışta bunlar çok az.
**Sonuç: nadir/düzleme-yakın sahnelerde self-calibration'a güvenmemek
lazım**, "iyi görünen" reprojection error yanıltıcı olabiliyor.

**C, D) Resmi (checkerboard) kalibrasyon.** Sen bize MATLAB
`cameraParameters` çıktısı olarak, gerçek bir **satranç tahtası deseni**
(checkerboard) 33 farklı açı/mesafeden fotoğraflanarak yapılmış resmi
kalibrasyon parametrelerini verdin. Checkerboard kalibrasyonu, dejenerasyon
sorunu yaşamaz çünkü kalibrasyon **hedefi bilerek farklı açı/mesafelerden**
çekilir — yani "düzlem + ölçek" belirsizliğini kırmak için özel olarak
tasarlanmıştır. İki resmi dosya vardı:
  - `Kamera_Kalibrasyon_Parametreleri_2026.txt` (RGB Camera 1080p):
    `fx=1389.7, fy=1387.1` — önce bunu kullandık (kalibrasyon C).
  - `Kamera_Kalibrasyon_Parametreleri_2024.txt` (33 pattern, 0.645px
    reprojection error): `fx=1413.3, fy=1418.8` — **verinin ait olduğu
    yılın kalibrasyonu bu** olduğu için asıl doğrusu bu (kalibrasyon D).

Kalibrasyon dosyaları hâlâ **EXIF/metadata değil** — bunlar kameranın kendi
resmi, bağımsız ölçüm süreciyle (fiziksel bir satranç tahtası çekilerek)
elde edilmiş kalibrasyon sabitleri; görüntü dosyalarının içine gömülü
otomatik meta veri değiller. Bu yüzden proje kuralını ihlal etmiyor.

---

## 5. Arkadaki Matematik: Monoküler Ölçek Belirsizliği ve 3B Umeyama Hizalaması

Bu kısım `visualodometry_aciklama.md`'deki Umeyama bölümüyle **aynı fikri**
kullanıyor, ama burada 2B değil **3B** (çünkü ORB-SLAM3'ün ürettiği poz tam
3 boyutlu: x, y, z).

### Problem

Tek kamera (monoküler) ile SLAM yaptığınızda, üretilen yörünge:

1. **Ölçeksizdir** — "5 birim ilerledi" bilgisi var ama bu 5 birimin kaç
   metre olduğunu görüntüden bilemezsiniz (aynı piksel hareketini hem
   "yakın bir sahne, az hareket" hem "uzak bir sahne, çok hareket" açıklar).
2. **Keyfi bir başlangıç ekseni ve yönelimi vardır** — ORB-SLAM3 ilk kareyi
   `(0,0,0)` kabul eder, dünyanın gerçek kuzey/doğu ekseniyle hiçbir ilgisi
   yoktur.

### Çözüm: Umeyama Benzerlik Dönüşümü (3B)

Elimizde ilk 200 karenin **gerçek konumu** (GT) olduğu için, ORB-SLAM3'ün
ürettiği (ölçeksiz, keyfi eksenli) 3B pozları bu gerçek konumlara **en iyi
oturacak şekilde** dönüştürebiliriz:

```
gerçek_nokta ≈ s · R · slam_nokta + t
```

- `s`: tek bir ölçek sayısı (SLAM biriminden metreye çevirme çarpanı)
- `R`: 3×3 dönüş matrisi (SLAM'in keyfi eksenini gerçek dünya eksenine
  çevirir)
- `t`: 3 boyutlu öteleme (kayma) vektörü

**Adım adım (kod: `scripts/align_and_evaluate.py`, `umeyama()` fonksiyonu):**

1. İki nokta kümesinin (SLAM pozları ve GT konumları) **ağırlık merkezini**
   (centroid) çıkar, her iki kümeyi kendi merkezine göre ortala.
2. **Çapraz kovaryans matrisi**ni hesapla: `cov = (dst_c^T · src_c) / n`
   — bu matris, iki kümenin birlikte nasıl değiştiğini özetler.
3. Bu matrisi **SVD** (Singular Value Decomposition) ile ayrıştır:
   `cov = U · D · Vᵀ`.
4. En iyi dönüş: `R = U · S · Vᵀ` (S, kümelerden biri "ayna görüntüsü" gibi
   ters çevrilmişse bunu düzeltmek için kullanılan bir işaret düzeltme
   matrisi — `det(U)·det(Vᵀ) < 0` ise devreye girer).
5. En iyi ölçek: `s = trace(D · S) / var(src)` (trace = köşegen toplamı,
   var = kaynak kümenin varyansı).
6. En iyi öteleme: `t = ortalama(dst) − s · R · ortalama(src)`.

Bu, iki 3B nokta bulutunu üst üste bindirmenin **en küçük kareler anlamında
matematiksel olarak en iyi** yoludur — tek bir örnek noktaya değil,
**bütün** hizalama penceresine (170-200 kareye) birden dayanır, bu yüzden
tek bir gürültülü ölçüme karşı çok daha dayanıklıdır.

### GT'de z olmadığı için ne yaptık?

GT CSV'sinde sadece x,y var. Bu yüzden hizalama hedefini `[gt_x, gt_y, 0]`
olarak kurduk — yani **"drone sabit bir yükseklikte uçuyor" varsayımıyla
GT'yi z=0 düzlemine oturttuk.** Bu bir yaklaşıklama: gerçek irtifa
değişiyorsa (özellikle dönüş sırasında drone alçalıp yükseliyorsa) bu
varsayım hizalamayı hafifçe bozabilir. SLAM'in ürettiği z değeri (tahmin
tablosundaki `pred_z`) bu yüzden **hiçbir zaman doğrulanamadı** — elimizde
karşılaştıracak gerçek irtifa verisi yok.

### Uygulama: GPS koptuktan sonrasını tahmin etme

Hizalama penceresinden (`s, R, t`) elde edilen dönüşüm, artık GPS'siz
bölümdeki her SLAM pozuna uygulanıyor:

```
tahmini_konum(kare) = s · R · slam_poz(kare) + t
```

Bu tek formül, GPS koptuktan sonraki her karenin metre cinsinden dünya
konumunu veriyor.

---

## 6. Kurduğumuz Dosyalar ve Ne İşe Yaradıkları

| Dosya | Ne yapar |
|---|---|
| `tools/ORB_SLAM3/` | ORB-SLAM3'ün kendisi, kaynaktan derlenmiş hali |
| `tools/ORB_SLAM3/Examples/Monocular/mono_tuyz.cc` | Bizim özel sürücümüz — her karenin **online** (anlık, geleceğe bakmadan) pozunu TUM formatında kaydeder |
| `calib/` | COLMAP self-calibration çalışma klasörü (görüntüler, veritabanı, kestirilen 3B model) |
| `config/tuyz_video1*.yaml` | ORB-SLAM3'e verilen kamera+ORB ayarları (her kalibrasyon denemesi için ayrı dosya) |
| `config/video1_*.txt` | `prepare_list.py` ile üretilen, ORB-SLAM3'e verilecek görüntü listesi (zaman damgası + dosya yolu) |
| `scripts/prepare_list.py` | Kare klasöründen, ilk N kareyi zaman damgalarıyla birlikte listeleyen basit araç |
| `scripts/align_and_evaluate.py` | Umeyama hizalama + GPS'siz bölüm tahmini + hata hesabı (RMSE, eksen bazlı bias) + grafik çizimi |
| `data/translations/` | Ground truth CSV'leri (translations.zip'ten çıkarılmış) |
| `results_video1/calibA, calibB, calibC_official, calibD_2024official_600` | Her kalibrasyon denemesinin SLAM çıktısı (`traj.txt`), tahmin tablosu (`predictions.csv`) ve grafiği (`trajectory.png`) |

---

## 7. Sonuçlar

### 200 kare hizalama + 200 kare tahmin (üç kalibrasyon karşılaştırması)

| Varyant | Kalibrasyon kaynağı | Odak (px) | Hizalama fit RMSE | GPS'siz bölüm RMSE |
|---|---|---|---|---|
| A | COLMAP serbest model | fx=1743, fy=1818 | 0.54 m | **20.8 m** |
| B | COLMAP fx=fy kısıtlı | f=1534 | 0.32 m | **6.3 m** |
| C | Resmi kalibrasyon (2026 dosyası) | fx=1390, fy=1387 | 0.27 m | **3.07 m** |

Yorum: self-calibration (A, B) ne kadar "kısıtlanırsa" (fx=fy zorlanırsa)
gerçek kalibrasyona (C) o kadar yaklaşıyor, ama hâlâ ondan daha kötü. **Gerçek
kalibrasyon her zaman en güvenilir seçenek.**

### 200 kare hizalama + 400 kare tahmin (2024 resmi kalibrasyon, en doğru kaynak)

| Eksen | RMSE | Ortalama işaretli sapma (bias) |
|---|---|---|
| X | 17.99 m | +15.54 m |
| Y | 34.09 m | +29.76 m |
| Toplam (x,y) | 38.54 m | — |
| Z (irtifa) | **doğrulanamıyor** | GT'de `translation_z` yok |

**Ama bu sayı yanıltıcı olabilir — detaya bakınca:** hata, GPS koptuktan
sonraki ilk ~250 karede **~3 metre** civarında (üstteki C varyantıyla
tutarlı!) kalıyor. Sonra frame ~440-560 arasında (grafikte "toplam hata"
eğrisinin dikleştiği bölge) **20 metreden 65 metreye** sıçrıyor. Bu bölgeye
bakınca, ground truth yörüngesinde **sert bir dönüş** olduğunu görüyoruz
(`trajectory.png`'nin sol üst panelinde mavi çizginin keskin U dönüşü
yaptığı yer).

**Neden dönüşte hata büyüyor? — SONRADAN DÜZELTİLDİ:** İlk yorumumuz
"IMU'suz monoküler SLAM dönüşlerde zayıftır" idi. Sonraki deneyler
(aşağıdaki 2026-07-10 güncellemesi) bunun **yanlış** olduğunu gösterdi:
aynı 600-frame testi 2026 kalibrasyon dosyasıyla tekrarlanınca sert dönüş
dahil RMSE 38.5 m'den 5.0 m'ye düştü. Asıl suçlu, 2024 kalibrasyon
dosyasının **distorsiyon katsayılarının** bu videolara uymamasıydı: yanlış
distorsiyon düzeltmesi, özellikle dönüşlerde (görüntünün her bölgesi
kullanılırken) poz kestirimini sistematik olarak bükiyordu. Ders: **hangi
kalibrasyonun doğru olduğunu dosya adındaki yıla göre değil, ampirik testle
(hizalama fit hatası + tahmin hatasıyla) seç.**

**Önemli, dürüst bir not:** D varyantının (38.5 m) C varyantından (3.07 m)
"14 kat kötü kalibrasyon" olduğu **yanlış bir sonuç** olurdu — ikisi
farklı şeyler ölçüyor: C sadece düz uçuş segmentini (200 kare) test etti, D
aynı düz segmenti **artı** sert dönüşü (400 kare) kapsadı. Kalibrasyonun
kendisi D'de daha güvenilir (doğru yıla ait, checkerboard kalibrasyonu);
kötüleşmenin sebebi test penceresinin zorlaştırılmasıdır.

---

## 8. Ne İyi Gitti, Ne Öğrendik

- **Online poz kaydı önemliydi.** Eğer `SaveTrajectoryTUM`'u (ya da
  benzer bir "sona kadar bekle, sonra kaydet" yaklaşımını) kullanabilseydik,
  loop-closing/global-BA düzeltmeleri GPS'siz bölümdeki pozlara da sızıp
  sonuçları olduğundan iyi gösterebilirdi — gerçekte GPS kopması anında bu
  düzeltmeler mevcut olmazdı. `mono_tuyz` bunu doğru simüle ediyor.
- **Nadir bakışta self-calibration'a güvenilmez.** COLMAP'in iki farklı
  modeli (A, B) neredeyse aynı reprojection error'a rağmen odak uzaklığında
  %14 fark verdi — düzlem dejenerasyonu yüzünden. Gerçek/checkerboard
  kalibrasyon şart.
- **Düz uçuşta sistem gayet iyi:** ~3 m hata / ~50-80 m'lik GPS'siz uçuş
  (yaklaşık %4-6 hata oranı), IMU olmadan makul bir sonuç.
- **Sert dönüşler asıl zayıf nokta.** Bu, veri setinin doğasından kaynaklı
  bir sınır, kodun bir hatası değil.

---

## 9. Ne İyileştirilebilir? (Gelecek Fikirler)

1. **IMU verisi eklemek.** ORB-SLAM3'ün *Monocular-Inertial* modu var —
   IMU (özellikle dönüş hızı/gyroscope) eklenirse sert dönüşlerdeki zayıflık
   büyük ölçüde giderilir, çünkü ölçek de IMU'dan gözlemlenebilir hale gelir
   (görüntüye hiç ihtiyaç kalmadan). Şu an IMU verimiz olmadığı için
   kullanılamadı ama en yüksek potansiyelli iyileştirme budur.
2. **Kayan pencere (sliding window) Umeyama.** Şu an tek, sabit bir 200
   karelik hizalama penceresi kullanılıyor. GPS sağlıklıyken sürekli
   güncellenen bir hizalama, ölçekteki yavaş kaymaları (scale drift) daha
   iyi takip edebilir (`visualodometry_aciklama.md`'de aynı fikir öneriliyor).
3. **Loop closing'i tetikleyecek uçuş planı.** Eğer drone GPS'siz bölümde
   daha önce gördüğü bir alana kısa bir süreliğine dönerse, ORB-SLAM3'ün
   loop-closing mekanizması devreye girip birikmiş hatayı geriye doğru
   düzeltebilir — bu, sistemin en güçlü, şu an hiç kullanılmayan özelliği.
4. **Stereo veya RGB-D kamera.** Eğer ileride ikinci bir kamera eklenirse,
   ORB-SLAM3 stereo modda ölçeği **doğrudan** (Umeyama'ya, GT'ye hiç
   ihtiyaç duymadan) metre cinsinden hesaplayabilir — monoküler ölçek
   belirsizliği tamamen ortadan kalkar.
5. **Barometre/irtifa sensörü füzyonu.** GT'de z hiç olmadığı için şu an z
   tahminini doğrulayamıyoruz bile. Gerçek irtifa verisi (barometre gibi)
   loglanırsa hem doğrulama yapılabilir hem de (VO dokümanındaki z
   çözümüne benzer şekilde) görüntüden bağımsız, daha güvenilir bir irtifa
   kaynağı sisteme eklenebilir.
6. **Diğer videolarla çapraz doğrulama.** Şu ana kadar sadece
   `TUYZ_Video_1` test edildi. `data/translations/` altında GT'si olan 13
   video var — aynı kamera/kalibrasyonla diğer videolarda da test edilip
   sonucun genellenip genellenmediği (özellikle düz uçuş vs. dönüş içeren
   videolar ayrı ayrı) incelenmeli.
7. **ORB özellik sayısı/parametreleri ince ayarı.** Şu an genel amaçlı
   ayarlar kullanılıyor (`ORBextractor.nFeatures: 3000`); bu veri setinin
   doku/kontrast özelliklerine göre ayarlanmadı — potansiyel küçük bir
   iyileştirme alanı.
8. **Essential matrix ile çapraz kontrol.** `visualodometry_aciklama.md`'de
   önerilen fikirle aynı: nadir açı tam 90 derece değilse (örneğin 85
   derece), bunun ORB-SLAM3'ün 3B model varsayımını ne kadar etkilediğini
   ayrı bir essential-matrix tabanlı tahminle karşılaştırıp modelin ne zaman
   güvenilmez hale geldiğini önceden haber vermek mümkün olabilir.

---

## 9.5 Güncelleme (2026-07-10): İyileştirme Turu Sonuçları

Üç yılın kalibrasyonu ve yeni hizalama fikirleri sistematik test edildi
(detaylı tablo README.md'de). Özet:

1. **Kalibrasyon ampirik seçilmeli.** 2024 videoları için en iyi sonucu
   2026 kalibrasyon dosyası verdi (Video_1 200+200: 2024 kalib 20.8 m ↔
   2026 kalib **1.79 m**; 200+400 dönüş dahil: 38.5 m ↔ **5.0 m**).
   Fark odakta değil (%1.7), distorsiyon katsayılarında.
2. **plane2d hizalama modu eklendi** (`--mode plane2d`): kamera
   yönelimlerinden yer düzlemi kestirilip hizalama 2B'de yapılıyor.
   Hizalama penceresi düz bir hat olduğunda (Video_2) 3B Umeyama'nın
   kısıtlayamadığı serbest rotasyonu ortadan kaldırıyor: Video_2
   **31.9 → 6.85 m**. Video_1'de regresyon yok. Varsayılan öneri bu mod.
3. **Reddedilenler:** pencerenin sadece son K frame'iyle hizalama
   (K=50'de 44 m — dar pencere dönüşümü kısıtlayamıyor; tam pencere en
   iyi) ve ORB nFeatures 4000 + iniThFAST 15 (1.79→5.6 m — düşük kaliteli
   fazla nokta zarar veriyor).
4. **2026 THYZ örnek verisi** (webp) 2026 kalibrasyonla sorunsuz izleniyor
   (590/600 frame). x,y,z'li GT değerlendirme sunucusunda (10.3.128.2:1025);
   sunucu şu an kapalı olduğu için 3-eksen (z dahil) testi henüz yapılamadı —
   sunucu açılınca yörünge dosyası hazır.

## 10. Terimler Sözlüğü

| Terim | Anlamı |
|---|---|
| SLAM | Eş zamanlı konum belirleme + haritalama |
| ORB | Hızlı hesaplanan bir özellik noktası türü (Oriented FAST + Rotated BRIEF) |
| Monoküler | Tek kameralı |
| Keyframe (Anahtar Kare) | Haritaya eklenen, seçilmiş temsilci kareler |
| Bundle Adjustment | Birçok kare/nokta pozunu birlikte en küçük hataya oturtan optimizasyon |
| Triangulation (Üçgenleme) | İki farklı açıdan görülen aynı noktanın 3B konumunu hesaplama |
| Loop Closing (Döngü Kapama) | Daha önce görülen bir yerin tekrar tanınıp birikmiş hatanın düzeltilmesi |
| DBoW2 | Görüntüleri hızlı karşılaştırmak için kullanılan görsel "kelime dağarcığı" yöntemi |
| Tcw / Twc | Kameradan-dünyaya / dünyadan-kameraya dönüşüm matrisi |
| TUM formatı | `zaman tx ty tz qx qy qz qw` şeklinde standart yörünge kaydı biçimi |
| Kuaterniyon | 3B dönüşü 4 sayıyla ifade eden matematiksel gösterim |
| Ölçek Belirsizliği | Tek kameranın "kaç metre" olduğunu bilememesi sorunu |
| Umeyama Yöntemi | İki nokta kümesini en iyi hizalayan ölçek+dönüş+kayma bulma yöntemi |
| SVD | Bir matrisi üç basit matrisin çarpımına ayıran matematik yöntemi |
| COLMAP | Görüntülerden kamera parametrelerini ve 3B sahneyi birlikte kestiren SfM programı |
| Self-Calibration (Kendinden Kalibrasyon) | Hiçbir önceden bilinen ölçü olmadan, sadece görüntülerden kamera parametrelerini kestirme |
| Reprojection Error | 3B'den 2B'ye geri izdüşürülen noktaların gerçek piksel konumundan sapması (px) |
| Düzlem Dejenerasyonu | Düz bir yüzeye bakan kamerada odak uzaklığı ile mesafenin birbirinden ayırt edilememesi |
| Checkerboard Kalibrasyonu | Bilinen desenli bir tahtanın farklı açılardan çekilmesiyle yapılan resmi kalibrasyon |
| RMSE | Hata kareleri ortalamasının karekökü — ortalama hata büyüklüğü (metre) |
| Bias (Sapma) | Hep aynı yönde tekrar eden, rastgele olmayan sistematik hata |
| GT (Ground Truth) | Gerçek/doğru değer (burada: bilinen konum) |
| Drift | Zamanla/kareyle birikerek büyüyen hata |

---

## 11. Terimler Dışı — Hızlı Özet (Tek Paragrafta Her Şey)

ORB-SLAM3'ü kaynaktan derledik (küçük uyumluluk yamalarıyla), her karenin
**anlık** pozunu kaydeden özel bir sürücü yazdık, kamera kalibrasyonunu
önce görüntülerden (COLMAP) sonra da senin verdiğin resmi (checkerboard)
kalibrasyonla denedik ve resmi kalibrasyonun çok daha iyi sonuç verdiğini
gördük. İlk 200 GPS'li karede SLAM'in ölçeksiz/keyfi-eksenli yörüngesini
gerçek konumlara **Umeyama benzerlik dönüşümüyle** oturttuk, aynı dönüşümü
GPS'siz karelere uygulayıp konum tahmini ürettik. Düz uçuşta hata ~3 metre
gibi iyi bir seviyede kaldı; ground truth'ta sert bir dönüş olan bölgede
IMU'suz monoküler SLAM'in bilinen zayıflığı yüzünden hata büyüdü (65 m'ye
kadar). En büyük iyileştirme potansiyeli IMU eklemek ve/veya uçuş planını
loop-closing'i tetikleyecek şekilde tasarlamak.
