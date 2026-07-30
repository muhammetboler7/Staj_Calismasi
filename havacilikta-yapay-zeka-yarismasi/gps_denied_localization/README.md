# GPS Kopması Senaryosunda Görüntü Tabanlı Konum Kestirimi

70–90° açıyla yere bakan drone kamerasından, GPS (translation) verisi kesildikten
sonra drone'un gittiği yolu tahmin eden sistem. Teknofest HYZ-2024 konum kestirimi
veri formatıyla çalışır.

## Yöntem

1. **ORB-SLAM3 (monoküler)** tüm sekans boyunca kamera pozunu takip eder.
   IMU yok; monoküler SLAM'in **ölçeği ve dünya ekseni belirsizdir**.
2. **Hizalama penceresi (ilk 200 frame):** GPS translation'ları bilinirken,
   SLAM yörüngesi ile GT arasında **3B Umeyama benzerlik dönüşümü**
   (ölçek s + rotasyon R + öteleme t) kestirilir. GT z=0 düzlemine oturtulur.
3. **GPS'siz bölüm (sonraki 200 frame):** Aynı (s, R, t) dönüşümü SLAM'in online
   pozlarına uygulanır → metre cinsinden (x, y) konum tahmini.

Kamera iç parametreleri **EXIF kullanılmadan**, COLMAP self-calibration ile
görüntü piksellerinden kestirildi (proje kuralı: metadata yasak).

## Klasör yapısı

```
gps_denied_localization/
├── tools/ORB_SLAM3/          # kaynaktan derlendi (C++17 + Pangolin 0.9.3 yamaları)
│   └── Examples/Monocular/mono_tuyz   # özel driver: her frame'in online pozunu kaydeder
├── calib/                    # COLMAP self-calibration çalışması (58 frame)
├── config/
│   ├── tuyz_video1.yaml      # ORB-SLAM3 kamera/ORB ayarları
│   └── video1_400.txt        # görüntü listesi (timestamp + yol)
├── scripts/
│   ├── prepare_list.py       # media klasöründen görüntü listesi üretir
│   └── align_and_evaluate.py # Umeyama hizalama + tahmin + RMSE + grafik
└── results_video1/           # traj.txt (TUM), predictions.csv, trajectory.png
```

## Çalıştırma

```bash
# 1. Görüntü listesi (ilk 400 frame = 200 GPS'li + 200 GPS'siz)
python3 scripts/prepare_list.py <media_klasoru> config/video1_400.txt --n 400

# 2. SLAM (viewer kapalı: son argüman 0)
cd tools/ORB_SLAM3
./Examples/Monocular/mono_tuyz Vocabulary/ORBvoc.txt \
    ../../config/tuyz_video1.yaml ../../config/video1_400.txt \
    ../../results_video1/traj.txt 0

# 3. Hizalama + değerlendirme
python3 scripts/align_and_evaluate.py results_video1/traj.txt \
    <translations.csv> --align-n 200 --predict-n 200 --out-dir results_video1
```

## Sonuçlar (TUYZ_Video_1, hizalama: frame 0–199, tahmin: frame 200–399)

| Kalibrasyon | Kaynak | f (px) | Fit RMSE | GPS'siz RMSE |
|---|---|---|---|---|
| A | COLMAP serbest (OPENCV) | fx=1743, fy=1818 | 0.54 m | 20.8 m |
| B | COLMAP fx=fy kısıtlı | 1534 | 0.32 m | 6.3 m |
| **C** | **Resmi kalibrasyon (1080p RGB)** | **fx=1389.7, fy=1387.1** | **0.27 m** | **3.07 m** |

Resmi kalibrasyon: `havacilikta-yapay-zeka-yarismasi/Kamera_Kalibrasyon/`
`Kamera_Kalibrasyon_Parametreleri_2026.txt` → `config/tuyz_video1_official.yaml`.
Hata GPS koptuktan sonra mesafeyle birikir (0.6 m → 5.5 m @ 200. frame),
~200 m GPS'siz uçuşta gezilen yolun ~%1.5'i.

Ders: nadir bakışlı, düzleme yakın sahnede COLMAP self-calibration odak
uzaklığını güvenilir kestiremiyor (düzlem degenerasyonu); fx=fy kısıtı bile
%10 sapıyor. Gerçek kalibrasyon varsa onu kullan.

### GÜNCEL SONUÇLAR (2026-07-10): kalibrasyon seçimi ampirik yapılmalı + plane2d hizalama

Üç yılın resmi kalibrasyonu ve iki hizalama modu sistematik karşılaştırıldı.
**Kalibrasyon eşleşmesi çözünürlükle değil, ampirik testle seçilmeli:**
2024 videoları (1920×1080) için en iyi sonucu 2024 dosyası DEĞİL,
2026 dosyasının 1080p profili verdi (fark distorsiyon katsayılarında:
2024: k1=-0.009 k2=+0.067 ↔ 2026: k1=+0.138 k2=-0.256).

| Test | Kalibrasyon | Mod | GPS'siz RMSE |
|---|---|---|---|
| Video_1, 200+200 | 2024 resmi | 3d | 20.8 m |
| Video_1, 200+200 | 2026 resmi | 3d | **1.79 m** |
| Video_1, 200+200 | 2026 resmi | plane2d | 1.93 m |
| Video_1, 200+400 (sert dönüş dahil) | 2024 resmi | 3d | 38.5 m |
| Video_1, 200+400 (sert dönüş dahil) | 2026 resmi | 3d | **5.00 m** |
| Video_2, 200+200 (hizalama penceresi düz hat) | 2026 resmi | 3d | 31.9 m |
| Video_2, 200+200 (hizalama penceresi düz hat) | 2026 resmi | plane2d | **6.85 m** |

### Ana_Oturum testi (2026-07-10): 300 hizalama + 600 tahmin

Veri: `2024_TUYZ_Online_Yarisma_Ana_Oturum/` (2250 frame) + aynı adlı CSV.
Kalibrasyon: kök dizindeki `Kalibrasyon_Parametreleri.txt` (= 2024 dosyasıyla aynı değerler).

| Kalibrasyon | Mod | GPS'siz RMSE (600 fr) | X RMSE | Y RMSE |
|---|---|---|---|---|
| **Verilen (fx=1413)** | **plane2d** | **3.90 m** | 3.67 m | 1.30 m |
| Verilen (fx=1413) | 3d | 11.5 m | 2.8 m | 11.2 m |
| 2026 (fx=1390) | plane2d | 7.50 m | 7.2 m | 2.1 m |
| 2026 (fx=1390) | 3d | 29.2 m | 5.4 m | 28.7 m |

Bu videoda verilen (2024) kalibrasyon daha iyi — TUYZ_Video_1/2'nin tersine!
Videolar farklı kameralarla çekilmiş olmalı. **Kalibrasyon seçimi video başına
ampirik yapılmalı** (hizalama fit RMSE düşük olan seçilir). plane2d her iki
kalibrasyonda da 3d'yi büyük farkla geçti (3-4× daha iyi) — varsayılan bu.

### THYZ_2025_Oturum_1 testi (2026-07-10): İLK 3 EKSENLİ (x,y,z) DEĞERLENDİRME

Veri: `thyz_2025_veri/media/THYZ_2025_Oturum_1_az7vb/` (3840×2160 webp, ardışık 30fps)
+ kök dizindeki `THYZ_2025_Oturum_1_Translation.csv` (**translation_z içeriyor!**).
Kalibrasyon: 2025 dosyası, video moduna uyarlanarak (4000→3840 ölçek 0.96 + dikey
merkez kırpma → fx=2680.5, cy=1139.8); `config/thyz2025_oturum1.yaml`, içeride
1920×1080'e küçültme (`Camera.newWidth/newHeight`). Pencere: 600-800 hizalama,
800-1200 tahmin (kullanıcı speci).

| Eksen | RMSE | Bias | Maks |
|---|---|---|---|
| X | 4.49 m | −1.34 m | 8.40 m |
| Y | 4.78 m | +3.72 m | 14.38 m |
| Yatay (xy) | **6.56 m** | — | 16.11 m |
| **Z (irtifa)** | **0.42 m** | +0.35 m | 0.73 m |

Bulgular:
- **İki aday klasörden doğrusu az7vb** (yp1ks fit RMSE 18.7 m + 2 harita reset →
  CSV'nin videosu değil). Doğru klasörü hizalama fit hatası seçti.
- **İlk ~600 frame açık deniz** — dokusuz suda mono SLAM başlatılamıyor; harita
  ancak kara görününce (frame ~618) doğuyor. Parametreyle çözülmez, veri özelliği.
- Harita hizalama penceresinin başında çok taze olduğu için erken frame'ler
  gürültülü → bu videoda `--align-last 40-80` büyük kazanç (xy 12.6→6.6 m).
  Kural: **hizalama penceresi harita doğumundan hemen sonraysa pencerenin geç
  kısmını kullan; harita olgunsa tam pencereyi kullan.**
- **z için serbest eğimli regresyon** (plane2d'de z eğimini xy ölçeğine kilitlemek
  yerine `polyfit(h, gt_z, 1)`): z RMSE 19.0→0.42 m. Script güncellendi.
- 3d Umeyama modu burada da çöktü (xy 131 m) — plane2d şart.

### THYZ_2025_Oturum_1 tam video testi (2026-07-10): 650-1000 kalibrasyon, 1000-2249 tahmin

Kullanıcı `vo_xy_sapma.png` stiliyle grafik istedi (üst-sol geniş XY yörünge +
sağda X/Y/Z alt grafikleri + altta tam genişlik sapma) — yeni script:
`scripts/plot_vo_style.py`. Tam video (2250 kare) 3 kalibrasyonla koşuldu.

| Kalibrasyon | Kaynak | Test 3B RMSE (1250 kare, frame 1000-2249) |
|---|---|---|
| **B (kazanan)** | 2026 dosyası "RGB Camera 1080p" ×2 (temiz ölçekleme) | **36.4 m** |
| C | 2024 dosyası (tek profil) ×2 (temiz ölçekleme) | 48.0 m |
| A | 2025/2026 "RGB Camera 4K" profili (kırpma varsayımlı) | 56.6 m |

Ders: video 3840×2160 (16:9) olduğu için 1080p kaynaklı kalibrasyonlar (B, C)
**varsayımsız** 2× ölçekleniyor; 4K profili (A) 4:3→16:9 kırpma varsayımı
gerektirdiği için en kötü sonucu verdi. Video en-boy oranına native olarak
uyan kalibrasyon kaynağını tercih et.

Gözlemler (`results_2025o1_full/calibB/vo_plot.png`):
- Frame 1000-2050: X/Y neredeyse mükemmel takip.
- Frame ~2050-2249: ani sapma (30m→180m) — GT düz dönerken tahmin büyük bir
  çevrim çiziyor; üç kalibrasyonda da aynı yerde. Muhtemel sert manevra.
- Z: GT'de frame ~1600'de gerçek bir tırmanış var (30m→38m keskin basamak).
  plane2d'nin z ekseni hizalama penceresinden (düz uçuş) öğrenilen sabit
  eğimle çalıştığı için bu sonradan olan sıçramayı hiç yakalayamıyor —
  yöntemin bilinen bir sınırı: irtifa profili hizalama penceresinde
  görülmeyen bir davranışa dönüşürse tahmin kopuyor.

Reddedilen iyileştirmeler (test edildi, kötüleştirdi):
- `--align-last 50/100/150` (pencerenin sonuyla hizalama): 50'de felaket (44 m) —
  dar/düz pencerede benzerlik dönüşümü kısıtlanamıyor. Tam pencere en iyi.
- ORB nFeatures 4000 + iniThFAST 15: 1.79→5.6 m. Düşük kaliteli fazla nokta zarar.

Kabul edilen iyileştirme — **plane2d modu** (`--mode plane2d`): kamera
yönelimlerinden (kuaterniyon) ortalama optik eksen = yer düzlemi normali
kestirilir, SLAM konumları bu düzleme izdüşürülüp 2B Umeyama (yansıma testli)
yapılır. Düz-hat hizalama pencerelerinde 3B Umeyama'nın serbest kalan
rotasyon eksenini ortadan kaldırır: Video_2'de 31.9→6.85 m, Video_1'de
regresyon yok. **Varsayılan olarak plane2d önerilir.**

2026 THYZ örnek verisi (webp, 1920×1080): 2026 kalibrasyonla tracking 590/600
frame — GT (x,y,z) değerlendirme sunucusunda (10.3.128.2:1025, şu an kapalı);
sunucu açılınca 3-eksen testi `results_thyz2026/traj.txt` ile hazır.

### ESKİ: 200 hizalama + 400 tahmin (2024 resmi checkerboard kalibrasyonu, fx=1413.3 fy=1418.8)

`config/tuyz_video1_2024official.yaml` — verinin ait olduğu 2024 yarışma dönemine ait
resmi kalibrasyon (33 checkerboard pattern, reproj. hatası 0.645 px).

| Eksen | RMSE | Ortalama işaretli sapma (bias) |
|---|---|---|
| X | 17.99 m | +15.54 m |
| Y | 34.09 m | +29.76 m |
| Toplam (xy) | 38.54 m | — |
| Z (irtifa) | doğrulanamıyor | GT'de translation_z yok |

~~Hata, GPS koptuktan ~250 frame sonra sıçrıyor — sert dönüş monoküler SLAM'i
bozuyor.~~ **DÜZELTME (2026-07-10):** Bu yorum yanlıştı. Aynı 600-frame testi
2026 kalibrasyonuyla tekrarlanınca dönüş dahil RMSE 5.0 m çıktı — asıl sorun
2024 kalibrasyon dosyasının distorsiyon katsayılarının bu videolara uymamasıydı.
Dönüşler doğru kalibrasyonla gayet iyi takip ediliyor (yukarıdaki güncel tablo).

## Notlar / bilinen sınırlar

- **Ölçek kayması (scale drift):** Monoküler SLAM'de ölçek zamanla kayabilir;
  hizalama penceresinden uzaklaştıkça tahmin hatası büyür. Loop closure yakalanırsa
  düzelir. İrtifa değişimi büyükse GT z=0 varsayımı hizalamayı bozar.
- **Yeni video için:** Kamera farklıysa COLMAP kalibrasyonunu o videonun
  frame'leriyle tekrarlayıp yaml'ı güncelle.
- **Timestamp varsayımı:** Frame'ler 4'er atlanmış, orijinal video 30 fps
  (efektif 7.5 fps). Yanlışsa sadece zaman ekseni ölçeklenir, sonuç değişmez.
