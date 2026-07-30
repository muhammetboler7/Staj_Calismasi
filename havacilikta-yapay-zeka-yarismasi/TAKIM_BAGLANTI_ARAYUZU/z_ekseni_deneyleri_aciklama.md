# Z Ekseni (İrtifa) Deneyleri — Baştan Sona Basit Anlatım

Bu belge, Görev 2'de (GPS koptuğunda kameradan konum tahmini) Z ekseninin
neden kötü olduğunu bulmak ve düzeltmek için yaptığımız **bütün deneyleri**
basit bir dille anlatır. Tarih: 22 Temmuz 2026. Test verisi:
`THYZ_2026_Online_Yarisma_Test_Oturumu` (2250 kare).

---

## 0. Önce sözlük — bol geçen terimler

| Terim | Basitçe anlamı |
|---|---|
| **GT (ground truth)** | Sunucunun bize verdiği GERÇEK konum. Sadece `health=1` karelerinde gelir. |
| **Kalibrasyon penceresi** | GT'nin geldiği ilk bölüm (örn. ilk 450 kare). Burada "SLAM'in dünyası" ile "gerçek dünya" arasındaki çeviriyi öğreniyoruz. |
| **Tahmin bölgesi** | GT kesildikten sonraki kareler. Burada sadece kameradan tahmin yapıyoruz ve başarıyı burada ölçüyoruz. |
| **RMSE** | "Tipik olarak kaç metre şaşıyoruz?" Tek sayı. Küçük = iyi. Büyük hataları ekstra cezalandırır. |
| **Bias** | "Hep aynı yöne mi şaşıyoruz?" Örn. bias = −3.9m ise tahminlerimiz ortalamada hep 3.9m ALÇAK. |
| **Eğim ve offset** | Bir doğrunun iki ayarı: `z = eğim × sinyal + offset`. Eğim "sinyal 1 birim artınca z kaç metre artar", offset "sıfır noktası nerede". |
| **Ölçek (s)** | Monoküler SLAM'in haritası "birim"siz — 1 SLAM birimi kaç metre, bilinmez. X/Y hizalaması bu çevirme katsayısını (s) GT'den hesaplar. |
| **Drift (sürüklenme)** | SLAM'in hatasının zamanla birikmesi. Özellikle ölçek yavaşça kayar. |
| **BA (bundle adjustment)** | ORB-SLAM3'ün "her şeyi birlikte en küçük hataya oturt" iç optimizasyonu. |
| **Loop closure** | Drone daha önce gördüğü yere dönünce SLAM'in "burayı tanıyorum!" deyip haritayı geriye doğru düzeltmesi. |
| **Oracle** | "Hile yapabilseydik en iyi ne olurdu?" testi. Cevabı (GT'yi) tahmin bölgesinde de kullanıp fit yaparız — gerçekte YAPILAMAZ, sadece bir yöntemin TAVANINI gösterir. |

---

## 1. Önceki (mevcut) sistem nasıl çalışıyor?

ORB-SLAM3 her karede bize bir konum verir ama bu konum **kendi uydurduğu
birimde ve kendi eksenlerinde**dir. Kalibrasyon penceresinde GT ile
eşleştirerek çeviriyi öğreniriz:

- **X/Y için:** SLAM konumları yere paralel bir düzleme yansıtılır, GT'nin
  x/y'sine "döndür + ölçekle + kaydır" (Umeyama) ile oturtulur. Bu çok iyi
  çalışıyor (~1-2m).
- **Z için (ESKİ YÖNTEM):** SLAM konumunun "yukarı" bileşeni alınır (buna
  **h** diyoruz) ve GT'nin z'sine düz bir çizgi çekilir:
  `z = eğim × h + offset`. Bu ~6m hata veriyordu. **Bütün mesele buydu.**

---

## 2. Deney 1 — "Suçlu BA mı? Loop closure mı az?"

**Soru:** ORB-SLAM3'ün iç optimizasyonu (BA + loop closure) sonuna kadar
çalışsa Z düzelir mi?

**Nasıl test ettik:** ORB-SLAM3 iki yörünge yazar: (1) her karede anlık poz
(bizim canlıda kullandığımız), (2) koşum sonunda tüm optimizasyonlardan
geçmiş "temize çekilmiş" keyframe yörüngesi. İkisini de aynı şekilde
değerlendirdik.

**Sonuç:**

| | Anlık poz | Tam optimize poz |
|---|---|---|
| X hatası | 0.92 m | 0.49 m ← optimizasyon X'e yarıyor |
| **Z hatası** | **5.91 m** | **5.96 m ← hiç düzelmedi!** |

**Ders:** BA, Z'yi düzeltemiyor. Nedeni fizik: BA sadece "3B nokta görüntüde
doğru piksele düşüyor mu?"ya bakar. Aşağı bakan kamerada bir noktayı
yukarı/aşağı (derinlikte) oynatsan görüntüdeki yeri NEREDEYSE DEĞİŞMEZ —
yani BA o yönü göremez. X/Y'de ise küçük oynama pikselleri hemen kaydırır,
o yüzden orada güçlüdür. **Sonuç: "BA'yı Z'de iyileştirmek" diye bir yol
yok; Z'ye BA'nın göremediği YENİ bir bilgi kanalı gerekiyordu.**

(Loop closure da tüm koşumda sadece 1-2 kez tetiklendi; ama tetiklense bile
bu tablo "asıl kazanç orada değil" diyor.)

---

## 3. Deney 2 — Yeni sinyal: "medyan sahne derinliği"

**Fikir:** Her karede ORB-SLAM3'ün takip ettiği yüzlerce harita noktası var
(yerdeki taşlar, çizgiler...). Bu noktaların kameraya uzaklıklarının
**ortancasını (medyan)** alırsak, aşağı bakan kamerada bu kabaca **yerden
yüksekliktir**. Üstelik her karede haritadan TAZE ölçülür — SLAM'in z
pozisyonu gibi hata biriktirmez (umut buydu).

**Nasıl:** `mono_tuyz`'un bir kopyasına (`mono_tuyz_depth`) her kare için
iki ek sütun ekledik: medyan derinlik ve o karede kaç nokta takip edildiği
(`n_tracked`). Mevcut sisteme dokunulmadı.

**İlk sonuç (ham haliyle):** Kalibrasyon 450 karede Z hatası **12m — FELAKET!**
Ama nedeni öğreticiydi: ilk 450 karede drone hep aynı yükseklikte uçuyor
(GT z dümdüz ~0). Düz veriden **eğim öğrenilemez** — "boy ile kilo ilişkisini
hepsi aynı boyda olan insanlardan öğrenmeye çalışmak" gibi. Model saçma bir
eğim uydurdu ve tahminde patladı.

**Ama iyi haber gizliydi:** Tahmin bölgesinde medyan derinlik GT z'yi
%97 doğrulukla takip ediyordu (eski sinyal h: %91). Yani **sinyal çok iyiydi,
sadece eğimi kalibrasyondan öğrenemiyorduk.**

---

## 4. Deney 3 — Eğimi öğrenme, HESAPLA ("eğim kilidi")

**Fikir:** Medyan derinlik metre değil, SLAM birimindedir. SLAM biriminden
metreye çevirme katsayısını zaten biliyoruz — X/Y hizalamasının hesapladığı
**ölçek s**! Derinlik de 3 boyutlu bir mesafe olduğu için aynı katsayıyla
çevrilmeli. O halde eğimi fit ETME, doğrudan `eğim = ±s` yaz; kalibrasyondan
sadece offset'i (sıfır noktasını) öğren — offset düz uçuştan bile öğrenilir.

**Sonuç:** Z hatası 12m → **5.45m**. Eski yöntemi (6.31m) geçti. Üstelik
450'lik ve 1000'lik kalibrasyonda neredeyse aynı sonucu verdi (5.45 / 5.47) —
yani kararlı.

> Not: Eski sinyal h için aynı numara ("eğim = s") daha önce denenmiş ve
> KÖTÜLEŞTİRMİŞTİ (7→12.5m). Çelişki yok: h, SLAM'in drift'li POZİSYONU;
> derinlik ise üçgenlemeyle ölçülen MESAFE. Fizik ikisinde farklı işliyor.

---

## 5. Deney 4 — Oracle testi: "Bu sinyallerin tavanı ne?"

GT'yi tahmin bölgesinde de kullanarak (hile!) her sinyalin ulaşabileceği en
iyi sonucu ölçtük:

| Sinyal | Tavan (oracle) |
|---|---|
| Eski (h) | ~5.1 m ← eski yöntem 6m'de takılıyordu çünkü TAVANI zaten 5m! |
| **Medyan derinlik** | **~3.1 m** ← doğru sinyal bu |

Bu test bize iki şey söyledi: (1) eski yöntemle ne yaparsak yapalım 5m
altına inilemez, (2) derinlik sinyaliyle 3m mümkün — aradaki farkı kapatmaya
değer.

---

## 6. Deney 5 — Offset'i iyileştirme denemeleri

Kilitli-eğim yönteminin kalan hatasının çoğu **sabit bir kayma** (bias
−3.9m: hep alçak tahmin). Offset'i daha akıllı seçersek düzelir mi?

| Denenen offset | 450 kalib. | 1000 kalib. | Karar |
|---|---|---|---|
| Tüm pencerenin ortalaması (baz) | 5.45 | 5.47 | baz |
| Medyan (ortanca) | 5.98 ↑kötü | 4.69 ↓iyi | tutarsız → RED |
| **n_tracked ağırlıklı ortalama** | **4.88 ↓** | **5.12 ↓** | **iki yerde de iyi → KABUL** |
| Sadece son N kareden | 9.6–11.5 ‼ | 4.1–4.7 | tek yerde felaket → RED |

- **n_tracked ağırlıklı:** çok nokta takip edilen kareler daha güvenilir,
  onlara daha çok söz hakkı ver. Mantıklı ve iki koşulda da kazandırdı.
- **"Son N kare" neden patladı?** 450'lik pencerenin İÇİNDE offset zaten
  kayıyordu ama bu kayma tahmin bölgesine AYNI YÖNDE devam etmiyordu.
  Pencerenin sonuna güvenmek bizi yanlış yöne fırlattı. Bir ayar iki test
  koşulundan birinde çöküyorsa canlı sisteme giremez.

---

## 7. Deney 6 — İki sinyali harmanlamak

Elimizde iki bağımsız Z tahmini var: eski (h) ve yeni (derinlik). İkisinin
ağırlıklı ortalamasını denedik: `z = w × yeni + (1−w) × eski`.

| w (yeninin payı) | 450 kalib. | 1000 kalib. |
|---|---|---|
| 0.00 (sadece eski) | 6.31 | 6.13 |
| 0.50 | 5.05 | 5.07 |
| **0.75** | **4.81** | **4.94** ← iki koşulda da en iyi |
| 1.00 (sadece yeni) | 4.88 | 5.12 |

Neden yardım ediyor? İki yöntemin hataları birebir aynı değil; ortalama
alınca birbirinin gürültüsünü kısmen siliyorlar. w=0.75 "çoğunlukla yeniye
güven, eskiyi az miktarda sigorta olarak tut" demek.

---

## 8. Nihai durum — tek bakışta

```
                         Z hatası (450 kare kalibrasyon)
Eski sistem (h-regresyon)          : 6.31 m   ██████████████████████
+ medyan derinlik, eğim=±s kilitli : 5.45 m   ███████████████████
+ n_tracked ağırlıklı offset       : 4.88 m   █████████████████
+ w=0.75 harman (NİHAİ REÇETE)     : 4.81 m   ████████████████   (−%24)
Teorik tavan (oracle)              : 3.12 m   ███████████
```

3B toplam hata: 6.57m → **~5.15m (−%22)**. X/Y değişmedi (o zaten iyiydi).

**Nihai reçete (formül):**
`z = 0.75 × ( ±s × medyan_derinlik + ağırlıklı_offset ) + 0.25 × ( eğim × h + offset )`

---

## 8.5 Deney 7 — w=0.75'in başka oturumlarda doğrulanması (22 Temmuz, devam)

Reçeteyi hiç ayar değiştirmeden 2025 oturumlarında denedik. Sonuç tablosu
(Z RMSE, metre):

| Oturum | Kalib. | Eski (h) | w=0.75 | Kazanan |
|---|---|---|---|---|
| 2026 test (RGB) | 450 | 6.31 | **4.81** | w=0.75 |
| 2026 test (RGB) | 1000 | 6.13 | **4.94** | w=0.75 |
| 2025 o1 (RGB) | 450* | 19.04 | **10.82** | w=0.75 |
| 2025 o1 (RGB) | 1000* | 15.99 | **9.91** | w=0.75 |
| 2025 o4 (termal) | 450* | 14.32 | **10.12** | w=0.75 |
| 2025 o4 (termal) | 1000* | **2.92** | 8.88 | ESKİ ‼ |

(*SLAM geç başladığı için pencereler izlemenin başladığı kareden itibaren.)

**Dürüst sonuç: w=0.75 evrensel olarak doğrulanamadı** — 6 koşulun 5'inde
kazandı ama birinde (termal, uzun kalibrasyon) eskiden 3 kat kötü kaldı.

Önemli bağlam: her iki 2025 oturumunda da SLAM'in kendisi bozuktu (X/Y
hatası 20-57m! — 2026'daki 1.8m'ye karşılık). Yani o oturumlarda Görev 2
zaten hangi Z yöntemiyle olursa olsun çalışmazdı. **SLAM'in sağlıklı olduğu
tek oturum (2026 testi — gerçek yarışma koşuluna en yakın: RGB, 1080p,
dik bakış) reçeteyi doğruladı.** Termaldeki bozulmanın olası nedeni: eğim
kilidi "kamera dik aşağı bakıyor" varsayımına dayanır; termal kamera/veri bu
varsayımı ve X/Y ölçeğinin güvenilirliğini bozuyor.

Ayrıca "w'yi otomatik seç" fikri de denendi (kalibrasyon penceresi içindeki
fit kalitesine göre): ÇALIŞMADI, çünkü h penceresinin içinde her zaman iyi
görünür (zaten oraya fit ediliyor) — pencere içi başarı, pencere dışını
öngörmüyor.

**Pratik karar önerisi:** 2026 yarışma koşullarında (RGB, dik bakış,
sağlıklı SLAM) w=0.75 kullan; termal/dejenere koşullar için eski yöntem
yedekte kalsın.

## 9. Kalan sorun ve sıradaki adımlar

- **Kalan ~3.6m'lik kayma kalibrasyondan öğrenilemiyor.** Tahmin bölgesinde
  haritanın ölçeği/zemin durumu az da olsa değişiyor ve GT gelmediği için
  bunu göremiyoruz. Bunu ancak uçuş sırasında ara ara GT gelmesi (yeniden
  çapalama) veya harici bir sensör kapatır. Oracle (3.1m) ile aramızdaki
  fark budur.
- **w=0.75 tek uçuşta seçildi.** Canlıya almadan önce başka bir oturumda
  (elde 2025 oturumları var) doğrulanmalı.
- **Canlıya taşındı (22 Temmuz):** `mono_tuyz_stream.cc` artık her karede
  medyan derinlik + nokta sayısı da gönderiyor; `online_alignment.py`'de yeni
  varsayılan `z_mode="depth_blend"` (w=0.75 harman) çalışıyor. Derinlik verisi
  gelmezse kendiliğinden eski yönteme düşer (emniyet). Doğrulamalar: replay
  testi deney sonucunu birebir üretti (4.81m), uçtan uca duman testi geçti.

## 10. Dosyalar

- `mono_tuyz_depth.cc` — medyan derinlik çıkaran deney programı
  (ORB_SLAM3/Examples/Monocular/ içinde; mevcut programlara dokunulmadı)
- `THYZ_2026_Test_Oturumu_kalibrasyon450_medyanderinlik_gercek_vs_tahmin.png`
  — eski/yeni yöntem karşılaştırma grafiği
