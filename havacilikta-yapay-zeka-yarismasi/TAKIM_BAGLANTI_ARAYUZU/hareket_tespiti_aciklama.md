# Taşıt Hareket Tespiti — Nasıl Çalışıyor, Neyi Düzelttik

Bu belge Görev 1'in bonus kısmını — bir taşıtın **Hareketli mi Sabit mi**
olduğunu anlamayı — basit bir dille anlatır. Başta neden iyi çalışmadığını, bu
oturumda adım adım neyi değiştirdiğimizi ve sonucun nasıl düzeldiğini içerir.

---

## 0. Sözlük

| Terim | Basitçe |
|---|---|
| **bbox / kutu** | Taşıtın etrafına çizilen dörtgen (YOLO'nun bulduğu) |
| **Optical flow** | İki kare arasında piksellerin ne kadar kaydığını ölçen yöntem |
| **Homografi** | "Bütün zemin nasıl kaydı?" sorusunu tek bir matematiksel kalıba oturtan dönüşüm (drone'un hareketinin görüntüdeki tam modeli) |
| **GSD** | 1 piksel kaç metre? (metre/piksel çevirme oranı) |
| **SLAM** | Görev 2'deki ORB-SLAM3 — kameradan konum ve harita çıkaran sistem |
| **Adım (step)** | Bir kareden diğerine taşıtın yaptığı hareket |

---

## 1. Temel fikir (aslında basit)

Drone uçarken kamera sürekli hareket ediyor, o yüzden **duran bir araç bile
görüntüde kayıyormuş gibi** görünüyor. Amacımız şu ayrımı yapmak:

> **Aracın görünen hareketi − drone'un hareketi = aracın gerçek hareketi**

Eğer sonuç ~0 ise araç **Sabit**, belirgin bir hız çıkıyorsa **Hareketli**.

"Drone'un hareketi"ni çıkarmak için ayrı bir sensöre gerek yok: aracın
**dışındaki** zemin noktalarına bir **homografi** oturtuyoruz. Zemin düz olduğu
için bu homografi, drone'un dönme dahil bütün hareketinin görüntüdeki tam
karşılığıdır — çıkarma işlemini bizim için o yapıyor.

---

## 2. Başta neden iyi çalışmıyordu?

Eski yöntem kararı **tek bir kare çiftinden** veriyordu: aracın içine nokta
koyup optical flow ile bir kareye bakıyordu. Sorun:

- 7.5 fps'te 3 m/s giden bir araç karede sadece ~8 piksel kayar.
- Optical flow gürültüsü + kutu kenarından zemine "kaçan" noktalar birkaç
  piksel hata üretir.
- Yani **sinyal (8px) ile gürültü (birkaç px) birbirine çok yakındı** — karar
  güvenilmezdi, testte yetersiz kaldı.

---

## 3. Bu oturumda ne değiştirdik (adım adım)

### Adım 1 — Kararı zamana yaydık (pencere)
Tek kareye bakmak yerine ~1 saniyelik (8 kare) bir pencerede aracın **toplam
yer değiştirmesine** bakmaya başladık. Duran araç pencere boyunca yerinde
titrer (hata birikmez), hareketli araç düzenli kayar (hata birikir). Böylece
sinyal/gürültü oranı kökten iyileşti.

### Adım 2 — Metrik ölçeği (GSD) SLAM'den aldık
"Kaç piksel = kaç metre?" sorusunu önce optical flow'dan hesaplıyorduk ama bu
değer dönme/parallaks yüzünden savruluyordu (aynı yerde 0.027 ile 0.054
arasında). Bunu **SLAM irtifasından** (Görev 2'nin verisi: ölçek × sahne
derinliği ÷ odak) hesapladık — her karede taze ve doğru. Bağımsız kontrol:
tipik araç kutusu 142 piksel × SLAM-GSD = ~5 metre araç boyu (gerçekçi).

### Adım 3 — "Doğum kontaminasyonunu" temizledik
Araç kareye yeni girerken kutusu büyür; büyüyen kutunun **merkezi**, araç dursa
bile kayar → sahte Hareketli. İki önlem: (a) kutu boyu bir karede çok
değişirse geçmişi sıfırla, (b) hızı adımların **medyanından** al (tek tük bozuk
adım sonucu bozamaz).

### Adım 4 — Otoyol araçlarını ve çıkışı düzelttik
- **Otoyol:** Hızlı araç kareler arası kendi boyu kadar kayınca eski eşleştirme
  kopuyordu. Aracı **kendi hızıyla önce tahmin edip** öyle eşleştirdik →
  otoyol araçları artık kaybolmuyor.
- **Çıkış kilidi:** Kareden çıkarken yarım görünen hareketli araç "kararsız"
  kalıp Sabit'e düşüyordu. 2 kare üst üste Hareketli olan aracı **kilitledik**;
  çıkana kadar Hareketli kalıyor.

### Adım 5 — Giriş/çıkışta doğru kenarı seçtik (asıl kilit çözüm)
Merkez izlemek giriş/çıkışta hata veriyordu (kutu büyüyüp küçülüyor). Bunun
yerine her karede **çerçeve sınırına yapışmayan kenarı** ölçmeye başladık:
araç üstten girerken alt kenarı, alttan çıkarken üst kenarı sabittir ve gerçek
hareketi verir. Böylece hem giren araç hemen yakalanıyor, hem de **duran araç
çıkarken artık sahte Hareketli olmuyor**.

### Adım 6 — Hızlı araçlar için erken karar
Normalde karar için 3 adım (4 kare) bekliyorduk. Ekledik: hız eşiğin çok
üstündeyse (otoyol aracı ~11 m/s) **1 adımda bile hemen Hareketli** de. Bu
yalnızca açık hareketliyi hızlandırır; sınırdaki araçlar güvenlik için yine
bekler. Sonuç: otoyol aracının "Hareketli"ye geçişi **4 kareden 2 kareye** indi.

---

## 4. Nihai yapı (tek cümlede)

**Homografi** (drone hareketi) + **kenar-seçimli kutu adımı** (aracın gerçek
hareketi) + **SLAM ölçeği** (metrik hız) + **medyan / kilit / erken karar / oy
çoğunluğu** (kararlılık).

---

## 5. Sonuç

Tüm oturum (2250 kare) üzerinde, düzeltme turları boyunca gidişat:

| Sürüm | Ne eklendi | HAREKETLI | SABIT | Karar yok "?" |
|---|---|---|---|---|
| v1 | pencere + SLAM-GSD | 118 | 1920 | 1454 |
| v2 | çıkış kilidi + otoyol eşleştirme | 395 | 1920 | 1179 |
| v4 | kenar-seçimli adım | 426 | 2351 | 717 |
| **v5** | **erken karar** | **493** | **2353** | **648** |

- **"Karar yok" %55 azaldı** (1454 → 648): giren araçlar artık "?"de takılı
  kalmıyor, doğru karara geçiyor.
- Park halindeki araçlar SABIT, otoyol araçları HAREKETLI — doğruluk yüksek.
- Otoyolda "Hareketli"ye geçiş gecikmesi medyan **4 → 2 kare**.

Görsel kontrol: `gorev1_hareket_kontrol_TAM_v5.mp4` (tüm oturumun etiketli
videosu).

---

## 6. Dosyalar

- `src/object_detection_model.py` — hareket algoritması
  (`_estimate_vehicle_moving`, `_edge_step`)
- `src/online_alignment.py` — SLAM ölçeği (`scale`, `ensure_scale`)
- `src/orbslam_stream.py` — SLAM'den medyan derinlik okuma
- `gps_denied_localization/.../mono_tuyz_stream.cc` — medyan derinlik çıktısı
