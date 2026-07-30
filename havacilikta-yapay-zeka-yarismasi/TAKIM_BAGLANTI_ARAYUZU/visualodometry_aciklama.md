# visualodometry.py — Baştan Sona Ne Yaptık, Neden Böyle Yaptık

Bu belge `visualodometry.py` dosyasında yapılan her şeyi; başlangıç fikrinden bugünkü haline kadar, basit dille, matematiğiyle, neden diğer yöntemler değil de bunların seçildiğiyle ve ileride ne değiştirilebileceğiyle anlatır.

---

## 1. Amaç Ne?

Yarışmada "Görev 2" diye bir görev var: uçan aracın GPS'i bazen bozuluyor (kayıtlarda `health_status = "0"` olarak işaretleniyor). GPS bozulduğunda aracın konumunu (x, y, z — sırasıyla doğu-batı, kuzey-güney, yükseklik, metre cinsinden) **sadece kameradan gelen görüntüye bakarak** tahmin etmemiz gerekiyor.

Bunu yapan yönteme **görsel odometri** (visual odometry, kısaca VO) denir. "Odometri" kelimesi "yol ölçümü" demektir (arabanın kilometre sayacı gibi) — burada yolu tekerlek yerine kamera görüntüsünden ölçüyoruz.

**Temel fikir çok basit:** İki ardışık kare arasında görüntüdeki noktalar ne kadar kaymışsa, kamera (yani araç) da o kadar hareket etmiştir. Bu kaymayı ölçüp, kamera hareketine çeviriyoruz.

---

## 2. Başlangıç Denemesi: 3 Boyutlu Yöntem (Başarısız)

### Ne yapıyorduk?

İlk yazdığımız kod şu adımları izliyordu:

1. Görüntüde belirgin köşe noktaları bulur — buna **özellik noktası** (feature point) denir. OpenCV'nin `goodFeaturesToTrack` fonksiyonu, kontrast farkının yüksek olduğu (yani "köşe" gibi duran) pikselleri seçer.
2. Bu noktaları bir sonraki karede takip eder. Buna **optik akış** (optical flow) denir — "bu nokta bir sonraki karede nereye gitti?" sorusunun cevabı. Kullandığımız yöntem **Lucas-Kanade** algoritmasıdır (kısaca LK): bir noktanın etrafındaki küçük bir pencereyi (örneğin 21×21 piksel) alıp, bir sonraki karede o pencereye en çok benzeyen yeri arar.
3. Bulunan nokta çiftlerinden (kare-1'deki nokta, kare-2'deki karşılığı) bir **homografi matrisi** (homography matrix) çıkarır. Homografi, "bir düzlemin bir kameradan diğerine nasıl göründüğünü" anlatan 3×3'lük bir matristir.
4. Bu matrisi matematiksel olarak ayrıştırıp (`decomposeHomographyMat` fonksiyonu) kameranın 3 boyutlu dönüşünü (rotasyon) ve hareket yönünü (öteleme) çıkarır.

### Neden başarısız oldu?

Homografi ayrıştırmasının matematiksel bir kusuru var: **aynı görüntü hareketini açıklayan birden fazla "geçerli" 3 boyutlu çözüm olabilir** (genelde 2-4 tane). Kod bunlardan hangisinin doğru olduğunu tam güvenilir seçemiyordu; bazı karelerde yanlışını seçince pozisyon tahmini büyük bir sıçrama yapıyordu.

**Sonuç:** Kalibrasyon bölgesinde (üzerinde test ettiğimiz, gerçek konumu bildiğimiz kareler) bile 25-40 metre hata; 200 kare ileri tahmin edince 154 metre hata. Kullanılamaz seviyedeydi.

---

## 3. Dönüş Noktası: "Nadir Kamera" Fikri

Sen kritik bir gözlem yaptın: **drone kamerası neredeyse her zaman tam aşağı bakıyor.** Buna havacılıkta **nadir** (nadir açı = 90 derece, tam dikey aşağı) denir.

### Bunun matematiksel önemi

Kamera tam aşağı baktığında, altındaki zemin (yaklaşık olarak) düz bir yüzey gibi davranır. Bu durumda iki kare arasındaki görüntü hareketi, karmaşık bir 3 boyutlu dönüşüm değil, çok daha basit bir **2 boyutlu benzerlik dönüşümü** (similarity transform) olur. Benzerlik dönüşümü, bir şeklin:

- **kayması** (öteleme — x,y yönünde kayma),
- **dönmesi** (rotasyon — aracın kendi ekseni etrafında dönmesi, "yaw"),
- **büyüyüp küçülmesi** (ölçek — irtifa değişince görüntüdeki her şey aynı oranda büyür/küçülür)

dışında şeklinin bozulmadığı bir dönüşümdür. Formülü şöyledir:

```
[x']   [ s·cos(θ)   s·sin(θ) ] [x]   [tx]
[y'] = [-s·sin(θ)   s·cos(θ) ] [y] + [ty]
```

Burada `s` ölçek (büyüme/küçülme oranı), `θ` (theta) dönüş açısı, `tx,ty` kayma miktarı. Sadece **4 bilinmeyen** var (s, θ, tx, ty) — 3 boyutlu homografinin 8 bilinmeyenine göre çok daha az, çok daha kararlı bir problem.

### Yeni yöntem

- 3B homografi ayrıştırması yerine OpenCV'nin `cv2.estimateAffinePartial2D` fonksiyonunu kullandık — bu fonksiyon iki nokta kümesi arasındaki en iyi 2B benzerlik dönüşümünü hesaplar.
- **RANSAC** kullandık (Random Sample Consensus — "rastgele örnekle, uzlaşıyı bul" demek). Nokta eşleşmelerinin bir kısmı yanlış olabilir (örneğin hareket eden bir araç, gölge, yanlış eşleşen bir köşe). RANSAC, rastgele küçük gruplar seçip her grup için bir çözüm dener, en çok noktanın "uyduğu" (inlier — çözümle tutarlı nokta) çözümü seçer, uymayanları (outlier — aykırı değer) göz ardı eder.
- z (yükseklik) bilgisini bu görüntü tabanlı yöntemden tamamen çıkardık — sebebini birazdan (bölüm 6) detaylı anlatıyoruz.

Bu değişiklik büyük bir iyileşme getirdi ama üstüne iki kritik hata daha bulup düzelttik.

---

## 4. İki Kritik Hata ve Düzeltmeleri

### Hata 1: Ölçüm Noktası Yanlış Yerdeydi (en büyük tekil düzeltme)

`estimateAffinePartial2D` bize `tx, ty` (kayma miktarı) verirken bunu görüntünün **sol-üst köşesindeki (0,0) noktasının** nereye gittiğine göre hesaplıyor. Ama görüntü dönüşü fiziksel olarak kameranın **kendi merkezi** (principal point, yani optik eksenin görüntüyü kestiği nokta — kabaca görüntünün ortası) etrafında olur.

Bunun sonucu: kamera sadece hafifçe dönse bile (θ küçük olsa bile), köşedeki (0,0) noktası **merkeze göre çok uzak olduğu için** büyük bir yay çizerek hareket ediyormuş gibi görünür. Bu sahte, dönüşten kaynaklanan büyük bir "kayma" gibi ölçülüyordu — gerçek kamera hareketiyle hiç ilgisi olmayan bir hata.

**Matematiği:** Eğer kamera merkezi `(cx, cy)` ise ve dönüşüm matrisi `M` ise, doğru öteleme şu şekilde hesaplanır:

```
gerçek_oteleme = M · [cx, cy, 1]ᵀ − [cx, cy]ᵀ
```

Yani "merkez noktası M dönüşümünden geçtikten sonra nereye gitti, oradan merkezin eski yerini çıkar" — bu bize merkezin gerçek kaymasını verir. Köşe yerine bunu kullanınca hata **40 metreden 2 metreye** düştü.

### Hata 2: Dönüş Yönü Ters Uygulanıyordu

Araç kendi etrafında döndükçe (`heading` — "baktığı yön", kümülatif/toplam dönüş açısı), her karedeki küçük dönüşü (`dtheta`) bu toplam açıya doğru işaretle eklememiz gerekiyordu. İlk halde işaret (yön) tersti; bunu deneyerek (iki işareti de deneyip GT — gerçek konum verisiyle karşılaştırarak) doğru olanı bulup düzelttik.

---

## 5. Monoküler Ölçek Belirsizliği ve Umeyama Hizalaması

### Problem: Tek Kameranın Metre Bilmemesi

Tek kameradan (buna **monoküler** denir — "tek gözlü") elde ettiğimiz hareket sadece görüntü pikseli cinsindendir; "5 piksel kaydı" bilgisinden "bu 5 piksel kaç metreye denk gelir" bilgisini çıkaramayız — çünkü aynı piksel kayması hem "yakın bir cisim az hareket etti" hem de "uzak bir cisim çok hareket etti" ile açıklanabilir. Buna **monoküler ölçek belirsizliği** denir.

Ayrıca VO'nun kendi "başlangıç yönü" (kare 0'da heading=0 kabul ettik) ile gerçek dünyanın kuzey/doğu ekseni aynı olmak zorunda değil — aralarında bilinmeyen bir dönüş de var.

### Çözüm: Umeyama Hizalaması

Elimizde bir miktar kare için **gerçek konum** (GT — Ground Truth, "yer gerçeği") olduğundan (GPS sağlıklıyken), VO'nun ürettiği (ölçeksiz, keyfi-yönlü) yörüngeyi bu gerçek konumlara **en iyi oturacak şekilde** dönüştürebiliriz. Bunun için **Umeyama yöntemini** (1991 yılında S. Umeyama'nın yayınladığı kapalı-form/analitik çözüm) kullandık.

Umeyama, iki nokta kümesi arasında en iyi **benzerlik dönüşümünü** (ölçek `s`, dönüş `R`, öteleme `t`) bulur:

```
gerçek_nokta ≈ s · R · vo_nokta + t
```

**Matematiği adım adım:**

1. İki kümenin ortalamasını (centroid — ağırlık merkezi) çıkar, merkezle:
   `src_c = vo_noktalari − ortalama(vo_noktalari)`
   `dst_c = gerçek_noktalar − ortalama(gerçek_noktalar)`
2. **Çapraz kovaryans matrisi**ni hesapla — bu, iki kümenin birlikte nasıl değiştiğini özetleyen bir matris:
   `cov = (dst_c^T · src_c) / n`
3. Bu matrisin **SVD** (Singular Value Decomposition — Tekil Değer Ayrışımı, her matrisi üç basit matrisin çarpımına ayıran bir yöntem) ile çöz: `cov = U · D · Vᵀ`
4. En iyi dönüş matrisi: `R = U · S · Vᵀ` (S burada, eğer küme "ayna görüntüsü" gibi ters çevrilmişse bunu düzeltmek için kullanılan bir düzeltme matrisi)
5. En iyi ölçek: `s = iz(D·S) / vo_noktalarının_varyansı` (iz = matrisin köşegen toplamı)
6. En iyi öteleme: `t = gerçek_ortalama − s·R·vo_ortalama`

Bu yöntem, iki nokta bulutunu üst üste bindirmenin **matematiksel olarak en iyi (en az kareler hatası veren)** yoludur.

### Neden Umeyama, neden değil de basit bir "iki nokta al, oran kur" yöntemi?

Basit yöntem (örneğin sadece ilk ve son noktanın mesafe oranını almak) tek bir örneğe dayanır ve gürültüye (ölçüm hatasına) çok duyarlıdır. Umeyama **bütün kalibrasyon noktalarını birlikte** kullanarak en iyi ortalama çözümü bulur — gürültüye karşı çok daha dayanıklıdır. Ayrıca döndürme + ölçek + kaymayı **aynı anda, birbirini etkilemeden** çözer.

### Ekstra ince ayar: Yansıma (flip) arama

Görüntü koordinat sisteminde y ekseni aşağı doğru artar, dünya koordinat sisteminde ise yukarı doğru artabilir (haritalarda kuzey yukarı gibi). Bu bir **ayna görüntüsü** (yansıma) farkı yaratabilir. Kodumuzda (`fit_similarity_2d` fonksiyonu) hem normal hem y-ekseni ters çevrilmiş hâli deneyip, hangisi gerçek veriye daha iyi oturuyorsa onu seçiyoruz.

---

## 6. Yedi Adımlık Sistematik İyileştirme Süreci

Buradan sonra kodu iyileştirmek için 7 fikri **sırayla, ölçerek, işe yaramazsa geri alarak** test ettik. Her denemenin sonucunu **RMSE** (Root Mean Square Error — Hata Kareleri Ortalamasının Karekökü, tahminlerin gerçek değerden ortalama ne kadar uzaklaştığını metre cinsinden özetleyen standart bir hata ölçüsü) ile ölçtük.

Test yöntemi hep aynıydı: elimizdeki karelerin ilk yarısını (kare 0-199) **kalibrasyon** için (gerçek konumu görüp ölçek/hizalamayı ayarlamak), geri kalanını (kare 200-399) **kör test** için kullandık — yani o bölgenin gerçek konumunu kodun görmesine izin vermeden, sadece görüntüden tahmin ettirip sonra gerçekle karşılaştırdık. Bu, "ezberleme" (overfitting — modelin gördüğü veriye aşırı uyup görmediği veriyi kötü tahmin etmesi) riskine karşı dürüst bir test.

### Adım 1 — Sistematik Dönüş Sapması Düzeltmesi ✅ (KABUL EDİLDİ)

**Ne bulduk:** Kod her karede aracın dönüşünü ölçerken, çok küçük ama **sürekli aynı yönde tekrar eden** bir hata yapıyordu: kare başına ortalama +0.04 derece fazla dönüş ölçüyordu.

**Neden önemli:** Bu dönüş değeri **kümülatif** olarak toplanıyor (her karede bir öncekine ekleniyor). Küçük bir hata bile 200 kare boyunca toplanınca **8 dereceye** kadar büyüyor — ve yön 8 derece yanlışsa, o yönde ilerleyen her adım da yanlış yöne kayar.

**Nasıl bulduk:** Kalibrasyon bölgesinin bir kısmını (0-149) kullanıp, kalan kısmında (150-199) VO'nun ölçtüğü dönüşü, gerçek GPS verisinden hesaplanan gerçek dönüşle karşılaştırdık. Aradaki ortalama fark bize sapma değerini (`YAW_BIAS_RAD = 0.00070` radyan/kare) verdi.

**Düzeltme:** Her karede ölçülen dönüşten bu sabit miktarı çıkarttık.

**Sonuç:** Kör test hatası 6.40 metreden 3.91 metreye düştü.

### Adım 2 — Yükseklik (z) Tahmini: Başarısız Deneme + Güvenli Çözüm

**Denenen (BAŞARISIZ):** Görüntüdeki büyüme/küçülme oranını (`scale`) kare kare çarparak (biriktirerek) bir "göreli yükseklik" sinyali oluşturmayı, sonra bunu gerçek GPS yüksekliğine oturtmayı denedik. Bu, x/y'de kullandığımız yöntemin z'ye uygulanmış hali.

**Neden başarısız oldu:** Test verisinde gerçek yükseklik değişimi çok küçüktü (yaklaşık 200 karede 0.77 metre). Görüntüdeki karşılığı (`scale`) kare başına sadece **%0.1-0.3** büyüklüğünde bir değişim — bu, ölçüm gürültüsüyle (RANSAC/takip hatasıyla) aynı büyüklükte, yani **sinyal gürültüde kayboluyor**. Üstelik bu küçük değerleri 200 kere üst üste **çarparak** biriktirince (`zrel[n] = zrel[n-1] / scale[n]`), gürültü de birikip büyüyor (buna **rastgele yürüyüş** — random walk denir: her adımda rastgele küçük bir sapma eklenirse, toplam sapma zamanla büyür). Gerçek yükseklik GT'sine bu gürültülü sinyali oturtmaya çalışan regresyon (eğri uydurma) aşırı büyük bir katsayı üretti (13'e yakın) — yani ufacık bir gürültü kırıntısı bile metreye çevrilirken 13 kat büyütülüyordu.

**Sonuç (test edildi, atıldı):** Bu yöntemi x/y hesabına karıştırınca kalibrasyon hatası 37.8 metreye, kör test hatası 77.9 metreye çıktı — X/Y'yi de mahvetti. Kod tabanında bu **kesinlikle kullanılmıyor** (yorum satırlarında "REDDEDİLDİ" diye işaretli).

**Kabul edilen çözüm — Dikey Hız Ekstrapolasyonu:** Görüntüye hiç güvenmeden, sadece **son bilinen gerçek yükseklik** ve **kalibrasyon bölgesinden ölçülen sabit dikey hız** ile düz bir çizgi halinde ileri tahmin ediyoruz:

```
z_tahmin(kare) = son_bilinen_z + dikey_hız × (kare − son_bilinen_kare)
```

Dikey hız, kalibrasyon GT'sine bir doğru uydurarak (`numpy.polyfit` fonksiyonu — en küçük kareler yöntemiyle doğru denklemi bulma) hesaplanıyor. Bu, matematiksel olarak "eğim" (slope) çıkarmaktan başka bir şey değil.

**Sonuç:** z için test hatası sadece 0.17 metre — hem sıfır döndürmekten (0.46 m) hem de görüntü tabanlı yöntemden (0.60 m, ayrı bir deneyde ölçüldü) çok daha iyi.

### Adım 3 — Başarısız Karede "Dur" Değil "Devam Et" ✅ (KABUL EDİLDİ)

**Problem:** Bazı karelerde nokta takibi başarısız olabilir (görüntü bulanık, çok az nokta bulunmuş, vs.). Eski kodda bu durumda pozisyon güncellenmiyordu — yani "araç durdu" gibi davranıyorduk, oysa araç aslında hareket etmeye devam ediyor, biz sadece göremiyoruz.

**Çözüm:** Son bilinen hareketi (`last_step`) kısa bir süre (en fazla 5 kare, `MAX_FALLBACK_FRAMES`) daha uygulamaya devam ediyoruz — **sabit hız varsayımı**. 5 kareden uzun süre başarısız olursa, tahmin etmeyi bırakıp yerinde bekletiyoruz (rastgele yanlış yöne kaymaktansa yerinde durmak daha güvenli).

**Not:** Bu veri setinde hiç başarısız kare olmadığı için sayısal olarak bir fark yaratmadı ama gerçek yarışma sırasında (motion blur, ışık değişimi gibi durumlarda) sıçramaları önleyecek bir güvenlik önlemi.

### Adım 4 — "Çapa Kare" (Anchor) Yöntemi ❌ (REDDEDİLDİ)

**Fikir:** Normalde her karede yeni noktalar buluyoruz ve sadece bir önceki kareyle karşılaştırıyoruz (buna **ardışık kare** / frame-to-frame yöntemi denir). Bunun yerine, belirli bir kareyi ("çapa" / anchor kare) sabit referans alıp, aynı noktaları bu çapadan itibaren uzun süre (25-60 kare) kesintisiz takip edip çapa ile şimdiki kare arasındaki dönüşümü **tek seferde** hesaplamayı denedik.

**Mantık neden cazipti:** Her ardışık kare karşılaştırması kendi küçük hesaplama gürültüsünü taşıyor; bunları K defa tekrarlarsak gürültü birikir. Tek bir uzun-mesafe hesaplama yapılırsa, ara gürültüler eklenmez.

**Neden başarısız oldu:** Optik akış (Lucas-Kanade) yöntemi, bir noktayı **uzun mesafe** boyunca (25+ kare, 500-700 piksel birikmiş hareket) takip etmekte zorlanıyor — nokta çevresindeki görüntü deseni zamanla değiştiği için (aydınlatma, açı, perspektif farkları) takip **kayıyor/sürükleniyor**. RANSAC bu kaymayı tam temizleyemedi.

**Sonuç (test edildi, atıldı):** Kör test hatası 34.6 metreden 70.8 metreye çıktı — iki katın üzerinde kötüleşme. Bu yöntemin düzgün çalışması için ardışık kareler arasında düz optik akış yerine **öğrenilmiş bir eşleştirici** (örneğin SuperPoint+LightGlue gibi yapay-zeka tabanlı nokta eşleştirme modelleri) kullanmak gerekirdi — bu, kapsam dışı bırakıldı (aşağıda "gelecekte neler denenebilir" bölümünde tekrar geçiyor).

### Adım 5 — Izgara (Grid) Tabanlı Nokta Dağılımı ✅ (KABUL EDİLDİ — en büyük tekil kazanım)

**Problem:** Görüntüde tek bir `goodFeaturesToTrack` çağrısı yapınca, bulunan noktalar genelde görüntünün dokulu/kontrastlı bölgelerinde (örneğin bir bina kenarı, yol çizgisi) **kümeleniyor**. Eğer bütün noktalar görüntünün küçük bir köşesinde toplanırsa, dönüş ve öteleme hesaplaması için kullanılan "kaldıraç" (leverage — noktaların birbirinden ne kadar uzak/dağınık olduğu, bu da hesabın ne kadar hassas olacağını belirler) küçük kalır; özellikle dönüş açısını doğru kestirmek için noktaların görüntüye **yayılmış** olması gerekir (düşünsen: bir masayı döndürüyorsan, masanın sadece bir köşesindeki iki noktaya bakarak dönüş açısını kestirmek, karşılıklı iki köşesindeki noktalara bakmaktan çok daha az hassastır).

**Çözüm:** Görüntüyü 8×6'lık bir **ızgaraya** (grid) böldük (48 hücre). Her hücreden en fazla 45 nokta seçiliyor (`GRID_GFTT_PARAMS`), böylece noktalar görüntünün her yerine zorla dağıtılıyor. Ayrıca hücre içinde noktalar arasındaki minimum mesafeyi de artırdık (8 pikselden 25 piksele) — bu da aynı bölgede noktaların birbirine çok yakın, neredeyse aynı bilgiyi taşıyan yığınlar oluşturmasını engelliyor.

**Sonuç:** Kalibrasyon hatası 1.44 → 1.26 metre, kör test hatası **34.6 → 24.25 metre** (yaklaşık %30 iyileşme). Tüm süreçteki en büyük tekil kazanım bu oldu.

### Adım 6 — Kalman Filtresi ile Yumuşatma ❌ (REDDEDİLDİ)

**Fikir:** **Kalman filtresi**, geçmiş tahminlerle yeni ölçümü istatistiksel olarak "akıllıca" karıştıran, çok yaygın kullanılan bir yöntemdir (uydu navigasyonundan robotiğe kadar her yerde kullanılır). Temel fikri: "önceki tahminime biraz güven, yeni ölçüme de biraz güven, ikisini ölçüm güvenilirliğine göre ağırlıklı ortala." Sabit hız modeliyle (her karede hızın çok değişmeyeceği varsayımıyla) kare-başı hareketi yumuşatmayı denedik.

**Neden başarısız/nötr oldu:** Geniş bir parametre taraması yaptık (filtrenin ne kadar "geçmişe güveneceğini" belirleyen ayarları değiştirerek). Hiçbir ayar temel çizgiyi geçemedi — en iyi ihtimalle filtreyi neredeyse etkisizleştirince (yani ölçüme tam güvenince) aynı sonuç çıktı, aksi halde kötüleşti. Sebep basit: Kalman filtresi **tekil sıçramaları/aykırı değerleri** temizlemekte iyidir, ama bu veri setinde zaten hiç başarısız kare veya aşırı sıçrama yoktu (temizlenecek bir "kir" yoktu). Ayrıca filtre, **sistematik** (yani hep aynı yönde tekrar eden) hataya hiç çare değildir — bizim asıl sorunumuz (Adım 1'de bulduğumuz yaw sapması gibi) zaten sistematikti, rastgele değildi.

**Sonuç (test edildi, atıldı):** Hiçbir iyileşme yok; kod tabanında kullanılmıyor.

### Adım 7 — Her Adımda Ölç, Doğrula, Gerekirse Geri Al

Bu aslında ayrı bir teknik adım değil, yukarıdaki 6 adımın **hepsinde uyguladığımız disiplin**: her değişiklikten sonra aynı kalibrasyon/kör-test bölünmesiyle yeniden ölçtük. İyileşme varsa tuttuk (Adım 1, 3, 5), kötüleşme varsa nedenini anlayıp geri aldık (Adım 2'nin "dinamik ölçek" kısmı, Adım 4, Adım 6). Bu, "büyük ihtimalle iyi olur" diye körü körüne özellik eklememizi engelledi.

---

## 7. Şu Anki Kod Ne Yapıyor? (Adım Adım Özet)

1. **Görüntü hazırlığı:** Her kare griye çevrilir.
2. **Nokta bulma:** Görüntü 8×6 ızgaraya bölünür, her hücreden en fazla 45 köşe noktası bulunur (`_detect_grid`).
3. **Takip:** Bu noktalar Lucas-Kanade optik akışla bir sonraki karede aranır. **İleri-geri tutarlılık kontrolü** yapılır: bir nokta kare A'dan B'ye, sonra B'den tekrar A'ya takip edildiğinde başladığı yere yakın bir yere dönmüyorsa (1 pikselden fazla fark varsa), o nokta güvenilmez sayılıp atılır.
4. **Lens bozulması düzeltme:** Kameranın lensinden kaynaklanan görüntü eğriliği (`undistortPoints`) matematiksel olarak düzeltilir — kamera kalibrasyon dosyasındaki bozulma katsayıları (`k1, k2`) kullanılır.
5. **Dönüşüm hesabı:** Kalan güvenilir nokta çiftlerinden RANSAC ile en iyi 2B benzerlik dönüşümü (`estimateAffinePartial2D`) çıkarılır — dönüş açısı, ölçek, öteleme.
6. **Merkez düzeltmesi:** Öteleme, köşe yerine görüntü merkezine göre yeniden hesaplanır (Bölüm 4, Hata 1).
7. **Sapma düzeltmesi:** Ölçülen dönüşten sabit yaw-sapması çıkarılır (Bölüm 6, Adım 1).
8. **Konum güncelleme:** Bu karenin hareketi, o ana kadar birikmiş toplam dönüşe göre döndürülüp konuma eklenir (`self.pos += Rh @ (-t)`).
9. **(Ayrı olarak, GT varsa) Kalibrasyon:** Bir dizi karenin gerçek konumu biliniyorsa, VO'nun ürettiği (ölçeksiz, keyfi yönlü) yörünge Umeyama ile gerçek dünyaya oturtulur.
10. **z tahmini:** Görüntüden değil, ayrı ve bağımsız olarak, son bilinen yükseklik + ölçülen dikey hızla düz bir tahmin üretilir.

---

## 8. Neden Bu Yöntemler, Neden Alternatifleri Değil?

### x, y için: neden "2B benzerlik + RANSAC", neden değil...

- **Essential Matrix / Fundamental Matrix yöntemi** (klasik iki-görüntü geometrisi, genel amaçlı stereo/hareket kestirimi için kullanılır): Bu yöntemler kameranın **serbestçe** (her yöne) hareket ettiği genel durum için tasarlanmıştır ve homografi gibi yine çoklu-çözüm belirsizliği taşır. Bizim özel durumumuzda (nadir kamera, düz zemin) bu genel-amaçlı karmaşıklığa gerek yok; 2B benzerlik daha basit VE daha kararlı.
- **PnP (Perspective-n-Point)** yöntemi: Sahnedeki noktaların 3B konumunu **önceden bilmemizi** gerektirir (örneğin bilinen yer işaretleri). Bizim elimizde böyle bir harita yok, bu yüzden uygulanamaz.
- **Öğrenilmiş (yapay zeka tabanlı) yöntemler** (RAFT — yoğun optik akış ağı; SuperPoint+LightGlue — öğrenilmiş nokta bulma+eşleştirme; DROID-SLAM — uçtan uca öğrenilmiş konum kestirimi): Bunlar özellikle **dokusuz, tekrarlayan** görüntülerde (tarla, yol, çatı gibi) klasik yöntemlerden daha güçlü olabilir ve Adım 4'te bahsedilen "uzun mesafeli takip" sorununu çözebilir. Ama ek bağımlılık (GPU, model indirme, entegrasyon emeği) gerektirdiği için bu oturumda denenmedi — **gelecekte denenebilecek en güçlü aday budur.**

### z için: neden "dikey hız ekstrapolasyonu", neden değil "görüntüden hesaplama"

Bunu Bölüm 6 Adım 2'de detaylı anlattık: görüntüdeki yükseklik sinyali (büyüme/küçülme oranı) bu veri setinde gürültü seviyesinde kalıyor, güvenilir değil. Fiziksel bir öncül (son bilinen hız devam ediyor varsayımı) kısa vadede çok daha güvenilir çıktı.

### Kalibrasyon için: neden Umeyama, neden değil basit oran/ortalama

Umeyama tüm noktaları **birlikte, en küçük kareler anlamında en iyi** şekilde hizalar; gürültüye karşı çok daha dayanıklıdır ve döndürme+ölçek+kaymayı birbirini bozmadan aynı anda çözer. Basit "iki nokta arası oran" yöntemleri tek bir örneğe dayandığı için çok kırılgandır.

---

## 9. Ne Değiştirilse Daha İyi Olabilir? (Gelecek Fikirler)

Aşağıdakiler test edilmedi veya kapsam dışı bırakıldı, ama mantıklı sıradaki adımlar:

1. **Öğrenilmiş nokta eşleştirici (SuperPoint + LightGlue veya benzeri) ile Adım 4'ü (çapa/anchor yöntemi) yeniden dene.** Adım 4'ün başarısız olma sebebi düz optik akışın uzun mesafede kayması idi; öğrenilmiş bir eşleştirici bu sorunu çözüp, ardışık-kare gürültüsünün birikmesini gerçekten azaltabilir.

2. **Kayan pencere (sliding window) Umeyama.** Şu an tek bir sabit kalibrasyon bölgesi kullanılıyor. Bunun yerine, GPS sağlıklıyken sürekli güncellenen, "son N kareye göre" yeniden hesaplanan bir kalibrasyon, ölçek/hizalamadaki yavaş kaymaları (drift) daha iyi takip edebilir.

3. **Barometre veya IMU (atalet ölçüm birimi — ivme/dönüş sensörü) füzyonu.** Eğer araçta barometre (basınç sensörü, yüksekliği doğrudan ölçer) veya IMU varsa, z tahmini için görüntü yerine (ya da görüntüyle birlikte) bunlar kullanılmalı — çok daha güvenilir olurdu. Görsel veriyle bu sensörleri birleştirmeye **sensör füzyonu** denir, genelde bir **Genişletilmiş Kalman Filtresi (EKF)** ile yapılır (Adım 6'da denediğimiz basit KF'den farklı olarak, burada filtrenin temizleyeceği gerçek, farklı-kaynaklı ölçümler olurdu — o yüzden burada işe yarama ihtimali çok daha yüksek).

4. **Görev 3'teki referans nesnelerini "kısmi çapa" olarak kullanmak.** Daha önce konuştuğumuz gibi bu nesnelerin dünya koordinatı yok, ama sabit bir noktayı uzun süre görünür tutarak o pencere boyunca drift'in büyümesini yavaşlatabilirler (drifti sıfırlamaz ama frenler).

5. **Essential matrix ile çapraz kontrol.** 2B benzerlik varsayımı (nadir düzlem) küçük bir açı hatasına (kamera tam 90 derece değil de 85 derece bakıyorsa) karşı ne kadar dayanıklı, bunu essential-matrix tabanlı bir tahminle karşılaştırıp sapma varsa tespit etmek, modelin ne zaman güvenilmez hale geldiğini önceden haber verebilir.

6. **Hareket modeli ekleme (ivme/sabit hız varsayımı, x/y için de).** Adım 6'daki Kalman filtresi burada işe yaramadı çünkü temizlenecek gürültü yoktu; ama gerçek yarışma videosunda daha gürültülü kareler (motion blur, ışık değişimi) çıkarsa, aynı fikir orada işe yarayabilir — bu veri setinde "hayır" demek, her veri setinde "hayır" demek değildir.

---

## 10. Terimler Sözlüğü (Hızlı Bakış)

| Terim | Anlamı |
|---|---|
| Görsel Odometri (VO) | Kamera görüntüsünden hareket/konum ölçme |
| Nadir | Kameranın tam dikey aşağı baktığı açı |
| Optik Akış | Bir noktanın kareler arasında nereye kaydığının ölçümü |
| Lucas-Kanade (LK) | Optik akış hesaplayan yaygın bir algoritma |
| Homografi | İki görüntü arasındaki düzlemsel dönüşümü anlatan 3×3 matris |
| Benzerlik Dönüşümü | Şekli bozmadan kaydırma+döndürme+ölçekleme |
| RANSAC | Aykırı (yanlış) verileri eleyip en tutarlı çözümü bulan yöntem |
| Inlier / Outlier | Çözümle tutarlı / tutarsız veri noktası |
| Monoküler | Tek kameralı |
| Ölçek Belirsizliği | Tek kameranın "kaç metre" olduğunu bilememesi sorunu |
| Umeyama Yöntemi | İki nokta kümesini en iyi hizalayan ölçek+dönüş+kayma bulma yöntemi |
| SVD | Bir matrisi üç basit matrisin çarpımına ayıran matematik yöntemi |
| Centroid | Bir nokta kümesinin ağırlık merkezi (ortalaması) |
| Kovaryans | İki değişkenin birlikte nasıl değiştiğini ölçen istatistik |
| RMSE | Tahmin hatalarının ortalama büyüklüğünü özetleyen standart ölçü (metre cinsinden) |
| Drift | Zamanla/kareyle birikerek büyüyen hata |
| Bias (Sapma) | Rastgele değil, hep aynı yönde tekrar eden sistematik hata |
| Kalman Filtresi | Geçmiş tahmin ile yeni ölçümü güvenilirliklerine göre karıştıran filtre |
| GSD | Görüntüdeki bir pikselin gerçek dünyada kaç metreye denk geldiği |
| GT (Ground Truth) | Gerçek/doğru değer (burada: GPS'ten gelen gerçek konum) |
| Overfitting (Ezberleme) | Modelin gördüğü veriye aşırı uyup görmediğini kötü tahmin etmesi |
