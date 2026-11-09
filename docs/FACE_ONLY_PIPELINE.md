# Retake — Face-only identity transfer pipeline

Tarih: 9 Kasım 2026  
Hedef: iPhone 17 Pro Max + modern web tarayıcıları  
Durum: Web üzerinde NVIDIA FLUX.1-Kontext-dev ile deneysel görsel düzenleme ve eliptik face-only compositing eklendi. Gerçek kimlik korumalı Face Swap modeli ve kimlik doğrulaması hâlâ yoktur.

## 1. Kesin ürün kuralları

1. Kamera yoktur. Canlı kamera, video, kamera izni veya kamera akışı API'si eklenmez. Girdi yalnızca galeriden/dosyadan seçilmiş sabit fotoğraftır.
2. Kullanıcı hedef fotoğrafı seçer. Birden çok yüz varsa değiştirilecek yüzü kendisi seçer.
3. Yalnızca seçilen yüzün pikselleri değiştirilir. Tüm fotoğrafı yeniden üretmek kabul edilmez.
4. Kimlik havuzundaki 20 fotoğraf aynı kişiye ait olmalıdır. Farklı kişilere ait referanslar tek kimlik profiline karıştırılmaz.
5. Hedefin pozu ve baş açısı hedef fotoğraftan; kimlik özellikleri 20 referanstan alınır.
6. Göz kapalıysa uygun açık göz referansından doğal göz açıklığı üretilebilir. Yüz ifadesi yalnızca kimlik aktarımını/doğallığı iyileştirmek için gerektiğinde ve yalnızca yüz crop'unda düzeltilir.
7. Saç, saç çizgisi, arka plan, boyun/gövde, kıyafet, aksesuarlar, eller, diğer kişiler ve diğer yüzler korunur.
8. Sonuç orijinal fotoğrafın tam çözünürlüğüne geri yerleştirilir. Düzenleme maskesi dışındaki her piksel kaynak fotoğrafla aynı kalmalıdır.
9. Model/maske/kimlik kontrolü başarısız olursa işlem sonucu üretilmiş gibi gösterilmez.

## 2. 20 referanstan tek kimlik profili

Referans fotoğrafları yalnızca tek yüz içerme kontrolünden geçmekle kalmamalı; gerçek kimlik encoder'ı eklendiğinde aynı kişiye ait olup olmadıkları da karşılaştırılmalıdır.

### Ön işleme

- EXIF yönünü uygula, doğru renk uzayına dönüştür.
- Yüzü tespit et, göz/landmark ile hizala ve yüz crop'unu al.
- Her fotoğraf için blur, yüz boyutu, yaw/pitch/roll, göz görünürlüğü, örtülme ve ışık kalitesini ölç.
- Çok bulanık, aşırı profilden, ağır örtülü veya yüzü küçük görüntüleri kimlik profiline düşük ağırlıkla dahil et.

### Kimlik birleştirme

- Her referans için normalize edilmiş kimlik embedding'i çıkar.
- Kalite skoruyla ağırlıklandır; aykırı embedding'leri (yanlış kişiye ait veya çok kötü fotoğraf) robust medoid/trimmed-mean yaklaşımıyla dışarıda tut.
- Ortak identity prototype ile birlikte en iyi birkaç referansı da sakla. Tek bir ortalama embedding ince ayrıntıları silebileceğinden, inference sırasında seçili referans örnekleri kimlik conditioning'e eklenebilir.
- Hedef yüz pozuna en yakın referans(lar)ı seç; fakat hedef pozunu referans pozuna zorla dönüştürme.
- Embedding'ler biyometrik veri kabul edilir: public GitHub'a yazılmaz, yerel özel depoda tutulur.

**Mevcut sınırlama:** Bugünkü uygulama tek fotoğrafta tek yüz bulmayı doğrular. 20 fotoğrafın aynı kişiye ait olduğunu henüz doğrulayamaz; bunu ancak lisansı onaylanmış identity encoder entegre edildiğinde iddia edebilir.

## 3. Hedef yüz crop ve koşullandırma

- Kullanıcının seçtiği yüzü yüksek çözünürlüklü orijinal fotoğrafta yeniden tespit et.
- Crop'u yüz sınırından biraz daha geniş al: alın, iki kaş/göz, yanaklar, burun, ağız ve çene dahil olmalı. Crop gereksiz saç/arka plan alanı içermemeli.
- Pose/landmark, gaze/eye-open, ağız açıklığı ve ifade ölçümlerini hedef yüz üzerinden çıkar.
- Yüz kimliği 20 referanstan; poz/ışık/renk/kompozisyon hedef fotoğraftan condition edilir.
- Model sadece crop üzerinde çalışır. Tam fotoğrafı üretim girdisi olarak yeniden sentezleyip eski sahnenin üzerine yazamaz.

## 4. Göz ve ifade retargeting

Öncelik doğal kimlik aktarımıdır; ifade değişikliği sınırsız bir "beautify" filtresi değildir.

- Hedefte gözler doğal biçimde açıksa göz şeklini koru.
- Gözler kapalı veya belirgin biçimde kısılmışsa aynı kişinin açık göz referanslarından iris/kapak geometrisini kullanarak doğal açıklığa getir.
- İki gözün bakış yönü, kapak açıklığı ve göz bebeği konumu birlikte değerlendirilir; gözleri yapay şekilde fazla açma.
- Ağız, dudak, kaş ve çene ifadesi hedefle uyumlu tutulur. Model kimlik benzerliğini korumak için düzeltme gerektirirse düzeltme yalnızca yüz içindedir.
- Yalnızca açık göz referansı bulunmuyorsa göz açma zorlanmaz; güvenilirlik düşükse kullanıcıya sonuç üretilmez.
- LivePortrait eye/lip retargeting bu alt problem için araştırma adayıdır, ancak tek başına kimlik transferi motoru değildir.

## 5. Face parsing ve occlusion mask

Sınır kutusu tek başına yeterli değildir. Piksel-seviyesinde semantik yüz maskesi ve örtücü algısı gerekir.

Mevcut NVIDIA deneysel web akışında maske, hedef yüz landmark sınırından türetilen yumuşak elipstir. Semantik face parser/occlusion detector henüz bağlanmadı.

Maske yaklaşık yüz derisi, göz, burun, ağız ve çene çevresini kapsar. Elips saç tellerini, gözlük camını/çerçevesini veya yüzü örten eli semantik olarak ayırt edemez. Bu nedenle maske sınırı kontrollü tutulur ve sonuç kullanıcı tarafından incelenir.

Korunması hedeflenen bölgeler:
- saç, saç çizgisinin yüz dışında kalan bölümü ve saç telleri;
- gözlük çerçevesi/camı, el, mikrofon veya yüzün önünden geçen nesneler;
- boyun, kıyafet, beden, arka plan;
- hedefteki diğer kişiler ve diğer yüzler.

Maske kenarında feather işlemi yalnızca maskenin içinde uygulanır. Alpha 0 olan piksel orijinalden byte-byte kopyalanır.

## 6. Birleştirme ve orijinal fotoğrafın korunması

1. Generator sadece seçilen yüz crop'unu döndürür.
2. Üretilen crop, hedef crop ile geometrik olarak hizalanır.
3. Renk/ışık uyumu lokal olarak ve maskenin içinde yapılır.
4. Feather/seamless compositing yüz maskesinin iç sınırında uygulanır.
5. Patch, orijinal tam çözünürlüklü fotoğrafın aynı koordinatına yapıştırılır.
6. Çıktı boyutu orijinalle aynı kalır; JPEG/PNG olarak dışa aktarılır.

**Sert test:** mask == 0 olan pikseller için output == input olmalıdır. Yalnızca maskelenen yüzdeki piksel farkı kabul edilir. Dosya metadata'sı dışa aktarımda uygun şekilde korunur/yenilenir.

## 7. Kalite kapıları

Çıktı kaydedilmeden önce:

- Identity similarity: sonuç yüzü 20 referansın ortak profiline yeterince yakın mı?
- Pose: hedefin baş pozu korunmuş mu?
- Expression/gaze: göz ve ağız anatomik olarak tutarlı mı?
- Mask integrity: saç, gözlük, el ve nesneler bozulmuş mu?
- Outside-mask integrity: yüz maskesi dışındaki piksel değişimi sıfır mı?
- Resolution: çıktı kaynak fotoğrafla aynı piksel boyutunda mı?
- Human review: kullanıcı öncesi/sonrası karşılaştırmasını görüp kaydetmeyi seçebilmeli.

Skor eşikleri gerçek izinli encoder ve test kümesiyle kalibre edilmelidir; rastgele bir cosine eşiği uydurulmaz.

## 8. Model araştırması ve uygulanabilirlik

### APPLE (CVPR 2026)

APPLE, kimlik transferini hedef fotoğrafın poz, ifade, ışık ve cilt özelliklerini korumayla dengelemeyi amaçlayan diffusion tabanlı bir face-swap yaklaşımıdır. Resmî kod deposu Apache-2.0 olarak listelenir. Ancak resmî README teacher/student akışları için NVIDIA GPU, üç ayrı Conda ortamı, 3DMM ve gaze preprocessing, Arc2Face ve torchrun ile örnek student inference'ta 3 GPU tarif eder. Bu bir iPhone/Core ML paketinin hazır olduğu anlamına gelmez. Checkpoint ve tüm yardımcı modellerin lisansları da ayrıca incelenmelidir.

Kod: https://github.com/cvlab-kaist/APPLE

### LivePortrait

LivePortrait kaynak projesi göz ve dudak açıklığı retarget edebilen portre canlandırma modelidir. Kapalı gözü açma/ifade düzeltme alt aşaması için değerlendirilebilir; ancak referans kimliğini 20 fotoğraftan aktaracak ana Face Swap motoru değildir. Resmî örnekler PyTorch ve indirilmiş pretrained weights kullanır; iOS Core ML dönüşümü ayrıca gerekir.

Kod ve retarget örnekleri: https://github.com/KlingAIResearch/LivePortrait  
Proje sayfası: https://liveportrait.github.io/

### FaceFusion maskeleri

FaceFusion dokümanları face parser ve occlusion mask'i ayırır, göz/kaş/burun/ağız gibi bölgeler için maske seçenekleri sunar. Bu, maske tasarımı için teknik bir referanstır; FaceFusion'ın kendi lisansı OpenRAIL-AS'tır ve model varlıklarının her birinin lisansı ayrıdır. Örneğin dokümantasyonda ArcFace/INSwapper Non-Commercial olarak listelenir; bu ağırlıklar otomatik kullanılmamalıdır.

Masker: https://docs.facefusion.io/usage/cli-arguments/face-masker  
Lisans listesi: https://docs.facefusion.io/introduction/licenses

## 9. Mevcut kod ile fark

Web sürümünde:
- fotoğraf seçimi ve çoklu yüz tespiti vardır;
- kullanıcı hedef yüzü kendisi seçer;
- 20 kayıtlı referans görsel API'sinden otomatik okunur;
- açık onaydan sonra hedef yüz crop'u ve 20 referans panosu NVIDIA FLUX'a gönderilir;
- NVIDIA çıktısı yalnızca eliptik yüz maskesinin içinde orijinal fotoğrafa eklenir;
- maske dışı piksel bütünlüğü kontrol edilir;
- API anahtarı Firebase Functions Secret'ta tutulur; istemci koduna eklenmez;
- kamera API'si kullanılmaz.

Henüz olmayanlar:
- 20 fotoğrafın aynı kişiye ait olduğunu doğrulayan identity encoder;
- kalite ağırlıklı identity embedding/prototype;
- özel Face Swap generator ve güvenilir eye/expression retargeter;
- semantik face parser ve occlusion mask;
- kimlik benzerliği/poz için otomatik kalite kapısı;
- aynı NVIDIA deneysel hattının native iOS Core ML karşılığı.

FLUX çıktısı deneysel olarak üretilir; özel kimlik korumalı Face Swap başarısı garanti edilmez.

## 10. CI gereksinimleri

- Web/iOS kaynaklarında kamera API'si bulunmaması otomatik test edilir.
- Face-only model entegrasyonunda mask == 0 bölgelerinin piksel eşitliği unit/integration testine bağlanır.
- Referans verisi ve embedding'ler test fixture olarak public GitHub'a eklenmez. Sentetik test görselleri kullanılabilir.
