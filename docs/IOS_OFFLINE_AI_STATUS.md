# Retake — iOS AI Face Swap teknik durumu

Son denetim: 9 Kasım 2026  
Hedef cihaz: iPhone 17 Pro Max

## Kesin ürün davranışı

- Kamera/AVCapture/canlı video akışı yoktur. Yalnızca PhotosPicker veya dosya içe aktarma kullanılır.
- Kullanıcı yalnızca düzenlenecek yüzü seçer. Tüm fotoğraf yeniden üretilmez.
- En fazla 20 kimlik referansı aynı kişiye ait olmalıdır. Kalite ağırlıklı embedding profili oluşturulması gerekir; mevcut sürümde aynı kimlik doğrulaması henüz yoktur.
- Gözler kapalıysa uygun açık göz referanslarından doğal göz retarget yapılabilir. İfade düzeltmesi sadece ihtiyaç varsa ve sadece yüz crop'unda uygulanır.
- Saç, arka plan, kıyafet, vücut, gözlük/ön örtücüler, diğer yüzler ve maske dışındaki pikseller korunur.
- Sonuç orijinal çözünürlüğe geri birleştirilir; maske dışı piksel farkı sıfır olmalıdır.

Teknik kabul sözleşmesi: `docs/FACE_ONLY_PIPELINE.md`.

## Güncel ürün kararı

Native iOS uygulaması için ağ kullanımı yasaktır: model indirme/güncelleme, bulut inference ve fotoğraf yükleme yoktur. Model ağırlıkları lisans ve çalışma sözleşmesi doğrulandıktan sonra uygulama paketine yerel kaynak olarak eklenmelidir. Web/PWA sürümü ayrı bir ürün yüzeyidir ve henüz bu çevrimdışı garantiyi karşılamaz.

## Native temel

Bu dal `ios/Retake.xcodeproj` adlı SwiftUI iOS uygulamasını içerir.

- PhotosPicker ile hedef fotoğraf seçimi ve Vision ile cihaz içi yüz tespiti.
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde seçme.
- En fazla 20 referans fotoğrafını PhotosPicker veya Files üzerinden içe aktarma; yalnızca tek yüz tespit edilen referanslar kabul edilir. Bu kontrol kimlik eşleşmesini doğrulamaz.
- Referansları Application Support/Retake/ReferenceFaces altında özel yerel depolama.
- Cihazda bulunan derlenmiş `.mlmodelc` klasörünü Files üzerinden inceleme amacıyla içe aktarma.
- Core ML modelini yükleyip girdi/çıktı açıklamalarını inceleme.

## Henüz tamamlanmamış kritik parçalar

Bu dal **henüz gerçek yüz değiştirme yapmaz**. Mevcut web ve native iOS sürümleri kameraya erişmez; yalnızca galeri fotoğraflarını kullanır.

- Lisans zinciri tamamen onaylanmış gerçek Face Swap model ağırlıkları yok.
- Belirli bir modelin gerçek tensor sözleşmesine göre yüz hizalama/embedding ve inference adapter'ı uygulanmış değil.
- Inference çıktısının hedef fotoğrafa maske ile birleştirilmesi yapılmadı.
- Uygulama paketine eklenmiş gerçek ve lisans onaylı Face Swap model ağırlıkları yok.
- Model kaynağı, tensor sözleşmesi, Swift adapter'ı ve yerel inference hattı doğrulanmadı.
- Fotoğraf sonucunu Photos kütüphanesine kaydetme ve gerçek cihaz kalite/performance testleri yapılmadı.

Model eksikken işlem düğmesi bilerek devre dışıdır. Uygulama sahte sonuç göstermez.

## Web Studio iyileştirmeleri — 9 Kasım 2026

Web arayüzü eski kamera/Canvas affine-warp prototipinden fotoğraf odaklı, duyarlı bir PWA stüdyosuna dönüştürüldü.

- Hedef fotoğrafı dosya seçici veya sürükle-bırak ile açma; 25 MB dosya sınırı.
- MediaPipe Tasks Vision 1.2.2 ile tek kare görselde çoklu yüz landmark tespiti; hedef yüzü kutuya dokunarak veya düğmeyle seçme.
- 20'ye kadar tek-yüz referans fotoğrafını doğrulama ve IndexedDB içinde yerel olarak saklama.
- Hedef fotoğraf yalnızca oturum belleğinde tutulur; sunucuya gönderilmez.
- Koyu/açık tema, mobil/masaüstü responsive düzen, erişilebilir durum bildirimleri ve PWA kurulumu.
- Service Worker uygulama kabuğu, sabitlenmiş MediaPipe paket dosyaları ve model varlığını önbelleğe alır.
- Web kalite iş akışı Node.js 22 testleri, dosya doğrulaması ve JavaScript sözdizimi kontrolü çalıştırır.

MediaPipe giriş görselini cihaz üzerinde işler; kitaplık kullanım/performans ölçümlerini Google'a iletebilir. Web uygulamasında gerçek kimlik korumalı Face Swap motoru bulunmadığı için işlem ve indirme düğmeleri kapalıdır. Yüz landmark'ı üzerine çizilen kutular yüz değiştirme sonucu değildir.

Teknik web ayrıntısı: `docs/WEB_STUDIO.md`.

## Yeni aday: APPLE (CVPR 2026)

APPLE, kimlik aktarımını hedef fotoğrafın poz, ifade, ışık ve cilt özelliklerini korumayla dengeleyen diffusion tabanlı bir yöntemdir. Resmî kod deposu Apache-2.0 olarak işaretlidir; ancak README student inference için `torchrun` ile 3 GPU, teacher akışı için 3–4 GPU, üç ayrı Conda ortamı, 3DMM/gaze ön işleme ve Arc2Face yolu tarif eder. Bu yüzden hazır iPhone/Core ML modeli değildir. Checkpoint ve yardımcı ağırlıkların tüm lisansları ayrıca incelenmeden dağıtıma eklenemez.

## İfade / göz retarget adayı: LivePortrait

LivePortrait proje sayfası göz ve dudak açıklığı retarget örnekleri verir. Bu, kapalı gözü doğal açma ve ifade düzeltme alt aşaması için teknik olarak anlamlıdır; fakat tek başına 20 referanstan kimlik aktaracak bir face swapper değildir. Resmî kurulum PyTorch ve önceden eğitilmiş ağırlık indirimi gerektirir; hazır iOS Core ML paketi doğrulanmamıştır.

## Yüz maskesi araştırması

FaceFusion dokümanları face parser ve occlusion mask ayrımını, skin, iki göz, kaş, burun, ağız ve dudak bölgelerini ayrı maskeleyebilme olanağını gösterir. Bu mimari referans olarak kullanılabilir; ancak FaceFusion OpenRAIL-AS, model varlıkları ise birbirinden bağımsız lisanslara sahiptir. Özellikle ArcFace/INSwapper gibi bazı ağırlıklar Non-Commercial olarak listelenmiştir.

## Model adayları ve lisans kararı

### MobileFaceSwap

AAAI 2022 makalesi MobileFaceSwap IDN modelini 0,50 milyon parametre ve kare başına 0,33 GFLOPs olarak tanımlar; bu, iPhone üzerinde çalışmaya uygun bir aday olduğunu gösterir. Kaynak kod deposu Apache-2.0 lisanslıdır. Bununla birlikte resmi README, ayrı ArcFace ağırlıkları, MobileFaceSwap ağırlıkları ve InsightFace tabanlı landmark/alignment süreci ister. Ağırlık dosyalarının bağımsız lisans şartı resmi depoda net şekilde verilmediğinden tüm zincir onaylanmış değildir. PaddlePaddle modelini Core ML'e dönüştürmek ve Swift ön/son işleme hattını ayrıca geliştirmek gerekir.

### MobileFSGAN

Kaynak kod MIT lisanslıdır. README'de PyTorch modelinin toplam boyutu 10,2 + 4,9 MB, Core ML dönüştürülmüş modelin boyutu 15 MB olarak raporlanır. Ancak resmi depo mevcut haliyle ağırlıklı olarak eğitim kodu ve kullanıcı tarafından tamamlanacak dataset iskeleti sunar; indirilebilir, lisansı açıklanmış hazır Core ML inference paketi yoktur. Bu nedenle hazır model olarak entegre edilemez.

### GHOST

Ana kaynak kod Apache-2.0 olsa da resmi `download_models.sh` ArcFace `backbone.pth`, InsightFace antelope detector/landmark dosyaları ve swapper ağırlıklarını birlikte indirir. Her ağırlık/yardımcı model için ayrı lisans doğrulaması gerekir. Tüm bileşenler açıkça onaylanmadan bu zincir kurulmayacak.

### Sonuç

Bu üç adaydan hiçbiri şu an “lisansı ve iOS paketi doğrulanmış, direkt indirilebilir model” kriterini karşılamıyor. MobileFaceSwap küçük mimarisi nedeniyle ilk teknik PoC adayı olarak tutuluyor; fakat ağırlık lisansı yazılı olarak netleşmeden dağıtım yapılmayacak.

## Modelin çevrimdışı paketlenmesi

Gerçek model seçildiğinde lisans kanıtı, model sürümü, checksum, dönüşüm ayarları ve tensor sözleşmesi; kişisel veya kısıtlı ağırlıkları kaynak deposuna eklemeden belgelenmelidir. Dağıtım paketi, doğrulanmış `.mlmodel`/`.mlmodelc` varlığını uygulamayla birlikte taşımalıdır. Şu anda model varlığı bulunmadığından yerel model içe aktarma yalnızca metadata denetimi sağlar; inference veya yüz değiştirme yapmaz.

## NVIDIA API değerlendirmesi

NVIDIA kataloğundaki Qwen-Image-Edit genel amaçlı görüntü düzenleme modelidir; özel bir kimlik korumalı Face Swap servisi olarak belgelenmemiştir. NVIDIA API anahtarının varlığı, bu uygulamada kullanılabilir bir Face Swap endpoint'i olduğu anlamına gelmez. Şu anda kullanıcı fotoğrafı NVIDIA'ya gönderilmez ve anahtar istenmez.

## Kaynaklar

- MobileFaceSwap makalesi: https://ojs.aaai.org/index.php/AAAI/article/view/20203
- MobileFaceSwap resmi kod deposu: https://github.com/Seanseattle/MobileFaceSwap
- APPLE CVPR 2026: https://github.com/cvlab-kaist/APPLE
- LivePortrait: https://github.com/KlingAIResearch/LivePortrait
- FaceFusion Face Masker: https://docs.facefusion.io/usage/cli-arguments/face-masker
- FaceFusion licenses: https://docs.facefusion.io/introduction/licenses
- MobileFSGAN resmi kod deposu: https://github.com/hoim/mobilefsgan
- GHOST resmi kod ve model indirme betiği: https://github.com/ai-forever/ghost
- Apple Core ML model derleme API'si: https://developer.apple.com/documentation/coreml/mlmodel/compilemodel%28at%3A%29
- Apple CryptoKit SHA-256 akış özeti: https://developer.apple.com/documentation/cryptokit/hashfunction/update%28data%3A%29

## Gizlilik ve varlıklar

Public GitHub deposuna kişisel yüz fotoğrafları, yüz embedding'leri veya lisansı kısıtlı model ağırlıkları eklenmemelidir. Özel referans fotoğrafları ve lisanslı yerel model ağırlıkları uygulamanın cihaz depolamasında tutulmalıdır.

## Tamamlanma ölçütleri

- [x] Xcode projesi iOS Simulator CI için derleniyor (önceki CI başarılı; son değişiklikler için yeni CI bekleniyor).
- [x] iOS uygulamasındaki model indirme ve sağlayıcı anahtarı yolları kaldırıldı.
- [ ] iPhone 17 Pro Max üzerine kurulum ve fiziksel cihaz testi.
- [ ] 20 gerçek JPEG/PNG referans görüntüsünün cihazda doğrulanması.
- [ ] Aynı kişiye ait 20 referansın kimlik encoder'ı ile ortak profile dönüştürülmesi.
- [ ] Face-only parser/occlusion mask ve maske dışı piksel eşitliği testi.
- [ ] Göz açma ve sınırlı expression retarget alt hattı.
- [ ] Lisans zinciri doğrulanmış gerçek Face Swap modelinin seçilmesi ve uygulamayla yerel paketlenmesi.
- [ ] Modelin kesin input/output sözleşmesine göre adapter, hizalama ve maskeli compositing.
- [ ] Gerçek inference çıktısının Photos kütüphanesine kaydedilmesi.
- [ ] iPhone 17 Pro Max üzerinde kalite, bellek ve hız testleri.
