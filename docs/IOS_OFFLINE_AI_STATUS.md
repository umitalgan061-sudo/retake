# Retake — iOS AI Face Swap teknik durumu

Son denetim: 9 Kasım 2026  
Hedef cihaz: iPhone 17 Pro Max

## Güncel ürün kararı

İnternet kullanımı yasak değildir. Model ve yardımcı varlıklar HTTPS üzerinden indirilebilir ve güncellenebilir. Tercih edilen akış fotoğrafı cihazda işleyen Core ML modelidir. Kullanıcı açıkça onaylarsa daha sonra uygun bir uzak inference sağlayıcısı da eklenebilir.

## Native temel

Bu dal `ios/Retake.xcodeproj` adlı SwiftUI iOS uygulamasını içerir.

- PhotosPicker ile hedef fotoğraf seçimi ve Vision ile cihaz içi yüz tespiti.
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde seçme.
- En fazla 20 referans fotoğrafını içe aktarma; yalnızca tek yüz tespit edilen referanslar kabul edilir.
- Referansları Application Support/Retake/ReferenceFaces altında özel yerel depolama.
- Derlenmiş `.mlmodelc` klasörünü Files üzerinden içe aktarma.
- Core ML modelini yükleyip girdi/çıktı açıklamalarını inceleme.
- `ModelDownloadManager`: HTTPS manifesti ve model dosyası indirme, izin verilen host kontrolü, tam dosya boyutu kontrolü, SHA-256 doğrulaması, Core ML derlemesi ve başarılı model yükleme kontrolünden sonra yerel kurulumu.
- `KeychainStore`: sağlayıcı anahtarlarını `WhenUnlockedThisDeviceOnly` korumasıyla iOS Keychain'e kaydetme, okuma ve silme.

## Henüz tamamlanmamış kritik parçalar

Bu dal **henüz gerçek yüz değiştirme yapmaz**.

- Lisans zinciri tamamen onaylanmış, indirilebilir gerçek Face Swap model ağırlıkları yok.
- Belirli bir modelin gerçek tensor sözleşmesine göre yüz hizalama/embedding ve inference adapter'ı uygulanmış değil.
- Inference çıktısının hedef fotoğrafa maske ile birleştirilmesi yapılmadı.
- Model yayıncısının imzalı manifesti ve public key sabitlemesi yok. İndirici SHA-256 ile dosya bütünlüğünü kontrol eder; tek başına manifest kaynağının kimliğini kanıtlamaz.
- Otomatik model kataloğu/URL'si bağlanmadı; gerçek ve lisans onaylı model seçilmeden indirme açılmayacak.
- Fotoğraf sonucunu Photos kütüphanesine kaydetme ve gerçek cihaz kalite/performance testleri yapılmadı.

Model eksikken işlem düğmesi bilerek devre dışıdır. Uygulama sahte sonuç göstermez.

## Model adayları ve lisans kararı

### MobileFaceSwap

AAAI 2022 makalesi MobileFaceSwap IDN modelini 0,50 milyon parametre ve kare başına 0,33 GFLOPs olarak tanımlar; bu, iPhone üzerinde çalışmaya uygun bir aday olduğunu gösterir. Kaynak kod deposu Apache-2.0 lisanslıdır. Bununla birlikte resmi README, ayrı ArcFace ağırlıkları, MobileFaceSwap ağırlıkları ve InsightFace tabanlı landmark/alignment süreci ister. Ağırlık dosyalarının bağımsız lisans şartı resmi depoda net şekilde verilmediğinden tüm zincir onaylanmış değildir. PaddlePaddle modelini Core ML'e dönüştürmek ve Swift ön/son işleme hattını ayrıca geliştirmek gerekir.

### MobileFSGAN

Kaynak kod MIT lisanslıdır. README'de PyTorch modelinin toplam boyutu 10,2 + 4,9 MB, Core ML dönüştürülmüş modelin boyutu 15 MB olarak raporlanır. Ancak resmi depo mevcut haliyle ağırlıklı olarak eğitim kodu ve kullanıcı tarafından tamamlanacak dataset iskeleti sunar; indirilebilir, lisansı açıklanmış hazır Core ML inference paketi yoktur. Bu nedenle hazır model olarak entegre edilemez.

### GHOST

Ana kaynak kod Apache-2.0 olsa da resmi `download_models.sh` ArcFace `backbone.pth`, InsightFace antelope detector/landmark dosyaları ve swapper ağırlıklarını birlikte indirir. Her ağırlık/yardımcı model için ayrı lisans doğrulaması gerekir. Tüm bileşenler açıkça onaylanmadan bu zincir kurulmayacak.

### Sonuç

Bu üç adaydan hiçbiri şu an “lisansı ve iOS paketi doğrulanmış, direkt indirilebilir model” kriterini karşılamıyor. MobileFaceSwap küçük mimarisi nedeniyle ilk teknik PoC adayı olarak tutuluyor; fakat ağırlık lisansı yazılı olarak netleşmeden dağıtım yapılmayacak.

## Model paketinin teknik sözleşmesi

`ModelPackageManifest` alanları:

- `modelID`: yalnızca `RetakeFaceSwap`
- `version`
- `artifactURL`: izin verilen hostlardan HTTPS tek dosyalı `.mlmodel`
- `byteCount`
- `sha256`: 64 karakter hex
- `codeLicense` ve `weightsLicense`
- `licenseEvidenceURL`
- `licenseReviewStatus`: manuel inceleme sonrası `approved`

İndirici 1,5 GB üstü paketi reddeder; URL yönlendirmesini tekrar kontrol eder; exact byte-count ve SHA-256 tutmazsa kurulumu durdurur. Core ML derleme ve model input/output yükleme kontrolünden sonra yeni dosyayı yerel depoya geçirir. Mevcut yerel model, dosya doğrulaması başarısızsa değiştirilmez.

**Önemli güvenlik notu:** Şu anki SHA-256 kontrolü indirilen artifact'ın manifestte yazan hash ile aynı olduğunu doğrular. Manifestin kendisini yayınlayan kimliği kriptografik olarak doğrulamaz. Üretim kullanımı için model yayıncısının imzalı manifest formatı ve uygulama içinde sabitlenmiş doğrulama public key'i gereklidir; yayıncı seçilmediği için bu anahtar uydurulmamıştır.

## NVIDIA API değerlendirmesi

NVIDIA kataloğundaki Qwen-Image-Edit genel amaçlı görüntü düzenleme modelidir; özel bir kimlik korumalı Face Swap servisi olarak belgelenmemiştir. NVIDIA API anahtarının varlığı, bu uygulamada kullanılabilir bir Face Swap endpoint'i olduğu anlamına gelmez. Şu anda kullanıcı fotoğrafı NVIDIA'ya gönderilmez ve anahtar istenmez.

## Kaynaklar

- MobileFaceSwap makalesi: https://ojs.aaai.org/index.php/AAAI/article/view/20203
- MobileFaceSwap resmi kod deposu: https://github.com/Seanseattle/MobileFaceSwap
- MobileFSGAN resmi kod deposu: https://github.com/hoim/mobilefsgan
- GHOST resmi kod ve model indirme betiği: https://github.com/ai-forever/ghost
- Apple Core ML model derleme API'si: https://developer.apple.com/documentation/coreml/mlmodel/compilemodel%28at%3A%29
- Apple CryptoKit SHA-256 akış özeti: https://developer.apple.com/documentation/cryptokit/hashfunction/update%28data%3A%29

## Gizlilik ve anahtarlar

Public GitHub deposuna kişisel yüz fotoğrafları, yüz embedding'leri veya lisansı kısıtlı model ağırlıkları eklenmemelidir. iOS uygulamasındaki API anahtarları Keychain'de saklanmalıdır; `.env` dosyası iOS uygulama bundle'ı için gizli anahtar kasası değildir.

## Tamamlanma ölçütleri

- [x] Xcode projesi iOS Simulator CI için derleniyor (önceki CI başarılı; son değişiklikler için yeni CI bekleniyor).
- [x] Model paketi için HTTPS, boyut, SHA-256 ve Core ML doğrulama altyapısı eklendi.
- [x] API anahtarı saklama için iOS Keychain altyapısı eklendi.
- [ ] iPhone 17 Pro Max üzerine kurulum ve fiziksel cihaz testi.
- [ ] 20 gerçek JPEG/PNG referans görüntüsünün cihazda doğrulanması.
- [ ] Lisans zinciri doğrulanmış gerçek Face Swap modelinin seçilmesi ve indirilmesi.
- [ ] Modelin kesin input/output sözleşmesine göre adapter, hizalama ve maskeli compositing.
- [ ] Gerçek inference çıktısının Photos kütüphanesine kaydedilmesi.
- [ ] iPhone 17 Pro Max üzerinde kalite, bellek ve hız testleri.
