# iOS uygulaması

Açılacak proje: `Retake.xcodeproj`

Hedef: iPhone 17 Pro Max, iOS 17 veya üzeri.

## Mevcut uygulama özellikleri

- SwiftUI native uygulama kabuğu
- PhotosPicker ile hedef fotoğraf seçimi
- Vision ile cihaz içi yüz tespiti
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde seçme
- En fazla 20 referans görselini yerel depoya içe aktarma
- Her referans görüntüsünde tek yüz kontrolü
- Kaynak referans seçimi ve yerel silme
- Yerel Core ML model paketi içe aktarma ve girdi/çıktı metadata inceleme
- Tek dosyalı `.mlmodel` paketlerini HTTPS üzerinden indirme, boyut ve SHA-256 doğrulaması, Core ML derlemesi ve yerel kurulum
- Servis sağlayıcı anahtarlarını cihazın Keychain alanında saklayan altyapı

## Ağ kullanımı ve model kurulumu

İnternet modeli indirmek/güncellemek için kullanılabilir. Model indiricisi şu an gerçek bir Face Swap kataloğuna bağlı değildir; lisans incelemesi tamamlanmış, güvenilir bir HTTPS manifest URL'si verildiğinde çalışacak altyapıdır.

Manifest şu alanları taşımalıdır: `modelID`, `version`, `artifactURL`, `byteCount`, `sha256`, `codeLicense`, `weightsLicense`, `licenseEvidenceURL` ve `licenseReviewStatus`. `modelID` değeri `RetakeFaceSwap`, artifact uzantısı `.mlmodel`, lisans inceleme durumu ise `approved` olmalıdır. Model dosyası için yalnızca izin verilen GitHub/Hugging Face HTTPS hostları kabul edilir.

İndirici dosyanın tam boyutunu ve SHA-256 değerini kontrol eder, sonra Core ML ile derler, modelin giriş/çıkış tanımlarını yükleyerek kontrol eder ve ancak bundan sonra yerel modele geçirir. Geçersiz dosya mevcut modeli değiştirmez.

**Güven sınırı:** SHA-256, manifestte verilen dosyanın bütünlüğünü kontrol eder; manifesti yayınlayan tarafın kimliğini tek başına kanıtlamaz. Üretim dağıtımı öncesinde model yayıncısının imzalı manifesti ve açık anahtar sabitlemesi ayrıca tanımlanmalıdır. Şu an doğrulanmış model yayıncısı veya gerçek Face Swap manifest URL'si bulunmadığından otomatik indirme etkinleştirilmedi.

Varsayılan durumda fotoğraflar cihazda kalır. Uzak inference sağlayıcısı devreye alınırsa fotoğrafın üçüncü tarafa gönderileceği işlem öncesinde açıkça belirtilmeli ve kullanıcı onayı alınmalıdır.

## API anahtarları

iOS uygulamasına `.env` dosyası koymak güvenli bir sır saklama yöntemi değildir; uygulama bundle'ına giren dosyalar kullanıcıdan gizli kalmaz. `KeychainStore` API anahtarlarını `WhenUnlockedThisDeviceOnly` korumasıyla cihazın Keychain alanında saklamak için eklendi. Henüz kullanılmayan NVIDIA anahtarını istemiyoruz: elimizde bu anahtarla çağrılabilecek belgelenmiş bir Face Swap endpoint'i yok.

## Model adayları — 9 Kasım 2026 incelemesi

- **MobileFaceSwap:** Araştırma makalesi modeli 0,50 milyon parametre ve kare başına 0,33 GFLOPs olarak tanımlar; kaynak kod deposu Apache-2.0 lisansını gösterir. Ancak resmi kurulum dokümanı ayrı ArcFace ağırlıkları ve InsightFace tabanlı yüz/landmark hattı ister. Önceden eğitilmiş dosyaların ayrı ağırlık lisansı depoda net olarak belirtilmediği için bu paket henüz lisans onayından geçmedi.
- **MobileFSGAN:** Kaynak deposu MIT lisanslıdır ve makale deposunda Core ML dönüştürülmüş model boyutu 15 MB olarak raporlanır. Buna rağmen açık depoda indirilebilir, lisansı açıklanmış hazır inference ağırlığı/Core ML paketi yayımlanmamıştır; depo ağırlıklı olarak eğitim kodu içerir.
- **GHOST:** Ana kod Apache-2.0 olsa da resmi inference indirme betiği ArcFace ve InsightFace tabanlı ek ağırlıklar da indirir. Tüm ağırlıkların kullanım izni tek tek doğrulanmadan bu işlem hattı lisanslı kabul edilemez.

Bu nedenle hiçbir araştırma modelinin ağırlığı henüz uygulamaya otomatik indirilmedi. Kod lisansı tek başına model ağırlıklarının lisansını veya ticari/kişisel kullanım izinlerini kanıtlamaz.

## NVIDIA notu

NVIDIA Qwen-Image-Edit genel amaçlı image-edit modelidir. Mevcut dokümantasyonda deterministik, kimlik korumalı Face Swap endpoint'i olarak sunulmaz. NIM'in indirilebilir model paketini çalıştırmak için ayrıca desteklenen NVIDIA GPU altyapısı gerekir. Bu nedenle mevcut NVIDIA API anahtarı doğrudan yüz değiştirme çözümü olarak varsayılmamıştır.

## Derleme

1. macOS üzerinde Xcode 16 veya üstünü açın.
2. `Retake.xcodeproj` projesini açın.
3. Signing & Capabilities bölümünde kendi Apple Developer Team hesabınızı seçin.
4. iPhone 17 Pro Max cihazını hedef seçip Run düğmesine basın.

Bu depo içinden gerçek cihaz derlemesi ve çalışma testi yapılamaz; simulator CI ile fiziksel cihaz doğrulaması ayrı tutulmalıdır.
