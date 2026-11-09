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

## Ağ kullanımı

Uygulama internet kullanabilir. İlk hedef, doğrulanmış Core ML model paketlerini HTTPS üzerinden indirmek ve cihazda çalıştırmaktır. Model dosyaları için manifestte sürüm, kaynak, lisans, boyut ve SHA-256 zorunlu olmalıdır.

Varsayılan durumda fotoğraflar cihazda kalır. Uzak inference sağlayıcısı devreye alınırsa fotoğrafın üçüncü tarafa gönderileceği işlem öncesinde açıkça belirtilmeli ve kullanıcı onayı alınmalıdır. Anahtarlar iOS Keychain'de saklanmalı; uygulama bundle'ına gömülmemelidir.

## NVIDIA notu

NVIDIA Qwen-Image-Edit genel amaçlı image-edit modelidir. Mevcut dokümantasyonda deterministik, kimlik korumalı Face Swap endpoint'i olarak sunulmaz. NIM'in indirilebilir model paketini çalıştırmak için ayrıca desteklenen NVIDIA GPU altyapısı gerekir. Bu nedenle mevcut NVIDIA API anahtarı doğrudan yüz değiştirme çözümü olarak varsayılmamıştır.

## Model dosyası

Beklenen yerel paket adı `RetakeFaceSwap.mlmodelc`. Bu isim tek başına uyumluluk garantisi değildir. Seçilecek gerçek modelin lisansı, tensor sözleşmesi ve Core ML uyumluluğu doğrulanmalıdır.

## Derleme

1. macOS üzerinde Xcode 16 veya üstünü açın.
2. `Retake.xcodeproj` projesini açın.
3. Signing & Capabilities bölümünde kendi Apple Developer Team hesabınızı seçin.
4. iPhone 17 Pro Max cihazını hedef seçip Run düğmesine basın.

Bu depo içinden gerçek cihaz derlemesi ve çalışma testi yapılamaz; simulator CI ile fiziksel cihaz doğrulaması ayrı tutulmalıdır.