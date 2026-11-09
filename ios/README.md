# iOS uygulaması

Açılacak proje: `Retake.xcodeproj`

Hedef: iPhone 17 Pro Max, iOS 17 veya üzeri.

## Kesin davranış

- Kamera/AVCapture/canlı video yoktur. Hedef ve referanslar yalnızca PhotosPicker/FileImporter ile alınır.
- Hedef fotoğrafın yalnızca seçilen yüz bölgesi düzenlenecektir. Saç, arka plan, kıyafet ve diğer kişiler korunur.
- 20 kimlik referans fotoğrafı aynı kişiye ait olmalıdır; tek yüz tespiti aynı kimlik olduklarını doğrulamaz.
- Gözler kapalıysa uygun referanslardan doğal göz açma; gerektiğinde kontrollü yüz ifadesi düzeltmesi sadece yüz crop'unda çalışır.

Detaylı kabul sözleşmesi: `../docs/FACE_ONLY_PIPELINE.md`.

## Mevcut uygulama özellikleri

- SwiftUI native uygulama kabuğu
- PhotosPicker ile hedef fotoğraf seçimi
- Vision ile cihaz içi yüz tespiti; ImageIO üzerinden 2048 px sınırlandırılmış küçük resim üretimi ve arka plan iş parçacığında analiz.
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde seçme
- En fazla 20 referans görselini PhotosPicker veya Files üzerinden yerel depoya içe aktarma
- Her referans görüntüsünde tek yüz kontrolü
- Kaynak referans seçimi ve yerel silme
- Yerel Core ML model paketi içe aktarma ve girdi/çıktı metadata inceleme

## Çevrimdışı çalışma ve model kurulumu

Native iOS uygulamasında model veya fotoğraf için ağ isteği yapılmaz. 20 referans fotoğrafını iPhone Fotoğraflar'a veya Dosyalar'a ekleyip uygulamadaki ilgili içe aktarma düğmesini kullanın. Uygulama her görselde tek bir yüz algılandığını doğrular ve görseli `Application Support/Retake/ReferenceFaces` içine özel olarak kopyalar. Bu işlem aynı kimliği doğrulamaz. Face Swap ağırlıkları lisans ve gerçek iOS çalışma sözleşmesi doğrulandıktan sonra uygulama kaynaklarına yerel olarak eklenmelidir. Model kartındaki içe aktarma, yalnızca cihazda önceden bulunan derlenmiş `RetakeFaceSwap.mlmodelc` paketini inceleme içindir; uygulamada şu anda paketlenmiş bir swap modeli yoktur.

Fotoğraf seçimi, Vision yüz tespiti ve referans depolaması cihaz üzerindedir. Gerçek Face Swap modeli, adapter ve maskeli birleştirme tamamlanana kadar işlem düğmesi kapalı kalır. Web/PWA sürümünün çevrimdışı olup olmadığı bu native iOS garantisinin parçası değildir.

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

Bu depo içindeki Linux geliştirme ortamında Xcode ve iPhone donanımı bulunmadığından iOS derlemesi, gerçek cihaz inference'ı ve uçak modu testi burada doğrulanamaz. Simulator CI ile fiziksel cihaz doğrulaması ayrı tutulmalıdır.
