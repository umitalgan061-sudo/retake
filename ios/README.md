# iOS uygulaması

Açılacak proje: `Retake.xcodeproj`

Hedef: iPhone 17 Pro Max, iOS 17 veya üzeri.

## Şu anda uygulamada bulunanlar

- SwiftUI native uygulama kabuğu
- PhotosPicker ile hedef fotoğraf seçimi
- Vision ile cihaz içi yüz dikdörtgeni tespiti
- Birden fazla yüz olduğunda hedef yüzü seçme
- En fazla 20 referans fotoğrafını yerel Application Support klasörüne içe aktarma
- Referansları uygulama içinden silme
- Core ML model dosyası mevcutsa modelin yüklenmesi ve girdi/çıktı isimlerinin incelenmesi
- Eksik modelde gerçek işlem yapılmadığının açıkça bildirilmesi

## Model dosyası

Beklenen paket adı `RetakeFaceSwap.mlmodelc`. Bu isim tek başına bir uyumluluk garantisi değildir. Seçilecek gerçek modelin lisansı ve tensor sözleşmesi incelenmeli; Core ML'e çevrilmeli; gerçek cihazda test edilmelidir.

Mevcut sürümde genel amaçlı tahmini bir inference çağrısı yapılmaz. Yüz değiştirme modelleri birbirinden farklı girdi, ön işleme, yüz hizalama, embedding ve çıktı sözleşmelerine sahiptir. Doğru model adapter'ı olmadan inference yapmak sahte/bozuk çıktı üretir.

## Yerel referans fotoğrafları

GitHub deposundaki eski `reference-faces/` girdileri gerçek fotoğraf içeriği değil. Orijinal 20 fotoğrafı uygulamaya PhotosPicker ile cihazdan yeniden içe aktarın. Uygulama bunları Application Support/Retake/ReferenceFaces altında saklar.

## Derleme

1. macOS üzerinde Xcode 16 veya üstünü açın.
2. `Retake.xcodeproj` dosyasını açın.
3. Signing & Capabilities bölümünde kendi Apple Developer Team hesabınızı seçin.
4. iPhone 17 Pro Max cihazını hedef seçip Run düğmesine basın.

Bu depo içinden Xcode veya iPhone üzerinde derleme/çalıştırma testi yapılamadı; gerçek cihaz doğrulaması ayrıca yapılmalıdır.
