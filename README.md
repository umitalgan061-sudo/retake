# Retake

Kişisel kullanım için geliştirilen Face Swap projesi. Hedef cihaz iPhone 17 Pro Max.

> **Durum:** Native iOS temel uygulaması eklendi. Gerçek Face Swap inference modeli henüz entegre edilmedi.

## İnternet ve veri politikası

İnternet kullanımı serbesttir. Ağ bağlantısı şunlar için kullanılabilir:
- Doğrulanmış model paketlerini indirme ve güncelleme
- Uygulama güncellemelerini kontrol etme
- İsteğe bağlı, ayrıca açıkça etkinleştirilen uzak inference sağlayıcıları

Varsayılan fotoğraf işleme yolu cihaz üzerinde olmalıdır. Bir fotoğraf uzak bir AI servisine gönderilecekse uygulama bunu açıkça belirtmeli ve ilk gönderimden önce kullanıcı onayı almalıdır. API anahtarları kaynak koda veya GitHub'a yazılmamalı; iOS Keychain'de saklanmalıdır.

## Native iOS uygulaması

Xcode projesi: `ios/Retake.xcodeproj`

Mevcut temel özellikler:
- SwiftUI uygulama kabuğu
- PhotosPicker ile hedef fotoğrafı seçme
- Vision ile cihaz içi yüz tespiti
- Çok yüzlü fotoğraflarda hedef yüzü görsel olarak seçme
- En fazla 20 referans fotoğrafını telefondan içe aktarma; her referans tam bir yüz içeriyorsa kabul edilir
- Referanslar arasından kaynak yüzü seçme ve yerel kaydı silme
- Referansları özel Application Support klasöründe saklama
- Derlenmiş Core ML model klasörünü yerel olarak içe aktarma ve model girdi/çıktı açıklamalarını inceleme

## Henüz tamamlanmayanlar

- Gerçek, lisansı uygun Face Swap model ağırlıkları
- Modelin kesin girdi/çıktı sözleşmesine göre inference adapter'ı
- Yüz değiştirme çıktısının hedef fotoğrafa maskeli birleştirilmesi
- Doğrulanmış model paketi indirme/güncelleme akışı
- İsteğe bağlı bulut sağlayıcısı için gerçek Face Swap endpoint'i ve açık rıza akışı
- iPhone 17 Pro Max üzerinde gerçek cihaz testi ve sonuç fotoğrafını galeriye kaydetme

Model dosyası yokken uygulama yüz değiştirme yapıyormuş gibi davranmaz; işlem düğmesi devre dışıdır.

## NVIDIA değerlendirmesi

NVIDIA NIM kataloğundaki Qwen-Image-Edit genel amaçlı görüntü düzenleme modelidir; özel bir kimlik korumalı Face Swap modeli olarak belgelenmemiştir. Model sayfasında indirilebilir olarak listelenir. NIM belgeleri kendi GPU donanımında container çalıştırma akışını gösterir. Bu nedenle yalnızca bir NVIDIA API anahtarının bu iOS uygulamasına hazır bir Face Swap servisi sunduğu varsayılmayacaktır.

Kaynaklar:
- https://build.nvidia.com/qwen/qwen-image-edit
- https://docs.nvidia.com/nim/visual-genai/latest/overview.html
- https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html

## Eski web prototipi

`index.html` MediaPipe landmark + Canvas affine warp demosudur. Gerçek eğitilmiş Face Swap modeli değildir ve CDN bağlantısı gerektirir.

## Yerel varlık kontrolü

```bash
python3 tools/validate_assets.py
```

Teknik mimari, model lisansları ve eksik adımlar için [iOS geçiş planına](docs/IOS_OFFLINE_AI_STATUS.md) bakın.

## Gizlilik

Kişisel referans fotoğraflarını, biyometrik şablonları ve lisansı kısıtlı model ağırlıklarını herkese açık GitHub deposuna yüklemeyin.