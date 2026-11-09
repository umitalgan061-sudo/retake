# Retake

Kişisel kullanım için geliştirilen Face Swap projesi. Nihai hedef, iPhone 17 Pro Max üzerinde tamamen çevrimdışı çalışan yerel bir iOS uygulamasıdır.

> **Durum:** Native iOS temel uygulaması eklendi. Gerçek Face Swap inference modeli henüz entegre edilmedi.

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
- iPhone 17 Pro Max üzerinde gerçek cihaz ve uçak modu testleri
- Düzenlenmiş sonucun Fotoğraflar arşivine kaydedilmesi

Model dosyası yokken uygulama yüz değiştirme yapıyormuş gibi davranmaz; işlem düğmesi devre dışıdır.

## Eski web prototipi

`index.html` MediaPipe landmark + Canvas affine warp demosudur. Gerçek eğitilmiş Face Swap modeli değildir ve CDN bağlantısı gerektirir.

## Yerel varlık kontrolü

```bash
python3 tools/validate_assets.py
```

Teknik mimari, model lisansları ve eksik adımlar için [iOS çevrimdışı AI geçiş planına](docs/IOS_OFFLINE_AI_STATUS.md) bakın.

## Gizlilik

Kişisel referans fotoğraflarını, biyometrik şablonları ve lisansı kısıtlı model ağırlıklarını herkese açık GitHub deposuna yüklemeyin.