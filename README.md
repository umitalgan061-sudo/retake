# Retake

Kişisel kullanım için geliştirilen Face Swap projesi. Nihai hedef, iPhone 17 Pro Max üzerinde tamamen çevrimdışı çalışan yerel bir iOS uygulamasıdır.

> **Durum:** Depodaki mevcut web sürümü henüz gerçek AI Face Swap değildir ve çevrimdışı çalışmaz.

## Mevcut web prototipi

- MediaPipe Face Landmarker ile yüz landmark tespiti
- 20 referans yüz seçimi arayüzü
- Canvas üzerinde üçgen/affine warp tabanlı geometrik yüz kaplama
- Kamera önizlemesi ve PNG indirme arayüzü
- Koyu/açık tema

Bu yaklaşım eğitilmiş bir yüz değiştirme modeli değildir. Mevcut `index.html`, MediaPipe kitaplığını jsDelivr'den ve model dosyasını Google Storage'dan yüklediği için çevrimdışı değildir.

## Depo varlıklarının durumu

Son depo denetiminde `models/` altındaki dosyalar gerçek ONNX/MediaPipe ağırlıkları olarak, `reference-faces/` altındaki 20 dosya da gerçek JPEG/PNG görüntüleri olarak doğrulanamadı. Dosyalar çok küçük API/asset referans metinleri biçiminde görünüyor.

Kontrol etmek için:

```bash
python3 tools/validate_assets.py
```

## Nihai hedef

- SwiftUI / PhotosUI tabanlı native iOS uygulaması
- Gerçek, lisansı doğrulanmış Face Swap modeli
- Core ML üzerinde cihaz içi inference
- Yerel referans fotoğrafları
- Uçak modunda uçtan uca çalışma
- Fotoğrafların üçüncü taraf servislere gönderilmemesi

Teknik durum ve tamamlanma testleri için [iOS çevrimdışı AI geçiş planına](docs/IOS_OFFLINE_AI_STATUS.md) bakın.

## Gizlilik

Bu GitHub deposu herkese açıktır. Kişisel yüz fotoğraflarınızı veya lisans kısıtlı model ağırlıklarını herkese açık depoya commit etmeyin. Bunları yerel Xcode build kaynaklarına veya uygulama içi güvenli yerel içe aktarma alanına ekleyin.
