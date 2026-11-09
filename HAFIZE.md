# Retake — proje geliştirme talimatları

## Nihai hedef

iPhone 17 Pro Max üzerinde çalışan, tamamen çevrimdışı, gerçek eğitilmiş AI Face Swap modeli kullanan kişisel iOS uygulaması.

## Değişmez şartlar

1. Fotoğraflar hiçbir uzak API'ye veya bulut servisine gönderilmez.
2. Model inference cihaz üzerinde gerçekleşir.
3. Gerçek inference yoksa işlem başarılı gibi gösterilmez.
4. Canvas, landmark ve affine warp tek başına AI Face Swap olarak adlandırılmaz.
5. Model kodu ile model ağırlıklarının lisansları ayrı ayrı doğrulanır.
6. Kullanıcıya ait referans yüz fotoğrafları ve lisans kısıtlı model ağırlıkları public GitHub deposuna commit edilmez.
7. Uçak modu testi, uygulama kurulumu ve ilk model hazırlığı bittikten sonra yapılır.

## Mevcut kodun durumu

Mevcut `index.html` bir web/kamera prototipidir. MediaPipe Face Landmarker kullanır ve yüzü Canvas üzerinde geometrik üçgen/affine warp ile kaplar. MediaPipe JavaScript ve model dosyası CDN üzerinden yüklendiğinden mevcut sürüm offline değildir.

Depodaki `models/` dosyaları gerçek model ağırlıkları olarak doğrulanmamıştır. `reference-faces/` içindeki 20 dosya da gerçek JPEG/PNG ikili içeriği olarak doğrulanmamıştır. Bu nedenle şimdiki depodan gerçek model inference yapılamaz.

## Hedef mimari

- Native uygulama: SwiftUI
- Fotoğraf seçimi: PhotosUI
- Yüz algılama / landmark: iOS Vision veya lisansı/uyumluluğu doğrulanmış yerel model
- Face Swap: gerçek, lisansı uygun, iOS üzerinde çalıştırılabilir model
- Inference: Core ML; CPU/GPU/Neural Engine çalışma birimi gerçek cihazda profillenerek doğrulanır
- Veri: yerel dosya sistemi ve Photos kütüphanesi
- Web arayüzü gerekiyorsa WKWebView; fakat AI işleminin sahibi native Swift olmalı

Modelin gerçek girdi/çıktı tensörleri ve dönüştürülmüş Core ML dosyası elde edilmeden model adapter'ı uydurma şekillerle yazma. Bir model dosyası eksikse açık hata göster.

## Model ve lisans

InsightFace kaynak kodunun MIT lisansı, önceden eğitilmiş ağırlıklarının da aynı lisansa sahip olduğu anlamına gelmez. InSwapper modelleri için ayrıca lisans koşulları kontrol edilmelidir. FaceFusion'da listelenen her modelin de ayrı lisansı vardır.

Kaynaklar:
- https://github.com/deepinsight/insightface
- https://www.insightface.ai/
- https://docs.facefusion.io/introduction/licenses

## Yerel doğrulama

```bash
python3 tools/validate_assets.py
```

Bu kontrol gerçek görsel imzalarını, model dosyalarının temel ikili bütünlüğünü, CDN bağımlılıklarını ve iOS proje yapısının varlığını kontrol eder. Modelin inference kalitesini kanıtlamaz; bu test gerçek iPhone üzerinde ayrıca yapılmalıdır.

Ayrıntılı durum ve kabul kriterleri: `docs/IOS_OFFLINE_AI_STATUS.md`.
