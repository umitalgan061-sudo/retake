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

## Mevcut native iOS temel

`ios/Retake.xcodeproj` SwiftUI uygulama projesidir. Fotoğraf seçimi PhotosPicker, yüz tespiti Vision, 20 referans fotoğrafının yerel depolanması Application Support ve Core ML model paketi incelemesi Swift ile yapılır.

Mevcut sürüm fotoğraftaki hedef yüzü algılayıp görsel olarak seçtirebilir. Gerçek yüz değiştirme modeli ve inference adapter'ı henüz yoktur. Bu nedenle yüz değiştirme butonu bilinçli olarak devre dışıdır.

## Eski web prototipi

`index.html` MediaPipe Face Landmarker ve Canvas üçgen/affine warp ile geometrik yüz kaplama yapar. Gerçek eğitilmiş Face Swap modeli değildir ve CDN bağımlılıkları nedeniyle offline değildir.

## Depo varlıkları

GitHub'daki eski `models/` dosyaları 51 baytlık API metin referanslarıdır; gerçek model ağırlıkları değildir. `reference-faces/` girdileri de 36 baytlık API metin referanslarıdır; gerçek JPEG/PNG fotoğraf dosyaları değildir. Orijinal fotoğraflar PhotosPicker ile iPhone'a içe aktarılmalıdır.

## Hedef mimari

- Native uygulama: SwiftUI
- Fotoğraf seçimi: PhotosUI
- Yüz tespiti: Vision
- Face Swap: lisansı uygun, iOS üzerinde çalıştırılabilir gerçek model
- Inference: Core ML; CPU/GPU/Neural Engine çalışma birimi gerçek cihazda profillenerek doğrulanır
- Veri: yerel dosya sistemi ve Photos kütüphanesi
- AI işlemi: ağ bağlantısına ihtiyaç duymadan cihaz üzerinde

Modelin gerçek girdi/çıktı tensörleri ve dönüştürülmüş Core ML dosyası elde edilmeden model adapter'ı uydurma şekillerle yazma.

## Model lisansı

InsightFace ve FaceFusion model varlıklarının lisansları birbirinden farklıdır. Modelin kendisi kadar algılama, yüz embedding, dönüştürme, segmentasyon ve restorasyon bileşenlerinin lisanslarını da doğrula.

Kaynaklar:
- https://github.com/deepinsight/insightface
- https://www.insightface.ai/
- https://docs.facefusion.io/introduction/licenses
- https://github.com/apple/coremltools

## Yerel doğrulama

```bash
python3 tools/validate_assets.py
```

Ayrıntılı durum ve kabul kriterleri: `docs/IOS_OFFLINE_AI_STATUS.md`.
