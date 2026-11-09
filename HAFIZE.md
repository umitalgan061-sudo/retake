# Retake — proje geliştirme talimatları

## Nihai hedef

iPhone 17 Pro Max üzerinde gerçek eğitilmiş AI Face Swap modeli kullanan, yüksek kaliteli fotoğraf çıktısı üreten kişisel iOS uygulaması.

İnternet kullanımı serbesttir. İnternet; model indirme, model güncelleme ve kullanıcı açıkça etkinleştirirse uzak inference için kullanılabilir. Varsayılan tercih cihaz üzerinde inference ve fotoğrafların cihazda kalmasıdır.

## Değişmez şartlar

1. Canvas, landmark ve affine warp tek başına AI Face Swap olarak adlandırılmaz.
2. Gerçek inference yoksa işlem başarılı gibi gösterilmez.
3. Model kodu ile model ağırlıklarının lisansları ayrı ayrı doğrulanır.
4. Kullanıcıya ait referans yüz fotoğrafları ve lisans kısıtlı model ağırlıkları public GitHub deposuna commit edilmez.
5. Bir fotoğraf harici AI servisine gönderilecekse kullanıcıya hangi servise, hangi amaçla gönderileceği gösterilir ve açık onay alınır.
6. API anahtarları kaynak koduna, Info.plist'e veya GitHub'a eklenmez. iOS Keychain kullanılır.
7. Model indirirken HTTPS, SHA-256 bütünlük kontrolü, beklenen dosya boyutu ve lisans manifesti doğrulanır.
8. NVIDIA API yalnızca gerçek Face Swap görevini destekleyen bir endpoint bulunursa kullanılır. Genel görüntü düzenleme endpoint'i kesin kimlik korumalı Face Swap gibi sunulmaz.

## Mevcut native iOS temel

`ios/Retake.xcodeproj` SwiftUI uygulama projesidir. Fotoğraf seçimi PhotosPicker, yüz tespiti Vision, 20 referans fotoğrafının yerel depolanması Application Support ve Core ML model paketi incelemesi Swift ile yapılır.

Mevcut sürüm fotoğraftaki hedef yüzü algılayıp görsel olarak seçtirebilir; referans yüzleri yerel olarak saklayıp kaynak yüz olarak seçtirebilir. Gerçek yüz değiştirme modeli ve inference adapter'ı henüz yoktur. Bu nedenle yüz değiştirme butonu bilinçli olarak devre dışıdır.

## Hedef mimari

- Native uygulama: SwiftUI
- Fotoğraf seçimi: PhotosUI
- Yüz tespiti: Vision
- Face Swap: lisansı uygun, iOS üzerinde çalıştırılabilir gerçek model
- Varsayılan inference: Core ML ile cihaz üzerinde
- Model dağıtımı: doğrulanmış HTTPS indirmesi ve SHA-256 kontrolü
- İsteğe bağlı uzak inference: yalnızca desteklenen gerçek Face Swap endpoint'i ve kullanıcı onayıyla
- API anahtarı saklama: iOS Keychain
- Veri: yerel dosya sistemi ve Photos kütüphanesi

Modelin gerçek girdi/çıktı tensörleri ve dönüştürülmüş Core ML dosyası elde edilmeden model adapter'ı uydurma şekillerle yazma.

## NVIDIA API değerlendirmesi

NVIDIA API kataloğundaki Qwen-Image-Edit genel amaçlı görsel düzenleme modelidir; dokümantasyonunda kesin kimlik korumalı yüz değiştirme özelliği tanımlanmamıştır. NIM dokümanları Qwen-Image-Edit'i indirilebilir model olarak gösterir ve self-hosted NIM kurulumu için uygun NVIDIA GPU ile container çalıştırmayı anlatır. Bu sebeple kullanıcıya ait yüz fotoğraflarını Qwen Image Edit'e otomatik yükleme ve bu modeli Face Swap motoru olarak kabul etme.

Kaynaklar:
- https://build.nvidia.com/qwen/qwen-image-edit
- https://docs.nvidia.com/nim/visual-genai/latest/overview.html
- https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html

## Model lisansı

InsightFace, FaceFusion ve diğer model kataloglarında kod ile ağırlık lisansları farklı olabilir. Yüz algılama, embedding, dönüştürme, segmentasyon, restorasyon ve yüz değiştirme bileşenlerinin tamamını ayrı ayrı denetle.

## Yerel doğrulama

```bash
python3 tools/validate_assets.py
```

Ayrıntılı durum ve kabul kriterleri: `docs/IOS_OFFLINE_AI_STATUS.md`.
