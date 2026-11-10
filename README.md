# Retake — Kişisel Fotoğraf Stüdyosu

Retake, galeriden seçilen bir hedef fotoğraftaki yüzü aynı kişiye ait referans fotoğraflarıyla değiştirmek için geliştirilen kişisel bir fotoğraf stüdyosudur. Kamera açmaz ve hedef fotoğrafın tamamını yeniden üretmez.

## Mevcut web motoru

- Yerel FaceFusion `3.9.2` ile iki geçişli hibrit motor: kimlik için `alphaface_256`, görünüm/sakal için `uniface_256`.
- 25 fotoğrafın tümü ortak göz koordinatına hizalanır ve netlik, açı, yüz alanı, ışık ölçütlerine göre 1–3 arası ağırlık alır.
- Pixel boost varsayılanı `768x768`; daha yüksek ayrıntı için işlem süresini ve VRAM kullanımını artırır.
- Tüm geçerli referanslardan ağırlıklı kimlik gömmesi ortalaması ve ağırlıklı hizalanmış piksel görünüm profili oluşturulur.
- Profilin kırpılmış referansları ve hizalanmış görünüm görseli tarayıcı IndexedDB alanında cihazda saklanır; aynı referans kümesiyle sayfa yenilendiğinde tekrar hizalama yapılmaz.
- AlphaFace geçişi tüm referanslardan ağırlıklı kimlik ortalaması alır; UniFace geçişi 25 fotoğrafın tamamından oluşan tek hizalı görünüm profiliyle çalışır.
- RTX GPU üzerinde CUDA ile çıkarım; NVIDIA hosted API, Firebase Functions veya ücretli inference API kullanılmaz.
- Retake, tek yüz sonucu orijinal fotoğrafa maskeli olarak birleştirir ve maske dışındaki pikseller için bütünlük kontrolü yapar.
- Yerel FaceFusion servisi yalnızca `127.0.0.1:8765` üzerinde dinler.

**Apple Retake ile aynı model değildir.** Retake, 25 referanstan ağırlıklı kimlik ortalaması ve hizalanmış görünüm ortalaması üretir; AlphaFace kimliği, UniFace ise görünüm/sakal katmanını işler. Bu, kişiye özel diffusion modeli eğitimi veya LoRA fine-tuning değildir; kimlik benzerliği ve sakal aktarımı garanti edilmez.

## Windows kurulumu

Gereksinimler: Windows 10/11, NVIDIA CUDA destekli GPU (bu geliştirme yapılandırması RTX A4000 16 GB), Conda, Git ve FFmpeg.

PowerShell'de Retake deposunu açıp:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\tools\\install_facefusion_windows.ps1
```

Kurulum FaceFusion 3.9.2, Python 3.12, CUDA 12.9 runtime, cuDNN 9.10 ve CUDA destekli ONNX Runtime ortamını hazırlar.

Ardından yerel servisi başlat:

```powershell
.\\tools\\start_facefusion_retake.ps1
```

Servis sağlık kontrolü: `http://127.0.0.1:8765/health`

Ayrıntılı kurulum ve sorun giderme: [docs/LOCAL_FACEFUSION_SETUP.md](docs/LOCAL_FACEFUSION_SETUP.md).

## Web uygulaması

- `index.html` statik web arayüzüdür.
- MediaPipe, tarayıcıda hedef ve referans fotoğraflarındaki yüzleri algılar; profil aynı sekmede bir kez oluşturulup sonraki hedeflerde bellekte yeniden kullanılır.
- Yüz tespiti olmayan referanslar FaceFusion'a gönderilmez.
- Hedef yüz crop'u, yüzü bulunan referans crop'ları, ağırlıklar ve 25 görüntüden hizalanmış profil aynı bilgisayardaki FaceFusion servisine gönderilir.
- AlphaFace ve UniFace sonuçları tek hibrit yüze birleştirilir; alt yüz bölgesinde sakal/bıyık görünümü desteklenir. Sonuç hedefe maske ile uygulanır ve maske dışı piksel bütünlüğü kontrol edilir.
- İlk kez açarken tarayıcı `retake06.web.app` için yerel ağ erişimi isterse izin verilmelidir.

## Firebase Hosting

Firebase projesi: `hafize-23465`  
Hosting hedefi: `retake06`  
Canlı adres: https://retake06.web.app

Sadece bu siteyi dağıtmak için:

```bash
npx --yes firebase-tools@latest deploy --only hosting:retake06 --project hafize-23465
```

Diğer Hosting sitelerini güncellememek için `--only hosting:retake06` hedefini koru. Firebase CLI oturumu yoksa önce `npx --yes firebase-tools@latest login --no-localhost` ile doğrulama yap.

## Gizlilik ve referans görselleri — önemli

Yerel FaceFusion çıkarımı, hedef ve referans fotoğraflarının NVIDIA/başka bir AI bulut servisine gönderilmediği anlamına gelir. Ancak bu deponun önceki sürümlerinde `reference-faces/` içindeki 25 JPEG dosyası public GitHub deposuna eklenmiştir. Güncel depo ağacı veya Firebase Hosting bu fotoğrafları herkese açık sunabilir; geçmiş Git commit'lerinde de kalmış olabilir. Bu nedenle bu referans fotoğraflarını gizli kabul etme. Referansların yalnızca kullanıcının cihazından seçildiği ve IndexedDB/yerel disk üzerinde saklandığı profil yükleme aşaması ayrıca tamamlanmalıdır. Git geçmişinden fotoğraf silmek ayrı bir geçmiş temizleme işlemidir; yalnızca yeni commit'ten silmek eski commit'leri gizlemez.

Model ağırlıklarının lisansları farklıdır. FaceFusion kataloğunda `alphaface_256` Non-Commercial, `uniface_256` ise Unknown olarak listelenir. Ticari kullanım hakkı olduğunu varsayma; iki modelin ve yardımcı ağırlıkların lisansını ayrı ayrı doğrula.

## Testler

```bash
npm test
npm run validate
```

Yerel FaceFusion Windows servisi Python tarafında `tools/facefusion_local_server.py` ile çalışır; Retake web arayüzünün Node sunucusuna veya Firebase Functions'a ihtiyacı yoktur.

## Native iOS

`ios/Retake.xcodeproj` SwiftUI uygulama kabuğu ve PhotosPicker/Vision altyapısını içerir. FaceFusion'ın Windows/CUDA motoru iOS üzerinde çalışmaz; native iOS için ayrı Core ML uyumlu model, model lisansları ve performans çalışması gereklidir. Şu an native iOS inference tamamlanmış değildir.
