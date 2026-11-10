# Retake — Kişisel Fotoğraf Stüdyosu

Retake, galeriden seçilen bir hedef fotoğraftaki yüzü aynı kişiye ait referans fotoğraflarıyla değiştirmek için geliştirilen kişisel bir fotoğraf stüdyosudur. Kamera açmaz ve hedef fotoğrafın tamamını yeniden üretmez.

## Mevcut web motoru

- Yerel FaceFusion `3.9.2` + `alphaface_256` yüz değiştirme modeli.
- 25 referans fotoğrafına kadar havuz; yüzü bulunan en az 10 fotoğrafın tamamı tek kimlik temsiline katılır.
- Referans kaynakları FaceFusion'a ayrı görseller olarak gönderilir; kolaj oluşturulmaz.
- RTX GPU üzerinde CUDA ile çıkarım; NVIDIA hosted API, Firebase Functions veya ücretli inference API kullanılmaz.
- Retake, tek yüz sonucu orijinal fotoğrafa maskeli olarak birleştirir ve maske dışındaki pikseller için bütünlük kontrolü yapar.
- Yerel FaceFusion servisi yalnızca `127.0.0.1:8765` üzerinde dinler.

**Apple Retake ile aynı model değildir.** FaceFusion çoklu kaynak yüz kimlik temsillerinin ortalamasını alır; kişiye özel diffusion model eğitimi/fine-tuning yapmaz. Kimlik benzerliği, saç/aksesuar uyumu ve her fotoğraftaki kalite garanti edilmez. İlk denemelerde sonucu kullanıcı kontrol etmelidir.

## Windows kurulumu

Gereksinimler: Windows 10/11, NVIDIA CUDA destekli GPU (bu geliştirme yapılandırması RTX A4000 16 GB), Conda ve Git.

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
- MediaPipe, tarayıcıda hedef ve referans fotoğraflarındaki yüzleri algılar.
- Yüz tespiti olmayan referanslar FaceFusion'a gönderilmez.
- Hedef yüz crop'u ve yüzü bulunan referans crop'ları kullanıcının aynı bilgisayarındaki FaceFusion servisine gönderilir.
- İşlenmiş hedef crop'u mevcut maskeli compositing yöntemiyle orijinal fotoğrafa eklenir. Maske dışı piksel değişikliği kabul edilmez.
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

Model ağırlıklarının lisansları birbirinden farklıdır. Bu yapılandırmadaki `alphaface_256` FaceFusion model kataloğunda Non-Commercial olarak listelenmiştir. Ticari kullanım için uygun olduğunu varsayma.

## Testler

```bash
npm test
npm run validate
```

Yerel FaceFusion Windows servisi Python tarafında `tools/facefusion_local_server.py` ile çalışır; Retake web arayüzünün Node sunucusuna veya Firebase Functions'a ihtiyacı yoktur.

## Native iOS

`ios/Retake.xcodeproj` SwiftUI uygulama kabuğu ve PhotosPicker/Vision altyapısını içerir. FaceFusion'ın Windows/CUDA motoru iOS üzerinde çalışmaz; native iOS için ayrı Core ML uyumlu model, model lisansları ve performans çalışması gereklidir. Şu an native iOS inference tamamlanmış değildir.
