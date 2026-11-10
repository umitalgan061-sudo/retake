# Retake — Kişisel Fotoğraf Stüdyosu

Retake, galeriden seçilen bir hedef fotoğraftaki yüzü aynı kişiye ait referans fotoğraflarıyla değiştirmek için geliştirilen kişisel bir fotoğraf stüdyosudur. Kamera açmaz ve hedef fotoğrafın tamamını yeniden üretmez.

## Mevcut web motoru

- Yerel FaceFusion `3.9.2` ile iki geçişli hibrit motor: kimlik için `alphaface_256`, görünüm/sakal için `uniface_256`.
- 25 fotoğrafın tamamı ortak göz koordinatına hizalanır ve netlik, açı, yüz alanı, ışık ölçütlerine göre ağırlık alır. 478 noktalı FaceLandmarker başarısız olursa BlazeFace'in altı anahtar noktalı yedek hizalaması kullanılır; yedek hizalamaların ağırlığı en fazla 2'dir.
- Pixel boost ilk denemesi `1024x1024`; CUDA bellek hatasında `768x768`, ardından `512x512` denenir. Bu yüz crop inference ölçeğidir, native çıktı büyütme değildir.
- FaceFusion ArcFace ONNX ile 25 crop için medoid/medyan tutarlılık taraması yapılır; güçlü aykırı referansların kimlik oyu 1'e düşürülür.
- UniFace'tan alt yüze düşük güçlü yüksek frekanslı ayrıntı aktarılır; AlphaFace geometrisi baskın kalır.
- GFPGAN 1.4 son iyileştirmesi isteğe bağlıdır ve varsayılan kapalıdır.
- Tüm geçerli referanslardan ağırlıklı kimlik gömmesi ortalaması ve ağırlıklı hizalanmış piksel görünüm profili oluşturulur.
- Profilin kırpılmış referansları ve hizalanmış görünüm görseli tarayıcı IndexedDB alanında cihazda saklanır; aynı referans kümesiyle sayfa yenilendiğinde tekrar hizalama yapılmaz.
- AlphaFace geçişi tüm referanslardan ağırlıklı kimlik ortalaması alır; UniFace geçişi 25 fotoğrafın tamamından oluşan tek hizalı görünüm profiliyle çalışır.
- RTX GPU üzerinde CUDA ile çıkarım; NVIDIA hosted API, Firebase Functions veya ücretli inference API kullanılmaz.
- Retake, tek yüz sonucu orijinal fotoğrafa maskeli olarak birleştirir ve maske dışındaki pikseller için bütünlük kontrolü yapar.
- Yerel FaceFusion servisi yalnızca `127.0.0.1:8765` üzerinde dinler.
- FaceFusion XSeg occlusion + BiSeNet region parser maskeleri iki üretim geçişinde de aktiftir; gözlük sınıfı değiştirme maskesinin dışında tutulur.
- Son hibrit PNG, gerçek değişen piksellerden türetilmiş alfa matı taşır. Tarayıcı bu maskeyi kullanarak yalnızca değişen pikselleri orijinal fotoğrafa uygular.

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
- **Kendi 25 fotoğrafını seç** ile varsayılan havuz yerine kişisel referans fotoğraflarını kullanabilirsin. 25 JPG, PNG veya WebP dosyası gerekir; her dosya en fazla 25 MB ve görsel en fazla 20 MP olabilir. Referans görselleri işlem belleğini sınırlamak için 2048 px kenara küçültülür; kaynaklar profil anahtarı için değişmeden kullanılır. Yeni sayfa oturumunda fotoğrafları gizlilik nedeniyle yeniden seçmen gerekir.
- MediaPipe, tarayıcıda hedef ve referans fotoğraflarındaki yüzleri algılar; profil aynı sekmede bir kez oluşturulup sonraki hedeflerde bellekte yeniden kullanılır.
- Her referans önce FaceLandmarker, gerekirse BlazeFace kısa menzil ve tam menzil modelleriyle yeniden algılanır. Bir fotoğraf yine de hizalanamazsa işlem durur ve dosya adı gösterilir; 19/25 gibi eksik bir profille devam edilmez.
- Hedef yüz crop'u, 25 referansın tamamından alınan yüz crop'ları, ağırlıklar ve hizalanmış profil aynı bilgisayardaki FaceFusion servisine gönderilir. Sunucu bu crop'ların ArcFace tutarlılık raporunu da çıkarır.
- AlphaFace ve UniFace sonuçları tek hibrit yüze birleştirilir; alt yüz bölgesinde sakal/bıyık görünümü desteklenir. Sonuç hedefe maske ile uygulanır ve maske dışı piksel bütünlüğü kontrol edilir.
- İlk kez açarken tarayıcı `retake06.web.app` için yerel ağ erişimi isterse izin verilmelidir.
- Seçilen kişisel fotoğraflar tarayıcıda açılır ve yüz crop'ları çıkarılır. Yalnızca kullanıcı düzenlemeyi başlatıp onay verdikten sonra hedef yüz crop'u ve referans crop'ları bu bilgisayardaki `127.0.0.1:8765` servisine gönderilir; üçüncü taraf çıkarım API'si kullanılmaz.
- Referans fotoğrafları sırayla açılıp küçültülür; profil hazırlama ilerlemesi görünür. **Fotoğrafları ve profili temizle** seçili fotoğrafları oturum belleğinden ve önbelleğe alınmış kırpılmış profili IndexedDB'den siler. Kaynak fotoğrafları cihazdaki dosyalarından silmez.

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

Yüz profili ve çıkarım yerel bilgisayarda yapılır; kırpılmış profil görselleri ve ağırlıklar tarayıcının IndexedDB alanında cihazda saklanır. Ancak 25 orijinal referans JPEG'i public GitHub deposunda ve Firebase Hosting alanında bulunabilir; eski Git commit'lerinde de kalmış olabilir. Yerel profil önbelleği bu kaynak fotoğrafları public olmaktan çıkarmaz. Bunları gizli kabul etme. Fotoğrafları public depodan çıkarmak ve Git geçmişini temizlemek ayrı bir güvenlik çalışmasıdır; yalnızca yeni commit'ten silmek geçmiş kopyaları ortadan kaldırmaz.

Model ağırlıklarının lisansları farklıdır. FaceFusion kataloğunda `alphaface_256` Non-Commercial, `uniface_256` ise Unknown olarak listelenir. Ticari kullanım hakkı olduğunu varsayma; iki modelin ve yardımcı ağırlıkların lisansını ayrı ayrı doğrula.

## Testler

```bash
npm test
npm run validate
```

Yerel FaceFusion Windows servisi Python tarafında `tools/facefusion_local_server.py` ile çalışır; Retake web arayüzünün Node sunucusuna veya Firebase Functions'a ihtiyacı yoktur.

## Native iOS

`ios/Retake.xcodeproj` SwiftUI uygulama kabuğu ve PhotosPicker/Vision altyapısını içerir. FaceFusion'ın Windows/CUDA motoru iOS üzerinde çalışmaz; native iOS için ayrı Core ML uyumlu model, model lisansları ve performans çalışması gereklidir. Şu an native iOS inference tamamlanmış değildir.
