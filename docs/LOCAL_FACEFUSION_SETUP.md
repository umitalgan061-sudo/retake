# Retake — Yerel FaceFusion kurulumu (Windows)

Bu sürüm, FLUX.1-Kontext-dev yerine FaceFusion 3.9.2 kullanır. Retake, 25 referans fotoğrafından yüz bulunanları ayrı kaynaklar olarak gönderir. Varsayılan `uniface_256` görünüm odaklı modeldir: ilk kaynak fotoğrafını doğrudan görsel girdi olarak kullanır. Arayüzde sakal/bıyığın en belirgin olduğu fotoğrafı **Ana görünüm** olarak seç; uygulama bu fotoğrafı ilk sıraya koyar. `alphaface_256` alternatifi kaynak yüz kimlik gömmelerini ortalar. Bu yaklaşımların hiçbiri kişiye özel model eğitimi veya fine-tuning değildir.

## Sistem

- Windows 10/11
- NVIDIA RTX A4000 16 GB VRAM
- Anaconda veya Miniconda
- Git for Windows
- FFmpeg (kurulum betiği winget varsa otomatik kurmayı dener)
- İnternet bağlantısı: ilk kurulum ve FaceFusion model ağırlıklarını indirmek için

FaceFusion'ın Windows kurulum yönergeleri Git, Conda ve FFmpeg gerektirir. Kurulum betiği FFmpeg'i winget ile kurmayı dener. Resmî Windows yönergeleri için https://docs.facefusion.io/installation/platform/windows adresine bak. FaceFusion'ın yayımladığı donanım rehberinde 12 GB VRAM ve üzeri önerilir. RTX A4000 16 GB bu öneriyi karşılar; gerçek performans kullanılan yüz algılayıcı, model ve fotoğraf boyutuna bağlıdır.

## 1. Kurulumu yap

Retake deposunu Codespaces'ten veya bilgisayarındaki Git çalışma klasöründen aç. FaceFusion deposu `C:\AI\FaceFusion` dizinine kurulacaktır.

PowerShell'i (Anaconda Prompt değil, Conda komutunun erişilebilir olduğu PowerShell de olur) aç ve Retake deposunda şu dosyayı çalıştır:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\tools\\install_facefusion_windows.ps1
```

Betik:
1. Ayrı bir `facefusion` Conda ortamı oluşturur (Python 3.12).
2. FaceFusion'ın sabitlenmiş `3.9.2` sürümünü indirir.
3. CUDA 12.9 runtime ve cuDNN 9.10 kurar.
4. FaceFusion'ın CUDA bağımlılıklarını yükler.
5. ONNX Runtime `CUDAExecutionProvider` sağlayıcısını doğrular.

Kurulum başarısız olursa hata veren son komutu ve terminaldeki son 30 satırı sakla. ComfyUI'nin gömülü Python ortamına FaceFusion kurma; iki ortam birbirinden ayrı kalmalı.

## 2. Model ve lisans

İlk yüz değiştirme işlemi sırasında FaceFusion gerekli model ağırlıklarını indirebilir. Retake varsayılan olarak şu görünüm odaklı ayarı kullanır:

- Face swapper: `uniface_256` (ana görünüm fotoğrafını görsel kaynak olarak kullanır)
- Pixel boost: `768x768`
- Swap weight: `1.0`
- Execution provider: `cuda`
- Device: `0`
- Face selector: `one`
- Face mask: `box`

**Lisans:** FaceFusion kaynak kodu ve model ağırlıkları ayrı lisanslara sahiptir. `uniface_256` FaceFusion model kataloğunda `Unknown` lisansıyla listelenir; bu nedenle kullanım hakkı net kabul edilmemelidir. `alphaface_256` ve `blendswap_256` Non-Commercial olarak listelenir. Kişisel/ticari olmayan kullanım dışına çıkmadan önce her modelin lisansını ve kaynak koşullarını tek tek doğrula.

## 3. Ana görünüm fotoğrafını seç ve yerel servisi çalıştır

Web arayüzündeki referans kartlarında sakal/bıyığın en net göründüğü, yüzü kapalı olmayan ve mümkünse önden/üç çeyrek açıyla çekilmiş fotoğrafın altındaki **Ana görünüm seç** düğmesine bas. UniFace bu ilk kaynak görselinden sakal, bıyık ve yüz dokusu gibi görünüş ipuçlarını alır; diğer fotoğrafların tamamı aynı işleme gönderilse de UniFace bunları birleştirip yeni bir kişisel model eğitmez.

FaceFusion kurulumu bittikten sonra Retake deposundaki PowerShell'den:

```powershell
.\\tools\\start_facefusion_retake.ps1
```

Bu komut `uniface_256` ve `768x768` ile başlar. Kimlik öncelikli eski AlphaFace davranışına dönmek için:

```powershell
.\\tools\\start_facefusion_retake.ps1 -FaceSwapperModel alphaface_256 -FaceSwapperPixelBoost 512x512
```

UniFace modelinin ağırlıkları ilk çalıştırmada indirilebilir. Pixel boost seçenekleri `512x512`, `768x768` ve `1024x1024` değerleridir; daha yüksek değer daha fazla süre/VRAM gerektirir.

Servis `http://127.0.0.1:8765` adresinde çalışır ve **yalnızca bilgisayarın loopback arayüzünü** dinler. Modemden port yönlendirme yapma, Windows güvenlik duvarında bu portu genel ağa açma. PowerShell penceresini işlem boyunca açık bırak.

Tarayıcıdan şu adresi açarak sağlık kontrolü yapılabilir:

```text
http://127.0.0.1:8765/health
```

`ok: true`, `cuda: true` ve `CUDAExecutionProvider` görülmelidir. `cuda: false` ise Retake yüz değiştirme düğmesini etkinleştirmez.

## 4. Retake'i çalıştır

1. Firebase'de `https://retake06.web.app` adresini aç.
2. Tarayıcı `retake06.web.app` için yerel ağ erişimi isterse izin ver.
3. 25 referans fotoğrafının aynı kişiye ait olduğundan emin ol.
4. Yüzü bulunan en az 10 referans bulunmalı; 10–25 arası fotoğrafın tamamı FaceFusion'a ayrı ayrı gönderilir.
5. Hedef fotoğrafı ekle ve değiştirilecek yüzü seç.
6. Açık işlem onayını kabul et. Hedefin yalnızca yüz crop'u ve kaynak referans crop'ları bu bilgisayardaki `127.0.0.1:8765` servisine gönderilir.

İlk model indirme işlemi, sonraki yüz değiştirme çalıştırmasından daha uzun sürebilir.

## 5. Motorun yaptığı işlem

- MediaPipe tarayıcıda hedef yüzü ve referans yüzleri algılar.
- Her referans fotoğrafında en büyük yüz bölgesi kırpılır.
- 10–25 referans fotoğrafı yerel FaceFusion servisine ayrı dosyalar olarak aktarılır.
- Retake seçili ana görünüm fotoğrafını kaynak listesinin ilk sırasına yerleştirir.
- Varsayılan `uniface_256`, ilk kaynak fotoğrafın yüz görünümünü doğrudan girdi olarak kullanır; bu, sakal/bıyık gibi görsel özelliklerin aktarılmasını deneyen görünüm odaklı yoldur.
- `alphaface_256` seçildiğinde kaynak yüz kimlik gömmeleri ortalanır; sakal/bıyık dokusu doğrudan taşınmayabilir.
- FaceFusion, hedef yüz crop'una seçili swapper modelini ve `768x768` pixel boost ayarını uygular.
- Retake tek portre sonucu alır ve mevcut yüz maskesiyle orijinal fotoğrafa geri birleştirir.
- Maske dışındaki piksellerin orijinalle aynı olması kontrol edilir.

UniFace'de ana görünüm fotoğrafı görsel kaynak olarak kullanılır; AlphaFace'de kaynak yüz gömmeleri ortalanır. **Bu işlem Apple Retake'teki gibi kişiye özel üretken model eğitimi değildir.** Apple'ın iç mimarisi açıklanmadığı için birebir aynılık iddiasında bulunmuyoruz. Kimlik benzerliği ve yüz kıllarının aktarımı hedeflenir ama garanti edilmez.

## 6. Sorun giderme

### CUDAExecutionProvider bulunamadı
FaceFusion ortamında:
```powershell
conda run -n facefusion python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Çıktıda `CUDAExecutionProvider` bulunmalı. Kurulum betiğini tekrar çalıştır; `pip` ile rastgele ONNX Runtime paketleri yükleyip ortamı karıştırma.

### Tarayıcı yerel servise bağlanamıyor
- PowerShell'de FaceFusion servisi çalışıyor olmalı.
- `http://127.0.0.1:8765/health` yerel tarayıcıda açılmalı.
- Edge/Chrome site izinlerinde `retake06.web.app` için yerel ağ erişimi verilmeli.
- Servis loopback'te kalmalı; `0.0.0.0` kullanma.

### Referans yüz sayısı 10'dan az
En az 10 fotoğrafta yüzün yeterince büyük, görünür ve net olması gerekir. Gözlük, maske, aşırı yan profil, çok küçük yüz veya bulanıklık tespit başarısını düşürebilir.

### Bellek / işlem hatası
İlk olarak FaceFusion modelinin tam indirildiğini ve `cuda` sağlayıcısının seçildiğini doğrula. İşlem sırasında GPU kullanan başka ağır modelleri (ComfyUI/FLUX gibi) kapat. FaceFusion hata kaydının son satırları Retake sonuç alanına aktarılır.

## 7. Gizlilik ve fotoğraf depolama

FaceFusion çıkarımı yerelde yapılır; bu, kaynak fotoğrafların her durumda özel olduğu anlamına gelmez. Bu deponun önceki sürümlerinde `reference-faces/` altındaki 25 JPEG dosyası public GitHub deposuna eklenmiştir ve geçmiş commit'lerde de bulunabilir. Firebase Hosting'e dağıtılan kopyalar da herkese açık olabilir. Bu görselleri özel kabul etme. Sonraki güvenli profil aşamasında referanslar tarayıcı/yerel disk üzerinde seçilerek public depodan çıkarılmalıdır; Git geçmişini silmek ayrı ve dikkatli bir işlemdir.

Hedef fotoğrafın tamamı yerel servisle paylaşılmaz; seçilen yüz crop'u gönderilir. Kaynak referans crop'ları da yerel FaceFusion sunucusuna gönderilir. Retake'in FaceFusion servisi `127.0.0.1` dışında dinlemez ve dış ağa açılmamalıdır.
