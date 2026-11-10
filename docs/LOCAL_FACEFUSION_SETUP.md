# Retake — Yerel FaceFusion kurulumu (Windows)

Bu sürüm, FLUX.1-Kontext-dev yerine FaceFusion 3.9.2 kullanır. FaceFusion yüz değiştirme için tasarlanmış bir işlem hattıdır. Retake, 25 referans fotoğrafından yüz bulunanların tamamını FaceFusion'a ayrı kaynak görüntüler olarak verir. FaceFusion kaynak kimlik temsillerini ortalar; bu işlem kişiye özel model eğitimi veya fine-tuning değildir.

## Sistem

- Windows 10/11
- NVIDIA RTX A4000 16 GB VRAM
- Anaconda veya Miniconda
- Git for Windows
- İnternet bağlantısı: ilk kurulum ve FaceFusion model ağırlıklarını indirmek için

FaceFusion'ın yayımladığı donanım rehberinde 12 GB VRAM ve üzeri önerilir. RTX A4000 16 GB bu öneriyi karşılar; gerçek performans kullanılan yüz algılayıcı, model ve fotoğraf boyutuna bağlıdır.

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

İlk yüz değiştirme işlemi sırasında FaceFusion gerekli model ağırlıklarını indirebilir. Retake şu başlangıç ayarını kullanır:

- Face swapper: `alphaface_256`
- Pixel boost: `512x512`
- Swap weight: `1.0`
- Execution provider: `cuda`
- Device: `0`
- Face selector: `one`
- Face mask: `box`

**Lisans:** FaceFusion kaynak kodu ve model ağırlıkları ayrı lisanslara sahiptir. `alphaface_256` modelinin FaceFusion model kataloğundaki lisansı Non-Commercial olarak belirtilmiştir. Bu yapılandırmayı kişisel/ticari olmayan kullanım için değerlendir; ürünü ücretli/ticari hizmete dönüştürmeden önce tüm model ve yardımcı ağırlıklarının lisanslarını tek tek doğrula. HyperSwap gibi ResearchRAIL lisanslı modeller bu yapılandırmada kullanılmaz.

## 3. Yerel servisi çalıştır

FaceFusion kurulumu bittikten sonra Retake deposundaki PowerShell'den:

```powershell
.\\tools\\start_facefusion_retake.ps1
```

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
- FaceFusion her kaynak fotoğrafta yüz kimlik temsili çıkarır ve kaynak kimliğini ortalama temsil ile birleştirir.
- FaceFusion, hedef yüz crop'una `alphaface_256` uygular.
- Retake tek portre sonucu alır ve mevcut yüz maskesiyle orijinal fotoğrafa geri birleştirir.
- Maske dışındaki piksellerin orijinalle aynı olması kontrol edilir.

FaceFusion çoklu kaynak görüntülerini bir kimlik ortalamasında birleştirir; **bu işlem Apple Retake'teki gibi kişiye özel üretken model eğitimi değildir.** Apple'ın iç mimarisi açıklanmadığı için birebir aynılık iddiasında bulunmuyoruz. Kimlik benzerliği hedeflenir ama garanti edilmez.

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
