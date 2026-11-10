# Retake — Yerel FaceFusion kurulumu (Windows)

Bu sürüm, FaceFusion 3.9.2 üzerinde 25 referanslı iki geçişli bir yüz profili oluşturur. 25 referansın tamamı FaceLandmarker ile algılanır; gerekirse BlazeFace yedek algılayıcısı kullanılarak ortak göz koordinatına hizalanır. Netlik, açı, yüz alanı ve pozlama puanlarına göre 1–3 arası ağırlık alır. Ağırlıklı kimlik ortalaması ve hizalanmış görünüm ortalaması ayrı ayrı hazırlanır. Bu işlem kişiye özel model eğitimi veya LoRA fine-tuning değildir.

## Sistem

- Windows 10/11
- NVIDIA RTX A4000 16 GB VRAM
- Anaconda veya Miniconda
- Git for Windows
- FFmpeg
- İlk kurulum ve model ağırlıkları için internet bağlantısı

FaceFusion'ın Windows kurulum yönergeleri Git, Conda ve FFmpeg gerektirir. Resmî yönergeler: https://docs.facefusion.io/installation/platform/windows

## 1. Kurulumu yap

PowerShell'de Retake deposuna geç:

    Set-ExecutionPolicy -Scope Process Bypass
    .\tools\install_facefusion_windows.ps1

Betik:
1. Ayrı bir facefusion Conda ortamı oluşturur (Python 3.12).
2. FaceFusion 3.9.2 sürümünü indirir.
3. CUDA 12.9 runtime ve cuDNN 9.10 kurar.
4. FaceFusion CUDA bağımlılıklarını yükler.
5. ONNX Runtime CUDAExecutionProvider sağlayıcısını doğrular.

ComfyUI'nin gömülü Python ortamına FaceFusion kurma; iki ortam birbirinden ayrı kalmalı.

## 2. İki model ve lisans

Hibrit profil iki ayrı geçiş kullanır:

- Kimlik geçişi: alphaface_256. 25 kaynak yüzün embedding ortalamasını kullanır. Referanslar kalite/açı ağırlıklarıyla kaynak listesinde tekrarlanır; her kabul edilen fotoğraf en az bir oy alır.
- Görünüm geçişi: uniface_256. 25 fotoğrafın hizalanmış ve ağırlıklı piksel ortalamasını kaynak yüz olarak kullanır. Bu geçiş alt yüz, bıyık ve sakal görünümünü desteklemek için kullanılır.
- Pixel boost varsayılanı 768x768'dir. CUDA bellek hatasında her geçiş 512x512 ile yeniden denenir.
- Face swapper weight: 1.0.
- Execution provider: cuda, device 0.
- Face selector: one.
- Face mask types: `box`, `occlusion`, `region`.
- Face occluder: `xseg_1`; face parser: `bisenet_resnet_34`.
- Parser regions: skin, eyebrows, eyes, nose, mouth and lips; `glasses` is intentionally excluded to preserve target eyewear.
- Final PNG includes an alpha matte derived from pixels actually changed by the two FaceFusion passes. The browser uses this matte instead of the old broad ellipse when available.

Lisanslar model bazında değerlendirilmelidir. FaceFusion kataloğunda alphaface_256 Non-Commercial, uniface_256 ise Unknown lisans durumuyla listelenir. Ticari kullanım hakkı olduğunu varsayma; model ve yardımcı ağırlıkların lisansını ayrı ayrı doğrula.

## 3. Yerel servisi çalıştır

Retake deposunda PowerShell'den:

    .\tools\start_facefusion_retake.ps1

Varsayılan çözünürlük 768x768'dir. Daha düşük bellek kullanımı için:

    .\tools\start_facefusion_retake.ps1 -FaceSwapperPixelBoost 512x512

Servis http://127.0.0.1:8765 üzerinde çalışır ve yalnızca bilgisayarın loopback arayüzünü dinler. Bu PowerShell penceresini işlem sırasında açık bırak. Modemden port yönlendirme yapma ve Windows güvenlik duvarında bu portu genel ağa açma.

Sağlık kontrolü:

    http://127.0.0.1:8765/health

Beklenen değerler: ok=true, cuda=true, model=retake_hybrid_25_reference_v3, identity_model=alphaface_256, appearance_model=uniface_256.

## 4. Retake'i çalıştır

1. Firebase'de https://retake06.web.app adresini aç.
2. Tarayıcı retake06.web.app için yerel ağ erişimi isterse izin ver.
3. Referans havuzundaki 25 fotoğrafın aynı kişiye ait olduğundan emin ol.
4. Hedef fotoğrafı yükle ve değiştirilecek yüzü seç.
5. Yerel işlem onayını kabul et.

Uygulama profil oluşturmayı ilk kullanımda yapar. Kırpılmış referans yüz görselleri, ağırlıklar ve hizalanmış görünüm profili tarayıcının IndexedDB alanında cihazda saklanır. Aynı 25 referans kümesiyle sayfa yenilendiğinde profil yeniden hizalanmaz; kayıtlı yerel profil tekrar kullanılır. Referans fotoğraflarından herhangi biri değişirse profil anahtarı değişir ve profil yeniden oluşturulur. Tarayıcı site verileri temizlenirse yerel profil de silinir; profil başka tarayıcı veya cihaza otomatik taşınmaz.

## 5. Profilin nasıl oluştuğu

- MediaPipe FaceLandmarker 478 noktalı yüz ağı ile önce referansları algılar. Başarısız kalanlarda BlazeFace FaceDetector önce kısa menzil, sonra tam menzil modeliyle düşük eşikte çalışır ve altı anahtar nokta (iki göz, burun, ağız ve iki kulak) üzerinden yedek hizalama kurar. Yedek hizalanan görsellerin kalite ağırlığı en fazla 2 ile sınırlandırılır.
- Her referans yüzü göz işaretlerine göre ortak 512×682 tuvale hizalanır.
- Netlik Laplacian varyansı ile ölçülür; yüz alanı, frontalness ve pozlama da puanlanır.
- 25/25 yüz profile katılır; FaceLandmarker hizalamaları kaliteye göre 1–3 oy, BlazeFace yedek hizalamaları en fazla 2 oy alır. Bir referans hâlâ hizalanamazsa uygulama dosya adını ve iki algılayıcının hata nedenlerini gösterir, eksik profille FaceFusion çalıştırmaz.
- Ağırlıklı ortalama RGBA görünüm profili oluşturulur. Aşırı farklı pozlar ve ifadeler bu ortalamayı yumuşatabilir.
- Aynı ağırlıklar FaceFusion kaynak listesinde tekrarlanarak AlphaFace'in 512 boyutlu kimlik embedding ortalamasına uygulanır.
- AlphaFace kimlik geçişi üretir.
- UniFace, tüm referansların hizalanmış piksel harmanını kaynak görüntü olarak kullanarak görünüm geçişi üretir.
- UniFace çıktısının alt yüz alanı AlphaFace çıktısına yumuşak eliptik maske ile karıştırılır. Kimlik geçişi ana yüzü, görünüm geçişi bıyık/sakal/çene dokusunu destekler.
- Tek hibrit PNG hedef crop'a uygulanır; orijinal fotoğrafın geri kalanı korunur ve maske dışı piksel bütünlüğü kontrol edilir.

Bu yöntem 25 fotoğrafı bir yüz görseli gibi körlemesine yan yana koymaz. Her fotoğraf aynı hizalama düzlemine taşınır ve ağırlıklı bir profil bileşeni olur. Sonuç kalitesi referans fotoğrafların netliğine, poz çeşitliliğine, ışığına ve sakal/bıyığın kaç fotoğrafta belirgin olduğuna bağlıdır; kimlik benzerliği garanti edilmez.

## 6. Sorun giderme

### CUDAExecutionProvider bulunamadı

    conda run -n facefusion python -c "import onnxruntime as ort; print(ort.get_available_providers())"

Çıktıda CUDAExecutionProvider bulunmalı. Kurulum betiğini tekrar çalıştır; ortamı rastgele ONNX Runtime paketleriyle karıştırma.

### Tarayıcı yerel servise bağlanamıyor

- FaceFusion PowerShell penceresi açık olmalı.
- http://127.0.0.1:8765/health yerel tarayıcıda açılmalı.
- Edge/Chrome site izinlerinde retake06.web.app için yerel ağ erişimi verilmelidir.
- Servis loopback'te kalmalı; 0.0.0.0 kullanma.

### 25 referansın tamamı hizalanamadı

Profil 25/25 zorunludur. Hata mesajında dosya adı ve FaceLandmarker/BlazeFace kısa ve tam menzil sonuçları görünür. İlgili fotoğrafı daha yüksek çözünürlüklü, yüzün daha büyük ve net göründüğü bir sürümle değiştir; aşırı yan profil, yüzün kapanması veya çok küçük yüz her iki algılayıcıyı da başarısız bırakabilir. Yeni algılama zinciri nedeniyle IndexedDB'deki eski 19/25 profil önbelleği kullanılmaz; ilk yeni işlem profili baştan oluşturur.

### Bellek / işlem hatası

Önce ComfyUI/FLUX gibi ağır GPU süreçlerini kapat. Sunucu CUDA bellek hatasında 768x768 ayarı için 512x512 yeniden denemesi yapar. Her iki geçişte de hata varsa FaceFusion PowerShell kayıtlarının son satırlarını sakla.

## 7. Gizlilik ve fotoğraf depolama

Profil oluşturma ve çıkarım yerel bilgisayarda yapılır. Ancak bu deponun önceki sürümlerinde reference-faces altındaki 25 JPEG dosyası public GitHub deposuna eklenmiştir. Güncel repo ve Firebase Hosting bu fotoğrafları herkese açık sunabilir; geçmiş Git commit'lerinde de bulunabilir. Yerel çıkarım bu dosyaları public olmaktan çıkarmaz. Bu nedenle referans fotoğraflarını gizli kabul etme. Fotoğrafların özel yerel depoya taşınması ve Git geçmişinin dikkatli temizlenmesi ayrı bir güvenlik çalışmasıdır.

Hedef fotoğrafın tamamı gönderilmez; seçilen yüz crop'u gönderilir. Referans crop'ları ve hizalanmış profil aynı bilgisayardaki 127.0.0.1:8765 servisine gönderilir. Servis dış ağa açılmamalıdır.
