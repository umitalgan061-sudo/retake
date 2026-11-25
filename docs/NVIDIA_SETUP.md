# Retake yerel NVIDIA NIM görüntü düzenleyicileri

Ana motor FaceFusion 3.9.2, CUDA, 25 fotoğraf, ArcFace ve yerel kompozisyondur. API anahtarını tarayıcıya koyma. Eski Cloudflare/hosted FLUX akışı devre dışıdır.

Yeni isteğe bağlı finalist katmanı, NVIDIA'nın [resmî görüntü düzenleme API'sini](https://docs.nvidia.com/nim/visual-genai/latest/api/openai-image-editing.html) kullanır. Modeller ayrı yerel NIM konteynerinde kurulu ve hazır olmalıdır; otomatik GPU kurulumu veya ücretsiz hosted endpoint garantisi yoktur.

| Model | Rol | Durum |
|---|---|---|
| Qwen Image Edit 2511 / 2509 / temel | Referanslı yüz ayrıntısı | Yerel API adaptörü |
| FLUX 3 Image Klein / Base | Çoklu görselden ayrıntı | Yerel API adaptörü |
| FLUX.2 Klein 4B | Daha hafif çoklu görsel düzenleme | Yerel API adaptörü |
| FLUX.1 Kontext dev | Tek fotoğrafta fotogerçekçilik | Yerel API adaptörü |
| Nemotron Super VL, Mistral Large 4, Step 5, Gemma 4 | İsteğe bağlı görsel kalite yorumu | Aday, henüz bağlı değil |
| Relighting, 3D Body Pose, Video Super Resolution | İleride video/ışık/poz işlemleri | Aday, henüz bağlı değil |

NVIDIA'nın [kurulum rehberi](https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html) ve GPU destek matrisi üzerinden kendi donanımına uygun imajı, lisansı ve sürümü seç. Her servisi yalnız yerel 127.0.0.1 üzerinde ayrı porta bağla. Örnek, Qwen gerçekten 8001'de hazırsa PowerShell:

```powershell
$env:RETAKE_NIM_EDITORS = "qwen/qwen-image-edit-2511=8001"
.\tools\start_facefusion_retake.ps1
```

Birden fazla hazır servis için virgülle model=port eklenebilir. Hiç ayarlanmazsa yalnız mevcut FaceFusion çalışır. İstekler sabit loopback adresine gider, URL/anahtar tarayıcıdan alınmaz. Port sağlık testi ve çıktı biçim/ölçü kontrolü uygulanır. Fotoğrafın tamamı buluta gönderilmez; yerel editöre yalnız yüz kırpımları gider. İzinli mevcut alfa maskesinin dışı korunur. Modeller yalnız Ultra finalistinde denenir; ArcFace ve doku kontrolü yeterli değilse elenir. Gerçek modelin seçildiği, yalnız doğrulanmış final başlığında gösterilir.

Blueprint değerlendirmesi: RAG (referans kayıtlarının ve kalite kanıtlarının ayrılması), NemoClaw ajan değerlendirmesi (aday ve hata testleri), DocsFab (sürüm/test belgeleri), CUDA Nsight (profil çıkarma) bu proje için yararlı yaklaşımlar. Bunlar resmî Blueprint dağıtımı değildir. Biyoloji, finans ve konuşma modelleri sabit fotoğraf düzenleme işine doğrudan yarar sağlamaz. Bu sürümde gerçek NIM GPU servisi kurulmadı ve fiziksel fotoğraf testi yapılmadı.

Birebir hedef dosyası bankada varsa tüm 25 fotoğrafın doğrulaması korunur; bağımsız 24 fotoğraf alternatif havuzu denenir ve birebir hedef doğrudan gerçek doku / NIM referansından çıkarılır. Son teslimde yüz içi piksel farkı ölçülür. Piksel farkı görüntü güzelliğinin garantisi değildir; eşik sadece değişmeyen sonucu ayıklar.

## RTX A4000 16 GB: gerçek donanım sınırı

NVIDIA'nın [güncel NIM destek tablosu](https://docs.nvidia.com/nim/visual-genai/latest/support-matrix.html) (25 Kasım 2026 kontrolü): Qwen-Image-Edit NIM ailesi en az 80 GB VRAM; FLUX.2 Klein ve FLUX.3 Image Klein NIM en az 48 GB VRAM ister. FLUX.1 Kontext NIM genel minimumu 16 GB olsa da RTX A4000 için destekli hazır TensorRT profili doğrulanmadı; FaceFusion ile eşzamanlı bellek ihtiyacı da hesaba katılmalı. Bu nedenle 16 GB sistemde bu NIM konteynerlerini otomatik başlatma veya satın alma yapılmaz. Daha düşük bellekli bağımsız açık ağırlık/quantization çalıştırıcıları ayrı entegrasyon ve gerçek GPU testi ister; resmi NIM ile eşdeğer değildir.

Mevcut ücretsiz yerel FaceFusion + isteğe bağlı kişisel LoRA/FaceID gerçekçilik yolu varsayılan kalır. NIM model ismi listelenmiş olması çalıştığı anlamına gelmez. Bir modeli yalnız gerçek yerel servis hazır ve örnek düzenleme başarılıysa etkinleştir; gerçek kişisel fotoğrafları hosted endpoint'lere otomatik gönderme.

Kabul testi: (1) bankadaki birebir dosya, (2) yeniden JPEG kodlanmış aynı fotoğraf, (3) bankada olmayan fotoğraf. Her biri için yüz içi değişim oranı, ortalama RGB farkı, orijinal ve sonuç kimlik cosine, maskenin dışındaki değişen piksel sayısı, doğal sakal/çene/cilt A/B incelemesi kaydedilir. Raporlanamayan veya anlamlı değişim üretmeyen örnek başarılı sayılmaz. Test resimleri ve biyometrik veriler commit edilmez.
