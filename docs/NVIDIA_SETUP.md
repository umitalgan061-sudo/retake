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
