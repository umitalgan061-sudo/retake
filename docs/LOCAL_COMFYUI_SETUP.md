# Retake — Yerel ComfyUI / FLUX kurulumu

Bu akış, Retake web sayfasını bilgisayarda çalışan ComfyUI sunucusuna bağlar. Görsel düzenleme NVIDIA'nın hosted API'sine gönderilmez; FLUX.1-Kontext-dev modeli kullanıcının kendi RTX GPU'sunda çalışır. Firebase Hosting Spark planı yeterlidir; Cloud Functions, Cloudflare Worker ve NVIDIA API anahtarı gerekmez.

## Gereksinimler

- Windows üzerinde ComfyUI Windows Portable.
- ComfyUI-GGUF özel düğümü.
- `flux1-kontext-dev-Q4_K_M.gguf`, `clip_l.safetensors`, `t5xxl_fp8_e4m3fn_scaled.safetensors` ve `ae.safetensors` model dosyaları.
- Retake sayfası: `https://retake06.web.app`.

## ComfyUI'yi başlatma

PowerShell'de:

```powershell
Set-Location "C:\AI\ComfyUI_windows_portable"
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-cors-header https://retake06.web.app
```

ComfyUI arayüzü `http://127.0.0.1:8188` adresinde açılır. Terminal penceresi işlem boyunca açık kalmalıdır. Sunucu varsayılan olarak loopback/localhost üzerinde dinler; portu modem veya genel ağa açmayın.

## Tarayıcı izni

Retake HTTPS sayfası, bilgisayardaki `127.0.0.1:8188` servisine bağlanır. Edge/Chrome yerel ağ erişimi için izin sorarsa `retake06.web.app` sayfasına izin verin. Bu izin verilmezse sayfa ComfyUI durumunu okuyamaz ve düzenleme düğmesi etkinleşmez. Tarayıcı izin vermediyse site izin ayarlarından yerel ağ/loopback iznini açın ve Retake sayfasını yenileyin.

ComfyUI CORS ayarında tam olarak `https://retake06.web.app` origin'i bulunmalıdır. Başka bir domain veya Codespaces önizlemesi farklı origin kullanır; gerektiğinde ComfyUI'yi o origin için ayrı başlatın.

## İşlem akışı

- Retake, hedef fotoğrafın yalnızca seçilen yüz çevresindeki crop'unu hazırlar.
- Retake, aynı kişiye ait 25 referansı 5×5 panoya yerleştirir.
- İki PNG dosyası ComfyUI'nin `/upload/image` uç noktasına yüklenir.
- `workflows/retake-flux-api.json` iş akışı `/prompt` üzerinden kuyruğa gönderilir. Bu dosya 191 ve 192 kimlikli `LoadImage` düğümlerini kullanır ve görüntüleri 146 kimlikli `ImageStitch` düğümünde birleştirir.
- Retake `/history/{prompt_id}` ile sonucu izler, `/view` üzerinden PNG çıktısını alır ve seçili yüz maskesiyle orijinal hedef fotoğrafa birleştirir.

ComfyUI'nin HTTP API uç noktaları arasında `/upload/image`, `/prompt`, `/history/{prompt_id}` ve `/view` bulunur. Resmî rota belgeleri: https://github.com/Comfy-Org/docs/blob/main/development/comfyui-server/comms_routes.mdx

## Gizlilik notları

Model çıkarımı yereldir: hedef crop ve referans panosu ComfyUI'nin bu bilgisayardaki loopback adresine gider; NVIDIA hosted API çağrısı yapılmaz. Ancak Retake'in referansları yüklediği kaynak ayrıca önemlidir: `reference-faces/manifest.json` ve fotoğrafları herkese açık Firebase Hosting'e koyulursa bu dosyalar web'den erişilebilir olur. Kişisel fotoğrafları public GitHub'a veya public Hosting'e eklemeyin; özel referans deposu/kimlik doğrulamalı kaynak ayrıca yapılandırılmalıdır.

FLUX.1-Kontext-dev ağırlıkları ticari olmayan kullanım lisansına tabidir. Model çıktısı belirli bir kimliği doğru biçimde yeniden üretmeyi garanti etmez.

## Sorun giderme

- **ComfyUI bağlantısı kurulamadı:** ComfyUI terminalinin açık olduğunu ve `http://127.0.0.1:8188/system_stats` adresinin yerel tarayıcıda açıldığını kontrol edin.
- **CORS hatası:** ComfyUI'yi `--enable-cors-header https://retake06.web.app` ile yeniden başlatın.
- **Yerel ağ izni:** Edge/Chrome'da site için Local Network / Loopback iznini verin, ardından sayfayı yenileyin.
- **UnetLoaderGGUF bulunamadı:** ComfyUI-GGUF kurulumunu kontrol edin ve ComfyUI'yi yeniden başlatın.
- **Model veya metin kodlayıcı bulunamadı:** dosyaların ComfyUI'nin `models/unet`, `models/text_encoders` ve `models/vae` klasörlerinde olduğunu doğrulayın.
- **VRAM yetersiz:** önce 1024 yerine daha küçük test görselleriyle deneyin; KSampler adımlarını düşürmek VRAM tepe kullanımını her durumda azaltmaz.
