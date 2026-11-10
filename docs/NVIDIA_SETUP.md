# NVIDIA Hosted API kurulumu artık kullanılmıyor

Retake'in web sürümü, FLUX.1-Kontext-dev işleme için NVIDIA hosted API / Cloudflare Worker yolunu yerel ComfyUI + GGUF akışıyla değiştirdi.

Bu belge eski NVIDIA API anahtarı ve Cloudflare Worker kurulumunu anlatır; yeni kurulumda bu adımları uygulamayın. API anahtarını Retake sayfasına veya HTML/JavaScript kaynaklarına eklemeyin.

Güncel kurulum için: [LOCAL_COMFYUI_SETUP.md](LOCAL_COMFYUI_SETUP.md).

Yeni akışta:
- FLUX.1-Kontext-dev GGUF modeli kullanıcının bilgisayarındaki RTX GPU üzerinde çalışır.
- Retake tarayıcıdan `http://127.0.0.1:8188` adresindeki ComfyUI API'sine bağlanır.
- NVIDIA hosted API anahtarı ve Firebase Functions gerekmez.
- Firebase Hosting Spark planı korunabilir.
