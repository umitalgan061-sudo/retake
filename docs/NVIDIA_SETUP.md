# NVIDIA / ComfyUI / FLUX — eski kurulum, artık kullanılmıyor

Retake'in güncel web yüz değiştirme motoru NVIDIA hosted API, Cloudflare Worker, ComfyUI veya FLUX değildir. Web arayüzü yerel FaceFusion 3.9.2 servisine bağlanır.

Yeni kurulum: [LOCAL_FACEFUSION_SETUP.md](LOCAL_FACEFUSION_SETUP.md).

Eski ComfyUI/FLUX model dosyalarını bilgisayarından silmek zorunda değilsin; yeni Retake akışı onları çağırmaz. Yerel FaceFusion servisi yalnızca `127.0.0.1:8765` üzerinde çalıştırılmalıdır. Firebase Functions veya ücretli NVIDIA API anahtarı gerekmez.
