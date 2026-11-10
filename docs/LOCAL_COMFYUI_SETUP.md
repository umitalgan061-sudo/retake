# ComfyUI / FLUX kurulumu — eski ve artık kullanılmıyor

Retake'in web yüz değiştirme akışı ComfyUI + FLUX.1-Kontext-dev kullanımından FaceFusion'a taşınmıştır.

Yeni kurulum:
- [Yerel FaceFusion Windows kurulumu](./LOCAL_FACEFUSION_SETUP.md)
- Başlatma betiği: `tools/start_facefusion_retake.ps1`
- Yerel servis: `http://127.0.0.1:8765`

ComfyUI'yi veya FLUX model dosyalarını bilgisayarından silmen gerekmiyor; Retake'in yeni web akışı bunları çağırmaz. Önce FaceFusion'ı çalışır hâle getirip yeni akışın çıktısını doğrula. Eski ComfyUI portu 8188 ve eski workflow JSON dosyası artık Retake işlemi için kullanılmaz.
