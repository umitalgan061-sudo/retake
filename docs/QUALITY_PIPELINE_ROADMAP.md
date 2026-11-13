# Retake kalite hattı — uygulananlar ve sınırlar

## Uygulananlar

- 25/25 referans zorunluluğu; FaceLandmarker ve BlazeFace kısa/tam menzil fallback algılama.
- FaceFusion 3.9.2 XSeg/BiSeNet semantik yüz ve örtülme maskeleri.
- Yerel FaceFusion ArcFace ONNX ile 25 crop için medoid/medyan tutarlılık taraması. Aykırı fotoğraflar raporlanır, kimlik oyları 1'e düşürülür; görünüm havuzundan sessizce silinmez.
- Pixel boost ilk denemesi 1024x1024, bellek hatasında 768x768 ve sonra 512x512.
- UniFace alt yüz high-frequency detay aktarımı düşük güçlü; AlphaFace kimlik geometrisi baskın.
- Near-no-op koruması: neredeyse değişmeyen çıktı yüz değişimi gibi sunulmaz.
- GFPGAN 1.4 opsiyonel, %25 blend, varsayılan kapalı.
- Dikişsiz birleştirme: kenar ton farkının harmonik (Poisson benzeri) düzeltmesi, Laplacian çok bantlı harmanlama, netlik/gren eşitleme (`tools/retake_compositing.py`).
- Kafa oranı oturtma: 25 referansın yüz oranına göre kafayı yumuşak sönümlü çarpıtmayla uzatma/kısaltma/genişletme/daraltma.
- Hedef kırpımın yeniden örneklemesiz gönderimi (1024–2048 px) ve iki aşamalı hedef yüz algılama.

## Henüz entegre edilmeyenler

- LivePortrait göz/dudak retargeting: sabit görsel retarget API'si ayrı post-stage olarak kalibre edilmeli. Video akışındaki eye/lip flag'leri WIP olduğundan etkinleştirilmez.
- ControlNet/IP-Adapter diffusion harmonization: model lisansları, VRAM maliyeti ve kimlik kayması değerlendirilmeden varsayılan hatta eklenmez.
- Native iOS Core ML inference: mevcut web akışı Windows CUDA yerel servisine bağlı; native iOS ayrı model dönüştürme, tensor adapter, compositing ve cihaz testi gerektirir.

## Kalibrasyon notları

Dikişsiz birleştirme ve kafa oturtma sentetik görüntülerle birim testinden ve sahte motorla uçtan uca tarayıcı testinden geçti; gerçek FaceFusion/CUDA çıktısıyla ton düzeltme gücü (`HARMONIZE_STRENGTH`), kenar payı (`RIM_INSET_RATIO`) ve kafa oturtma sönümü (`estimateHeadFit` damping) aynı hedeflerde A/B incelenmelidir. Ton düzeltmesi yüzün genel ten rengini hedef fotoğrafa yaklaştırır; referans kişinin teni belirgin farklıysa bu bilinçli bir tercihtir.

ArcFace çıktısı grup içi tutarlılık taramasıdır; biyometrik kimlik kanıtı değildir. 0.75 gibi sabit bir eşik kullanılmamıştır; poz, ışık ve açı çeşitliliği gerçek 25 fotoğrafta incelenmelidir.

1024x1024 FaceFusion pixel boost, iç yüz kırpımı inference ölçeğidir; 1024 px native çıktı veya 4x upscale anlamına gelmez. GFPGAN ve LivePortrait kimlik, sakal veya göz geometrisini değiştirebilir; sonuçlar aynı hedefte A/B test edilmelidir.

Model lisansları model ağırlığı bazında kontrol edilmelidir. AlphaFace 256 Non-Commercial, UniFace 256 Unknown olarak işaretlidir; ticari kullanım hakkı varsayılmamalıdır.
