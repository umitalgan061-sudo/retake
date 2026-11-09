# Retake - AI Face Swap Uygulaması

## Amaç
Kullanıcının yüklediği fotoğraftaki yüzü, hazır 20 referans yüzünden birine MediaPipe Face Landmarker (468 landmark) + Delaunay triangulation + affine warp ile değiştiren, tamamen client-side çalışan bir web uygulaması.

## Özellikler
- Fotoğraf yükleme (drag-drop, tıklama) — JPG/PNG/WebP, max 10 MB
- 20 referans yüz (reference-faces/ klasöründen)
- MediaPipe Face Landmarker (GPU delegate, WASM)
- 68 ana landmark üzerinden Delaunay triangulation (hardcoded triangles)
- Her üçgen için affine warp ile referans yüz dokusunu kullanıcı yüz geometrisine taşıma
- Sonuç önizleme + PNG indirme
- Karanlık/açık tema (localStorage + system preference)
- PWA manifest (offline çalışmaz, model CDN'den gelir)
- Tamamen client-side, veri sunucuya gitmez

## Teknik Detaylar
- Model: `face_landmarker.task` (Google CDN, float16)
- Landmark: 468 nokta → 68 kritik nokta (dlib stil) mapping
- Triangulation: 68 nokta için önceden hesaplanmış 100+ üçgen
- Warp: Her üçgen için affine transform (canvas 2D context clip + transform)
- Referans yüzler: `reference-faces/face_01.jpg` ... `face_20.jpg` (depoda mevcut)

## Geliştirme Notları
- Canlı kamera KULLANILMAZ, sadece dosya yükleme
- GPU delegate isteğe bağlı, fallback CPU
- 10 MB dosya limiti
- Model yükleme ~2-3 saniye (ilk ziyaret)
- Referans yüz landmarkları başlangıçta precompute edilir

## Hedef Kitle
Eğlence, sosyal medya içeriği, prototype/demo amaçlı.

## Kısıtlar
- Yüz bulunamazsa hata verir
- Çok yanlı yüzler, aşırı gölge, maske ile çalışmaz
- Sadece 1 yüz desteklenir (numFaces: 1)
- Mobilde GPU delegate bazen CPU'ya düşer