# Face Swap

Yapay zeka destekli gerçek zamanlı yüz değiştirme uygulaması.

## Özellikler
- MediaPipe Face Landmarker ile 468 noktalı yüz landmark tespiti
- 20 referans yüz seçeneği
- Gerçek zamanlı yüz değiştirme (Delaunay triangulation + affine warp)
- Fotoğraf çekme ve indirme
- Ön/arka kamera geçişi
- Karanlık/açık tema
- PWA desteği
- Mobil uyumlu

## Canlı Demo
https://retake06.web.app

## Kullanım
1. Sayfa yüklendiğinde kamera izni verin
2. Alt kısımda 20 referans yüzdenden birini seçin
3. Kameradaki yüzünüz anında seçilen yüzle değişir
4. "Fotoğraf Çek" ile anı kaydedin, "İndir" ile PNG kaydedin

## Teknolojiler
- MediaPipe Tasks Vision (Face Landmarker)
- HTML5 Canvas 2D affine warp
- Vanilla JavaScript (ES modules)
- CSS Custom Properties (theming)