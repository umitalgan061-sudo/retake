# Retake - AI Face Swap

Yüz değiştirme web uygulaması. Kullanıcı fotoğraf yükler, 20 referans yüzden birini seçer, AI yüzü değiştirir.

## Canlı
https://retake06.web.app

## Teknolojiler
- MediaPipe Face Landmarker (Tasks Vision)
- Canvas 2D affine warp
- Vanilla ES modules
- Firebase Hosting

## Yerel Çalıştırma
```bash
npx serve .
```
Sonra `http://localhost:3000` açın.

## Referans Yüzler
`reference-faces/` klasöründe 20 adet yüz fotoğrafı bulunur. Bu dosyalar depoda dahilildir (gitignore), ancak Firebase deploy ile birlikte yüklenir.

## Lisans
MIT