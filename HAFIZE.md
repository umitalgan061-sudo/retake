# Proje hedefi

Tek sayfalık, mobil uyumlu bir **Face Swap / Yüz Değiştirme** uygulaması.

Özellikler:
1. MediaPipe Face Landmarker (WASM/GPU) ile gerçek zamanlı 468 noktalı yüz landmark tespiti
2. Depoda `reference-faces/` klasöründeki **20 referans yüz** dosyasından seçim
3. Seçilen referans yüzün landmarkları ile kamera yüz landmarkları arası **Delaunay triangulation + affine warp** ile yüz değiştirme
4. Gerçek zamanlı önizleme (kamera akışında anında swap)
5. Fotoğraf çekme ve PNG indirme
6. Ön/arka kamera geçişi
7. Karanlık/açık tema (localStorage)
8. PWA manifest ile "Uygulamaya Ekle" desteği
9. Klavye erişilebilir referans yüz seçimi
10. Tek dosya (index.html), CSS/JS inline, CDN'den MediaPipe yükleme

Bağımlılıklar: @mediapipe/tasks-vision (CDN), referans yüz görselleri (reference-faces/ klasörü)

Model: MediaPipe Face Landmarker (face_landmarker.task) - Google CDN'den yüklenir
Referans yüzler: Depodaki 20 dosya (JPEG/PNG)

Hedef: Kullanıcı kamerasındaki yüzü, veritabanındaki 20 referans yüzdenden seçilenle anında değiştirir.