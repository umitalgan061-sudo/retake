# Retake Web Studio — teknik yapı

Son güncelleme: 9 Kasım 2026

## Kamera yok

Web ve iOS uygulamalarında kamera açılmaz. `getUserMedia`, WebRTC canlı video, `AVCaptureSession` ve kamera izni eklenemez. Yalnızca fotoğraf yükleme/PhotosPicker desteklenir.

## Kullanıcı akışı

1. Kullanıcı hedef fotoğrafını dosya seçiciyle veya sürükle-bırakla açar.
2. Dosya türü ve 25 MB üst sınırı doğrulanır. Hedef görsel yalnızca sayfa belleğinde tutulur.
3. MediaPipe Face Landmarker birden fazla yüzü cihazda algılar.
4. Kullanıcı yüz kutusuna veya yüz seçim düğmesine basarak hedef yüzü seçer.
5. Kullanıcı aynı kişiye ait en fazla 20 adet tek yüz içeren kimlik referansı seçer.
6. Referans görseller IndexedDB içinde yerel tarayıcı depolamasına kaydedilir ve sonraki oturumda geri yüklenir. Şu an tek yüz kontrolü vardır; aynı kimlik doğrulaması henüz eklenmemiştir.

## Modüller

- \`index.html\`: Anlamsal, Türkçe ve erişilebilir sayfa iskeleti.
- \`styles.css\`: Responsive masaüstü/tablet/mobil düzen, koyu/açık tema ve hareket azaltma tercihleri.
- \`app.js\`: MediaPipe yaşam döngüsü, dosya seçimi, hedef seçim, IndexedDB ve PWA etkileşimleri.
- \`src/core.js\`: Saf ve test edilebilir görsel doğrulama, landmark sınır kutusu, aspect-ratio ve dosya boyutu yardımcıları.
- \`sw.js\`: Uygulama kabuğu ve açıkça izin verilen MediaPipe varlıkları için önbellek.
- \`manifest.webmanifest\`: PWA adı, başlangıç URL'si, kapsam ve tema.
- \`tests/core.test.js\`: Node.js yerleşik test koşucusuyla yardımcı fonksiyon testleri.

## Yüz algılama

MediaPipe Tasks Vision sürümü CDN URL'sinde sabitlenmiştir: \`@mediapipe/tasks-vision@1.2.2\`.

Görsel, ara bir Image/ImageData sarmalayıcısına çevrilmeden doğrudan \`FaceLandmarker.detect(HTMLImageElement)\` metoduna verilir. Öncelikle GPU delegate denenir; başlatılamazsa CPU ile tekrar denenir. MediaPipe giriş görüntüsünü cihazda işler. Kütüphane çalışma/performans ölçümlerini Google'a iletebilir; ayrıntı için https://goo.gle/mediapipe-privacy adresi kullanılır.

## Yerel depolama ve gizlilik

- Hedef fotoğraf IndexedDB'ye yazılmaz; sayfa oturumu sona erdiğinde bellek referansı serbest bırakılır.
- Referans fotoğrafları IndexedDB'de blob olarak saklanır. Uygulama fotoğraf blob'larını herhangi bir Face Swap servisine veya sunucu API'sine göndermez.
- Tarayıcı depolamasının temizlenmesi veya özel gezinme modu referans fotoğraflarını silebilir. Önemli dosyaların ayrıca yedeklenmesi kullanıcı sorumluluğundadır.
- Service Worker ağ üzerinden yalnızca uygulama kaynakları, sabitlenmiş MediaPipe paket yolu ve resmi model dosyasını önbelleğe alır. Fotoğraf verileri cache'e eklenmez.

## PWA ve çevrimdışı davranış

İlk kurulumda HTTPS/localhost üzerinden çevrimiçi açılış gerekir. Service Worker uygulama kabuğunu, MediaPipe WASM dosyalarını ve model dosyasını indirip cache'e ekler. Bu varlıklar önbelleğe alındıktan sonra desteklenen tarayıcılarda arayüz ve yüz algılama çevrimdışı da çalışabilir. Tarayıcı cache'i her zaman silebilir; çevrimdışı kullanım garantisi disk alanı ve tarayıcı politikalarına bağlıdır.

## Yüz dışı piksel koruma

Üretim hattı sadece seçilen yüz crop'unda çalışır. 20 referans aynı kimlikten olmalı ve kalite ağırlıklı tek kimlik profiline dönüştürülmelidir. Hedef fotoğraf poz/ışık/kompozisyonu sağlar. Kapalı gözler varsa uygun açık göz referanslarından doğal göz açıklığı üretilir; ifade değişimi yalnızca gerektiğinde yapılır. Face parsing ve occlusion mask saç, gözlük, eller, nesneler, arka plan ve diğer yüzleri dışarıda bırakmalıdır. Sonuç orijinal çözünürlüklü fotoğrafa yalnızca yüz maskesinin içine birleştirilir; maskenin dışı piksel olarak değişmemelidir.

## Gerçek Face Swap motoru neden kapalı?

MediaPipe Face Landmarker, yüz landmark'ı ve sınırlarını verir; yüz kimliğini başka bir fotoğrafa aktaran bir Face Swap modeli değildir. Eski Canvas affine-warp demosu kaldırıldı; çünkü gerçek bir kimlik korumalı model çıktısı üretmiyordu. Web sürümünde işlem/indirme düğmeleri, model ve model-özel inference/kompozitleme hattı doğrulanana kadar devre dışıdır.

Web gerçek AI entegrasyonu için modelin lisans zinciri, WebAssembly/WebGPU uyumluluğu, bellek/performans sınırı, 20-referans identity aggregation, expression retargeting ve yüz maskeli birleştirme ayrıca doğrulanmalıdır. iOS Core ML modeli tek başına tarayıcıda çalıştırılamaz.

## Geliştirici doğrulaması

Node.js 22 veya üstü:

\`\`\`bash
npm test
npm run validate
node --check app.js
node --check sw.js
\`\`\`

GitHub Actions aynı kontrolleri \`.github/workflows/web-quality.yml\` üzerinden yürütür.