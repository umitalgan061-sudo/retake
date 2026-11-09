# Retake Web Studio — teknik yapı

Son güncelleme: 9 Kasım 2026

## Kamera yok

Web ve iOS uygulamalarında kamera açılmaz. `getUserMedia`, WebRTC canlı video, `AVCaptureSession` ve kamera izni eklenemez. Yalnızca fotoğraf yükleme/PhotosPicker desteklenir.

## Kullanıcı akışı

1. Kullanıcı hedef fotoğrafını dosya seçiciyle veya sürükle-bırakla açar.
2. Dosya türü ve 25 MB üst sınırı doğrulanır. Hedef görsel yalnızca sayfa belleğinde tutulur.
3. MediaPipe Face Landmarker birden fazla yüzü cihazda algılar.
4. Kullanıcı yüz kutusuna veya yüz seçim düğmesine basarak hedef yüzü seçer.
5. Uygulama önceden kaydedilmiş 20 referansı `GET /api/images/{id}` üzerinden otomatik çeker.
6. Referanslar salt okunur gösterilir; kullanıcıdan yeniden yükleme istenmez. Fotoğraflar IndexedDB'ye yazılmaz ve Service Worker Cache Storage'a alınmaz.

## Modüller

- \`index.html\`: Anlamsal, Türkçe ve erişilebilir sayfa iskeleti.
- \`styles.css\`: Responsive masaüstü/tablet/mobil düzen, koyu/açık tema ve hareket azaltma tercihleri.
- \`app.js\`: MediaPipe yaşam döngüsü, hedef fotoğraf seçimi, veritabanı referansları, NVIDIA isteği ve sonuç dışa aktarma.
- \`src/nvidia-pipeline.js\`: Hedef yüz crop'u, 20 fotoğraflı kimlik panosu ve yumuşak eliptik yüz maskesi.
- \`src/face-only.js\`: RGBA kompozitleme ve maske dışı piksel bütünlüğü kontrolü.
- \`functions/index.js\`: NVIDIA API anahtarını Secret Manager'dan alan sunucu proxy'si.
- \`src/core.js\`: Görsel doğrulama, landmark sınır kutusu, aspect-ratio ve dosya boyutu yardımcıları.
- \`sw.js\`: Uygulama kabuğu ve açıkça izin verilen MediaPipe varlıkları için önbellek.
- \`manifest.webmanifest\`: PWA adı, başlangıç URL'si, kapsam ve tema.
- \`tests/core.test.js\`: Node.js yerleşik test koşucusuyla yardımcı fonksiyon testleri.

## Yüz algılama

MediaPipe Tasks Vision sürümü CDN URL'sinde sabitlenmiştir: \`@mediapipe/tasks-vision@1.2.2\`.

Görsel, ara bir Image/ImageData sarmalayıcısına çevrilmeden doğrudan \`FaceLandmarker.detect(HTMLImageElement)\` metoduna verilir. Öncelikle GPU delegate denenir; başlatılamazsa CPU ile tekrar denenir. MediaPipe giriş görüntüsünü cihazda işler. Kütüphane çalışma/performans ölçümlerini Google'a iletebilir; ayrıntı için https://goo.gle/mediapipe-privacy adresi kullanılır.

## Yerel depolama ve gizlilik

- Hedef fotoğraf IndexedDB'ye yazılmaz; sayfa oturumu sona erdiğinde bellek referansı serbest bırakılır.
- Referans fotoğrafları mevcut API üzerinden salt okunur çekilir; uygulama bunları yeniden yüklemez veya IndexedDB'ye yazmaz.
- API varsayılan olarak aynı origin'de `/api/images/{id}` yolunda bulunmalıdır. Farklı origin kullanılıyorsa `meta[name="retake-api-base"]` ve CSP `connect-src` birlikte yapılandırılmalıdır.
- Service Worker `/api/images/` yanıtlarını önbelleğe almaz; yüz görselleri Cache Storage'a kaydedilmez.
- Tarayıcı depolamasının temizlenmesi veya özel gezinme modu referans fotoğraflarını silebilir. Önemli dosyaların ayrıca yedeklenmesi kullanıcı sorumluluğundadır.
- Service Worker yalnızca uygulama kabuğunu, sabitlenmiş MediaPipe paket yolunu ve resmi landmark modelini önbelleğe alır. `/api/images/` altındaki fotoğraflar cache'e eklenmez.

## PWA ve çevrimdışı davranış

İlk kurulumda HTTPS/localhost üzerinden çevrimiçi açılış gerekir. Service Worker uygulama kabuğunu, MediaPipe WASM dosyalarını ve model dosyasını indirip cache'e ekler. Bu varlıklar önbelleğe alındıktan sonra desteklenen tarayıcılarda arayüz ve yüz algılama çevrimdışı da çalışabilir. Tarayıcı cache'i her zaman silebilir; çevrimdışı kullanım garantisi disk alanı ve tarayıcı politikalarına bağlıdır.

## NVIDIA FLUX deneysel düzenleme hattı

Web sürümü, kullanıcı onayından sonra seçili hedef yüz crop'unu ve veritabanındaki 20 referansı tek bir 1568×1024 panoda birleştirir. Pano \`functions/index.js\` üzerinden NVIDIA NVCF Asset API'ye yüklenir ve FLUX.1-Kontext-dev endpoint'ine gönderilir. Tam hedef fotoğrafın geri kalanı NVIDIA'ya gönderilmez.

FLUX.1-Kontext-dev özel bir Face Swap/kimlik encoder modeli değildir. Yirmi referansı tek bir kimlik profiline dönüştürmez ve kimlik benzerliğini garanti etmez; prompt ve görsel panoyla yönlendirilen deneysel bir görsel düzenleme yöntemidir. Model kartı ticari olmayan kullanıma işaret eder; dağıtım öncesinde NVIDIA ve Black Forest Labs'ın güncel koşulları incelenmelidir.

Sonuçta yalnızca hedef yüz crop'u alınır ve tarayıcıda yumuşak eliptik maskeyle orijinal fotoğrafa yerleştirilir. \`inspectOutsideMaskIntegrity\` maskenin dışındaki her pikselin kaynakla aynı olduğunu kontrol eder. Maske semantik face parsing değildir; gözlük, saç teli, el veya başka örtücüleri otomatik tanımaz. Bu nedenle maskenin dışındaki piksel garantisi vardır, ancak maskenin içindeki bölgede model hatası oluşabilir.

NVIDIA API anahtarı istemci kodunda tutulmaz. Firebase Function, \`NVIDIA_API_KEY\` Secret'ını okur; tarayıcı yalnızca Function URL'sine istek gönderir. Kurulum ve izinler: \`docs/NVIDIA_SETUP.md\`.

## Gerçek kimlik korumalı Face Swap ile arasındaki fark

MediaPipe yalnızca yüz landmark'ı tespit eder. FLUX genel amaçlı görsel düzenleme modelidir; özel yüz kimlik aktarımı, biyometrik referans karşılaştırması veya kalite skoru sunmaz. Mevcut 20 fotoğrafın tek kişiye ait olduğu varsayılır; kod bunu henüz bir identity encoder ile doğrulamaz. Bu akış deneysel sonuç verebilir, ancak deterministik veya profesyonel Face Swap garantisi değildir.

## Geliştirici doğrulaması

Node.js 22 veya üstü:

\`\`\`bash
npm test
npm run validate
node --check app.js
node --check src/nvidia-pipeline.js
node --check functions/index.js
node --check functions/lib/nvidia.js
node --check sw.js
\`\`\`

GitHub Actions aynı kontrolleri \`.github/workflows/web-quality.yml\` üzerinden yürütür.