# Retake

Retake, kamera kullanmayan ve yalnızca galeriden seçilen fotoğraflar üzerinde çalışan kişisel yüz düzenleme uygulamasıdır. Hedef fotoğrafta sadece seçilen yüz bölgesi değiştirilir; saç, arka plan, kıyafet, diğer insanlar ve yüz dışındaki alanlar korunur. Kimlik için aynı kişiye ait en fazla 20 referans fotoğrafı birlikte kullanılır.

> **Gerçek durum:** Web sürümünde NVIDIA FLUX.1-Kontext-dev ile deneysel görsel düzenleme akışı bulunur. Bu model özel bir Face Swap modeli değildir; kimlik benzerliği garanti edilmez. Firebase sunucu fonksiyonunun ve NVIDIA erişiminin yapılandırılması gerekir. Native iOS uygulamasına aynı inference henüz bağlanmamıştır.

## Kesin ürün kuralları

- Kamera, canlı video ve kamera izinleri yoktur; sadece fotoğraf seçici/dosya aktarımı vardır.
- 20 kimlik fotoğrafı aynı kişiye ait olmalıdır; tek yüz tespiti tek başına aynı kimlik doğrulaması değildir.
- Gözler kapalıysa aynı kişiye ait açık göz referanslarıyla doğal göz açma yapılabilir. İfade düzeltmesi yalnızca gerektiğinde ve yüz alanında uygulanır.
- Sonuç, yumuşak yüz maskesiyle orijinal fotoğrafa birleştirilir. Maske dışındaki piksellerin değişmemesi zorunludur.

Detaylı model sözleşmesi: `docs/FACE_ONLY_PIPELINE.md`.

## Web Studio

Statik web uygulaması: \`index.html\`

- Telefon, tablet ve masaüstüne uyumlu arayüz.
- Fotoğraf seçme ve sürükle-bırak desteği.
- MediaPipe Face Landmarker 1.2.2 ile cihazda çoklu yüz algılama ve hedef seçimi.
- Veritabanında önceden kayıtlı 20 referansı `GET /api/images/{id}` üzerinden otomatik ve salt okunur yükleme.
- Referans fotoğrafları yeniden yükletilmez, IndexedDB'ye kopyalanmaz; hedef görsel yalnızca oturum belleğinde tutulur.
- NVIDIA FLUX için tek panoda 20 referans ve seçili hedef yüz crop'u hazırlar; sonuç tarayıcıda yüz maskesiyle birleştirilir.
- NVIDIA API anahtarı Firebase Cloud Functions Secret'ta saklanır; ön yüzde API anahtarı bulunmaz.
- Koyu/açık tema ve kurulabilir Progressive Web App (PWA).
- Uygulama kabuğu, MediaPipe WebAssembly dosyaları ve yüz modeli için servis çalışanı önbelleği. İlk açılışta internet gerekir; önbellek oluştuğunda desteklenen tarayıcılarda çevrimdışı açılış mümkündür.

Hedef fotoğraf ve referans görselleri Face Swap API'sine gönderilmez. NVIDIA düzenlemesi başlatılırsa tam hedef fotoğraf değil, seçili yüz crop'u ve 20 referansın birleşik panosu NVIDIA'ya gönderilir; önce açık onay alınır. Referanslar mevcut görsel API'sinden okunur. `meta[name="retake-api-base"]` mevcut görsel API'sini, `meta[name="retake-ai-base"]` ise dağıtılmış Firebase Function URL'sini belirler. Kurulum: [docs/NVIDIA_SETUP.md](docs/NVIDIA_SETUP.md). MediaPipe yüz algılamayı cihazda çalıştırır; kütüphane ölçümleri Google'a gönderilebilir: https://goo.gle/mediapipe-privacy

### Web'i çalıştırma

Node.js 22 veya üzeri:

\`\`\`bash
npm test
npm run validate
\`\`\`

Statik sunucuyla \`index.html\` dosyasını HTTPS veya \`localhost\` üzerinden açın. Örneğin:

\`\`\`bash
npx serve .
\`\`\`

Kamera/Service Worker özellikleri için \`file://\` yerine güvenli bağlam (HTTPS veya localhost) kullanın.

## Native iOS

Xcode projesi: \`ios/Retake.xcodeproj\`

- SwiftUI uygulama kabuğu.
- PhotosPicker ile hedef fotoğraf ve referans seçimi.
- Vision ile cihaz içi yüz tespiti ve çoklu yüz seçimi.
- 20 adede kadar referans görselini uygulamaya özel klasörde saklama.
- Core ML modeli içe aktarma, metadata inceleme ve SHA-256 doğrulamalı model indirme altyapısı.
- Sağlayıcı kimlik bilgileri için iOS Keychain altyapısı.

Gerçek iOS inference modeli, adapter, maskeli compositing ve Photos'a dışa aktarma henüz tamamlanmadı.

## AI motoru için kabul kriterleri

Face-only crop, 20 referans kimlik birleştirme, göz/ifade retargeting, face parsing/occlusion mask ve maske dışı piksel doğrulaması tek bir uçtan uca testte geçmelidir.

1. Face swapper, kimlik encoder, yüz hizalama/tespit ve gerekirse restorasyon/maske bileşenlerinin tüm ağırlık lisansları ayrı ayrı doğrulanmalı.
2. Modelin gerçek tensor giriş/çıkış sözleşmesi belgelenmeli; varsayımsal adapter yazılmamalı.
3. iOS için Core ML dönüşümü ve Web için ayrı bir çalıştırma yolu/uyumlu model bulunmalı.
4. Fotoğraf kalitesi, kimlik benzerliği, poz/ışık uyumu, bellek ve hız testleri gerçek cihazlarda yapılmalı.
5. Harici inference sağlayıcısı kullanılacaksa fotoğraf gönderimi öncesinde açık onay alınmalı; anahtarlar istemci kaynak koduna gömülmemeli.

## Model ve fotoğraf varlıkları

Kişisel referans fotoğrafları, yüz embedding'leri ve lisansı kısıtlı ağırlıklar public depoya eklenmemelidir. Yerel dosyalar \`.gitignore\` ile hariç tutulur. Model kod lisansı ile önceden eğitilmiş ağırlık lisansı ayrı kabul edilir.

## Doğrulama

\`\`\`bash
npm test
python3 tools/validate_assets.py
\`\`\`

iOS simulator derlemesi \`.github/workflows/ios-build.yml\`, web doğrulaması \`.github/workflows/web-quality.yml\` ile çalışır.