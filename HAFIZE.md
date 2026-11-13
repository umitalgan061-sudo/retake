# Retake — proje geliştirme talimatları

## Nihai hedef

iPhone 17 Pro Max ve modern tarayıcılarda çalışan, yalnızca fotoğraftaki seçilen yüz bölgesini değiştiren kişisel uygulama. Kamera/canlı video özelliği yoktur.

## Değişmez şartlar

0. Kamera API'si, canlı video, kamera önizlemesi veya kamera izni kesinlikle eklenmez. Yalnızca fotoğraf seçici kullanılır.
0.1. Değişiklik yalnızca seçilen yüz crop'u ve alfa matı içinde yapılır; mat dışı pikseller orijinalle aynı kalır. Kullanıcı isteğiyle (12 Kasım 2026) mat, yüzün kafaya oturması için kafayı yumuşakça uzatan/kısaltan/genişleten/daraltan çarpıtma bölgesini de kapsar; bu bölge yüz crop'unun kenarına varmadan biter.
0.2. En çok 20 referans fotoğrafı aynı kimliğe ait olmalıdır; farklı kişiler bir identity bank içinde karıştırılmaz.
0.3. Göz/ifade retarget yalnızca ihtiyaç varsa ve yalnızca hedef yüz crop'unda çalışır. Kapalı göz doğal açılabilir, fakat saç/arka plan/diğer kişiler değiştirilmez.


1. Canvas, landmark ve affine warp tek başına AI Face Swap değildir.
2. Gerçek inference yoksa işlem başarılı gibi gösterilmez; ilgili düğme devre dışı kalır.
3. Model kodu ile önceden eğitilmiş ağırlıkların lisansları ayrı ayrı doğrulanır.
4. Kullanıcının referans fotoğrafları, yüz embedding'leri ve lisansı kısıtlı ağırlıklar public GitHub'a commit edilmez.
5. Bir fotoğraf harici AI servisine gönderilecekse sağlayıcı, amaç ve gönderilecek veri açıkça belirtilir; kullanıcı onayı olmadan gönderim yapılmaz.
6. API anahtarları kaynak koduna, Info.plist'e veya GitHub'a eklenmez. Native iOS için Keychain; web uygulamasında gizli anahtar gerekiyorsa istemciye koymak yerine güvenli sunucu tarafı proxy gerekir.
7. Model indirme: HTTPS, boyut, SHA-256 ve yayıncı kimliği doğrulanmalıdır. SHA-256 tek başına manifest yayıncısını doğrulamaz.
8. NVIDIA API ancak gerçek Face Swap işlevi sunan belgelenmiş endpoint bulunursa değerlendirilir.

## Face-only generation contract

Detaylı işlem hattı: `docs/FACE_ONLY_PIPELINE.md`.

- 20 kaynak fotoğrafını kaliteye göre hizala ve tek identity profile oluştur.
- Hedefte kullanıcı tarafından seçilen yüz crop'unda kimlik aktarımı yap.
- Pose/ışık/skin target attributes korunur.
- Gerekirse eyes-open ve sınırlı expression retarget uygula.
- Face parser ve occlusion mask ile yüz dışını koru.
- Tam çözünürlüklü kaynak fotoğrafa maskeli patch yapıştır.
- Maske dışı piksel eşitliğini otomatik doğrula.

## Web Studio

\`index.html\`, \`styles.css\`, \`app.js\`, \`src/core.js\` ve \`sw.js\` statik PWA'yı oluşturur.

- MediaPipe Tasks Vision sürümü sabitlenir; doğru API ile \`HTMLImageElement\` üzerinden \`detect()\` çağrılır.
- Fotoğraf algılama tarayıcıda çalışır; görsel içeriği üçüncü tarafa yüklenmez.
- Hedef fotoğraf oturum belleğinde tutulur; hedef fotoğraf IndexedDB'ye kaydedilmez.
- Kullanıcının eklediği en fazla 20 referans fotoğrafı IndexedDB'de saklanır; tek yüz doğrulaması yapılır.
- Servis çalışanı yalnızca uygulama kabuğunu ve izin verilen MediaPipe/WASM/model varlıklarını önbelleğe alır.
- Web uygulamasında gerçek Face Swap modeli yoktur; swap ve indirme eylemleri kapalı kalır.

Web değişikliklerinde \`npm test\`, \`npm run validate\`, JavaScript sözdizimi ve GitHub Actions sonucu kontrol edilir.

## Native iOS temel

\`ios/Retake.xcodeproj\` SwiftUI uygulamasıdır. Fotoğraf seçimi PhotosUI, yüz tespiti Vision, yerel depolama Application Support, model metadata Core ML ve API anahtarı saklama Keychain ile yapılır.

- Yüz tespitinde büyük fotoğrafları ana iş parçacığında tam çözünürlükte rasterize etme; ImageIO thumbnail ve background task tercih edilir.
- Özel yüz fotoğrafları ve model varlıkları iCloud/device backup dışında tutulmalıdır.
- Gerçek modelin tensor sözleşmesi ve ağırlık lisansı doğrulanmadan adapter veya başarı durumu uydurma.

## Model seçimi

MobileFaceSwap, MobileFSGAN ve GHOST araştırma adaylarıdır; hiçbirinin tüm yardımcı ağırlık lisansları ve iOS dağıtım koşulları otomatik olarak onaylanmış sayılmaz. ArcFace/InsightFace gibi encoder ve landmark bileşenleri ayrıca denetlenmelidir.

Ayrıntılı teknik durum: \`docs/IOS_OFFLINE_AI_STATUS.md\` ve \`ios/README.md\`.