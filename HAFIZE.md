# Retake — proje geliştirme talimatları

## Nihai hedef

iPhone 17 Pro Max ve modern tarayıcılarda çalışan, yalnızca fotoğraftaki seçilen yüz bölgesini değiştiren kişisel uygulama. Kamera/canlı video özelliği yoktur.

## Değişmez şartlar

0. Kamera API'si, canlı video, kamera önizlemesi veya kamera izni kesinlikle eklenmez. Yalnızca fotoğraf seçici kullanılır.
0.1. Değişiklik yalnızca seçilen yüz crop'u ve alfa matı içinde yapılır; mat dışı pikseller orijinalle aynı kalır. Kullanıcı isteğiyle (12 Kasım 2026) mat, yüzün kafaya oturması için kafayı yumuşakça uzatan/kısaltan/genişleten/daraltan çarpıtma bölgesini de kapsar; bu bölge yüz crop'unun kenarına varmadan biter.
0.2. 25 referans fotoğrafın tamamı tek ve aynı kimliğe ait olmalıdır; farklı kişiler bir identity bank içinde karıştırılmaz.
0.3. Göz/ifade retarget yalnızca ihtiyaç varsa ve yalnızca hedef yüz crop'unda çalışır. Kapalı göz doğal açılabilir, fakat saç/arka plan/diğer kişiler değiştirilmez.


1. Canvas, landmark ve affine warp tek başına AI Face Swap değildir.
2. Gerçek inference yoksa işlem başarılı gibi gösterilmez; ilgili düğme devre dışı kalır.
3. Model kodu ile önceden eğitilmiş ağırlıkların lisansları ayrı ayrı doğrulanır.
4. Kullanıcının referans fotoğrafları, yüz embedding'leri ve lisansı kısıtlı ağırlıklar public GitHub'a commit edilmez.
5. Bir fotoğraf harici AI servisine gönderilecekse sağlayıcı, amaç ve gönderilecek veri açıkça belirtilir; kullanıcı onayı olmadan gönderim yapılmaz.
6. API anahtarları kaynak koduna, Info.plist'e veya GitHub'a eklenmez. Native iOS için Keychain; web uygulamasında gizli anahtar gerekiyorsa istemciye koymak yerine güvenli sunucu tarafı proxy gerekir.
7. Model indirme: HTTPS, boyut, SHA-256 ve yayıncı kimliği doğrulanmalıdır. SHA-256 tek başına manifest yayıncısını doğrulamaz.
8. NVIDIA API ancak gerçek Face Swap işlevi sunan belgelenmiş endpoint bulunursa değerlendirilir.
9. NVIDIA görsel jürisi (Hafize `/api/project-ai/retake/vision-judge`) yalnız danışmandır: piksel üretmez, yalnız ArcFace eşitlik bandındaki ve ayrıntı kapısını geçmiş en fazla 4 finalistin 512 px yüz kırpımını görür, yalnız kullanıcının işlem bazlı onayıyla çalışır; hata/kararsızlıkta yerel seçim korunur. Ham proje jetonu yalnız kullanıcının bilgisayarında, Hafize'de yalnız SHA-256 özeti tutulur.
10. Öğrenen Ultra yalnız varyantların deneme sırasını değiştirir; kimlik, ayrıntı, kenar ve tüm kabul kapıları aynen uygulanır. Öğrenme verisi yalnız sayıdır, bu bilgisayarda kalır, kişi/referans bankası başına ayrıdır. Öz-test (`selfTest`) hedef fotoğrafı tüm kimlik kaynaklarından çıkarır; öz-test ölçümü seçimde değil yalnız ölçüm ve sıralama öğrenmesinde kullanılır.
11. Kalite derinliği (standard/deep/max) yalnız daha çok ölçülmüş aday satın alır: daha uzun varyant araması, kazananın çevresinde aynı yüz değiştirici ağırlığıyla ince ayar ve daha çok gerçekçilik finalisti. Kabul kapıları ve seçim kuralları her derinlikte aynıdır; her aday orijinal hedef ve 25 referanstan başlar, üretilen bir yüz asla yeniden kaynak yapılmaz. `variationParams` ve `realismSeedBase` yalnız motor tarafından konur; tarayıcı metadata'sı beyaz listeden geçtiği için bunları gönderemez. Öz-test kıyaslaması karşılaştırılabilirlik için hep standard çalışır.
12. İnce bölge iyileştirmesi (göz, kirpik, kaş, dudak, burun ucu, diş, göz ışığı, göz akı) yalnız sonucun kendi ayrıntısını geri kazandırır veya aşırı keskinliği yumuşatır; hedef fotoğraftan yalnız sahne ışığı (iris içi speküler yansıma) ve göz akının ışık altındaki tonu alınır, doku/şekil alınmaz. Her iyileştirme ArcFace (en çok 0,003 kayıp), bölgesel ayrıntı kapısı ve ölçülebilir kazançla doğrulanmadan teslim edilmez; maske dışı pikseller değişmez.

## Face-only generation contract

Detaylı işlem hattı: `docs/FACE_ONLY_PIPELINE.md`.

- 25 kaynak fotoğrafını kaliteye göre hizala ve tek identity profile oluştur.
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
- Kullanıcının 25 referans fotoğrafı IndexedDB'de saklanır; tek yüz ve aynı kişi tutarlılık doğrulaması yapılır.
- Servis çalışanı uygulama kabuğunu ve izin verilen MediaPipe/WASM/model varlıklarını önbelleğe alır. Kullanıcı isteğiyle (13 Kasım 2026) varsayılan 25 referans fotoğrafı ilk açılışta bir kez indirilir ve tarayıcının Cache Storage alanında (`retake-reference-photos-v1`) kalıcı olarak saklanır; sonraki açılışlarda ve PWA'da yeniden indirilmez.
- Statik Firebase Hosting tek başına Face Swap inference çalıştırmaz. Yalnız eşleştirilmiş yerel FaceFusion/CUDA motoru hazır ve sağlıklıysa gerçek fotoğraf üretme/indirme açılır; aksi halde başarı taklidi yapılmaz.

Web değişikliklerinde \`npm test\`, \`npm run validate\`, JavaScript sözdizimi ve GitHub Actions sonucu kontrol edilir.

## Native iOS temel

\`ios/Retake.xcodeproj\` SwiftUI uygulamasıdır. Fotoğraf seçimi PhotosUI, yüz tespiti Vision, yerel depolama Application Support, model metadata Core ML ve API anahtarı saklama Keychain ile yapılır.

- Yüz tespitinde büyük fotoğrafları ana iş parçacığında tam çözünürlükte rasterize etme; ImageIO thumbnail ve background task tercih edilir.
- Özel yüz fotoğrafları ve model varlıkları iCloud/device backup dışında tutulmalıdır.
- Gerçek modelin tensor sözleşmesi ve ağırlık lisansı doğrulanmadan adapter veya başarı durumu uydurma.

## Model seçimi

MobileFaceSwap, MobileFSGAN ve GHOST araştırma adaylarıdır; hiçbirinin tüm yardımcı ağırlık lisansları ve iOS dağıtım koşulları otomatik olarak onaylanmış sayılmaz. ArcFace/InsightFace gibi encoder ve landmark bileşenleri ayrıca denetlenmelidir.

Ayrıntılı teknik durum: \`docs/IOS_OFFLINE_AI_STATUS.md\` ve \`ios/README.md\`.
## Retake yüz gerçekçiliği kalite kapısı (2026-11-23)

- ArcFace cosine skorunu "gerçekçilik yüzdesi" ya da biyometrik kanıt sayma.
- Ultra adaylar, anlamlı kimlik farkı varsa önce kimliği korur; yalnız çok yakın
  kimlik sonuçlarında yüz çevresi koyu kontur oranını ve bölgesel mikrodetayı
  tie-break amacıyla karşılaştırır.
- Son dosyada kalan halo, gerçek çıktı üzerinden ölçülür. Kontur düzeltmesi
  ArcFace kontrolünden geçmediyse uygulanmış gibi gösterilmez.
- Çıktı fotoğrafları ve 25 referans kişisel veridir: kamuya açık GitHub,
  test fixture'ı, GitHub Actions artifact'ı veya loglara yüklenmez.
- Hafize'nin GitHub App üzerinden Retake'e yazma devri için önce ayrı dal,
  dosya, taslak PR ve canlı onay akışının uçtan uca doğrulanması gerekir.
  Self-development agent merge/deploy yapmaz; gizli anahtarları görmez.
- Model gerçeği ancak aynı referanslarla, aynı orijinal hedefe karşı yerel GPU
  A/B testi ve kimlik + yüz dokusu + kontur ölçümleriyle doğrulanabilir.
