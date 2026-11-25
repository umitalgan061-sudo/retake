# Retake — Kişisel Fotoğraf Stüdyosu

Retake, galeriden seçilen bir hedef fotoğraftaki yüzü aynı kişiye ait referans fotoğraflarıyla değiştirmek için geliştirilen kişisel bir fotoğraf stüdyosudur. Kamera açmaz ve hedef fotoğrafın tamamını yeniden üretmez.

## Mevcut web motoru

- Yerel FaceFusion `3.9.2` ile iki geçişli hibrit motor: kimlik için `alphaface_256`, görünüm/sakal için `uniface_256`.
- 25 fotoğrafın tamamı ortak göz koordinatına hizalanır ve netlik, açı, yüz alanı, ışık ölçütlerine göre ağırlık alır. 478 noktalı FaceLandmarker başarısız olursa BlazeFace'in altı anahtar noktalı yedek hizalaması kullanılır; yedek hizalamaların ağırlığı en fazla 2'dir.
- Pixel boost ilk denemesi `1024x1024`; CUDA bellek hatasında `768x768`, ardından `512x512` denenir. Bu yüz crop inference ölçeğidir, native çıktı büyütme değildir.
- FaceFusion ArcFace ONNX ile 25 crop için medoid/medyan tutarlılık taraması yapılır; güçlü aykırı referansların kimlik oyu 1'e düşürülür.
- UniFace'tan alt yüze düşük güçlü yüksek frekanslı ayrıntı aktarılır; AlphaFace geometrisi baskın kalır.
- GFPGAN 1.4 son iyileştirmesi isteğe bağlıdır ve varsayılan kapalıdır.
- Tüm geçerli referanslardan ağırlıklı kimlik gömmesi ortalaması ve ağırlıklı hizalanmış piksel görünüm profili oluşturulur.
- Profilin kırpılmış referansları ve hizalanmış görünüm görseli tarayıcı IndexedDB alanında cihazda saklanır; aynı referans kümesiyle sayfa yenilendiğinde tekrar hizalama yapılmaz.
- AlphaFace geçişi tüm referanslardan ağırlıklı kimlik ortalaması alır; UniFace geçişi 25 fotoğrafın tamamından oluşan tek hizalı görünüm profiliyle çalışır.
- RTX GPU üzerinde CUDA ile çıkarım; yüzü üreten hiçbir adım NVIDIA hosted API, Firebase Functions veya ücretli inference API kullanmaz. İsteğe bağlı **NVIDIA görsel jürisi** yalnız kimliği eşit finalistler arasında seçim yapar (aşağıya bak).
- Retake, tek yüz sonucu orijinal fotoğrafa maskeli olarak birleştirir ve maske dışındaki pikseller için bütünlük kontrolü yapar.
- Yerel FaceFusion servisi yalnızca `127.0.0.1:8765` üzerinde dinler.
- FaceFusion XSeg occlusion + BiSeNet region parser maskeleri iki üretim geçişinde de aktiftir; gözlük sınıfı değiştirme maskesinin dışında tutulur.
- FaceFusion 3.9.2'nin desteklediği yüz parser bölgeleri arasında saç sınıfı bulunmadığından saç stili kaynak fotoğraftan korunur; saç transferi bu motorda yapılmaz.
- **Dikişsiz birleştirme (`tools/retake_compositing.py`, üretken model değil, klasik görüntü işleme):** FaceFusion çıktısı fotoğrafa üç adımda yeniden oturtulur. (1) Değişen bölgenin iç kenarında ölçülen ton/ışık farkı yüzün içine harmonik olarak yayılıp çıkarılır (Poisson benzeri gradyan alanı düzeltmesi); yeni yüz çevresindeki cildin tonunu ve sahne ışığını alır, kendi hatlarını korur. (2) Laplacian piramidiyle çok bantlı harmanlama: renk farkı geniş bir kuşakta, ince ayrıntı dar bir kuşakta geçiş yapar; kesilmiş gibi görünen sınır çizgisi oluşmaz. (3) Yüzün netliği ve sensör greni fotoğrafın geri kalanına eşitlenir (yumuşaksa mikro kontrast + fotoğrafın kendi greni, fazla keskinse hafif yumuşatma).
- **Hız (önbellekler, `tools/retake_cache.py`):** 25 referansın ArcFace raporu, referanslar değişmedikçe `%USERPROFILE%\.retake\identity` altında saklanır ve sonraki işlemlerde bu adım atlanır. Aynı fotoğraf için iki FaceFusion geçişinin çıktısı motor açık kaldıkça bellekte tutulur; "Yeniden oluştur" alternatiflerinin ilk üçü aynı kimlik ağırlığını kullandığı için GPU geçişlerini yeniden çalıştırmadan saniyeler içinde hazırlanır, dördüncüsü daha yumuşak ayrı bir geçiş yapar. Her aşamanın bu bilgisayardaki süresi `%USERPROFILE%\.retake\timings.json` dosyasında hatırlanır ve bekleme ekranında kalan süre tahmini olarak gösterilir.
- **Çene hattı:** Tarayıcı, karşıdan çekilmiş referansların 478 noktalı ağından kişinin çene/yanak hattını (21 nokta) gözlere göre normalize edip ağırlıklı medyanını alır. Hedef fotoğrafta aynı noktalar bu hatta doğru kaydırılır: çenede tam, yanaklarda %35 güçle; yüz genişliğinin %2'sinden küçük farklar poz gürültüsü sayılıp yok sayılır, her kayma %7 ile sınırlıdır ve baş belirgin dönükse uygulanmaz. Motor kaymayı gözler, burun ve ağız sabit kalacak şekilde yumuşak bir alan olarak uygular (`warp_jawline`); alan kırpım kenarında sıfırdır. "Fotoğraftaki kafa oranı" alternatifi çeneye dokunmaz.
- **Kişiye birebir benzerlik (kimlik öncelikli birleştirme):** Ölçümle bulundu: FaceFusion'ın ham AlphaFace çıktısı kişinin 25 fotoğrafına ArcFace'le 0,80 benziyor (fotoğrafların kendi aralarında 0,77), ama eski birleştirme bunu 0,62'ye düşürüyordu. Nedenleri: dikiş gizlemek için yüzün dış bandının (yüz genişliğinin %8'i) asıl fotoğrafa geri döndürülmesi; böylece yüz hatları, kaşlar ve çene hedef kişide kalıyordu. Bir de güçlü ten rengi ve ışık-gölge aktarımı. Şimdi harmanlama %3'te başlıyor, ton farkı %5'te FaceFusion'ın yumuşak kenarının ötesinde maskeli bulanıklaştırmayla ölçülüyor; ışık-gölge 0,35, ten rengi %40 güçte. Dikiş testleri geçmeye devam ediyor.
- **Kimlik öncelikli birleştirme seçimi:** Motor her işte üç birleştirme üretir. İlki en dikişsizidir; ton eşitleme 0,9'dur, yani asıl fotoğrafın ten tonu çekilir. Diğer ikisi swap'ın kendi cildini daha çok korur (0,6 ve 0,35). Her birinin kenar geçişi ölçülür (`matte_border_step`, yüz genişliğinin 100 pikseli başına). En dikişsizinin iki katını aşan varyant elenir. Kalanlar ArcFace ile ölçülür ve kişiye en çok benzeyen kullanılır. Ölçüm yapılamazsa en dikişsizi kalır.
- **Kişinin kendi sakal tonu:** Yüz değiştiriciler sakalı açık ve ince çizer. Motor, 25 fotoğrafın hizalanmış ortalama profilinden sakal bölgesinin yanağa göre ne kadar koyu olduğunu ölçer; bu oran sahne ışığından bağımsızdır. Çıktıda aynı bölgeyi bu orana yaklaştırır. Kıl gibi koyu pikseller değişimin çoğunu alır, çıplak cilt neredeyse hiç değişmez. Referanslardaki renk ilişkisi ışığa bağlı olduğu için yalnızca %25'i taşınır. Kullanıcının fotoğraflarında referanslar −70 L, swap −46 L çıktı; düzeltme sonrası −64 L oldu, kimlik benzerliği değişmedi. Tarayıcı hedef yüzün 478 noktalı ağını (`targetLandmarks`) bunun için gönderir.
- **Ölçülüp eklenmeyenler:**
  - 25 yerine en tutarlı 8, 12 veya 16 referansı kullanmak (0,789–0,800; 25'in tamamıyla 0,803).
  - Yüz iyileştiriciler GPEN, CodeFormer, GFPGAN ve RestoreFormer; hepsi benzerliği düşürdü.
  - Üçüncü bir değiştirici geçişi (+0,005; süreye değmez).
  - 3B yüz şekli bükmesi (güçlü swap'ta benzerliği düşürdü).
- **Kimlik pekiştirme:** AlphaFace sonucunun üstüne ikinci bir değiştirici (InSwapper 128) uygulanır. İki sonuç ArcFace ile ölçülür ve kişiye daha çok benzeyen kullanılır (`X-Retake-Identity-Reinforced`). Ölçümde ham benzerlik 0,803'ten 0,832'ye çıktı. Bahçe fotoğrafında zincirin tamamı (birleştirme, kafa oranı, fotoğrafa yapıştırma) eski ayarlarla 0,627 verdi. Pekiştirmeyle 0,764'e, birleştirme seçimi ve sakal tonuyla **0,801**'e çıktı; kişinin kendi fotoğraflarının birbirine benzerliği 0,766. Sonuç rozeti bu karşılaştırmayı gösterir ("fotoğraflarınla aynı düzeyde"). "Hafif" alternatifte uygulanmaz. **Netlik koruması (v22.1):** InSwapper yüzü 128 px'te çizer; büyük yüzlerde gözler ve dudaklar bulanıklaşır, göz kapakları çift görünür, ArcFace ise (InSwapper aynı ArcFace ailesiyle eğitildiği için) yine yükselir. Bu yüzden pekiştirme yalnız göz ve ağız bölgesindeki ince ayrıntının en az %85'ini 256 px geçişe göre koruyorsa kabul edilir; aksi halde daha net 256 px sonuç kalır (`X-Retake-Reinforce-Guard`). Küçük yüzlerde 128 px kayıp yaratmadığı için pekiştirme eskisi gibi çalışır.
- **Işık-gölge ve ten rengi (3B görünüm):** Yüz değiştirme modelleri yüzü düz, ortalama bir ışıkla üretir; yüz "kağıt gibi" görünür. Retake, asıl fotoğraftaki yüzün orta ölçekli ışık-gölge kabartmasını (burun sırtı ışığı, yanak ve göz çukuru gölgeleri, alın parlaması) logaritmik parlaklıkta yeni yüze aktarır; ince doku ve hatlar yeni yüzde kalır. Aynı yöntemle her bölgenin lokal ten rengi kısmen (%40) asıl fotoğraftakine çekilir; modelin bıraktığı kırmızı burun ucu, pembe ağız çevresi gibi lekeler azalır. Tam çekmek, hedef kişinin dudak ve kaş rengini de geri getirip benzerliği düşürdüğü için yapılmaz. Yalnızca yeni yüzde olan yapılar (sakal, bıyık) ve koyu bölgeler korunur. Güç `-ShadingStrength` ile ayarlanır (0 kapalı, 0.35 varsayılan, 1.2 en çok): `.\tools\start_facefusion_retake.ps1 -ShadingStrength 0.9`.
- **Fotoğraftan doku nakli (deneysel, varsayılan kapalı):** Varsayılan yol artık aşağıdaki fotoğraf gerçekçiliği katmanıdır; bu nakil bazı fotoğraflarda maske gibi durabilir. Yüz değiştirme modelleri yüzü 256 px'te üretip büyütür; burun, dudak ve cilt yumuşak ve "yapay" görünür. Retake, 25 referans içinden hedef yüzle baş açısı ve ifadesi en yakın 3 fotoğrafı seçer (478 noktalı yüz ağı ile). Motor her adayı yüz ağının üçgenleriyle hedef yüzün geometrisine büker; gözleri, kaşları, burun delikleri ve dudak çizgisi fotoğraftakilerle en iyi çakışanı (gözlüklü, eli yüzünde veya farklı ifadeli fotoğraflar böylece elenir), eşitlikte ışığı en benzer olanı kullanır. Yüz, alından çeneye kadar bu gerçek pikselle değiştirilir; alında yalnızca kişinin kendi yanak-burun ten rengine uyan pikseller alınır, böylece referanstaki perçem fotoğrafa taşınmaz. Gerçek fotoğrafın geniş ölçekli ışığı ve rengi sahneye eşitlenir (gözenek, kıl ve kırışıklık gibi ince doku fotoğrafın kendisinden kalır); büyük referans fotoğraf önce alan ortalamasıyla küçültülür, böylece sakal ve saç kumlanmaz. Kapalı dudaklar da gerçek fotoğraftan gelir. Hiçbir adayın hatları fotoğrafla yeterince çakışmıyorsa (uyum < 0,3) yapay zekâ sonucu kullanılır. Saç, el ve gözlük gibi modelin koruduğu bölgeler ile açık ağız fotoğrafta kalır. Ardından ışık-gölge, ten rengi ve gren eşleştirmesi bu yüze uygulanır. Yeterince yakın açılı referans yoksa veya fotoğraf işlenemezse yapay zekâ sonucu kullanılır. Ayar, "Yüzümü uygula" düğmesinin üstündeki **Fotoğraftan doku nakli** kutusuyla açılır. Motor kullanılan yolu `X-Retake-Real-Face: on/off` başlığıyla bildirir.
- **Kafa oranı oturtma:** Tarayıcı 25 referansın 478 noktalı ağından kişinin yüz uzunluğu/genişliği oranını (3B ölçüm, baş dönüklüğüne dayanıklı ağırlıklı medyan) ve hedef fotoğraftaki kafanın oranını ölçer. Fark %1,5'ten büyükse kafa, yüz merkezinden dışa doğru yumuşakça sönen bir çarpıtmayla uzatılır/kısaltılır/genişletilir/daraltılır; oran farkının tamamı uygulanır (eksen başına en çok %15, güçlü alternatifte %18), çene hattı düzeltmesiyle birlikteyken dörtte üçü. Yüz, saç çizgisi, çene ve hemen çevresi birlikte hareket ettiği için yüz kenarında boşluk veya çift kenar oluşmaz; çarpıtma kırpım kenarına varmadan sıfırlanır.
- Hedef kırpım yerel motora mümkün olduğunda piksel piksel (1024–2048 px yükseklik aralığında yeniden örneklemesiz) gönderilir; yüz çevresi küçült-büyüt bulanıklığına uğramaz.
- Hedef fotoğrafta yüz iki aşamada aranır: FaceLandmarker, ardından BlazeFace tam/kısa menzil. İlk geçişin kaçırdığı küçük, uzak veya dönük yüzlerde bölgeye yakınlaşılıp 478 noktalı ağ yeniden çıkarılır.
- **Her zaman Ultra-420 (v16):** 24 birbirinden farklı ağırlık, geometrik uyarlama ve alt yüz doku ayarı denenir. İlk fotoğraf ve 25 referans her turda korunur; ara AI çıktıları tekrar kaynak yapılmaz. Yerel ArcFace kimlik puanı + göz/burun/ağız/alt yüz fotoğraf ayrıntısı (gerçekçilik için sezgisel ölçüm) kullanılır. **Yalnızca en iyi doğrulanmış fotoğraf** gösterilir. Yeni adaylara 420 saniye içinde başlanır; kullanıcı isteğine göre başarılı sonuç **en az 7 dakika sonra** sunulur. GPU aramayı daha erken bitirirse kalan süre dürüstçe bekleme aşamasıdır, aynı görüntü defalarca işlenmez. Gerçek GPU süresi bazen 7 dakikayı aşabilir. Kimlik ve gerçekçilikte yüzde yüz eşleşme garanti edilemez.
- Sayfanın en üstünde ve işlem sürerken sonuç panelinde, dokununca sıvı gibi dağılıp yeniden toplanan parçacık heykel animasyonu (`src/forma-sculpture.js`, WebGL2) ve geçen süre gösterilir. Motor ve görseller, Guglielmo Giannattasio'nun "Forma — Liquid Sculpture" 21st.dev bileşeninden alınmıştır (https://guglielmogiannattasio.it); görseller `assets/forma/` altında barındırılır ve servis çalışanı tarafından önbelleğe alınır.
- Arayüz buzlu cam yüzeyleri ve temaya uyumlu uyarı/onay pencereleriyle hem açık hem koyu modda çalışır.
- Son PNG, birleştirmenin ve kafa oturtmanın dokunduğu alanı kapsayan alfa matı taşır. Tarayıcı bu matı kullanarak yalnızca o alanı orijinal fotoğrafa uygular; alfa 0 olan her piksel bayt bayt korunur.

**Apple Retake ile aynı model değildir.** Retake, 25 referanstan ağırlıklı kimlik ortalaması ve hizalanmış görünüm ortalaması üretir; AlphaFace kimliği, UniFace ise alt yüz görünüm/sakal detay katmanını işler. Bu, kişiye özel diffusion modeli eğitimi veya LoRA fine-tuning değildir; kimlik benzerliği, sakal aktarımı ve saç transferi garanti edilmez.

### Kimlik yeterliliği: fotoğraflarının düzeyinden sonra gerçekçilik seçer

ArcFace benzerliği bir noktadan sonra "daha çok sen" demek değildir. Motor, 25 referansının her birini *diğer* 24'ün ortalamasıyla karşılaştırarak senin gerçek bir fotoğrafının kimlik merkezine tipik benzerliğini ölçer (`X-Retake-Identity-Photo-Level`, birini dışarıda bırakarak; fotoğraf kendine oy vermez). Bir aday bu düzeyin 0,02 altına kadar geldiyse kimlik "yeterli" sayılır:

- Yeterli adaylar arasında karar sırası: 128 px pekiştirme olmayan sonuç → daha temiz saç/çene çizgisi → göz/burun/ağızda gerçek fotoğrafın ayrıntısını daha çok koruyan sonuç → ancak sonra ArcFace.
- Yeterli bir sonuç, sırf daha yüksek kosinüslü (çoğunlukla 128 px pekiştirmeli) bir aday çıktı diye geri çekilmez; ek finalist araması da yapılmaz.
- Düzeyin altındaki adaylarda eski kimlik öncelikli kurallar aynen geçerlidir. Ölçüm yoksa (eski önbellek) davranış değişmez; kimlik raporu sürümü `centroid-v2` olduğu için ilk işte referanslar bir kez yeniden ölçülür.
- NVIDIA jürisine giden finalistler de artık yalnız ArcFace'e yakınlıkla değil, yeterli adaylar arasından gerçekçilik sırasıyla seçilir.
- Sonuç rozeti "Kimlik fotoğraflarının düzeyinde; adaylar arasından en doğal görünen seçildi" der; "fotoğraflarınla aynı düzeyde" karşılaştırması da artık aynı ölçekteki bu değere göre yapılır.

Aynı kural tek bir işlemin içinde de geçerli: 256 px geçiş zaten bu düzeye ulaştıysa 128 px InSwapper pekiştirmesi hiç uygulanmaz (`X-Retake-Reinforce-Guard: identity-sufficient`), ve başlık uydurma/ham-kafa/fotoğraf-dengesi gibi iç adaylar arasından gerçek fotoğraf ayrıntısını en çok koruyan seçilir.

**Öz-test ölçümü.** Hedef fotoğraf 25 referansından biriyse (aynı dosya), orijinal yüz doğru cevaptır. Bu durumda motor sonucu orijinalle karşılaştırır: yüz bölgesinde yapısal benzerlik (SSIM, 1 = aynı) ve ortalama renk farkı (ΔE, 0 = aynı); sonuç rozetinde "Öz-test" satırı olarak görünür (`X-Retake-Self-Fidelity`). Ölçek için: aynı fotoğrafı JPEG olarak yeniden kaydetmek ≈ 0,97 / 1,3 verir; 25 Kasım testindeki yapay görünen sonuç 0,69 / 7,2 idi (σ=2 bulanıklaştırılmış orijinal bile 0,77 / 2,4). Bu yalnız ölçümdür; seçimde kullanılmaz, yoksa test kendi cevabını görmüş olurdu.

Neden: Kendi referans fotoğrafınla yapılan bir testte sonuç ArcFace'te 0,85'ten 0,91'e çıktı ama gözler bulanık, burun delikleri lekeli, cilt mumsu göründü. Yüksek puan, gerçek bir fotoğrafından daha "sen" olduğu anlamına gelmiyordu.

### NVIDIA görsel jürisi (isteğe bağlı, Hafize üzerinden)

Ultra araması kimlik için yerel ArcFace'i, doku için bölgesel ayrıntı ölçümünü kullanır; ama "hangisi gerçek fotoğraf gibi duruyor" sorusunu bir insan gözü daha iyi yanıtlar. Jüri bu boşluğu NVIDIA'nın ücretsiz barındırılan görsel modelleriyle doldurur:

- Ultra, ArcFace eşitlik bandındaki (en iyi skordan en çok 0,002 düşük) ve bölgesel ayrıntı kapısını geçmiş adayları saklar. Yerel kazanan ve en güçlü 3 eşdeğer aday, yüz çevresi bağlamıyla 512 px JPEG kırpım olarak Hafize'nin `/api/project-ai/retake/vision-judge` ucuna gider.
- Hafize, NVIDIA anahtarını kendisi tutar; `nemotron-3.5-super-vl`, `llama-3.2-11b-vision` ve `nemotron-3-nano-omni` modellerine adayları iki farklı sırayla gösterir (konum yanlılığı dengelenir). Modeller dikiş/hale, ten ve ışık uyumsuzluğu, mumsu cilt, bozuk göz/diş, çarpıklık ve netlik farkına puan verir; kişiyi tanıması istenmez.
- En az 2 geçerli oy, ⅔ uzlaşı ve belirgin puan farkı yoksa jüri karar vermez; motor da bu eşikleri yerelde yeniden uygular. Hata, zaman aşımı veya kararsızlıkta yerel seçim korunur. Jüri piksel üretmez, kimlik kapısını aşamaz.
- Varsayılan kapalıdır. Motor yapılandırılmışsa sitede **NVIDIA görsel jürisi** kutusu görünür; kutu açıkken her işlemden önceki onay penceresi neyin nereye gittiğini ayrıca yazar. Sonuç rozeti gerçek sonucu gösterir: "daha doğal görünen finalisti seçti", "yerel seçimi onayladı", "kararsız kaldı" veya "ulaşılamadı".
- Jüri 7 dakikalık Ultra süresinin içinde çalışır; GPU araması süreyi zaten doldurduysa en çok ~2 dk ekler (Render uykudaysa uyanma süresi dahil).

Kurulum (bir kez, PowerShell, Retake klasöründe):

```powershell
.\tools\setup_hafize_judge.ps1
```

Betik rastgele bir proje jetonu üretir, ham jetonu yalnız `%USERPROFILE%\.retake\hafize-project-token` dosyasında (yalnız senin kullanıcın okuyabilir) saklar ve Hafize için `{"retake":"<sha256>"}` özetini verir. Bu özeti Render'da `hafize-free` servisine `HAFIZE_PROJECT_AI_TOKEN_HASHES` ortam değişkeni olarak ekle ve servisi yeniden dağıt (Cloud Run kullanılırsa aynı değer Hafize deposunda GitHub Secret olur). Motor varsayılan olarak `https://hafize-free.onrender.com` adresine bağlanır; Render Free uykudan uyanırken ilk istek bir dakika kadar sürebilir. Sonra Retake motorunu yeniden başlat; `/health` içinde `nvidia_judge: "configured"` görünür. Kontrol: `.\tools\setup_hafize_judge.ps1 -Check`. Kapatmak: `-Disable` veya `RETAKE_NVIDIA_JUDGE=off`. Farklı Hafize adresi için `RETAKE_HAFIZE_JUDGE_URL` yalnız `hafize-ua.web.app`, `hafize-free.onrender.com` veya `*.run.app` üzerindeki aynı yolu kabul eder.

Neden NVIDIA'nın görüntü düzenleme modelleri (FLUX.2 Klein, FLUX.1 Kontext, Qwen-Image-Edit) yüzü üretmek için kullanılmıyor: ücretsiz barındırılan uçlar düzenleme modunda yalnız NVIDIA'nın önceden tanımlı örnek görsellerini kabul ediyor, kişisel fotoğraf yüklenemiyor; yerel NIM konteynerleri ise 48–80 GB VRAM istiyor (bkz. [docs/NVIDIA_SETUP.md](docs/NVIDIA_SETUP.md)).

## Windows kurulumu

Gereksinimler: Windows 10/11, NVIDIA CUDA destekli GPU (bu geliştirme yapılandırması RTX A4000 16 GB), Conda, Git ve FFmpeg.

PowerShell'de Retake deposunu açıp:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\tools\\install_facefusion_windows.ps1
```

Kurulum FaceFusion 3.9.2, Python 3.12, CUDA 12.9 runtime, cuDNN 9.10 ve CUDA destekli ONNX Runtime ortamını hazırlar.

Ardından yerel servisi başlat:

```powershell
.\\tools\\start_facefusion_retake.ps1
```

Servis sağlık kontrolü: `http://127.0.0.1:8765/health`

Ayrıntılı kurulum ve sorun giderme: [docs/LOCAL_FACEFUSION_SETUP.md](docs/LOCAL_FACEFUSION_SETUP.md).

### Fotoğraf gerçekçiliği katmanı (önerilir, tek seferlik kurulum)

FaceFusion'ın yüz modelleri yüzü 256 px'te çizer: cilt mumsu, dudak düz ve parlak, gözler bulanık kalır. Bu katman, yüzü gerçek fotoğraflarla eğitilmiş bir difüzyon modeliyle (Realistic Vision 6, SD 1.5 inpainting) yeniden işler. Model **kişinin kendi ArcFace kimliğiyle** (25 referansın kimlik merkezi, IP-Adapter FaceID) yönlendirilir; böylece gerçekçilik yüzü "genel bir insana" çekmez. Sonuç sahnenin ışığını ve rengini FaceFusion'dan alır, gözenek, kıl, dudak kıvrımı ve göz parıltısı gibi ince dokuyu modelden.

Motor her işte üç sürüm üretir, hepsini ArcFace ile ölçer ve kişiye en yakın olanı kullanır. Hiçbiri FaceFusion sonucundan iyi değilse FaceFusion sonucu kalır. Kullanıldığında sonuç rozetinde "fotoğraf gerçekçiliği" yazar; motor bunu `X-Retake-Realism: on` başlığıyla bildirir. Kullanıcının fotoğraflarıyla yapılan ölçümde ArcFace benzerliği FaceFusion'da 0,394 iken bu katmanla 0,45–0,47 oldu.

```powershell
.\tools\install_realism.ps1
```

Kurulum `retake-realism` adlı ayrı bir conda ortamı açar; FaceFusion ortamına dokunmaz. Modeller `C:\AI\RetakeRealism\models` altına iner (~2,6 GB). NVIDIA GPU (~4 GB bellek) gerekir. Kurulumdan sonra `start_facefusion_retake.ps1` katmanı kendiliğinden açar; kapatmak için `-DisableRealism` kullanılır.

### Kişisel model (en yüksek benzerlik için, tek seferlik eğitim)

Gerçekçilik katmanı, kişinin 25 fotoğrafıyla eğitilen bir LoRA ile "kişiyi tanıyan" bir modele dönüşür. Model sakalı (yanaklara ne kadar yayıldığı dahil), cildi, kaşları ve dudakları öğrenir. Fotoğraflardan yüz kırpımları otomatik çıkarılır; gözleri görünmeyen fotoğraflar (güneş gözlüğü) eğitime alınmaz. Eğitim ayarları GPU belleğine göre seçilir: 512 px, yaklaşık 1000 adım, 10 GB altında bellek tasarrufu açık. Eğitilen model varsa motor onu kendiliğinden kullanır ve her sürüm yine ArcFace ile ölçülür.

Kişisel model varken her render'ın bir de **hibrit** sürümü üretilir: yanaklar, çene, sakal, bıyık ve dudaklar kişisel modelden; gözler, kaşlar ve burun swap'tan gelir. Gözler, kaşlar ve burun ArcFace kimliğinin çoğunu taşır. Kullanıcının fotoğraflarıyla ölçümler (CPU'da 600 adımlık eğitim):
- Hiç fotoğraf verilmeden yazıyla üretilen yüzler kişiye 0,58 ve 0,49 benziyor; genel modelde 0,07 ve 0,006.
- Bahçe fotoğrafında tam kişisel render 0,69, swap zinciri 0,794, hibrit **0,788**. Hibritte sakal, referanslardaki gibi yanaklara kadar dolu.

```powershell
.\tools\train_personal_model.ps1
```

Önce `install_realism.ps1` kurulmuş olmalı. Temel model (~2,1 GB) ilk seferde iner. Fotoğraflar değişirse eğitim yeniden çalıştırılır.

## Web uygulaması

- `index.html` statik web arayüzüdür.
- Ana akışta yalnızca fotoğraf yükleme, sonuç ve "Yeniden oluştur" bulunur. Kişisel referans yönetimi kapalı bir "Yüz profili ayarları" alanındadır; varsayılan 25 referans otomatik yüklenir.
- Her düzenleme veya yeniden oluşturma öncesinde tema uyumlu onay penceresi gösterilir; hedef ve referans yüz kırpımları gönderilmeden önce işlem bazında açık onay alınır.
- **Kendi 25 fotoğrafını seç** ile varsayılan havuz yerine kişisel referans fotoğraflarını kullanabilirsin. 25 JPG, PNG veya WebP dosyası gerekir; her dosya en fazla 25 MB ve görsel en fazla 20 MP olabilir. Referans görselleri işlem belleğini sınırlamak için 2048 px kenara küçültülür; kaynaklar profil anahtarı için değişmeden kullanılır. Yeni sayfa oturumunda fotoğrafları gizlilik nedeniyle yeniden seçmen gerekir.
- MediaPipe, tarayıcıda hedef ve referans fotoğraflarındaki yüzleri algılar; profil aynı sekmede bir kez oluşturulup sonraki hedeflerde bellekte yeniden kullanılır.
- Her referans önce FaceLandmarker, gerekirse BlazeFace kısa menzil ve tam menzil modelleriyle yeniden algılanır. Bir fotoğraf yine de hizalanamazsa işlem durur ve dosya adı gösterilir; 19/25 gibi eksik bir profille devam edilmez.
- Hedef yüz crop'u, 25 referansın tamamından alınan yüz crop'ları, ağırlıklar ve hizalanmış profil aynı bilgisayardaki FaceFusion servisine gönderilir. Sunucu bu crop'ların ArcFace tutarlılık raporunu da çıkarır.
- AlphaFace ve UniFace sonuçları tek hibrit yüze birleştirilir; alt yüz bölgesinde sakal/bıyık görünümü desteklenir. Sonuç hedefe maske ile uygulanır ve maske dışı piksel bütünlüğü kontrol edilir.
- İlk kez açarken tarayıcı `retake06-ua.web.app` için yerel ağ erişimi isterse izin verilmelidir.
- Arka planı silinmiş (saydam) bir fotoğraf seçilirse uyarı gösterilir: Retake yalnız yüzü değiştirir, silinmiş arka planı geri getiremez. Böyle bir fotoğraf JPG'de siyah arka plana döneceği için PNG olarak kaydedilir.
- Seçilen kişisel fotoğraflar tarayıcıda açılır ve yüz crop'ları çıkarılır. Yalnızca kullanıcı düzenlemeyi başlatıp açık onay verdikten sonra hedef yüz crop'u ve referans crop'ları bu bilgisayardaki `127.0.0.1:8765` servisine gönderilir; üçüncü taraf çıkarım API'si kullanılmaz.
- Varsayılan 25 referans fotoğrafı ilk açılışta bir kez indirilir ve tarayıcı önbelleğinde (Cache Storage) kalıcı olarak saklanır; sonraki açılışlarda ve ana ekrana eklenen uygulamada yeniden indirilmez. Yalnızca birkaç KB'lık liste dosyası kontrol edilir. **Varsayılan havuza dön** fotoğrafları yeniden indirir. Tarayıcı verileri silinirse fotoğraflar bir kez daha iner.
- Referans fotoğrafları sırayla açılıp küçültülür; profil hazırlama ilerlemesi görünür. **Fotoğrafları ve profili temizle** seçili fotoğrafları oturum belleğinden ve önbelleğe alınmış kırpılmış profili IndexedDB'den siler. Kaynak fotoğrafları cihazdaki dosyalarından silmez.

## Tasarım

- "Atelier" teması: porselen (açık) ve obsidyen (koyu) zemin, şampanya altını vurgu; Geist ve Instrument Serif yazı tipleri `assets/fonts/` altında barındırılır (SIL Open Font License 1.1, lisans metinleri aynı klasörde).
- Logo: heykelin halkasından esinlenen, tek şerit gibi akan "R" monogramı (`icons/retake.svg`, PNG simgeler bundan üretilir).
- Masaüstünde heykel tüm genişliğe yayılır ve yeri yalnızca hero kutusunun boyutuna göre hesaplanır; açık ve koyu temada aynıdır. Telefonda heykel kartında parmakla sürükleme fare gibi parçacıkları dağıtır.
- Açılışta önce heykel çizilir; yüz algılama modeli ve 25 referans fotoğrafı sayfa boşa çıkınca (ya da fotoğraf seçmeye uzanınca) yüklenir. Referanslar ikişer ikişer açılır.

## Telefon, tablet ve PWA ile kullanım

Yüz dönüşümü NVIDIA ekran kartlı bilgisayarda çalışır; telefon bu bilgisayardaki motoru şifreli bir tünel ve eşleştirme koduyla kullanır. Fotoğraflar bir AI servisine yüklenmez, yalnızca tünel sağlayıcısının (Cloudflare veya Tailscale) şifreli bağlantısından geçer.

1. Bilgisayarda motoru eşleştirme moduyla başlat:

   ```powershell
   .\tools\start_facefusion_retake.ps1 -EnablePairing
   ```

   Pencerede bir **EŞLEŞTİRME KODU** yazar (`%USERPROFILE%\.retake\pairing-token` dosyasında saklanır, her açılışta aynıdır).
2. Ayrı bir pencerede tüneli aç ([cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) kurulu olmalı):

   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8765
   ```

   Çıktıdaki `https://….trycloudflare.com` adresini not et. Tailscale kullanıyorsan `tailscale serve` ile verilen `https://….ts.net` adresi de çalışır.
3. Telefonda https://retake06-ua.web.app adresini aç → **Motor bağlantısı** → adresi ve kodu gir → **Bağlan**.
4. Safari'de **Paylaş → Ana Ekrana Ekle** (Android Chrome'da **Uygulamayı yükle**) ile Retake'i uygulama olarak kurabilirsin. Adres ve kod yalnızca o cihazın tarayıcısında saklanır.

Notlar:
- Eşleştirme açıkken motora gelen her istek kodu taşımak zorundadır; kodu bilen herkes motoru kullanabilir, paylaşma. Kodu yenilemek için `pairing-token` dosyasını silip motoru yeniden başlat.
- Ücretsiz `trycloudflare.com` adresi tünel her açıldığında değişir; telefonda adresi güncellemek gerekir.
- Uzun işlemler arka plan işi olarak çalışır; telefon ekranı kapansa bile işlem bilgisayarda sürer, uygulama yeniden açıldığında durum tekrar sorgulanır.

## Sonuç stüdyosu

- **Kimlik benzerliği:** Motor her sonucu ve orijinal yüz kırpımını, 25 referansın ortak ArcFace kimlik vektörüyle (aykırılar hariç ortalama) karşılaştırır. Sonuç panelinde "Kimlik benzerliği 66 · Çok güçlü · orijinal 21" gibi gösterilir; her görünümün küçük resminde puanı yazar, en benzer görünüm ★ ile işaretlenir. Bu bir benzerlik ölçüsüdür, kimlik kanıtı değildir; ölçüm başarısız olursa sonuç yine gelir, yalnızca puan gösterilmez.
- **Profil sağlığı:** Yüz profili ayarlarında 25 referansın birbirine uyumu ve diğerlerine benzemeyen fotoğraflar (etkileri otomatik azaltılır) listelenir.
- **Basılı tut, karşılaştır:** Sonuç fotoğrafına basılı tutunca orijinal, bırakınca sonuç görünür.

- **Önce / Sonra** kaydırıcısı orijinal ile sonucu aynı karede karşılaştırır.
- **Dönüşüm gücü** (%30–100) sonucu orijinalle harmanlar; indirme ve paylaşım bu ayarı kullanır.
- Son altı görünüm küçük resim olarak tutulur; aralarında geçiş yapılabilir.
- PNG (kayıpsız) veya JPG olarak indirilebilir; destekleyen telefonlarda **Paylaş** ile doğrudan Fotoğraflar'a kaydedilebilir.

## Firebase Hosting

Firebase projesi: `projelerim-8fecd`  
Hosting hedefi: `retake06` → site `retake06-ua`  
Canlı adres: https://retake06-ua.web.app

Sadece bu siteyi dağıtmak için:

```bash
npx --yes firebase-tools@latest deploy --only hosting:retake06 --project projelerim-8fecd
```

Diğer Hosting sitelerini güncellememek için `--only hosting:retake06` hedefini koru. Firebase CLI oturumu yoksa önce `npx --yes firebase-tools@latest login --no-localhost` ile doğrulama yap.

Eski adres https://retake06.web.app (`hafize-23465` projesi, `.firebaserc` içinde `legacy`) bant genişliği kotası dolduğu için yerini yeni siteye bıraktı. Yerel FaceFusion motoru iki adrese de izin verir.

## Gizlilik ve referans görselleri — önemli

Yüz profili ve çıkarım yerel bilgisayarda yapılır; kırpılmış profil görselleri ve ağırlıklar tarayıcının IndexedDB alanında cihazda saklanır. Ancak 25 orijinal referans JPEG'i public GitHub deposunda ve Firebase Hosting alanında bulunabilir; eski Git commit'lerinde de kalmış olabilir. Yerel profil önbelleği bu kaynak fotoğrafları public olmaktan çıkarmaz. Bunları gizli kabul etme. Fotoğrafları public depodan çıkarmak ve Git geçmişini temizlemek ayrı bir güvenlik çalışmasıdır; yalnızca yeni commit'ten silmek geçmiş kopyaları ortadan kaldırmaz.

Model ağırlıklarının lisansları farklıdır. FaceFusion kataloğunda `alphaface_256` Non-Commercial, `uniface_256` ise Unknown olarak listelenir. Ticari kullanım hakkı olduğunu varsayma; iki modelin ve yardımcı ağırlıkların lisansını ayrı ayrı doğrula.

## Testler

```bash
npm test
npm run validate
```

Yerel FaceFusion Windows servisi Python tarafında `tools/facefusion_local_server.py` ile çalışır; Retake web arayüzünün Node sunucusuna veya Firebase Functions'a ihtiyacı yoktur.

## Native iOS

`ios/Retake.xcodeproj` SwiftUI uygulama kabuğu ve PhotosPicker/Vision altyapısını içerir. FaceFusion'ın Windows/CUDA motoru iOS üzerinde çalışmaz; native iOS için ayrı Core ML uyumlu model, model lisansları ve performans çalışması gereklidir. Şu an native iOS inference tamamlanmış değildir.

## 7 dakikalık Ultra-420 v16'ya geçiş

Github'a kod eklenmesi **Firebase Hosting** sürümünü ya da Windows üzerinde zaten açık FaceFusion sürecini değiştirmez. Canlı sayfada eski `app.js` ve `/health.model=retake_hybrid_25_reference_v11` görülüyorsa yeni sürüm hâlâ kullanılmıyor demektir. Retake klasöründe PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\tools\\update_retake_ultra.ps1
```

Bu komut `git pull --ff-only origin main` yapar, Firebase CLI mevcut ve oturum açıksa `projelerim-8fecd` projesindeki `hosting:retake06` hedefine yayınlamayı dener; yalnızca Retake'nin eski Python köprüsünü durdurup FaceFusion sunucusunu v16 ile yeniden başlatır. Eşleştirme açıksa korunmaya çalışılır. CUDA/FaceFusion ve isteğe bağlı gerçekçilik modelleri zaten kurulu olmalıdır; ilk kez kurmak için `docs/LOCAL_FACEFUSION_SETUP.md` dosyasına bak. Firebase yoksa `npm install -g firebase-tools` ve `firebase login` kullan.

Güncel motorun sağlık değerleri: `model=retake_hybrid_25_reference_ultra_v16`, `ultra_mode=true`, `ultra_rounds=24`, `ultra_min_seconds=420`. ArcFace'in `0.83` değeri kalibre edilmiş %83 kimlik ihtimali değildir.

## Retake v17: ince koyu yüz konturu ve Ultra zaman tahmini

v16 ile yapılan gerçek fotoğraf denemesinde, aktarılan yüzün alın–yanak–çene boyunca ince koyu bir kesme çizgisi oluşabildiği gözlendi. v17, yalnızca **değişmiş yüzün kenarındaki** yeni ve belirgin kararmaları tespit ederek 5×5 medyan tabanlı dar bir kontur düzeltmesini alternatif aday olarak oluşturur (`tools/retake_rim_cleanup.py`). Fotoğrafın dışı ve PNG alfa kanalı değiştirilmez. **ArcFace** ile kişisel profil benzerliği yeniden ölçülür; 0.004 mutlak kosinüs puanından daha büyük bir kayıp varsa düzeltme atılır. Bu tek başına kişisel kimlik benzerliğinin artacağını garanti etmez; çizgi artefaktını azaltmayı amaçlar.

Ultra-420 tahmini kalan süre artık teorik 24 turun tamamını (ör. 73 dakika) toplamaktan kaçınır. Yeni GPU turu 420 saniyelik arama penceresi dolduğunda başlatılmaz; devam eden CUDA geçişi bu süreyi aşabilir. Eksik hız geçmişinde veya final süre beklemesinde ETA hesabı hata vermez. Ekrandaki ArcFace sayı değeri artık 83 yerine örneğin **82,8** ya da **83,3** gibi bir ondalıkla gösterilir; bu sayı doğrulanmış kişinin kimlik olasılığı değildir.

Yeni sürüme geçmek için `.\\tools\\update_retake_ultra.ps1` çalıştırılmalı ve gerekirse Firebase yayını tamamlanmalıdır. Sağlık kontrolü `model=retake_hybrid_25_reference_ultra_v17` göstermelidir. Web arayüzü geçiş süresince hem v16 hem v17 Ultra motorlarına bağlanabilir.


## Retake v18 — otomatik sakal, bıyık ve çene transferi (Ultra-420)

Önceki v17 sürümünde sakal çoğunlukla açık/koyu renk dengelemesiyle işleniyordu; gerçek referans fotoğrafını yüzün üzerine taşıyan seçenek ise varsayılan olarak kapalıydı. v18 ile MediaPipe 478 işaretli yüzde **pozla eşleşen en fazla 3 gerçek referans fotoğrafı**, varsayılan olarak *yalnızca sakal / bıyık / çene* adayı için hazırlanır. `tools/lower_face_transfer.py` kaynak fotoğraftaki gerçek kıl/doku bilgisini hedef yüze 478 noktalı üçgenlerle taşır, dudakları, ağız açıklığını, gözleri, saç ve kıyafetleri değiştirmez. Yüzün tümünü gerçek fotoğraftan aktarmak hâlâ ayrı bir kullanıcı seçeneği olarak kalır.

Ayrıca, referans yüzün ölçülen çene konturunu daha güçlü uygulayan bazı Ultra varyasyonları bulunur. Ortaya çıkan alternatifler yine v18 motorundaki ArcFace kimlik, kenar dikişi ve gerçekçilik kontrollerinden geçirilir; değişiklik yüzü daha az benzetiyorsa önceki güvenilir adaya dönülür. Pozu eşleşen referans bulunamadığında alt yüz nakli devreye girmez. **Bu algoritma 83 skorunu kesin olarak yükseltme garantisi vermez** ve salt koyuluk/eşleştirme ölçüleri biyometrik çene/sakal kimlik skoru değildir.

GitHub `main` güncellenmesi tek başına **Firebase web yayını veya Windows'ta açık FaceFusion motorunu** güncellemez:

```powershell
Set-Location "C:\\AI\\retake"
.\\tools\\update_retake_ultra.ps1
```

Yeni yerel motorun `/health` yanıtında `model=retake_hybrid_25_reference_ultra_v18` görünmelidir. Canlı `app.js` dosyasında `X-Retake-Lower-Face-Transfer` bulunması web yayınının güncelliğine işarettir. Güncel arayüz, v18 motorunu zorunlu kılar.

## Retake v19 — full-face identity reconstruction (Ultra-420)

Retake v19 now compares **three distinct kinds of identity-preserving candidate** automatically: FaceFusion's original masked AI identity, a 478-point triangulated transfer of the person's own *beard/moustache/chin*, and a broader reference **full-face** transfer including visible skin, eyes, nose and lower-face detail. The reference face is warped to the target pose and lighting; the target's mouth opening, hairline, scene, clothes and outer image are preserved. Since FaceFusion's official `--face-mask-regions` choices do **not** include a beard region, the real-reference candidate explicitly covers beard pixels **even where the skin-only FaceFusion swap left the original facial hair unchanged**. Masks fade within the mesh; mismatched poses and visible seam failures are rejected.

**Selection is not forced.** Every reference candidate competes against the pure AI candidate using local ArcFace, invisible seam, texture and realism gates. A near-tied complete-reference candidate can win, but never at the cost of a significant verified identity drop. The source reference shortlist now tolerates moderate pose differences, which are separately verified by the real-photo landmark fit. Some occlusions (a hand over the face) cannot be reliably detected without additional segmentation: those photos may need a different source/pose.

**GPU time:** Up to 24 identity variants still use a **420-second search budget and a minimum 420-second elapsed time** for a successful result. The heavyweight diffusion/personal LoRA photographic-realism stage now runs **once, on the leading verified finalist** rather than once in each candidate. If it worsens identity/texture, the original best stays. Reference triangle warp results are cached within each round rather than computed three times for every tone setting. A running GPU pass can exceed 420 seconds; no absolute completion SLA or exact 100% identity guarantee is possible.

**Activate the already-merged code on your Windows PC and Firebase site**:

```powershell
Set-Location "C:\\AI\\retake"
Set-ExecutionPolicy -Scope Process Bypass
.\\tools\\update_retake_ultra.ps1
```

The helper pulls main, tries `firebase deploy --only hosting:retake06 --project projelerim-8fecd` if CLI/authentication is available, and restarts only the Retake Python bridge. It **does not** magically install missing CUDA, diffusion checkpoints or train a personal LoRA. To confirm the new engine: `(Invoke-RestMethod "http://127.0.0.1:8765/health").model` must equal `retake_hybrid_25_reference_ultra_v19`. The live `app.js` should contain `X-Retake-Complete-Identity`. GitHub merge does not automatically publish or restart the user's computer.


## Retake v20 — Gerçek GPU logundan doğrulanan kritik Ultra hataları

23 Kasım 2026 tarihli **canlı v19 GPU testi** iki ayrı kalite sorununu doğruladı. İlk olarak FaceFusion 3.9.2 yalnızca **0.05 aralıklarla** verilen `--face-swapper-weight` değerlerini kabul ederken, v19'un 24 adaylık listesinde `0.94, 0.99, 0.98, 0.97, 0.96, 0.92` gibi geçersiz değerler vardı. Bu turlar sessizce eleniyor, gerçekte çok daha az varyasyon çalışıyordu. v20, tüm 24 varyasyonu doğrudan geçerli değerlerle tanımlar (1.00 / 0.95 / 0.90). `supported_facefusion_weight()` CLI'ye gönderim **ve GPU önbellek anahtarı** için ikinci bir güvenlik katmanı olarak 0.05'e yuvarlama yapar; tüm varyasyonların CLI uyumluluğu regresyon testleriyle sınanır.

İkincisi, v19'daki gerçek sakal/çene ve tam yüz nakli **ara aday** olarak başarıyla oluşturulsa da ArcFace skorları (sakal 0.797–0.817, tam yüz yaklaşık 0.632–0.652) saf kimlik yedeğinin (0.833) altında kalınca **nihai çıktı sürekli `identity-only`** oluyordu. Böylece gerçek referans sakalı ve çene şekli geri alınmış oluyordu. v20, güçlü kimlik yedeğinin ÜZERİNE doğrudan, iki ayrı güçte (0.35, 0.65) gerçek sakal/çene adayı, iki düşük güçte (0.20, 0.40) tam yüz referans dokusu adayı ve referansa göre ayrı bir çene geometrisi adayı oluşturur. Bölgesel sakal adayları, mevcut kimlik yedeğinin sınır dikişine göre belirgin daha kötü kenar oluşturmuyorsa ölçülür. **Sadece 0.003 veya daha küçük mutlak ArcFace kaybı** varsa, referans sakalı/çene varyantı kimliği değiştirmeyen yedeğin yerine geçebilir. Büyük kimlik kaybı varsa yine `identity-only` korunur; başarı garantisi veya sahte skor yükseltmesi yapılmaz.

Üçüncüsü, ana gerçekçilik/difüzyon modelleri üç render üretmiş ama ArcFace 0.730 / 0.722 / 0.703 ölçtüğünden v19 bunları doğru biçimde elemişti. v20 bu kimlik koruma filtresini **gevşetmez**. Bu skorlar kalibre edilmiş kimlik yüzdeleri değildir.

Sonuç ekranı, saf kimlik yedeği seçildiğinde **referans sakal/çene aktarımının finalde kullanılmadığını** dürüstçe gösterir. `/health.model` v20'yi, `X-Retake-Identity-Selection` ise gerçekten seçilen final yöntemini belirtir. Örneğin `identity-beard` veya `identity-jaw` ancak ArcFace ve kenar koşulları sağlanınca döner.

Windows kurulumunda eski `C:\AI\retake` klasörü GitHub geçmişiyle ayrışmıştı. Onun yerine 23 Kasım'da oluşturulan güvenli worktree kullanılır:

```powershell
Set-Location "C:\AI\retake-v19-20261123-143224"
git pull --ff-only origin main
.\tools\update_retake_ultra.ps1 -SkipPull
```

Devam eden iş yokken motor yeniden başlatılmalıdır. Firebase deployment başarılıysa tarayıcıda Ctrl+F5; motor `(Invoke-RestMethod "http://127.0.0.1:8765/health").model` ile `retake_hybrid_25_reference_ultra_v20` olarak kontrol edilir. Bir GitHub merge işlemi **bilgisayarda çalışan servisi veya canlı siteyi** kendiliğinden güncellemez.


## Retake v21 — ölçülen kontur onarımı

Gerçek v20 GPU çıktısında saç çizgisi, şakaklar ve çene altında koyu yapay sınır görülürken motor "6496 piksel yüz konturu temizlendi" diyordu. Önceki 5x5 medyan filtre piksel sayısını başarı olarak bildiriyor, 2–4 piksel kalınlığındaki koyu çizgiyi tam kaldırmıyordu.

v21, sınır bandını genişleterek **yalnızca orijinal hedef fotoğraftan belirgin şekilde daha koyu hale gelmiş, alfa içindeki** kontur piksellerini Telea ile tamamlar; medyan filtre alternatifini de karşılaştırır. Düzenleme başarılı kabul edilmek için tespit edilen koyu çizginin **en az %25 azalması** gerekir. Göz, sakalın içi, fotoğraf çevresi ve alfa maskesi korunur. Son ArcFace testi önceki **en fazla 0,004 skor kaybı** koşulunu korur. Loglarda artık tespit edilen koyu sınırın gerçekte yüzde kaç azaldığı da gösterilir.

Bu, görsel yüz kenarı iyileştirmesidir; **kimlik skorunu artırma veya kusursuz eşleşme garantisi değildir**. Sentetik testler gerçek fotoğraftaki bütün hataları dışlamaz. Beklenen `/health.model`: `retake_hybrid_25_reference_ultra_v21`.

Bilgisayarındaki çalışan güncel worktree klasöründen (eski `C:\AI\retake` Git geçmişi ayrıştığı için kullanılmamalı), fotoğraf işi yokken:

```powershell
Set-Location "C:\AI\retake-v19-20261123-143224"
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "Güncelleme tamamlanmadı" }
.\tools\update_retake_ultra.ps1 -SkipPull
```

GitHub merge tek başına yerel GPU motorunu veya Firebase Hosting'i yayımlamaz.


### Retake v22 — ince yüz/çene çizgisi ve karar kayıtları

v21 gerçek fotoğraf testinde motor ve Firebase doğru sürümde çalışsa da v20 ve v21 çıktıları piksel bazında neredeyse aynı kaldı; kazanan ilk tur için kontur temizliği uygulanmamıştı. v21 koyu çizginin geniş 11–27 piksellik sınır bandının en az %3'ünü kaplamasını beklediği için ince fakat uzun bir çizgi elenebiliyordu. v22, şüpheli koyulaşmayı **yalnızca 7–13 piksellik dar gerçek kontur şeridinde** puanlar, en az %1,5 şerit payı / asgari koyu piksel koşulunu kullanır ve işe yarayan onarımı yine ArcFace doğrulamasına gönderir. Çizgi hedef fotoğrafın kendisinde de varsa onarılmaz.

Her doğrulanan turda `[Retake] Kontur taraması: koyu piksel=...; kenar payı=%...; karar=...; azalma=%...` yazılır. `karar=repaired` olsa dahi sonraki `Kontur adayı kimliği koruyamadı; atlandı` kaydı olasıdır: bu durumda onarım **nihai sonuçta kullanılmamıştır**. Bu yeni tanı kayıtları gerçek kaynak fotoğraf olmadan önceki atlanmanın kesin sebebini varsaymak yerine sonraki testte ölçmeye yarar.

Güncelleme, tamamlanan fotoğraf işlemi yokken: `git pull --ff-only origin main`, ardından `.\tools\update_retake_ultra.ps1 -SkipPull`. Motor `retake_hybrid_25_reference_ultra_v22` dönmelidir. Görsel kalitenin arttığı ancak yeni üretilmiş gerçek sonuçla doğrulanabilir.
