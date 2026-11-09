# Retake — ücretsiz NVIDIA Worker kurulumu

Bu depo, Firebase Hosting'de yalnızca `retake06` sitesini kullanmaya devam eder. NVIDIA istekleri Firebase Functions yerine Cloudflare Workers Free üzerinden iletilir. Firebase projesini Spark planında tutmak için Cloud Functions deploy edilmez.

## Mimari ve ücretsiz kotalar

- Firebase Hosting: mevcut `hafize-23465` projesinde Spark planı.
- NVIDIA proxy: Cloudflare Workers Free.
- NVIDIA API anahtarı: yalnızca Cloudflare Worker Secret.
- Kullanıcı erişimi: Worker üzerinde doğrulanan özel parola; 12 saatlik imzalı oturum.
- Görsel aktarımı: tarayıcı PNG dosyasını doğrudan Worker'a gönderir; Base64'e çevrilmez. Worker, NVIDIA NVCF asset akışını kullanır ve başarılı/başarısız işlem sonrasında geçici asset'i silmeyi dener.
- Worker CPU işlemi hafif tutulur; NVIDIA ağ isteği beklenirken Cloudflare CPU kotası tüketilmez.

Cloudflare Workers Free güncel limitleri: 100.000 istek/gün, istek başına 10 ms CPU ve 128 MB bellek. NVIDIA Developer Program, API Catalog'daki uygun hosted NIM endpoint'lerine prototipleme için ücretsiz erişim sunar; fakat model/endpoint bazında kullanılabilirlik, hız sınırı ve kota değişebilir. Önemli: NVIDIA kataloğunda FLUX.1-Kontext-dev şu anda "Downloadable" olarak listeleniyor, "Free Endpoint" olarak işaretli değil. Bu nedenle ücretsiz API anahtarının bu model için NVIDIA-hosted inference erişimi verdiği garanti değildir. İlgili endpoint hesabında etkin değilse 403/404 alınabilir; bu durumda bu modelin NVIDIA üzerinde ücretsiz hosted kullanımı mümkün olmayabilir. Ücretsiz erişim, sınırsız kullanım garantisi değildir.

## İlk kurulum (Codespaces terminali)

1. Codespaces terminalinde Wrangler kurulum onayına `y` yaz.
2. Cloudflare hesabın yoksa ücretsiz hesap aç: https://dash.cloudflare.com/sign-up
3. Giriş yap:

```bash
npx --yes wrangler@latest login
```

Tarayıcıda Cloudflare hesabına izin ver.

4. Worker'ı ilk kez dağıt:

```bash
npx --yes wrangler@latest deploy
```

Terminal, `https://retake-nvidia-edit.<hesap-adin>.workers.dev` biçiminde Worker adresini gösterecek.

5. NVIDIA API anahtarını sunucu tarafında Secret olarak ekle:

```bash
npx --yes wrangler@latest secret put NVIDIA_API_KEY
```

Komutun istediği alana NVIDIA API anahtarını yapıştır. Anahtarı GitHub'a, `.env` dosyasına, HTML'e veya bana gönderme.

6. Retake için uzun ve benzersiz bir erişim parolası oluştur:

```bash
npx --yes wrangler@latest secret put RETAKE_ACCESS_PASSWORD
```

Bu parolayı yalnızca Retake'te giriş yaparken sen kullanacaksın. En az 24 karakterlik, başka yerde kullanmadığın bir parola seç. Worker parola değerini Git'e yazmaz. Başarılı girişten sonra 12 saat geçerli, imzalı bir oturum belirteci tarayıcı sekmesinin sessionStorage alanında tutulur.

7. Bana `wrangler deploy` çıktısındaki tam `workers.dev` adresini gönder. Bu adres gizli değildir; NVIDIA API anahtarı veya Retake erişim parolasını gönderme. Adresi `index.html` içindeki `retake-ai-base` ayarına bağlayıp Firebase Hosting dağıtımı için kesin komutu verebilirim.

## Hosting dağıtımı

Worker URL'si uygulamaya eklendikten sonra Codespaces terminalinde:

```bash
npx --yes firebase-tools@latest deploy --only hosting:retake06 --project hafize-23465
```

Bu komut yalnızca `retake06` Hosting hedefini yayımlar. Diğer yedi Hosting sitesi hedeflenmez. Firebase Spark planında kalınır; Firebase Functions dağıtılmaz.

## Güvenlik

- NVIDIA anahtarı tarayıcıya gönderilmez.
- Worker API'si yalnızca `retake06.web.app` ve `retake06.firebaseapp.com` origin'lerine CORS izni verir.
- Origin kontrolüne ek olarak Worker parola doğrulaması ve 12 saatlik imzalı Bearer oturumu uygular.
- Oturum açma denemeleri ve düzenleme çağrıları için yardımcı IP tabanlı sınırlar bulunur. Bu bellek içi sınır global bir WAF/rate-limit kuralının yerini tutmaz.
- NVIDIA'ya görsel gönderiminden önce uygulama açık onay ister.
- Worker yalnızca PNG panosunu kabul eder (en fazla 10 MB). Orijinal hedef fotoğrafın tamamı gönderilmez.
- NVIDIA geçici asset'i işlem sonunda silinmeye çalışılır. Silme başarısız olursa NVIDIA'nın saklama koşulları geçerli olabilir.
- Bu, kişisel uygulama için erişim bariyeridir; XSS riskine karşı tarayıcı ve hesap güvenliği yine önemlidir.

## 20 kayıtlı referans görseli

Bu değişiklik mevcut referans veri kaynağını otomatik taşımaz. Hafize'deki `img_...` kayıtları sahibinin oturumuyla korunan yerel SQLite/API verisindedir; herkese açık `retake06.web.app` bunlara doğrudan ulaşamaz. 20 fotoğrafı özel ve erişim kontrollü bir depoya bir defalık aktaracak ayrı işlem tamamlanana kadar uygulamadaki referans havuzu yüklenmeyebilir. Fotoğrafları public GitHub deposuna veya Hosting içine koyma.

## Sorun giderme

- `RETAKE_ACCESS_PASSWORD Worker Secret olarak tanımlanmamış`: `wrangler secret put RETAKE_ACCESS_PASSWORD` komutunu tekrar çalıştır.
- `NVIDIA_API_KEY Worker Secret olarak tanımlanmamış`: NVIDIA secret'ını ekle.
- NVIDIA 401/403: API anahtarını, `invoke_function` yetkisini ve hosted endpoint erişimini doğrula.
- NVIDIA 404/422: bu endpoint NVIDIA hesabında etkin olmayabilir veya endpoint girdiyi desteklemiyor olabilir.
- NVIDIA 402/429: NVIDIA tarafındaki kota, ücretsiz erişim veya hız sınırı uygulanmıştır.
- Cloudflare 10 ms CPU aşımı: Worker kodunda görsel base64 dönüştürme veya büyük JSON ayrıştırma ekleme; mevcut akış görseli ikili veri olarak geçirir ve NVIDIA yanıtını akış olarak tarayıcıya döndürür.
