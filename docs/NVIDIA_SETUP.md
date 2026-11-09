# Retake — Cloudflare Worker ve yerel NVIDIA anahtarı

Firebase Hosting, mevcut `hafize-23465` projesinde Spark planında kalır. NVIDIA API trafiği Cloudflare Workers Free üzerinden gider; Firebase Functions kullanılmaz.

## İstenen kullanım: parola yok, API anahtarı dosyada

- Retake uygulamasında kullanıcı girişi veya ek parola yok.
- NVIDIA anahtarını depo kökündeki yerel `.dev.vars` dosyasına bir kez yaz.
- `.dev.vars` Git tarafından yok sayılır; GitHub'a commit edilmez.
- `npm run deploy:worker` komutu dosyadaki anahtarı alır, Worker kodunu dağıtır ve anahtarı Cloudflare Secret olarak aktarır. Anahtar Worker koduna veya tarayıcıya konmaz.
- `.dev.vars` yerel/özel kaynak dosyadır. Canlı Worker anahtarı Cloudflare'ın şifrelenmiş Secret alanından okur; deploy betiği dosyadaki değeri otomatik olarak Secret'a aktarır.

## Kurulum

### 1) Cloudflare girişi
```bash
npx --yes wrangler@latest login
```
Tarayıcıda Cloudflare hesabına izin ver.

### 2) NVIDIA anahtarı dosyası
Depo kökünde:
```bash
cp .dev.vars.example .dev.vars
```
VS Code Explorer'dan `.dev.vars` dosyasını aç. `NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE` satırındaki yer tutucuyu kendi NVIDIA API anahtarınla değiştir. Dosyayı kaydet. Anahtarı terminale veya sohbete yapıştırma.

> Dosya `.gitignore` içindedir. `git status` çıktısında görünmemelidir. `.dev.vars` dosyasını hiçbir şekilde GitHub'a gönderme.

### 3) Worker kodu ve anahtarı tek komutla dağıt
```bash
npm run deploy:worker
```
Bu komut sırasıyla Worker kodunu dağıtır ve `.dev.vars` içindeki NVIDIA anahtarını Wrangler stdin üzerinden `NVIDIA_API_KEY` Worker Secret olarak kaydeder. Hassas anahtarı terminal çıktısına yazdırmaz. `wrangler secret put` gizli değerin stdin'den aktarılmasını destekler.

İlk seferde Wrangler'ın npm paketi için onay sorması hâlinde `y` yazabilirsin. Daha sonra `workers.dev` adresi terminalde görünür.

### 4) Site bağlantısı
Dağıtım sonunda oluşan tam `https://retake-nvidia-edit.<hesap-adin>.workers.dev` adresini al. Bu adres gizli değildir. `index.html` içindeki `retake-ai-base` değerine bu adres bağlanacak. Ardından yalnızca Retake Hosting'i dağıt:
```bash
npx --yes firebase-tools@latest deploy --only hosting:retake06 --project hafize-23465
```
Diğer yedi Firebase Hosting sitesi hedeflenmez. Firebase Spark planında kalır; Firebase Functions dağıtılmaz.

## Erişim ve güvenlik uyarısı

Kullanıcı parolası kaldırılmıştır. Worker, tarayıcı CORS için yalnızca `retake06.web.app` ve `retake06.firebaseapp.com` origin'lerine izin verir ve yardımcı olarak IP başına 15 dakikada 5 düzenleme sınırı uygular.

**CORS bir kimlik doğrulama yöntemi değildir.** Herhangi biri Workers URL'sini öğrenip doğrudan HTTP istemcisiyle çağrı yapabilir veya Origin başlığını taklit edebilir. Kullanıcı parolası istemediğin için Worker endpoint'i fiilen herkese açık bir API haline gelir. Bellek içi hız sınırı dağıtık veya güçlü bir kötüye kullanım önlemi değildir. NVIDIA ücretsiz kotan kötüye kullanılabilir veya tükenebilir. Parolasız kullanımın kaçınılmaz riski budur.

NVIDIA API anahtarı yine de tarayıcıda, HTML/JS'de veya GitHub'da yer almaz; yalnızca Cloudflare Secret'ta saklanır. NVIDIA'ya görsel gönderilmeden önce uygulama açık onay ister. Worker hedef crop + referans panosunu kabul eder, maksimum 10 MB, ve NVIDIA geçici asset'ini işlem sonunda silmeyi dener.

## Ücretsiz kullanım notları

Cloudflare Workers Free planında günlük ve CPU limitleri vardır. NVIDIA API Catalog'da ücretsiz hosted erişim model bazında değişir. FLUX.1-Kontext-dev katalogda `Downloadable` olarak listelendiğinden, NVIDIA API key bu hosted endpoint'e erişim garantisi vermez. İlk gerçek düzenleme çağrısı ile 403/404/422 yanıtlarını kontrol et.

## 20 kayıtlı yüz referansı

Bu değişiklik 20 Hafize görselini taşımaz. Eski `img_...` kayıtları Hafize'nin yerel/oturum korumalı SQLite görsel deposunda. `retake06.web.app` yerel SQLite veritabanına erişemez. Referansları özel depoya güvenli biçimde taşıyan ayrı bir aktarım adımı gerekir; fotoğrafları GitHub'a veya public Hosting'e kopyalama.

## Sorun giderme

- `NVIDIA_API_KEY Worker Secret olarak tanımlanmamış`: `.dev.vars` satırını kontrol edip `npm run deploy:worker` çalıştır.
- `401`: NVIDIA API anahtarı geçersiz veya süresi dolmuş olabilir.
- `403`: NVIDIA anahtarının `invoke_function` yetkisini ve hosted endpoint erişimini kontrol et.
- `404/422`: FLUX.1-Kontext-dev bu hesapta hosted API olarak açık olmayabilir.
- `402/429`: NVIDIA kota, hız sınırı veya ücretsiz erişim limiti.
- `npm run deploy:worker` Cloudflare oturumu istiyorsa önce `npx --yes wrangler@latest login` çalıştır.
