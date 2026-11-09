# NVIDIA FLUX entegrasyonu

## Ne kullanılıyor?

Bu entegrasyon başka bir AI sağlayıcısı kullanmaz. NVIDIA NVCF Asset API ile görsel yüklenir ve NVIDIA'nın FLUX.1-Kontext-dev görsel düzenleme uç noktasına gönderilir.

Model özel bir Face Swap motoru değildir. Çıktı deneysel kabul edilir; kimlik benzerliği garanti edilmez. FLUX.1-Kontext-dev NVIDIA model kartında indirilebilir ve ticari olmayan kullanım için listelenir. Kişisel deneme dışında kullanım için model sahibinin koşulları ayrıca incelenmelidir.

## Veri akışı

1. Site, kullanıcının seçtiği hedef fotoğrafta yüz tespiti yapar.
2. Veritabanından gelen 20 kayıtlı fotoğrafı alır.
3. Seçili hedef yüzün crop'unu ve 20 referansı tek bir 1568×1024 PNG panoda birleştirir.
4. Açık onaydan sonra bu pano Firebase Function üzerinden NVIDIA'ya gönderilir. Hedef fotoğrafın tam hâli gönderilmez.
5. NVIDIA sonucu döndürür.
6. Tarayıcı, oluşturulan yüzü yaklaşık oval bir yüz maskesinin içinde orijinal fotoğrafa birleştirir.
7. Maske dışındaki pikseller test edilir. Herhangi bir dış piksel değişimi görülürse sonuç reddedilir.
8. NVIDIA geçici asset kimliği işlemden sonra silinmeye çalışılır.

Maskenin şekli yüz landmark sınırından hesaplanan yaklaşık bir elipstir; semantik saç/örtü segmentasyonu değildir. Sonucu indirmeden önce kontrol et.

## NVIDIA API anahtarını güvenli ekleme

Anahtarı bana gönderme, HTML/JavaScript'e yazma ve GitHub'a commit etme.

Firebase CLI ile:

```bash
firebase login
firebase use <FIREBASE_PROJECT_ID>
firebase functions:secrets:set NVIDIA_API_KEY
```

Komut sorunca NVIDIA API anahtarını terminale yapıştır.

İzin verilen web origin adreslerini Functions ortamına eklemek için `functions/.env` dosyası oluştur:

```dotenv
RETAKE_ALLOWED_ORIGINS=https://retake06.web.app,https://retake06.firebaseapp.com
```

Site başka bir adreste çalışıyorsa o adresi ekle. Bu dosya Git tarafından yok sayılır; commit etme.

Functions dağıtımı:

```bash
firebase deploy --only functions:retakeNvidiaEdit --project <FIREBASE_PROJECT_ID>
```

Dağıtım sonunda Function URL'si şu şekildedir:

```text
https://europe-west1-<FIREBASE_PROJECT_ID>.cloudfunctions.net/retakeNvidiaEdit
```

Bu URL'yi `index.html` içindeki `meta[name="retake-ai-base"]` alanına yaz ve mevcut Firebase Hosting hedefini kullanarak siteyi yayımla. Örneğin:

```html
<meta name="retake-ai-base" content="https://europe-west1-PROJECT_ID.cloudfunctions.net/retakeNvidiaEdit">
```

Functions doğrudan URL'si kullanılınca `RETAKE_ALLOWED_ORIGINS` içine sitenin tam origin'i eklenmelidir.

## Gerekli NVIDIA yetkisi

NVCF varlık oluşturma çağrısı API anahtarında `invoke_function` scope ister. Anahtar 403 döndürürse NVIDIA hesabında bu yetkinin ve görsel endpoint erişiminin açık olması gerekir.

FLUX.1-Kontext-dev model kartında Downloadable olarak listelendiği için tüm NVIDIA API anahtarlarının bu barındırılan image endpoint'ine erişeceği garanti değildir. 403/404/422 yanıtlarında uygulama işlemi tamamlanmış gibi göstermez; endpoint bu anahtarla kullanılamıyorsa NVIDIA GPU üzerinde kendi NIM dağıtımı gerekir. Bu da yalnızca API anahtarından ibaret değildir.

## Faturalandırma ve güvenlik

- Firebase Cloud Functions dağıtımı için Firebase projesinin Blaze planında olması gerekebilir.
- Function tek instance, 15 dakikada IP başına en fazla 5 düzenleme isteği ve 10 MB giriş sınırı kullanır. Bu, kişisel kullanım için başlangıç korumasıdır; güçlü kullanıcı kimlik doğrulaması/App Check yerine geçmez.
- NVIDIA anahtarı sadece Cloud Secret Manager üzerinden Function'a bağlanır.
- Görsel içerikleri log'lara yazılmaz.
- Geçici NVCF asset'i istek sonunda silinir; silme başarısızsa NVIDIA'nın saklama politikası geçerlidir.
