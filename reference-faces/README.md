# Yerel referans fotoğrafları

Site 25 görseli bu klasördeki `manifest.json` dosyasından yükler. Manifest, klasördeki 25 WhatsApp JPEG dosyasının mevcut adlarını listeler. Dosyaları yeniden adlandırır veya değiştirirsen `manifest.json` içindeki `images` listesini de eşitle. JPG, PNG ve WebP desteklenir; her dosya 25 MB'tan küçük olmalı. Tarayıcı statik bir klasörü kendiliğinden listeleyemediği için manifest gereklidir.

Siteyi yerel HTTP sunucusuyla aç; `file://` üzerinden çalıştırma. Fotoğraflar GitHub deposunda ve Firebase Hosting'de yayımlanır; bu URL'lere erişebilen herkes görselleri indirebilir. Bu davranış yalnızca fotoğrafların yayımlanmasına açık onay verildiği için yapılandırılmıştır. Fotoğrafları gizli tutmak istersen Hosting'den kaldırıp kimlik doğrulamalı özel depolama kullan.

Native iOS uygulaması için fotoğrafları iPhone Fotoğraflar veya Dosyalar uygulamasından içe aktar. Uygulama her görselde tek yüz bulunduğunu kontrol eder; bunun aynı kişiyi gösterdiğini doğrulamaz.