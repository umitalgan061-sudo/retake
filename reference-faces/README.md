# Yerel referans fotoğrafları

Site 25 görseli bu klasördeki `manifest.json` dosyasından yükler. Fotoğrafları aşağıdaki adlarla bu klasöre koy:

`01.jpg` … `25.jpg`

Manifestte dosya adlarını değiştirebilirsin; yalnızca JPG, PNG ve WebP desteklenir. Her dosya 25 MB'tan küçük olmalı. Yeni isimleri `manifest.json` içindeki `images` listesinde güncelle. Tarayıcı statik bir klasörü kendiliğinden listeleyemediği için manifest gereklidir.

Siteyi yerel HTTP sunucusuyla aç. `file://` üzerinden çalıştırma. Firebase Hosting, kişisel yüz fotoğraflarını herkese açık yayımlamamak için `reference-faces/**` yolunu dağıtım dışında bırakır; canlı sitede bu fotoğrafları kullanmak için kimlik doğrulamalı özel depolama gerekir. Fotoğraflar Git tarafından yok sayılır; herkese açık depoya ekleme.

Native iOS uygulaması için fotoğrafları iPhone Fotoğraflar veya Dosyalar uygulamasından içe aktar. Uygulama her görselde tek yüz bulunduğunu kontrol eder; bunun aynı kişiyi gösterdiğini doğrulamaz.