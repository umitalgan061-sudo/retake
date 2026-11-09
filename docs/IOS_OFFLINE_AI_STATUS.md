# Retake — iOS çevrimdışı AI geçiş planı

Son denetim: 9 Kasım 2026  
Hedef cihaz: iPhone 17 Pro Max  
Hedef çalışma şekli: fotoğraflar ve model inference tamamen cihazda; ağ üzerinden fotoğraf/model işleme yok.

## Mevcut deponun doğrulanmış durumu

Bu depo henüz çevrimdışı iOS AI Face Swap uygulaması değildir.

- Depoda Xcode projesi, Swift kaynak kodu veya Swift Package bulunmuyor.
- `index.html`, MediaPipe kitaplığını jsDelivr CDN üzerinden ve `face_landmarker.task` modelini Google Storage üzerinden yüklüyor. Dolayısıyla mevcut web sürümü çevrimdışı değildir.
- Mevcut yüz değiştirme yaklaşımı eğitilmiş bir Face Swap modeli değildir. MediaPipe landmark tespiti ve Canvas üzerinde üçgen/affine warp kullanır.
- `models/det_10g.onnx`, `models/w600k_r50.onnx` ve `models/face_landmarker.task` depoda model ağırlığı olarak doğrulanamadı. GitHub ağacında her biri 51 bayt görünüyor; blob içeriği model değil, `/api/documents/...` biçiminde bir referans metni.
- `reference-faces/` altındaki 20 dosya gerçek JPEG/PNG içeriği olarak doğrulanamadı. GitHub ağacında her biri 36 bayt; blob içerikleri `/api/images/...` biçiminde referans metinleri.
- Bu nedenle mevcut depodan model inference veya 20 referansla çalışan bir işlem üretmek mümkün değil. Önce gerçek ikili dosyalar edinilmeli ve doğrulanmalı.

Yerel kontrolü çalıştır:

```bash
python3 tools/validate_assets.py
```

JSON raporu:

```bash
python3 tools/validate_assets.py --json
```

## Uygulama mimarisi

1. **iOS kabuğu:** SwiftUI + PhotosUI. Fotoğraf seçimi ve galeriye kaydetme native iOS API'leriyle yapılır.
2. **Yerel görüntü analizi:** Vision yüz tespiti/landmark; hedef fotoğrafta birden fazla yüz varsa kullanıcının hedefi seçmesi.
3. **Yerel inference:** lisansı ve iOS uyumluluğu doğrulanmış, Core ML'e çevrilmiş gerçek Face Swap modeli. Model girdisi/çıktısı belirlenmeden sahte bir adapter veya uydurma tensor şekli yazılmamalı.
4. **İşleme sonrası:** yüz maskesi, renk/ışık uyumu ve kontrollü restorasyon; arka plan ve hedef fotoğrafın diğer bölgeleri korunur.
5. **Gizlilik:** özel referans fotoğraflar ve model dosyaları herkese açık Git deposuna yüklenmez. Uygulama paketine yerel Xcode build aşamasında eklenir veya cihazda kullanıcı tarafından içe aktarılır.
6. **Ağ doğrulaması:** ilk kurulumdan sonra uçak modunda fotoğraf seçimi, model yükleme, inference ve dışa aktarma uçtan uca test edilir.

## Model lisansı kapısı

Model kodunun lisansı ile önceden eğitilmiş ağırlıkların lisansı ayrı ayrı incelenmelidir. InsightFace, kodunun MIT olduğunu ancak önceden eğitilmiş modellerin araştırma amaçlı ticari olmayan kullanım koşullarına tabi olduğunu belirtiyor; InSwapper modelleri için ayrıca lisanslama/iletişim yönlendirmesi bulunuyor. FaceFusion da kendi dağıttığı her model varlığının ayrı lisansını listeliyor.

- https://github.com/deepinsight/insightface
- https://www.insightface.ai/
- https://docs.facefusion.io/introduction/licenses

Açık kaynak kod görmek, model ağırlıklarını kişisel uygulamaya kopyalama veya yeniden dağıtma izni vermez. Uygun lisans doğrulanmadan model ağırlıkları depoya eklenmemeli.

## Tamamlanma ölçütleri

Proje ancak aşağıdakiler gerçek cihaz üzerinde kanıtlandığında tamamlanmış sayılır:

- Xcode projesi başarıyla derlenir ve iPhone 17 Pro Max'e kurulur.
- 20 yerel referansın tümü gerçek görüntü dosyalarıdır ve yüz içerdikleri doğrulanır.
- Gerçek eğitilmiş yüz değiştirme modeli yerel olarak yüklenir.
- Kullanıcı fotoğrafından hedef yüz seçilir ve seçilen referansla gerçek inference yapılır.
- Modelin gerçek Core ML çalışma birimi ve süreleri ölçülür; Neural Engine kullanımı varsayılmaz, doğrulanır.
- Uçak modunda, uygulamayı yeniden başlattıktan sonra da fotoğraf işleme tamamlanır.
- Sonuç yerel Fotoğraflar arşivine kaydedilir.
- İşleme sonucu olmayan bir fotoğraf başarı gibi gösterilmez.

## Önemli sınır

Bu plan, eksik model ağırlıklarını veya kullanıcıya ait orijinal fotoğrafları uydurmaz. Bu dosyalar sağlanmadan ve doğru lisansla edinilmeden gerçek AI Face Swap inference tamamlanmış olarak sunulamaz.
