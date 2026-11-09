# Retake — iOS çevrimdışı AI geçiş planı

Son denetim: 9 Kasım 2026  
Hedef cihaz: iPhone 17 Pro Max  
Hedef çalışma şekli: fotoğraflar ve model inference tamamen cihazda; ağ üzerinden fotoğraf/model işleme yok.

## Bu dalda geliştirilen native temel

Bu dal artık `ios/Retake.xcodeproj` adlı bir Xcode uygulama projesi içeriyor.

- SwiftUI uygulama kabuğu.
- PhotosPicker ile hedef fotoğrafı seçme.
- Vision ile fotoğraftaki yüzleri cihaz üzerinde algılama.
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde çerçeveleyip seçme.
- En fazla 20 referans fotoğrafını PhotosPicker üzerinden içe aktarma.
- Referansları Application Support/Retake/ReferenceFaces altında özel yerel depolama.
- Referans silme.
- Derlenmiş `.mlmodelc` klasörünü Files üzerinden yerel içe aktarma.
- Core ML modelini yükleyip girdi/çıktı açıklamalarını inceleme.

## Hâlâ tamamlanmamış kritik parçalar

Bu dal **henüz gerçek yüz değiştirme yapmaz**.

- Depoda geçerli Face Swap model ağırlıkları yok.
- 20 referans fotoğrafının GitHub girdileri gerçek JPEG/PNG dosyaları değil; küçük `/api/images/...` referans metinleri.
- Model girdileri, çıktı tensörleri, yüz hizalama ve embedding sözleşmesi doğrulanmış değil.
- `RetakeFaceSwap.mlmodelc` bulunursa yalnızca yüklenip model açıklaması okunuyor; yüz değiştirme inference çağrısı uygulanmış değil.
- Gerçek iPhone 17 Pro Max derleme, GPU/Neural Engine profilleme ve uçak modu testi henüz yapılmadı.

Model eksikken işlem düğmesi bilerek devre dışıdır. Uygulama sahte görüntü üretmez.

## Eski web prototipinin durumu

`index.html` hâlâ MediaPipe Face Landmarker ve Canvas üçgen/affine warp kullanan web prototipidir. Eğitilmiş Face Swap modeli değildir ve jsDelivr / Google Storage CDN'lerine bağımlıdır. Offline iOS uygulamasının AI motoru olarak kullanılmamalıdır.

## Mevcut depo varlıkları

GitHub ağacındaki `models/det_10g.onnx`, `models/w600k_r50.onnx` ve `models/face_landmarker.task` dosyalarının her biri 51 bayt. Blob içerikleri model ağırlığı değil, `/api/documents/...` metnidir.

`reference-faces/` altındaki 20 dosyanın her biri 36 bayt. İçerikleri JPEG/PNG değil, `/api/images/...` metnidir.

Yerel varlık kontrolü:

```bash
python3 tools/validate_assets.py
```

## Model seçimi ve lisans kapısı

Modelin adı kadar yardımcı modellerin lisansları da önemlidir.

- InsightFace, kaynak kodu için MIT lisansı belirtirken önceden eğitilmiş modelleri non-commercial academic research olarak sınırlıyor; InSwapper serisi için ayrıca lisanslama yönlendirmesi yayımlıyor.
- FaceFusion, kendi uygulama lisansı ile model varlıklarının lisanslarının ayrı olduğunu açıkça belirtiyor. Model kataloğunda InSwapper non-commercial, GHOST face-swap ağırlıkları Apache-2.0, fakat bazı GHOST yapılandırmalarında kullanılan CrossFace embedding dönüştürücüsü ayrı lisansa sahip.
- Bu nedenle yalnızca face-swap ağırlığının lisansına bakmak yeterli değildir. Yüz algılama, landmark, tanıma/embedding, embedding dönüştürücü, restorasyon ve segmentasyon bileşenlerinin tamamı incelenmelidir.

Kaynaklar:
- https://github.com/deepinsight/insightface
- https://www.insightface.ai/
- https://docs.facefusion.io/introduction/licenses
- https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py
- https://github.com/ai-forever/ghost

Apple'ın güncel Core ML Tools rehberi öncelikle PyTorch/TensorFlow dönüşüm akışını anlatır; eski ONNX-Core ML dönüştürücüsü artık tercih edilen yol değildir. Bu yüzden ONNX modelinin Core ML'e doğrudan sorunsuz çevrileceği varsayılmamalı; gerçek model grafiği macOS'ta çevrilip test edilmelidir.

- https://github.com/apple/coremltools
- https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html

## Xcode ile açma

1. macOS üzerinde Xcode 16 veya üstünü açın.
2. `ios/Retake.xcodeproj` projesini açın.
3. Signing & Capabilities bölümünde kendi Apple Team hesabınızı seçin.
4. iPhone 17 Pro Max cihazını hedefleyin.
5. Uygulamayı derleyip çalıştırın.
6. Fotoğraf seçimini ve 20 referans fotoğrafı yerel içe aktarmayı test edin.
7. Uçak modunda tekrar açıp yerel fotoğraf seçimi/yüz algılamayı doğrulayın.

Model paketi içe aktarmak için Dosyalar uygulamasından derlenmiş `RetakeFaceSwap.mlmodelc` klasörü seçilebilir. Modelin yalnızca yüklenebilmesi, onun yüz değiştirme modeli olduğunu veya Neural Engine üzerinde çalıştığını kanıtlamaz.

## Tamamlanma ölçütleri

- [ ] Xcode projesi iOS Simulator için derlenir.
- [ ] iPhone 17 Pro Max üzerine kurulur.
- [ ] Gerçek 20 JPEG/PNG referans görüntüsü cihazda açılır.
- [ ] Çok yüzlü fotoğrafta seçilen yüz görsel olarak işaretlenir.
- [ ] Lisansı doğrulanmış gerçek Face Swap modeli Core ML'e dönüştürülür.
- [ ] Modelin kesin girdi/çıktı sözleşmesi ve ön/son işleme kodu uygulanır.
- [ ] Gerçek inference sonucunun hedef fotoğrafa maskeli biçimde birleştirilmesi yapılır.
- [ ] Modelin gerçek Core ML çalışma birimleri Instruments / Core ML araçlarıyla profillenir.
- [ ] Uçak modunda uygulamayı yeniden başlattıktan sonra fotoğraf işleme tamamlanır.
- [ ] Çıktı Fotoğraflar arşivine kaydedilir.
- [ ] Sonuç olmayan durumlar başarılı işlem gibi gösterilmez.

## Güvenlik ve gizlilik

Bu GitHub deposu herkese açıktır. Kişisel yüz fotoğraflarını, yüz embedding'lerini veya lisans kısıtlı model ağırlıklarını public depoya commit etmeyin. Uygulama bu verileri yerel uygulama klasöründe tutmak üzere tasarlanmıştır.
