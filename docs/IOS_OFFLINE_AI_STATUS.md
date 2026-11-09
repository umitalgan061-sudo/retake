# Retake — iOS AI Face Swap teknik durumu

Son denetim: 9 Kasım 2026  
Hedef cihaz: iPhone 17 Pro Max

## Güncel ürün kararı

İnternet kullanımı yasak değildir. Model ve yardımcı varlıklar HTTPS üzerinden indirilebilir ve güncellenebilir. Tercih edilen akış fotoğrafı cihazda işleyen Core ML modelidir. Kullanıcı açıkça onaylarsa daha sonra uygun bir uzak inference sağlayıcısı da eklenebilir.

## Native temel

Bu dal `ios/Retake.xcodeproj` adlı SwiftUI iOS uygulamasını içerir.

- PhotosPicker ile hedef fotoğrafı seçme.
- Vision ile fotoğraftaki yüzleri cihaz üzerinde algılama.
- Birden fazla yüz varsa hedef yüzü görüntü üzerinde seçme.
- En fazla 20 referans fotoğrafını içe aktarma; yalnızca tek yüz tespit edilen referanslar kabul edilir.
- Kaynak referans yüzünü seçme ve yerel depodan silme.
- Referansları Application Support/Retake/ReferenceFaces altında özel yerel depolama.
- Derlenmiş `.mlmodelc` klasörünü Files üzerinden yerel içe aktarma.
- Core ML modelini yükleyip girdi/çıktı açıklamalarını inceleme.

## Henüz tamamlanmamış kritik parçalar

Bu dal **henüz gerçek yüz değiştirme yapmaz**.

- Geçerli, lisansı uygun Face Swap model ağırlıkları yok.
- Gerçek modelin kesin girdi/çıktı sözleşmesi, yüz hizalama/embedding ve ön/son işleme adapter'ı uygulanmış değil.
- İnference çıktısının hedef fotoğrafa maske ile birleştirilmesi yapılmadı.
- Doğrulanmış, imzalı/hash kontrollü model indirme kataloğu yok.
- Fotoğraf sonucunu Photos kütüphanesine kaydetme akışı yok.
- iPhone 17 Pro Max üzerinde gerçek cihaz, kalite ve performans testi yapılmadı.

Model eksikken işlem düğmesi bilerek devre dışıdır. Uygulama sahte görüntü üretmez.

## Depo varlıkları

Önceki depoda `models/` ve `reference-faces/` altındaki kayıtlar gerçek ikili dosyalar değil, küçük API metin referanslarıydı. Bu placeholder kayıtları geliştirme dalından kaldırıldı. Kullanılabilir model ağırlıkları ve orijinal yirmi JPEG/PNG fotoğrafı şu anda depoda yok. Fotoğraflar PhotosPicker ile iPhone'a içe aktarılmalıdır.

## NVIDIA API değerlendirmesi

NVIDIA kataloğundaki Qwen-Image-Edit genel amaçlı image editing modelidir ve sayfada kimlik korumalı Face Swap görevi olarak belgelenmemiştir. Model yaklaşık 27.1B parametreli Qwen-Image-Edit mimarisidir; NVIDIA sayfası onu ticari/ticari olmayan image editing kullanımları için tarif eder. Ancak bu açıklama hedef kişiyi deterministik şekilde başka bir referans kimlikle değiştirme garantisi değildir.

Model sayfası Qwen-Image-Edit'i "Downloadable" gösterir. NVIDIA Visual GenAI NIM belgeleri indirilebilir container'ların GPU destekli sunucuda çalıştırılmasını ve NGC API anahtarının NGC'den model/container edinmede kullanılmasını anlatır. Bu, bir NVIDIA API anahtarının otomatik olarak barındırılan Face Swap endpoint'i verdiği anlamına gelmez.

Bu yüzden NVIDIA API anahtarı şu anda uygulamaya eklenmedi; Qwen-Image-Edit'e kullanıcı fotoğrafları gönderilmedi. Gerçek Face Swap endpoint'i bulunduğunda uzak sağlayıcı, fotoğraf gönderimi için ayrı onay ve Keychain anahtar saklama ile eklenebilir.

Kaynaklar:
- https://build.nvidia.com/qwen/qwen-image-edit
- https://docs.nvidia.com/nim/visual-genai/latest/overview.html
- https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html

## Model lisansı ve adaylar

Model zinciri tek bir dosyadan ibaret değildir. Face detector, landmark, identity encoder/embedding, embedding converter, swapper, segmentation, restoration ve upscaling bileşenlerinin lisansları ayrı ayrı kontrol edilmelidir.

FaceFusion'in güncel katalog metadata'sı GHOST swapper ağırlıklarını Apache-2.0 olarak etiketler; aynı GHOST işleminde kullanılan `crossface_ghost` embedding converter ayrıca gereklidir. O dönüştürücü ve kaynak embedding modelinin kullanım şartları ayrıca doğrulanmadan tüm işlem hattının lisansı uygun kabul edilemez.

InsightFace kaynak kodu MIT olsa da önceden eğitilmiş model ağırlıklarının kullanımı ayrıca kısıtlıdır. InSwapper için ticari olmayan koşullar belirtilmiştir. Bu dosyalar otomatik olarak seçilmeyecektir.

Kaynaklar:
- https://github.com/ai-forever/ghost
- https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py
- https://github.com/deepinsight/insightface
- https://www.insightface.ai/

## Model dağıtım şartları

Model download manager geliştirilirken:
- Yalnızca HTTPS ve izin verilen kaynak hostları.
- Sabitlenmiş sürüm ve dosya boyutu.
- SHA-256 checksum.
- Kaynak URL, model adı, sürüm, kod lisansı ve ağırlık lisansı için manifest.
- İndirme tamamlanmadan model dosyasını aktif etme.
- Hatalı checksum'da dosyayı silme ve hata gösterme.
- API anahtarı gerekiyorsa Keychain saklama.
- Kullanıcı fotoğrafını model indirme servisine asla ekleme.

## Tamamlanma ölçütleri

- [ ] Xcode projesi iOS Simulator için derlenir.
- [ ] iPhone 17 Pro Max üzerine kurulur.
- [ ] 20 gerçek JPEG/PNG referans görüntüsü cihazda açılır.
- [ ] Çok yüzlü fotoğrafta seçilen yüz görsel olarak işaretlenir.
- [ ] Kaynak referans yüz seçimi çalışır.
- [ ] Lisansı doğrulanmış gerçek Face Swap modeli indirilir veya içe aktarılır.
- [ ] Modelin kesin girdi/çıktı sözleşmesi ve ön/son işleme kodu uygulanır.
- [ ] Gerçek inference çıktısı hedef fotoğrafa maskeli biçimde birleştirilir.
- [ ] Core ML CPU/GPU/Neural Engine kullanım birimleri ölçülür.
- [ ] Ağ olmadan model cihazda kayıtlıysa işlem devam eder.
- [ ] Sonuç Photos kütüphanesine kaydedilir.
- [ ] Uzak inference varsa işlem öncesinde kullanıcı onayı ve Keychain anahtar saklama çalışır.
- [ ] Başarısız işlem hiçbir zaman başarı gibi gösterilmez.

## Gizlilik

Public GitHub deposuna kişisel yüz fotoğrafları, yüz embedding'leri veya lisansı kısıtlı model ağırlıkları eklenmemelidir.