# Retake — fotogerçekçilik ve kimlik koruma kalite denetimi

## Kapsam

Retake FaceFusion 3.9.2 yerel CUDA motoru, 25 doğrulanmış referans
üzerinden AlphaFace/UniFace ve mümkünse isteğe bağlı difüzyon
ile yalnız seçili yüz bölgesini değiştirir. Hedef fotoğrafın gövdesi,
kıyafeti, saçları ve arka planı korunur.

Kullanıcı testinde 2026-11-23 tarihli 0.838 ArcFace benzerliği
(uygulama arayüzünde yaklaşık 83,8) bildirildi. Bu puan ArcFace
cosine ölçümüdür; **yüzde 83,8 gerçeğe uygunluk** veya kesin
kişilik/kimlik kanıtı değildir.

İncelenen örnekte asıl görsel kalite açığı, saç çizgisi/çene boyunca
ince koyu kontur ile yüz derisinin çevredeki fotoğrafa kıyasla farklı
doku ve ışık karakteri taşımasıdır. v20/v21 çıktıları görünüşte çok
benzerdir; v22 onarımının canlı gerçek CUDA çıktısındaki etkisi henüz
ölçülmemiştir.

## Yeni kalite kapısı

1. Her aday mevcut kontur onarımını kullanır; onarım ArcFace ile
   doğrulanmazsa uygulanmaz.
2. Sonuç dosyasındaki **kalan** koyu kontur miktarı ayrıca incelenir.
   Yeni `X-Retake-Rim-Fraction` başlığı 0..1 arasında, yalnız
   doğrulanabilir görüntü için bir *diagnostik* değer döndürür.
   Yüksek değer, orijinal hedefe göre koyulaşmış ince kenar izi
   ihtimalini anlatır. Bu **genel bir fotoğraf gerçekçilik puanı değildir**.
3. Ultra araması, ArcFace skorları arasında anlamlı (0.002'den
   fazla cosine) fark varsa eskisi gibi kimliği öncelemeye devam eder.
   Yakın eşitlikte ve iki adayda da geçerli metrik varken en az
   0.025 daha az kenar kararması gösteren sonucu tercih edebilir.
4. Kenar metrikleri yakınsa mevcut bölgesel göz/burun/ağız/alt yüz
   mikrodetay karşılaştırması devrededir.
5. Ölçülemeyen halo *sıfırmış gibi* değerlendirilmez.

## Yerel gerçek görüntülerle A/B testi

Özel fotoğrafları GitHub'a veya buluta yüklemeden kendi Windows
bilgisayarındaki Retake klasöründe, OpenCV ve NumPy kurulu ortamda
şu komutu çalıştır:

```powershell
py -3 -m tools.audit_photo_quality "C:\\local\\input.jpg" "C:\\local\\new-output.png" --previous "C:\\local\\old-output.png" --bounds 500 600 230 280
```

Örnekteki yüz sınırı sayıları yalnız sözdizimini gösterir;
**ilgili resimde gerçekten algılanan** x/y/genişlik/yükseklik
değerlerini kullan. Resimler aynı orijinal çözünürlükte olmalıdır.
A/B için referans kişi ve hedef fotoğraf aynı kalmalıdır.

Rapor:
- `residual_dark_seam_fraction`: daha küçük değer, benzer kimlik
  korunuyorsa, genelde daha iyi bir sınır geçişine işaret eder.
- `face_detail_score`: bölgesel mikrodetay dengesi (0..1).
- `seam_fraction_change`: negatif ise yeni sonuçta kontur azaldı.
- `diagnostic_stage`: ölçümün neden yapılamadığı / seçildiği.

Çıktı **sadece JSON sayısal istatistik** üretir; dosyaları
değiştirmez, fotoğrafı kaydetmez, kimliği doğrulamaz.

## Kabul kriteri / Retake'i Hafize'ye devretme

- Mevcut `npm test`, `tools/validate_assets.py`,
  Python sözdizimi ve `unittest discover` başarıyla tamamlanmalı.
- Aynı hedef ve 25 referansta eski/yeni GPU çıktıları gerçek
  dosyalarla incelenmeli. Çene, saç çizgisi, göz çevresi ve sakal
  doğal kalmalı, referans kimlik skoru gerilememeli.
- Retake'de inference başlatabilmek için kullanıcı bilgisayarında
  yerel GPU motoru bulunmalıdır; Firebase statik site tek başına
  FaceFusion çalıştırmaz.
- Hafize sohbet ajanının GitHub App dal-dosya-taslak PR yolu
  canlıda kullanıcı onayıyla doğrulanmadan Retake allowlist'ine
  eklenmez. Yapılabilecek ilk görev: bu PR'ın diff/test raporunu
  salt-okunur inceleyip önerileri yorumlamak.
- Onaylanmamış commit'i `main` üzerine merge etme veya otomatik
  deploy etme. Bu PR kod seviyesinde kalite iyileştirmesidir,
  gerçek GPU'da daha yüksek yüz benzerliği **garanti etmez**.
