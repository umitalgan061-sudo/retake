import Foundation
import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class RetakeViewModel: ObservableObject {
    @Published private(set) var referenceFaces: [ReferenceFace] = []
    @Published private(set) var targetImage: UIImage?
    @Published private(set) var detectedFaces: [DetectedFace] = []
    @Published var selectedFaceIndex: Int?
    @Published private(set) var modelInspection: ModelInspection = .missing
    @Published private(set) var isBusy = false
    @Published var statusMessage = "Fotoğraf seçerek başlayın."

    private let assetStore = LocalAssetStore()
    private let faceDetector = FaceDetectionService()
    private let modelInspector = LocalModelInspector()

    var canAttemptFaceSwap: Bool {
        // Intentionally false until a licensed model and its verified adapter exist.
        false
    }

    func startup() async {
        await refreshReferences()
        refreshModelInspection()
    }

    func refreshReferences() async {
        do {
            let urls = try await assetStore.referenceFiles()
            referenceFaces = urls.map(ReferenceFace.init(url:))
            if referenceFaces.isEmpty {
                statusMessage = "Referans yüzlerinizi cihazdan içe aktarın."
            }
        } catch {
            statusMessage = "Yerel referans klasörü açılamadı: \(error.localizedDescription)"
        }
    }

    func importReferences(_ items: [PhotosPickerItem]) async {
        guard !items.isEmpty else { return }
        guard !isBusy else { return }

        let remaining = max(0, 20 - referenceFaces.count)
        guard remaining > 0 else {
            statusMessage = "20 referans yüz sınırına ulaşıldı."
            return
        }

        isBusy = true
        defer { isBusy = false }

        var imported = 0
        var failed = 0
        for item in items.prefix(remaining) {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      UIImage(data: data) != nil else {
                    failed += 1
                    continue
                }
                let ext = item.supportedContentTypes.first?.preferredFilenameExtension ?? "jpg"
                _ = try await assetStore.saveReference(data: data, fileExtension: ext)
                imported += 1
            } catch {
                failed += 1
            }
        }

        await refreshReferences()
        statusMessage = "İçe aktarılan: \(imported) · Atlanan: \(failed) · Toplam: \(referenceFaces.count)/20"
    }

    func removeReference(_ face: ReferenceFace) async {
        do {
            try await assetStore.deleteReference(fileName: face.fileName)
            await refreshReferences()
        } catch {
            statusMessage = "Referans silinemedi: \(error.localizedDescription)"
        }
    }

    func installModelBundle(from url: URL) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }

        do {
            try await assetStore.installCompiledModel(from: url)
            refreshModelInspection()
            statusMessage = "Model paketi kopyalandı. Dosyanın yüklenmesi ve model sözleşmesi ayrıca doğrulanmalıdır."
        } catch {
            statusMessage = "Model içe aktarılamadı: \(error.localizedDescription)"
        }
    }

    func loadTarget(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        guard !isBusy else { return }

        isBusy = true
        defer { isBusy = false }

        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                throw TargetImageError.invalidImage
            }

            targetImage = image
            detectedFaces = try faceDetector.detectFaces(in: image)
            selectedFaceIndex = detectedFaces.isEmpty ? nil : detectedFaces[0].index

            if detectedFaces.isEmpty {
                statusMessage = "Fotoğrafta yüz bulunamadı. Başka bir fotoğraf seçin."
            } else if detectedFaces.count == 1 {
                statusMessage = "Bir yüz tespit edildi. Referans yüzünüzü seçebilirsiniz."
            } else {
                statusMessage = "\(detectedFaces.count) yüz tespit edildi. Değiştirilecek yüzü seçin."
            }
        } catch {
            targetImage = nil
            detectedFaces = []
            selectedFaceIndex = nil
            statusMessage = "Fotoğraf işlenemedi: \(error.localizedDescription)"
        }
    }

    func clearTarget() {
        targetImage = nil
        detectedFaces = []
        selectedFaceIndex = nil
        statusMessage = "Yeni bir fotoğraf seçebilirsiniz."
    }

    func refreshModelInspection() {
        modelInspection = modelInspector.inspect()
    }
}

private enum TargetImageError: LocalizedError {
    case invalidImage

    var errorDescription: String? {
        "Seçilen dosya geçerli bir görüntü olarak açılamadı."
    }
}
