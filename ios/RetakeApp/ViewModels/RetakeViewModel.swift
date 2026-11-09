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
    private let modelInspector = LocalModelInspector()
    private let maximumImageBytes = 100 * 1024 * 1024

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
                      !data.isEmpty,
                      data.count <= maximumImageBytes else {
                    failed += 1
                    continue
                }

                let ext = item.supportedContentTypes.first?.preferredFilenameExtension ?? "jpg"
                if await saveValidatedReference(data: data, fileExtension: ext) {
                    imported += 1
                } else {
                    failed += 1
                }
            } catch {
                failed += 1
            }
        }

        await finishReferenceImport(imported: imported, failed: failed)
    }

    func importReferenceFiles(_ urls: [URL]) async {
        guard !urls.isEmpty else { return }
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
        for url in urls.prefix(remaining) {
            do {
                let (data, fileExtension) = try await readReferenceFile(at: url)
                if await saveValidatedReference(data: data, fileExtension: fileExtension) {
                    imported += 1
                } else {
                    failed += 1
                }
            } catch {
                failed += 1
            }
        }

        await finishReferenceImport(imported: imported, failed: failed)
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
                  !data.isEmpty,
                  data.count <= maximumImageBytes,
                  let image = UIImage(data: data) else {
                throw TargetImageError.invalidImage
            }

            let faces = try await Task.detached(priority: .userInitiated) {
                try FaceDetectionService().detectFaces(in: data)
            }.value

            // Replace the current target only after the new image and detection both succeed.
            targetImage = image
            detectedFaces = faces
            selectedFaceIndex = detectedFaces.isEmpty ? nil : detectedFaces[0].index

            if detectedFaces.isEmpty {
                statusMessage = "Fotoğrafta yüz bulunamadı. Başka bir fotoğraf seçin."
            } else if detectedFaces.count == 1 {
                statusMessage = "Bir yüz tespit edildi. Referans yüzünüzü seçebilirsiniz."
            } else {
                statusMessage = "\(detectedFaces.count) yüz tespit edildi. Değiştirilecek yüzü seçin."
            }
        } catch {
            statusMessage = "Fotoğraf işlenemedi: \(error.localizedDescription). Önceki fotoğraf korunuyor."
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

    private func readReferenceFile(at url: URL) async throws -> (Data, String) {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard fileSize > 0, fileSize <= maximumImageBytes else {
            throw ReferenceImportError.invalidImageSize
        }

        let data = try await Task.detached(priority: .userInitiated) {
            try Data(contentsOf: url, options: .mappedIfSafe)
        }.value
        guard data.count <= maximumImageBytes else {
            throw ReferenceImportError.invalidImageSize
        }

        return (data, url.pathExtension)
    }

    private func saveValidatedReference(data: Data, fileExtension: String) async -> Bool {
        guard !data.isEmpty, data.count <= maximumImageBytes else { return false }

        do {
            let faceCount = try await Task.detached(priority: .userInitiated) {
                try FaceDetectionService().detectFaces(in: data).count
            }.value
            guard faceCount == 1 else { return false }
            _ = try await assetStore.saveReference(data: data, fileExtension: fileExtension)
            return true
        } catch {
            return false
        }
    }

    private func finishReferenceImport(imported: Int, failed: Int) async {
        await refreshReferences()
        statusMessage = "Tek yüz içerdiği doğrulanan: \(imported) · Atlanan: \(failed) · Toplam: \(referenceFaces.count)/20"
    }
}

private enum TargetImageError: LocalizedError {
    case invalidImage

    var errorDescription: String? {
        "Seçilen görsel okunamadı veya 100 MB boyut sınırını aşıyor."
    }
}

private enum ReferenceImportError: LocalizedError {
    case invalidImageSize

    var errorDescription: String? {
        "Referans görsel boş veya 100 MB sınırını aşıyor."
    }
}