import CoreML
import CryptoKit
import Foundation

struct ModelInstallResult: Sendable {
    let version: String
    let inputs: [String]
    let outputs: [String]
}

/// Downloads a single-file .mlmodel only after validating its manifest,
/// source host, exact byte count, SHA-256 and manual license-review gate.
actor ModelDownloadManager {
    private let assetStore: LocalAssetStore

    init(assetStore: LocalAssetStore = LocalAssetStore()) {
        self.assetStore = assetStore
    }

    func install(from manifestURL: URL) async throws -> ModelInstallResult {
        guard ModelPackageManifest.isTrustedHTTPSURL(manifestURL) else {
            throw ModelDownloadError.untrustedManifestURL
        }

        let (manifestData, manifestResponse) = try await URLSession.shared.data(from: manifestURL)
        guard let manifestHTTP = manifestResponse as? HTTPURLResponse,
              (200...299).contains(manifestHTTP.statusCode),
              let finalManifestURL = manifestHTTP.url,
              ModelPackageManifest.isTrustedHTTPSURL(finalManifestURL) else {
            throw ModelDownloadError.manifestRequestFailed
        }

        let manifest = try JSONDecoder().decode(ModelPackageManifest.self, from: manifestData)
        try manifest.validate()
        return try await install(manifest)
    }

    func install(_ manifest: ModelPackageManifest) async throws -> ModelInstallResult {
        try manifest.validate()

        let (temporaryModelURL, response) = try await URLSession.shared.download(from: manifest.artifactURL)
        defer { try? FileManager.default.removeItem(at: temporaryModelURL) }

        guard let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode),
              let finalURL = http.url,
              ModelPackageManifest.isTrustedHTTPSURL(finalURL),
              finalURL.pathExtension.lowercased() == "mlmodel" else {
            throw ModelDownloadError.artifactRequestFailed
        }

        let attributes = try FileManager.default.attributesOfItem(atPath: temporaryModelURL.path)
        let actualByteCount = (attributes[.size] as? NSNumber)?.int64Value ?? -1
        guard actualByteCount == manifest.byteCount else {
            throw ModelDownloadError.byteCountMismatch
        }

        let actualHash = try Self.sha256(of: temporaryModelURL)
        guard actualHash.caseInsensitiveCompare(manifest.sha256) == .orderedSame else {
            throw ModelDownloadError.checksumMismatch
        }

        let compiledURL = try await MLModel.compileModel(at: temporaryModelURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)

        let inputs = model.modelDescription.inputDescriptionsByName
            .sorted { $0.key < $1.key }
            .map { "\($0.key): \(String(describing: $0.value.type))" }
        let outputs = model.modelDescription.outputDescriptionsByName
            .sorted { $0.key < $1.key }
            .map { "\($0.key): \(String(describing: $0.value.type))" }

        guard !inputs.isEmpty, !outputs.isEmpty else {
            throw ModelDownloadError.invalidCoreMLModel
        }

        try await assetStore.installCompiledModel(from: compiledURL)

        return ModelInstallResult(
            version: manifest.version,
            inputs: inputs,
            outputs: outputs
        )
    }

    private static func sha256(of fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            guard let block = try handle.read(upToCount: 1_048_576), !block.isEmpty else {
                break
            }
            hasher.update(data: block)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

enum ModelDownloadError: LocalizedError {
    case untrustedManifestURL
    case manifestRequestFailed
    case artifactRequestFailed
    case byteCountMismatch
    case checksumMismatch
    case invalidCoreMLModel

    var errorDescription: String? {
        switch self {
        case .untrustedManifestURL:
            return "Manifest yalnızca izin verilen HTTPS kaynaklarından indirilebilir."
        case .manifestRequestFailed:
            return "Model manifesti indirilemedi veya güvenilir kaynağa yönlendirilmedi."
        case .artifactRequestFailed:
            return "Model dosyası indirilemedi veya izin verilmeyen bir kaynağa yönlendirildi."
        case .byteCountMismatch:
            return "İndirilen modelin dosya boyutu manifest ile eşleşmiyor."
        case .checksumMismatch:
            return "Model SHA-256 kontrolünden geçemedi; dosya kurulmadı."
        case .invalidCoreMLModel:
            return "Dosya derlendi ancak geçerli Core ML giriş/çıkış tanımları bulunamadı."
        }
    }
}
