import Foundation

/// Describes one self-contained Core ML model file.
/// The manifest is data, not a trust root: the publisher and its signing key
/// must be independently approved before a production manifest is distributed.
struct ModelPackageManifest: Decodable, Sendable {
    let modelID: String
    let version: String
    let artifactURL: URL
    let byteCount: Int64
    let sha256: String
    let codeLicense: String
    let weightsLicense: String
    let licenseEvidenceURL: URL
    let licenseReviewStatus: String

    func validate() throws {
        guard modelID == "RetakeFaceSwap" else {
            throw ModelManifestError.unsupportedModelID
        }

        guard !version.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelManifestError.invalidVersion
        }

        guard Self.isTrustedHTTPSURL(artifactURL),
              artifactURL.pathExtension.lowercased() == "mlmodel" else {
            throw ModelManifestError.untrustedArtifactURL
        }

        guard Self.isTrustedHTTPSURL(licenseEvidenceURL) else {
            throw ModelManifestError.untrustedLicenseEvidenceURL
        }

        guard byteCount > 0, byteCount <= Self.maximumArtifactBytes else {
            throw ModelManifestError.invalidArtifactSize
        }

        let normalizedHash = sha256.lowercased()
        guard normalizedHash.count == 64,
              normalizedHash.allSatisfy({ $0.isHexDigit }) else {
            throw ModelManifestError.invalidSHA256
        }

        guard Self.isDeclared(codeLicense),
              Self.isDeclared(weightsLicense),
              licenseReviewStatus.lowercased() == "approved" else {
            throw ModelManifestError.licenseNotApproved
        }
    }

    private static let maximumArtifactBytes: Int64 = 1_500_000_000

    private static let trustedHosts: Set<String> = [
        "github.com",
        "raw.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
        "huggingface.co",
        "cdn-lfs.huggingface.co"
    ]

    static func isTrustedHTTPSURL(_ url: URL) -> Bool {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.scheme?.lowercased() == "https",
              components.user == nil,
              components.password == nil,
              components.fragment == nil,
              components.port == nil || components.port == 443,
              let host = components.host?.lowercased() else {
            return false
        }

        return trustedHosts.contains(host)
    }

    private static func isDeclared(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !normalized.isEmpty && normalized != "unknown" && normalized != "tbd"
    }
}

enum ModelManifestError: LocalizedError {
    case unsupportedModelID
    case invalidVersion
    case untrustedArtifactURL
    case untrustedLicenseEvidenceURL
    case invalidArtifactSize
    case invalidSHA256
    case licenseNotApproved

    var errorDescription: String? {
        switch self {
        case .unsupportedModelID:
            return "Manifest, RetakeFaceSwap model kimliğiyle eşleşmiyor."
        case .invalidVersion:
            return "Model sürümü belirtilmemiş."
        case .untrustedArtifactURL:
            return "Model dosyası HTTPS ve izin verilen bir indirme kaynağından gelmelidir."
        case .untrustedLicenseEvidenceURL:
            return "Lisans kanıtı izin verilen HTTPS kaynaklarından gelmelidir."
        case .invalidArtifactSize:
            return "Model dosyasının boyutu geçersiz veya izin verilen sınırın üzerinde."
        case .invalidSHA256:
            return "Manifest içindeki SHA-256 değeri geçersiz."
        case .licenseNotApproved:
            return "Kod ve model ağırlığı lisansları ayrıca incelenip onaylanmadan model kurulamaz."
        }
    }
}
