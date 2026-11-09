import Foundation

actor LocalAssetStore {
    private let fileManager = FileManager.default
    private let rootURL: URL

    init() {
        let support = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        rootURL = support.appendingPathComponent("Retake", isDirectory: true)
    }

    private var referencesURL: URL {
        rootURL.appendingPathComponent("ReferenceFaces", isDirectory: true)
    }

    func prepareDirectories() throws {
        try fileManager.createDirectory(at: referencesURL, withIntermediateDirectories: true)
    }

    func referenceFiles() throws -> [URL] {
        try prepareDirectories()
        let files = try fileManager.contentsOfDirectory(
            at: referencesURL,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        let allowed = Set(["jpg", "jpeg", "png", "heic", "webp"])
        return files
            .filter { allowed.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    func saveReference(data: Data, fileExtension: String) throws -> URL {
        try prepareDirectories()
        let allowed = Set(["jpg", "jpeg", "png", "heic", "webp"])
        let normalizedExtension = allowed.contains(fileExtension.lowercased()) ? fileExtension.lowercased() : "jpg"
        let destination = referencesURL
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(normalizedExtension)
        try data.write(to: destination, options: [.atomic, .completeFileProtectionUnlessOpen])
        return destination
    }

    func deleteReference(fileName: String) throws {
        try prepareDirectories()
        let candidate = referencesURL.appendingPathComponent(fileName).standardizedFileURL
        guard candidate.deletingLastPathComponent() == referencesURL.standardizedFileURL else {
            throw LocalAssetStoreError.invalidFilePath
        }
        if fileManager.fileExists(atPath: candidate.path) {
            try fileManager.removeItem(at: candidate)
        }
    }
}

enum LocalAssetStoreError: LocalizedError {
    case invalidFilePath

    var errorDescription: String? {
        "Geçersiz yerel dosya yolu."
    }
}
