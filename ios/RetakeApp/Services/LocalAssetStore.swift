import Foundation

actor LocalAssetStore {
    private let fileManager = FileManager.default
    private let rootURL: URL

    init() {
        let supportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        rootURL = supportURL.appendingPathComponent("Retake", isDirectory: true)
    }

    private var referencesURL: URL {
        rootURL.appendingPathComponent("ReferenceFaces", isDirectory: true)
    }

    private var modelsURL: URL {
        rootURL.appendingPathComponent("Models", isDirectory: true)
    }

    func prepareDirectories() throws {
        try fileManager.createDirectory(at: referencesURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: modelsURL, withIntermediateDirectories: true)
        excludeFromBackup(rootURL)
        excludeFromBackup(referencesURL)
        excludeFromBackup(modelsURL)
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
        excludeFromBackup(destination)
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

    /// Installs a compiled model with a staging directory and rollback.
    /// The currently installed model remains intact unless the new copy succeeds.
    func installCompiledModel(from sourceURL: URL) throws {
        try prepareDirectories()
        guard sourceURL.pathExtension.lowercased() == "mlmodelc" else {
            throw LocalAssetStoreError.invalidModelBundle
        }

        let didAccess = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw LocalAssetStoreError.invalidModelBundle
        }

        let destination = modelsURL.appendingPathComponent("RetakeFaceSwap.mlmodelc", isDirectory: true)
        let staging = modelsURL.appendingPathComponent(".RetakeFaceSwap-\(UUID().uuidString).mlmodelc", isDirectory: true)
        let backup = modelsURL.appendingPathComponent(".RetakeFaceSwap-backup-\(UUID().uuidString).mlmodelc", isDirectory: true)

        do {
            try fileManager.copyItem(at: sourceURL, to: staging)
            excludeFromBackup(staging)

            let hadPreviousModel = fileManager.fileExists(atPath: destination.path)
            if hadPreviousModel {
                try fileManager.moveItem(at: destination, to: backup)
            }

            do {
                try fileManager.moveItem(at: staging, to: destination)
                excludeFromBackup(destination)
                if hadPreviousModel {
                    try? fileManager.removeItem(at: backup)
                }
            } catch {
                if hadPreviousModel,
                   fileManager.fileExists(atPath: backup.path),
                   !fileManager.fileExists(atPath: destination.path) {
                    try? fileManager.moveItem(at: backup, to: destination)
                }
                throw error
            }
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    private func excludeFromBackup(_ url: URL) {
        var mutableURL = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutableURL.setResourceValues(values)
    }
}

enum LocalAssetStoreError: LocalizedError {
    case invalidFilePath
    case invalidModelBundle

    var errorDescription: String? {
        switch self {
        case .invalidFilePath:
            return "Geçersiz yerel dosya yolu."
        case .invalidModelBundle:
            return "Lütfen geçerli, derlenmiş bir .mlmodelc klasörü seçin."
        }
    }
}