import Foundation
import Security

/// Stores provider credentials in this device's Keychain.
/// Do not mirror secrets into UserDefaults, Info.plist, .env files inside the app,
/// analytics, logs or source control.
struct KeychainStore {
    private let service: String

    init(service: String = Bundle.main.bundleIdentifier ?? "com.umitalgan.retake") {
        self.service = service
    }

    func save(_ secret: String, account: String) throws {
        guard !account.isEmpty, !secret.isEmpty else {
            throw KeychainStoreError.invalidInput
        }

        let key = baseQuery(account: account)
        let attributes: [String: Any] = [
            kSecValueData as String: Data(secret.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        let updateStatus = SecItemUpdate(key as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            throw KeychainStoreError.operationFailed(updateStatus)
        }

        var addQuery = key
        attributes.forEach { addQuery[$0.key] = $0.value }
        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw KeychainStoreError.operationFailed(addStatus)
        }
    }

    func read(account: String) throws -> String? {
        guard !account.isEmpty else {
            throw KeychainStoreError.invalidInput
        }

        var query = baseQuery(account: account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            throw KeychainStoreError.operationFailed(status)
        }
        return value
    }

    func delete(account: String) throws {
        guard !account.isEmpty else {
            throw KeychainStoreError.invalidInput
        }

        let status = SecItemDelete(baseQuery(account: account) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainStoreError.operationFailed(status)
        }
    }

    private func baseQuery(account: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
    }
}

enum KeychainStoreError: LocalizedError {
    case invalidInput
    case operationFailed(OSStatus)

    var errorDescription: String? {
        switch self {
        case .invalidInput:
            return "Keychain kaydı için hesap adı ve gizli anahtar boş olamaz."
        case .operationFailed(let status):
            return "Keychain işlemi başarısız oldu (OSStatus: \(status))."
        }
    }
}
