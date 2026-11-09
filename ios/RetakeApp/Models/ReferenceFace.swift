import Foundation

struct ReferenceFace: Identifiable, Hashable {
    let url: URL

    var id: String { url.lastPathComponent }
    var fileName: String { url.lastPathComponent }
}

struct DetectedFace: Identifiable {
    let index: Int
    let boundingBox: CGRect

    var id: Int { index }

    var title: String { "Yüz \(index + 1)" }
}
