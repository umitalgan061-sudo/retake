import Foundation
import ImageIO
import Vision

struct FaceDetectionService {
    /// Decodes a bounded ImageIO thumbnail and performs face detection on that image.
    /// The thumbnail keeps EXIF orientation while avoiding a full-resolution raster decode.
    func detectFaces(in imageData: Data, maximumPixelSize: Int = 2048) throws -> [DetectedFace] {
        guard maximumPixelSize >= 256,
              let source = CGImageSourceCreateWithData(imageData as CFData, nil) else {
            throw FaceDetectionError.imageDecodeFailed
        }

        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true
        ]

        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            options as CFDictionary
        ) else {
            throw FaceDetectionError.imageDecodeFailed
        }

        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: thumbnail, orientation: .up, options: [:])
        try handler.perform([request])

        return (request.results ?? []).enumerated().map { index, observation in
            DetectedFace(index: index, boundingBox: observation.boundingBox)
        }
    }
}

enum FaceDetectionError: LocalizedError {
    case imageDecodeFailed

    var errorDescription: String? {
        "Fotoğraf okunamadı veya yüz tespiti için uygun bir görüntü oluşturulamadı."
    }
}