import Foundation
import ImageIO
import UIKit
import Vision

struct FaceDetectionService {
    func detectFaces(in image: UIImage) throws -> [DetectedFace] {
        guard let cgImage = image.cgImage else {
            throw FaceDetectionError.imageDecodeFailed
        }

        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: CGImagePropertyOrientation(image.imageOrientation),
            options: [:]
        )
        try handler.perform([request])

        let observations = request.results ?? []
        return observations.enumerated().map { index, observation in
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

private extension CGImagePropertyOrientation {
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
