import SwiftUI
import UIKit

struct TargetFacePreview: View {
    let image: UIImage
    let faces: [DetectedFace]
    let selectedFaceIndex: Int?
    let onSelect: (Int) -> Void

    var body: some View {
        GeometryReader { proxy in
            let imageSize = image.size
            let safeWidth = max(imageSize.width, 1)
            let safeHeight = max(imageSize.height, 1)
            let scale = min(proxy.size.width / safeWidth, proxy.size.height / safeHeight)
            let displayedWidth = safeWidth * scale
            let displayedHeight = safeHeight * scale
            let originX = (proxy.size.width - displayedWidth) / 2
            let originY = (proxy.size.height - displayedHeight) / 2

            ZStack(alignment: .topLeading) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: displayedWidth, height: displayedHeight)
                    .position(
                        x: originX + displayedWidth / 2,
                        y: originY + displayedHeight / 2
                    )

                ForEach(faces) { face in
                    let box = face.boundingBox
                    let rect = CGRect(
                        x: originX + box.minX * displayedWidth,
                        y: originY + (1 - box.maxY) * displayedHeight,
                        width: box.width * displayedWidth,
                        height: box.height * displayedHeight
                    )
                    let selected = selectedFaceIndex == face.index

                    Button {
                        onSelect(face.index)
                    } label: {
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(selected ? Color.mint : Color.white.opacity(0.85),
                                    lineWidth: selected ? 3 : 1.5)
                            .background(selected ? Color.mint.opacity(0.08) : Color.clear)
                            .frame(width: max(rect.width, 2), height: max(rect.height, 2))
                            .overlay(alignment: .topLeading) {
                                Text(face.title)
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.black)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 3)
                                    .background(selected ? Color.mint : Color.white, in: Capsule())
                                    .fixedSize()
                                    .offset(y: -19)
                            }
                    }
                    .buttonStyle(.plain)
                    .position(x: rect.midX, y: rect.midY)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .accessibilityElement(children: .contain)
    }
}
