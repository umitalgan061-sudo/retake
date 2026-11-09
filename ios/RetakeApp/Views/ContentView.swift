import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = RetakeViewModel()
    @State private var targetPickerItem: PhotosPickerItem?
    @State private var referencePickerItems: [PhotosPickerItem] = []
    @State private var isImportingModel = false
    @State private var modelManifestURL = ""

    private let columns = [GridItem(.adaptive(minimum: 88), spacing: 12)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    modelStatusCard
                    targetPhotoCard
                    referenceFacesCard
                    privacyNote
                }
                .padding(16)
            }
            .background(Color(red: 0.055, green: 0.065, blue: 0.09))
            .navigationBarHidden(true)
            .task {
                await viewModel.startup()
            }
            .onChange(of: targetPickerItem) { _, newItem in
                Task { await viewModel.loadTarget(newItem) }
            }
            .onChange(of: referencePickerItems) { _, newItems in
                Task {
                    await viewModel.importReferences(newItems)
                    referencePickerItems = []
                }
            }
            .fileImporter(
                isPresented: $isImportingModel,
                allowedContentTypes: [.folder, .package],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        Task { await viewModel.installModelBundle(from: url) }
                    }
                case .failure(let error):
                    print("Model içe aktarma hatası: \(error.localizedDescription)")
                }
            }
        }
        .tint(Color.mint)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("RETAKE")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .tracking(2.4)
                    .foregroundStyle(.mint)
                Spacer()
                Image(systemName: "iphone.gen3")
                    .foregroundStyle(.secondary)
            }
            Text("Yerel AI Face Swap")
                .font(.system(size: 28, weight: .bold, design: .rounded))
            Text("iPhone üzerinde özel ve çevrimdışı fotoğraf işleme")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 8)
    }

    private var modelStatusCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Yapay zekâ motoru", systemImage: "cpu")
                .font(.headline)
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: modelStatusSymbol)
                    .font(.title2)
                    .foregroundStyle(modelStatusColor)
                VStack(alignment: .leading, spacing: 4) {
                    Text(viewModel.modelInspection.message)
                        .font(.subheadline.weight(.medium))
                    Text("Çalışma modu: Core ML · CPU/GPU/Neural Engine seçimi Core ML'e bırakılır.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if !viewModel.modelInspection.inputs.isEmpty {
                DisclosureGroup("Model girdi / çıktı bilgileri") {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Girdiler").font(.caption.bold())
                        ForEach(viewModel.modelInspection.inputs, id: \.self) {
                            Text($0).font(.caption2).foregroundStyle(.secondary)
                        }
                        Text("Çıktılar").font(.caption.bold()).padding(.top, 4)
                        ForEach(viewModel.modelInspection.outputs, id: \.self) {
                            Text($0).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 8)
                }
                .font(.caption)
            }
            HStack {
                Button {
                    isImportingModel = true
                } label: {
                    Label("Model klasörü içe aktar", systemImage: "folder.badge.plus")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy)

                Button {
                    viewModel.refreshModelInspection()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isBusy)
                .accessibilityLabel("Modeli yeniden denetle")
            }
            Text("Dosyalar uygulamasından derlenmiş RetakeFaceSwap.mlmodelc klasörünü seçin.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            Divider().overlay(Color.white.opacity(0.1))

            TextField("HTTPS model manifest URL'si", text: $modelManifestURL)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.URL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            Button {
                Task { await viewModel.installModelManifest(from: modelManifestURL) }
            } label: {
                Label("Manifestten modeli indir ve kur", systemImage: "arrow.down.circle")
                    .font(.caption.weight(.semibold))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(
                viewModel.isBusy ||
                modelManifestURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )

            Text("Yalnızca izin verilen HTTPS kaynaklarından gelen, lisans incelemesi onaylı ve SHA-256 bilgisi bulunan manifestler kabul edilir. Henüz hazır bir model kataloğu URL'si tanımlı değildir.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color.white.opacity(0.08)))
    }

    private var targetPhotoCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionHeading("1", "Düzenlenecek fotoğraf")
            if let image = viewModel.targetImage {
                TargetFacePreview(
                    image: image,
                    faces: viewModel.detectedFaces,
                    selectedFaceIndex: viewModel.selectedFaceIndex
                ) { index in
                    viewModel.selectedFaceIndex = index
                }
                .frame(height: 320)
                .clipShape(RoundedRectangle(cornerRadius: 14))

                if viewModel.detectedFaces.isEmpty {
                    Text("Yüz tespit edilmedi.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else {
                    Text("\(viewModel.detectedFaces.count) yüz tespit edildi · Hedef yüzü seçin")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(viewModel.detectedFaces) { face in
                                Button {
                                    viewModel.selectedFaceIndex = face.index
                                } label: {
                                    Text(face.title)
                                        .font(.caption.weight(.semibold))
                                        .padding(.horizontal, 13)
                                        .padding(.vertical, 8)
                                        .background(
                                            viewModel.selectedFaceIndex == face.index
                                            ? Color.mint.opacity(0.22)
                                            : Color.white.opacity(0.07),
                                            in: Capsule()
                                        )
                                        .overlay(Capsule().stroke(
                                            viewModel.selectedFaceIndex == face.index ? Color.mint : Color.clear
                                        ))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                HStack {
                    PhotosPicker(selection: $targetPickerItem, matching: .images) {
                        Label("Başka fotoğraf seç", systemImage: "photo")
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Kaldır", role: .destructive) {
                        viewModel.clearTarget()
                        targetPickerItem = nil
                    }
                    .buttonStyle(.bordered)
                }
            } else {
                PhotosPicker(selection: $targetPickerItem, matching: .images) {
                    VStack(spacing: 10) {
                        Image(systemName: "arrow.up.doc")
                            .font(.system(size: 30))
                        Text("Fotoğraf galerinden seç")
                            .font(.headline)
                        Text("HEIC, JPEG ve PNG görselleri")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 34)
                    .background(Color.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 15))
                    .overlay(RoundedRectangle(cornerRadius: 15).stroke(
                        Color.white.opacity(0.13),
                        style: StrokeStyle(lineWidth: 1, dash: [6, 4])
                    ))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(16)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 18))
    }

    private var referenceFacesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center) {
                sectionHeading("2", "Referans yüzler")
                Spacer()
                Text("\(viewModel.referenceFaces.count)/20")
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundStyle(.mint)
            }

            if viewModel.referenceFaces.isEmpty {
                Text("Depodaki referans kayıtları gerçek fotoğraf dosyaları değil. Orijinal 20 fotoğrafı buradan cihazına içe aktar.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(viewModel.referenceFaces) { face in
                        ReferenceFaceTile(
                            face: face,
                            isSelected: viewModel.selectedReferenceID == face.id,
                            onSelect: { viewModel.selectReference(face) },
                            onDelete: { Task { await viewModel.removeReference(face) } }
                        )
                    }
                }
            }

            PhotosPicker(selection: $referencePickerItems, maxSelectionCount: 20, matching: .images) {
                Label("Fotoğraflardan referans ekle", systemImage: "person.crop.square.badge.plus")
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(viewModel.isBusy || viewModel.referenceFaces.count >= 20)

            Text("Referanslar yalnızca bu uygulamanın özel yerel klasöründe saklanır.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 18))
    }

    private var privacyNote: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Gizlilik ve işlem durumu", systemImage: "lock.shield")
                .font(.headline)
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("Bu geliştirme sürümü fotoğraf seçimi ve yerel yüz tespitini yapar. Geçerli, lisansı uygun Face Swap modeli ve model adaptörü bulunmadığı için henüz yeni yüz üretmez. Sahte sonuç gösterilmez.")
                .font(.caption)
                .foregroundStyle(.orange)
            Button {
                // Deliberately no fake processing action.
            } label: {
                Label("Yüz Değiştir", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(true)
            .accessibilityHint("Gerçek, lisansı uygun Face Swap modeli kurulana kadar devre dışıdır.")
        }
        .padding(16)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 18))
    }

    private func sectionHeading(_ number: String, _ title: String) -> some View {
        HStack(spacing: 9) {
            Text(number)
                .font(.caption.bold())
                .foregroundStyle(Color.black)
                .frame(width: 24, height: 24)
                .background(Color.mint, in: Circle())
            Text(title).font(.headline)
        }
    }

    private var modelStatusSymbol: String {
        switch viewModel.modelInspection.state {
        case .missing: return "exclamationmark.triangle.fill"
        case .loadedButAdapterMissing: return "cpu"
        case .failed: return "xmark.octagon.fill"
        }
    }

    private var modelStatusColor: Color {
        switch viewModel.modelInspection.state {
        case .missing, .failed: return .orange
        case .loadedButAdapterMissing: return .yellow
        }
    }
}

private struct ReferenceFaceTile: View {
    let face: ReferenceFace
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(spacing: 5) {
            ZStack(alignment: .topTrailing) {
                Button(action: onSelect) {
                    Group {
                        if let image = UIImage(contentsOfFile: face.url.path) {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFill()
                        } else {
                            RoundedRectangle(cornerRadius: 11)
                                .fill(Color.white.opacity(0.08))
                                .overlay(Image(systemName: "person.crop.square").foregroundStyle(.secondary))
                        }
                    }
                    .frame(height: 88)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 11))
                    .overlay(
                        RoundedRectangle(cornerRadius: 11)
                            .stroke(isSelected ? Color.mint : Color.clear, lineWidth: 3)
                    )
                }
                .buttonStyle(.plain)

                Button(action: onDelete) {
                    Image(systemName: "xmark.circle.fill")
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.7))
                }
                .buttonStyle(.plain)
                .padding(5)
            }
            Text(face.fileName)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            if isSelected {
                Text("Kaynak yüz")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.mint)
            }
        }
    }
}
