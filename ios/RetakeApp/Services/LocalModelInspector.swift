import CoreML
import Foundation

struct ModelInspection {
    enum State {
        case missing
        case loadedButAdapterMissing
        case failed(String)
    }

    let state: State
    let location: String
    let inputs: [String]
    let outputs: [String]

    static var missing: ModelInspection {
        ModelInspection(
            state: .missing,
            location: "RetakeFaceSwap.mlmodelc",
            inputs: [],
            outputs: []
        )
    }

    var message: String {
        switch state {
        case .missing:
            return "Core ML modeli bulunamadı. Gerçek yüz değiştirme şu anda etkin değil."
        case .loadedButAdapterMissing:
            return "Core ML modeli yüklendi; ancak modelin yüz değiştirme girdi/çıktı sözleşmesi doğrulanmadığı için inference kapalı."
        case .failed(let reason):
            return "Core ML modeli yüklenemedi: \(reason)"
        }
    }
}

struct LocalModelInspector {
    func inspect() -> ModelInspection {
        let bundledURL = Bundle.main.url(forResource: "RetakeFaceSwap", withExtension: "mlmodelc")
        let supportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("Retake/Models/RetakeFaceSwap.mlmodelc", isDirectory: true)

        guard let modelURL = bundledURL ?? supportURL,
              FileManager.default.fileExists(atPath: modelURL.path) else {
            return .missing
        }

        do {
            let configuration = MLModelConfiguration()
            // .all permits Core ML to choose among CPU, GPU and Neural Engine.
            // It does not prove which compute unit actually executed each operation.
            configuration.computeUnits = .all
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            let inputs = model.modelDescription.inputDescriptionsByName
                .sorted { $0.key < $1.key }
                .map { "\($0.key): \(String(describing: $0.value.type))" }
            let outputs = model.modelDescription.outputDescriptionsByName
                .sorted { $0.key < $1.key }
                .map { "\($0.key): \(String(describing: $0.value.type))" }

            return ModelInspection(
                state: .loadedButAdapterMissing,
                location: modelURL.path,
                inputs: inputs,
                outputs: outputs
            )
        } catch {
            return ModelInspection(
                state: .failed(error.localizedDescription),
                location: modelURL.path,
                inputs: [],
                outputs: []
            )
        }
    }
}
