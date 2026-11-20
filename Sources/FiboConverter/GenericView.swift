import FiboCore
import SwiftUI

@MainActor @Observable
final class GenericModel {
  var input: URL?
  var name = ""
  var needsPrep = false
  var convertTextEncoders = false
  let run = RunController()

  func setInput(_ url: URL) {
    input = url
    if name.isEmpty || name == suggestedName { name = Self.suggest(url) }
    suggestedName = Self.suggest(url)
  }
  private var suggestedName = ""
  static func suggest(_ url: URL) -> String {
    url.deletingPathExtension().lastPathComponent
      .replacingOccurrences(of: "-DT", with: "")
  }

  var settings: Settings { Settings.shared }
  var inputSize: Int64 {
    guard let input, let size = try? input.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return 0 }
    return Int64(size)
  }
  /// The prepared copy, the ckpt and some margin.
  var neededBytes: Int64 { inputSize * (needsPrep ? 3 : 2) + 2_000_000_000 }
  var canStart: Bool {
    input != nil && !name.trimmingCharacters(in: .whitespaces).isEmpty && settings.modelsFolder != nil
      && Settings.helpersFolder != nil && !run.isRunning && (!needsPrep || settings.prepScript != nil)
      && (settings.freeBytes.map { $0 > neededBytes } ?? true)
  }

  func start() {
    guard canStart, let input, let models = settings.modelsFolder, let helpers = Settings.helpersFolder
    else { return }
    let work = models.appendingPathComponent("_generic-work")
    let job = GenericJob(
      input: input, name: name.trimmingCharacters(in: .whitespaces), models: models, work: work,
      helpers: helpers, prepScript: needsPrep ? settings.prepScript : nil,
      convertTextEncoders: convertTextEncoders)
    run.start(
      reason: "Converting a model", work: work, steps: GenericConversion.steps(job),
      success: "Done. Refresh the models list in your app to see \(job.name).")
  }
}

struct GenericView: View {
  @State private var model = GenericModel()
  var settings = Settings.shared

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            Text(
              "Converts any safetensors checkpoint Draw Things' engine can import into a ckpt and adds it to your models folder's custom.json. The model family is detected by the engine from the file."
            )
            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            PathRow(
              title: "Safetensors file", url: model.input, choose: {
                chooseFile(model.input, extensions: ["safetensors"]) { model.setInput($0) }
              })
            HStack {
              Text("Name in the library")
              TextField("Model name", text: $model.name).textFieldStyle(.roundedBorder)
            }
            Text("Use only models whose licence lets you convert them for your own use.")
              .font(.caption).foregroundStyle(.secondary)
          }.padding(4)
        } label: { Label("1. Model", systemImage: "doc") }

        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            Toggle(
              "The file is a ComfyUI-quantized checkpoint (FP8, int8, MXFP8, NVFP4): prepare it first",
              isOn: $model.needsPrep)
            if model.needsPrep {
              PathRow(
                title: "prep-model.py", url: settings.prepScript,
                placeholder: "Choose your copy of prep-model.py",
                choose: { chooseFile(settings.prepScript, extensions: ["py"]) { settings.prepScript = $0 } })
              Text(
                "This script is not part of the app. It needs Python 3 with numpy (python3 -m pip install numpy). It writes a dequantized copy first, so about three times the file's size is needed while converting."
              )
              .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Toggle(
              "Also convert the text encoders stored in the file (SD 1.x / SDXL style checkpoints)",
              isOn: $model.convertTextEncoders)
          }.padding(4)
        } label: { Label("2. Options", systemImage: "slider.horizontal.3") }

        GroupBox {
          ModelsFolderRow(needed: model.neededBytes).padding(4)
        } label: { Label("3. Output", systemImage: "folder") }

        RunBox(
          run: model.run, canStart: model.canStart,
          hint: model.input == nil ? "Choose a file first." : nil, start: { model.start() })
      }
      .padding(20)
    }
  }
}
