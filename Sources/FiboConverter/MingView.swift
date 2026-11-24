import FiboCore
import SwiftUI

@MainActor @Observable
final class MingModel {
  var sourceFolder: URL? { didSet { rescan(); UserDefaults.standard.set(sourceFolder?.path, forKey: "mingSourceFolder") } }
  var precision: Precision = .eightBitS
  var licenseAccepted = false
  private(set) var located = [RequiredFile: URL]()
  let run = RunController()
  var settings: Settings { Settings.shared }

  init() {
    sourceFolder =
      UserDefaults.standard.string(forKey: "mingSourceFolder").map { URL(fileURLWithPath: $0) }
      ?? FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    rescan()
  }

  func rescan() {
    located = sourceFolder.map { FiboSources.locate(MingSources.all, in: $0) } ?? [:]
  }

  var allFilesFound: Bool { MingSources.all.allSatisfy { located[$0] != nil } }
  /// Rough peak disk use on the models volume: the text encoder in full precision (35 GB) next to its 8-bit S
  /// copy (18 GB), then the model (12 GB, plus 6 GB for 8-bit S), the vision tower and the VAE.
  var neededBytes: Int64 {
    var total: Int64 = 35_000_000_000 + 17_800_000_000 + 1_500_000_000 + 300_000_000
    total = max(total, 17_800_000_000 + 1_800_000_000 + 12_400_000_000 + (precision.makesI8x ? 6_200_000_000 : 0))
    return total
  }
  var canStart: Bool {
    licenseAccepted && allFilesFound && settings.modelsFolder != nil && Settings.helpersFolder != nil
      && !run.isRunning && (settings.freeBytes.map { $0 > neededBytes } ?? true)
  }

  func start() {
    guard canStart, let models = settings.modelsFolder, let helpers = Settings.helpersFolder,
      let modelFile = located[MingSources.model], let encoder = located[MingSources.textEncoder],
      let vae = located[MingSources.vae]
    else { return }
    let work = models.appendingPathComponent("_ming-work")
    let job = MingJob(
      modelFile: modelFile, textEncoderFile: encoder, vaeFile: vae, models: models, work: work,
      helpers: helpers, precision: precision)
    run.start(
      reason: "Converting Ming-Image", work: work, steps: MingConversion.steps(job),
      success: "Done. Refresh the models list in your app. Use the fork's server (DT Hub 1.8.3 and later), 12 steps, no guidance.",
      finish: { [weak self] in self?.rescan() })
  }
}

struct MingView: View {
  @State private var model = MingModel()
  var settings = Settings.shared

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        GroupBox {
          VStack(alignment: .leading, spacing: 8) {
            Text(
              "Ming-Image 0.1 Design is released by inclusionAI under the MIT licence: free use, commercial included, keeping the copyright notice. This app contains and downloads no model weights: you download the ComfyUI files yourself from Hugging Face (Comfy-Org/Ming-Image)."
            )
            .fixedSize(horizontal: false, vertical: true)
            HStack {
              Link("Model page", destination: MingSources.repositoryURL)
              Link("Licence text", destination: MingSources.licenseURL)
            }
            Toggle("I have read the licence.", isOn: $model.licenseAccepted)
          }.padding(4)
        } label: { Label("Licence", systemImage: "doc.text") }

        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            Text(
              "Download these three files with your browser (about 49 GB). Save them anywhere in the folder below: the app finds them by name and size."
            )
            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(MingSources.all) { file in
              HStack {
                Image(systemName: model.located[file] != nil ? "checkmark.circle.fill" : "arrow.down.circle")
                  .foregroundStyle(model.located[file] != nil ? Color.green : Color.secondary)
                VStack(alignment: .leading) {
                  Text(file.name).font(.body.monospaced())
                  Text("\(file.repository) · \(file.folder) · \(byteString(file.size))").font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Download") { NSWorkspace.shared.open(file.url) }
              }
            }
            Divider()
            PathRow(
              title: "Folder with the downloaded files", url: model.sourceFolder,
              choose: { chooseFolder(model.sourceFolder) { model.sourceFolder = $0 } },
              extra: ("Check again", { model.rescan() }))
            if !model.allFilesFound {
              Text("Files still missing or not fully downloaded (the size must match exactly).")
                .font(.caption).foregroundStyle(.orange)
            }
          }.padding(4)
        } label: { Label("1. Download", systemImage: "arrow.down.doc") }

        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            ModelsFolderRow(needed: model.neededBytes)
            Picker("Result", selection: $model.precision) {
              Text("8-bit S (i8x): about 26 GB in all, recommended").tag(Precision.eightBitS)
              Text("Full precision model (f16): about 32 GB in all").tag(Precision.full)
              Text("Both").tag(Precision.both)
            }
            .pickerStyle(.radioGroup)
            Text(
              "The text encoder is always quantized to 8-bit S (the engine uses that copy); the vision tower (for editing) and the VAE stay in full precision. Ming-Image runs only on the Draw Things engine fork (the gRPCServerCLI of DT Hub 1.8.3 and later includes it): 12 steps, no guidance, 1024 or 2048 px. Z-Image LoRAs work on it."
            )
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if Settings.helpersFolder == nil {
              Text("The conversion tools are missing from this copy of the app.").foregroundStyle(.red)
            }
          }.padding(4)
        } label: { Label("2. Output", systemImage: "folder") }

        RunBox(
          run: model.run, canStart: model.canStart,
          hint: model.licenseAccepted ? nil : "Accept the licence first.", start: { model.start() })
      }
      .padding(20)
    }
  }
}
