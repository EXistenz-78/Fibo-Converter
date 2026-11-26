import FiboCore
import SwiftUI

@MainActor @Observable
final class MingLayerModel {
  var sourceFolder: URL? {
    didSet { rescan(); UserDefaults.standard.set(sourceFolder?.path, forKey: "mingLayerSourceFolder") }
  }
  var precision: Precision = .eightBitS
  var licenseAccepted = false
  private(set) var located = [RequiredFile: URL]()
  private(set) var missingDesign = [String]()
  let run = RunController()
  var settings: Settings { Settings.shared }

  init() {
    sourceFolder =
      UserDefaults.standard.string(forKey: "mingLayerSourceFolder").map { URL(fileURLWithPath: $0) }
      ?? UserDefaults.standard.string(forKey: "mingSourceFolder").map { URL(fileURLWithPath: $0) }
      ?? FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    rescan()
  }

  func rescan() {
    located = sourceFolder.map { FiboSources.locate(MingLayerSources.all, in: $0) } ?? [:]
    missingDesign = settings.modelsFolder.map { MingLayerConversion.missingDesignFiles(in: $0) } ?? []
  }

  var allFilesFound: Bool { MingLayerSources.all.allSatisfy { located[$0] != nil } }
  /// Rough peak disk use on the models volume: the Layer parts of the text encoder (2.7 GB in full precision, 1.4 GB
  /// at 8-bit S), the text encoder's own file (1.4 GB; its 18 GB of tensor data is a clone on APFS) and the model
  /// (12 GB, plus 6 GB for 8-bit S).
  var neededBytes: Int64 {
    2_700_000_000 + 2_800_000_000 + 12_400_000_000 + (precision.makesI8x ? 6_200_000_000 : 0)
  }
  var canStart: Bool {
    licenseAccepted && allFilesFound && missingDesign.isEmpty && settings.modelsFolder != nil
      && Settings.helpersFolder != nil && !run.isRunning && (settings.freeBytes.map { $0 > neededBytes } ?? true)
  }

  func start() {
    guard canStart, let models = settings.modelsFolder, let helpers = Settings.helpersFolder,
      let modelFile = located[MingLayerSources.model], let encoder = located[MingLayerSources.textEncoder]
    else { return }
    let work = models.appendingPathComponent("_ming-layer-work")
    let job = MingLayerJob(
      modelFile: modelFile, textEncoderFile: encoder, models: models, work: work, helpers: helpers,
      precision: precision)
    run.start(
      reason: "Converting Ming-Image Design-Layer", work: work, steps: MingLayerConversion.steps(job),
      success:
        "Done. Refresh the models list in your app. Use the fork's server (DT Hub 1.10.0 and later): the design as the start image, batch size = layers + 1, guidance 3, 12 steps.",
      finish: { [weak self] in self?.rescan() })
  }
}

struct MingLayerView: View {
  @State private var model = MingLayerModel()
  var settings = Settings.shared

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        GroupBox {
          VStack(alignment: .leading, spacing: 8) {
            Text(
              "Ming-Image 0.1 Design-Layer is released by inclusionAI under the MIT licence: free use, commercial included, keeping the copyright notice. It splits a finished design into transparent layers. This app contains and downloads no model weights: you download the ComfyUI files yourself from Hugging Face (Comfy-Org/Ming-Image)."
            )
            .fixedSize(horizontal: false, vertical: true)
            HStack {
              Link("Model page", destination: MingLayerSources.repositoryURL)
              Link("Licence text", destination: MingLayerSources.licenseURL)
            }
            Toggle("I have read the licence.", isOn: $model.licenseAccepted)
          }.padding(4)
        } label: { Label("Licence", systemImage: "doc.text") }

        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            Text(
              "Download these two files with your browser (about 49 GB). Save them anywhere in the folder below: the app finds them by name and size. Ming-Image Design must already be converted (tab Ming-Image): the Layer model uses its vision tower, VAE and tokenizer, and its text encoder gets only the parts that differ from the Layer file."
            )
            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(MingLayerSources.all) { file in
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
            if !model.missingDesign.isEmpty {
              Text(
                "Convert Ming-Image Design first (tab Ming-Image). Missing in the models folder: \(model.missingDesign.joined(separator: ", "))."
              )
              .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
            Picker("Result", selection: $model.precision) {
              Text("8-bit S (i8x): about 8 GB more, recommended").tag(Precision.eightBitS)
              Text("Full precision model (f16): about 14 GB more").tag(Precision.full)
              Text("Both").tag(Precision.both)
            }
            .pickerStyle(.radioGroup)
            Text(
              "The text encoder is always 8-bit S. Design-Layer runs only on the Draw Things engine fork (the gRPCServerCLI of DT Hub 1.10.0 and later): the design as the start image (strength 1), a prompt that lists the layers, batch size = layers + 1, 12 steps, guidance 3, 1024 px."
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
    .onAppear { model.rescan() }
  }
}
