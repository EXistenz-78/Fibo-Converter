import FiboCore
import SwiftUI

@MainActor @Observable
final class H3Model {
  var sourceFolder: URL? { didSet { rescan(); UserDefaults.standard.set(sourceFolder?.path, forKey: "h3SourceFolder") } }
  var precision: Precision = .eightBitS
  var licenseAccepted = false
  private(set) var located = [RequiredFile: URL]()
  let run = RunController()
  var settings: Settings { Settings.shared }

  init() {
    sourceFolder =
      UserDefaults.standard.string(forKey: "h3SourceFolder").map { URL(fileURLWithPath: $0) }
      ?? FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    rescan()
  }

  func rescan() {
    var found = sourceFolder.map { FiboSources.locate(H3Sources.all, in: $0) } ?? [:]
    // The text encoder may already be in the models folder.
    if found[H3Sources.textEncoder] == nil, let models = settings.modelsFolder {
      let url = models.appendingPathComponent(H3Sources.textEncoder.name)
      if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
        Int64(size) == H3Sources.textEncoder.size
      {
        found[H3Sources.textEncoder] = url
      }
    }
    located = found
  }

  var allFilesFound: Bool { H3Sources.all.allSatisfy { located[$0] != nil } }
  var textEncoderAlreadyThere: Bool {
    guard let url = located[H3Sources.textEncoder], let models = settings.modelsFolder else { return false }
    return url.standardizedFileURL.path == models.appendingPathComponent(H3Sources.textEncoder.name).standardizedFileURL.path
  }
  /// Rough peak disk use on the models volume.
  var neededBytes: Int64 {
    var total: Int64 = 40_300_000_000 + 1_200_000_000
    if !textEncoderAlreadyThere { total += 26_600_000_000 }
    if precision.makesI8x { total += 21_000_000_000 }
    return total
  }
  var canStart: Bool {
    licenseAccepted && allFilesFound && settings.modelsFolder != nil && Settings.helpersFolder != nil
      && !run.isRunning && (settings.freeBytes.map { $0 > neededBytes } ?? true)
  }

  func start() {
    guard canStart, let models = settings.modelsFolder, let helpers = Settings.helpersFolder,
      let modelFile = located[H3Sources.model], let vae = located[H3Sources.vae],
      let encoder = located[H3Sources.textEncoder]
    else { return }
    let work = models.appendingPathComponent("_h3-work")
    let job = H3Job(
      modelFile: modelFile, vaeFile: vae, textEncoderFile: encoder, models: models, work: work,
      helpers: helpers, precision: precision)
    run.start(
      reason: "Converting MiniMax H3 Image", work: work, steps: H3Conversion.steps(job),
      success: "Done. Refresh the models list in your app. Use the fork's server (see the README) and 6 steps.",
      finish: { [weak self] in self?.rescan() })
  }
}

struct H3View: View {
  @State private var model = H3Model()
  var settings = Settings.shared

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        GroupBox {
          VStack(alignment: .leading, spacing: 8) {
            Text(
              "MiniMax H3 Image is released under the MiniMax-H3-Image Community License. It grants use only in its \"Applicable Territory\", which excludes the European Union, the United Kingdom, the Republic of Korea and the United States of America. Converted and quantized copies are \"Model Derivatives\" under the same terms. This app contains and downloads no model weights: you download the files yourself from Hugging Face."
            )
            .fixedSize(horizontal: false, vertical: true)
            HStack {
              Link("Model page", destination: H3Sources.repositoryURL)
              Link("Licence text", destination: H3Sources.licenseURL)
            }
            Toggle(
              "I have read the licence and I am allowed to use this model where I live.",
              isOn: $model.licenseAccepted)
          }.padding(4)
        } label: { Label("Licence", systemImage: "exclamationmark.triangle") }

        GroupBox {
          VStack(alignment: .leading, spacing: 10) {
            Text(
              "Download these three files with your browser (about 68 GB). Save them anywhere in the folder below: the app finds them by name and size. The text encoder may also already be in your models folder."
            )
            .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(H3Sources.all) { file in
              HStack {
                Image(systemName: model.located[file] != nil ? "checkmark.circle.fill" : "arrow.down.circle")
                  .foregroundStyle(model.located[file] != nil ? Color.green : Color.secondary)
                VStack(alignment: .leading) {
                  Text(file.name).font(.body.monospaced())
                  Text("\(file.repository) · \(byteString(file.size))").font(.caption).foregroundStyle(.secondary)
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
              Text("8-bit S (i8x): about 21 GB, recommended").tag(Precision.eightBitS)
              Text("Full precision (f16): about 40 GB").tag(Precision.full)
              Text("Both").tag(Precision.both)
            }
            .pickerStyle(.radioGroup)
            Text(
              "H3 Image runs only on the Draw Things engine fork used by Atelier (the gRPCServerCLI of DT Hub 1.8 and later includes it). Use 6 steps: more crashes the server."
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
