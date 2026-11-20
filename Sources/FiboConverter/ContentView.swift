import FiboCore
import SwiftUI
import UniformTypeIdentifiers

struct FiboView: View {
  @State private var model = ConverterModel()

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        licenseBox
        downloadBox
        outputBox
        runBox
      }
      .padding(20)
    }
  }

  // MARK: Sections

  private var licenseBox: some View {
    GroupBox {
      VStack(alignment: .leading, spacing: 8) {
        Text(
          "Fibo 2.0 Edit Turbo is released by Bria under the CC BY-NC 4.0 licence: non-commercial use only, with credit to Bria. Converted and quantized copies are adaptations and follow the same licence, so do not share them or use them commercially. This app does not download or include any model weights: you download the files yourself from Hugging Face, after accepting the licence there."
        )
        .fixedSize(horizontal: false, vertical: true)
        HStack {
          Link("Model page", destination: FiboSources.repositoryURL)
          Link("Licence text", destination: FiboSources.licenseURL)
        }
        Toggle("I have read the licence and will use the converted model only as it allows.", isOn: $model.licenseAccepted)
      }
      .padding(4)
    } label: {
      Label("Licence", systemImage: "exclamationmark.triangle")
    }
  }

  private var downloadBox: some View {
    GroupBox {
      VStack(alignment: .leading, spacing: 10) {
        Text(
          "Log in to Hugging Face, accept the model's terms, then download these three files with your browser. Save them anywhere in the folder below: the app finds them by name and size."
        )
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
        ForEach(FiboSources.all) { file in
          HStack {
            Image(systemName: model.located[file] != nil ? "checkmark.circle.fill" : "arrow.down.circle")
              .foregroundStyle(model.located[file] != nil ? Color.green : Color.secondary)
            VStack(alignment: .leading) {
              Text(file.name).font(.body.monospaced())
              Text("\(file.folder) · \(Self.size(file.size))").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button("Download") { NSWorkspace.shared.open(file.url) }
          }
        }
        Divider()
        folderRow(
          "Folder with the downloaded files", model.sourceFolder,
          action: { chooseFolder(model.sourceFolder) { model.sourceFolder = $0 } }, extra: ("Check again", { model.rescan() }))
        if !model.allFilesFound {
          Text("Files still missing or not fully downloaded (the size must match exactly).")
            .font(.caption).foregroundStyle(.orange)
        }
      }
      .padding(4)
    } label: {
      Label("1. Download", systemImage: "arrow.down.doc")
    }
  }

  private var outputBox: some View {
    GroupBox {
      VStack(alignment: .leading, spacing: 10) {
        folderRow("Models folder (your Draw Things / Atelier / DT Hub folder)", model.modelsFolder, action: { chooseFolder(model.modelsFolder) { model.modelsFolder = $0 } }, extra: nil)
        Picker("Result", selection: $model.precision) {
          Text("8-bit S (i8x): about 13 GB, recommended").tag(Precision.eightBitS)
          Text("Full precision (f16): about 25 GB").tag(Precision.full)
          Text("Both").tag(Precision.both)
        }
        .pickerStyle(.radioGroup)
        if let free = model.freeBytes {
          Text("Free space on that drive: \(Self.size(free)). Needed while converting: about \(Self.size(model.neededBytes)).")
            .font(.caption)
            .foregroundStyle(model.enoughSpace ? Color.secondary : Color.red)
        }
        if model.helpersFolder == nil {
          Text("The conversion tools are missing from this copy of the app.").foregroundStyle(.red)
        }
      }
      .padding(4)
    } label: {
      Label("2. Output", systemImage: "folder")
    }
  }

  private var runBox: some View {
    GroupBox {
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          if model.isRunning {
            Button("Stop") { model.cancel() }
          } else {
            Button(model.failed ? "Continue" : "Convert") { model.start() }
              .keyboardShortcut(.defaultAction)
              .disabled(!model.canStart)
          }
          if !model.licenseAccepted { Text("Accept the licence first.").font(.caption).foregroundStyle(.secondary) }
        }
        ForEach(model.plannedSteps) { step in stepRow(step) }
        if let message = model.message {
          Text(message).foregroundStyle(model.failed ? Color.red : Color.green)
            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        }
        if !model.log.isEmpty {
          DisclosureGroup("Details") {
            ScrollView {
              Text(model.log.joined(separator: "\n"))
                .font(.caption.monospaced()).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 140)
          }
        }
      }
      .padding(4)
    } label: {
      Label("3. Convert", systemImage: "gearshape.2")
    }
  }

  // MARK: Pieces

  private func stepRow(_ step: Step) -> some View {
    let state = model.states[step] ?? .pending
    return HStack {
      switch state {
      case .pending: Image(systemName: "circle").foregroundStyle(.secondary)
      case .running: ProgressView().controlSize(.small)
      case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
      case .skipped: Image(systemName: "checkmark.circle").foregroundStyle(.secondary)
      case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
      }
      Text(step.title)
      if case .running(let fraction?) = state {
        ProgressView(value: fraction).frame(width: 140)
      }
      if state == .skipped { Text("already done").font(.caption).foregroundStyle(.secondary) }
    }
  }

  private func folderRow(
    _ title: String, _ url: URL?, action: @escaping () -> Void, extra: (String, () -> Void)?
  ) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.caption).foregroundStyle(.secondary)
      HStack {
        Text(url?.path ?? "Not chosen").lineLimit(1).truncationMode(.middle)
        Spacer()
        Button("Choose…", action: action)
        if let extra { Button(extra.0, action: extra.1) }
      }
    }
  }

  private func chooseFolder(_ current: URL?, _ apply: (URL) -> Void) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    panel.directoryURL = current
    if panel.runModal() == .OK, let url = panel.url { apply(url) }
  }

  private static func size(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }
}

struct ContentView: View {
  var body: some View {
    TabView {
      FiboView().tabItem { Text("Fibo 2.0 Edit Turbo") }
      GenericView().tabItem { Text("Any safetensors") }
      H3View().tabItem { Text("MiniMax H3 Image") }
    }
    .padding(.top, 8)
  }
}
