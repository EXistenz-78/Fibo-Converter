import AppKit
import FiboCore
import Foundation
import Observation
import SwiftUI

/// Settings shared by every tab (the models folder), remembered between launches.
@MainActor @Observable
final class Settings {
  static let shared = Settings()
  var modelsFolder: URL? {
    didSet { UserDefaults.standard.set(modelsFolder?.path, forKey: "modelsFolder") }
  }
  var prepScript: URL? {
    didSet { UserDefaults.standard.set(prepScript?.path, forKey: "prepScript") }
  }
  private init() {
    modelsFolder = UserDefaults.standard.string(forKey: "modelsFolder").map { URL(fileURLWithPath: $0) }
    prepScript = UserDefaults.standard.string(forKey: "prepScript").map { URL(fileURLWithPath: $0) }
  }

  var freeBytes: Int64? {
    guard let modelsFolder,
      let values = try? modelsFolder.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
    else { return nil }
    return values.volumeAvailableCapacityForImportantUsage
  }

  /// The conversion tools: inside the app, or in the folder FIBO_HELPERS names (development).
  static var helpersFolder: URL? {
    var candidates = [Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers")]
    if let override = ProcessInfo.processInfo.environment["FIBO_HELPERS"] {
      candidates.insert(URL(fileURLWithPath: override), at: 0)
    }
    return candidates.first {
      FileManager.default.isExecutableFile(atPath: $0.appendingPathComponent("ModelConverter").path)
        && FileManager.default.isExecutableFile(atPath: $0.appendingPathComponent("QuantI8x").path)
    }
  }
}

func byteString(_ bytes: Int64) -> String {
  ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

/// Runs a list of steps, keeps their state for the UI and prevents idle sleep meanwhile.
@MainActor @Observable
final class RunController {
  struct Row: Identifiable { var id: String; var title: String }
  private(set) var rows = [Row]()
  private(set) var states = [String: StepState]()
  private(set) var log = [String]()
  private(set) var isRunning = false
  private(set) var message: String?
  private(set) var failed = false
  private var runner: WorkRunner?
  private var activity: NSObjectProtocol?

  func start(
    reason: String, work: URL, steps: [WorkStep], success: String, finish: @escaping () -> Void = {}
  ) {
    guard !isRunning else { return }
    rows = steps.map { Row(id: $0.id, title: $0.title) }
    states = Dictionary(uniqueKeysWithValues: steps.map { ($0.id, StepState.pending) })
    log = []
    message = nil
    failed = false
    isRunning = true
    activity = ProcessInfo.processInfo.beginActivity(
      options: [.idleSystemSleepDisabled, .userInitiated], reason: reason)
    let runner = WorkRunner()
    self.runner = runner
    Task {
      do {
        try await runner.run(steps, work: work) { event in
          Task { @MainActor in self.handle(event) }
        }
        try? FileManager.default.removeItem(at: work)
        message = success
      } catch is CancellationError {
        message = "Stopped. Start again to continue from the step it was on."
        failed = true
      } catch {
        message = error.localizedDescription
        failed = true
        if let running = states.first(where: { if case .running = $0.value { true } else { false } }) {
          states[running.key] = .failed
        }
      }
      isRunning = false
      if let activity { ProcessInfo.processInfo.endActivity(activity) }
      activity = nil
      finish()
    }
  }

  func cancel() { runner?.cancel() }

  private func handle(_ event: WorkEvent) {
    switch event {
    case .started(let id): states[id] = .running(nil)
    case .progress(let id, let fraction): states[id] = .running(fraction)
    case .finished(let id): states[id] = .done
    case .skipped(let id): states[id] = .skipped
    case .log(let line):
      log.append(line)
      if log.count > 500 { log.removeFirst(log.count - 500) }
    }
  }
}

func chooseFolder(_ current: URL?, _ apply: (URL) -> Void) {
  let panel = NSOpenPanel()
  panel.canChooseDirectories = true
  panel.canChooseFiles = false
  panel.canCreateDirectories = true
  panel.allowsMultipleSelection = false
  panel.directoryURL = current
  if panel.runModal() == .OK, let url = panel.url { apply(url) }
}

func chooseFile(_ current: URL?, extensions: [String], _ apply: (URL) -> Void) {
  let panel = NSOpenPanel()
  panel.canChooseDirectories = false
  panel.canChooseFiles = true
  panel.allowsMultipleSelection = false
  panel.allowedFileTypes = extensions
  panel.directoryURL = current?.deletingLastPathComponent()
  if panel.runModal() == .OK, let url = panel.url { apply(url) }
}

/// A row with a title, the chosen path and Choose… (and one optional extra button).
struct PathRow: View {
  var title: String
  var url: URL?
  var placeholder = "Not chosen"
  var choose: () -> Void
  var extra: (String, () -> Void)?

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.caption).foregroundStyle(.secondary)
      HStack {
        Text(url?.path ?? placeholder).lineLimit(1).truncationMode(.middle)
        Spacer()
        Button("Choose…", action: choose)
        if let extra { Button(extra.0, action: extra.1) }
      }
    }
  }
}

struct ModelsFolderRow: View {
  var settings = Settings.shared
  var needed: Int64
  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      PathRow(
        title: "Models folder (your Draw Things / Atelier / DT Hub folder)", url: settings.modelsFolder,
        choose: { chooseFolder(settings.modelsFolder) { settings.modelsFolder = $0 } })
      if let free = settings.freeBytes {
        Text("Free space on that drive: \(byteString(free)). Needed while converting: about \(byteString(needed)).")
          .font(.caption)
          .foregroundStyle(free > needed ? Color.secondary : Color.red)
      }
    }
  }
}

/// The Convert / Stop button, the step list, the result message and the log.
struct RunBox: View {
  var run: RunController
  var canStart: Bool
  var hint: String?
  var start: () -> Void

  var body: some View {
    GroupBox {
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          if run.isRunning {
            Button("Stop") { run.cancel() }
          } else {
            Button(run.failed ? "Continue" : "Convert", action: start)
              .keyboardShortcut(.defaultAction)
              .disabled(!canStart)
          }
          if let hint { Text(hint).font(.caption).foregroundStyle(.secondary) }
        }
        ForEach(run.rows) { row in stepRow(row) }
        if let message = run.message {
          Text(message).foregroundStyle(run.failed ? Color.red : Color.green)
            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        }
        if !run.log.isEmpty {
          DisclosureGroup("Details") {
            ScrollView {
              Text(run.log.joined(separator: "\n"))
                .font(.caption.monospaced()).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 140)
          }
        }
      }
      .padding(4)
    } label: {
      Label("Convert", systemImage: "gearshape.2")
    }
  }

  private func stepRow(_ row: RunController.Row) -> some View {
    let state = run.states[row.id] ?? .pending
    return HStack {
      switch state {
      case .pending: Image(systemName: "circle").foregroundStyle(.secondary)
      case .running: ProgressView().controlSize(.small)
      case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
      case .skipped: Image(systemName: "checkmark.circle").foregroundStyle(.secondary)
      case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
      }
      Text(row.title)
      if case .running(let fraction?) = state { ProgressView(value: fraction).frame(width: 140) }
      if state == .skipped { Text("already done").font(.caption).foregroundStyle(.secondary) }
    }
  }
}
