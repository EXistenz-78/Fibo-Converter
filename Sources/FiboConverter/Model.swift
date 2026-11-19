import AppKit
import FiboCore
import Foundation
import Observation

enum StepState: Equatable {
  case pending, running(Double?), done, skipped, failed
}

@MainActor @Observable
final class ConverterModel {
  var sourceFolder: URL? { didSet { rescan(); save("sourceFolder", sourceFolder) } }
  var modelsFolder: URL? { didSet { updateSpace(); save("modelsFolder", modelsFolder) } }
  var precision: Precision = .eightBitS
  var licenseAccepted = false
  private(set) var located = [RequiredFile: URL]()
  private(set) var states = [Step: StepState]()
  private(set) var plannedSteps = [Step]()
  private(set) var log = [String]()
  private(set) var isRunning = false
  private(set) var message: String?
  private(set) var failed = false
  private(set) var freeBytes: Int64?

  private var pipeline: Pipeline?
  private var activity: NSObjectProtocol?

  init() {
    sourceFolder = Self.load("sourceFolder")
    modelsFolder = Self.load("modelsFolder")
    if sourceFolder == nil {
      sourceFolder = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    }
    rescan()
    updateSpace()
  }

  // MARK: Checks

  var allFilesFound: Bool { FiboSources.all.allSatisfy { located[$0] != nil } }

  /// Rough peak disk use on the models volume, in bytes.
  var neededBytes: Int64 {
    let merged: Int64 = 16_300_000_000
    let encoder: Int64 = 8_900_000_000
    var total = merged + merged + encoder
    if precision.makesI8x { total += 8_200_000_000 + 5_300_000_000 }
    return total
  }
  var enoughSpace: Bool { freeBytes.map { $0 > neededBytes } ?? true }
  var canStart: Bool {
    licenseAccepted && allFilesFound && modelsFolder != nil && enoughSpace && !isRunning
      && helpersFolder != nil
  }

  var helpersFolder: URL? {
    var candidates = [Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers")]
    if let override = ProcessInfo.processInfo.environment["FIBO_HELPERS"] {
      candidates.insert(URL(fileURLWithPath: override), at: 0)
    }
    return candidates.first {
      FileManager.default.isExecutableFile(atPath: $0.appendingPathComponent("ModelConverter").path)
        && FileManager.default.isExecutableFile(atPath: $0.appendingPathComponent("QuantI8x").path)
    }
  }

  func rescan() {
    guard let sourceFolder else { located = [:]; return }
    located = FiboSources.locate(in: sourceFolder)
  }

  func updateSpace() {
    guard let modelsFolder,
      let values = try? modelsFolder.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey
      ])
    else { freeBytes = nil; return }
    freeBytes = values.volumeAvailableCapacityForImportantUsage
  }

  // MARK: Run

  func start() {
    guard canStart, let modelsFolder, let helpers = helpersFolder,
      let shard1 = located[FiboSources.shard1], let shard2 = located[FiboSources.shard2],
      let encoder = located[FiboSources.textEncoder]
    else { return }
    let work = modelsFolder.appendingPathComponent("_fibo-work")
    let job = Job(
      transformerShards: [shard1, shard2], textEncoderFile: encoder, work: work,
      models: modelsFolder, precision: precision, helpers: helpers)
    plannedSteps = job.steps
    states = Dictionary(uniqueKeysWithValues: job.steps.map { ($0, StepState.pending) })
    log = []
    message = nil
    failed = false
    isRunning = true
    activity = ProcessInfo.processInfo.beginActivity(
      options: [.idleSystemSleepDisabled, .userInitiated], reason: "Converting Fibo 2.0")
    let pipeline = Pipeline()
    self.pipeline = pipeline
    Task {
      do {
        try await pipeline.run(job) { event in
          Task { @MainActor in self.handle(event) }
        }
        try? FileManager.default.removeItem(at: work)
        message = "Done. Refresh the models list in your app to see Fibo 2.0 Edit Turbo."
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
      updateSpace()
    }
  }

  func cancel() { pipeline?.cancel() }

  private func handle(_ event: Event) {
    switch event {
    case .started(let step): states[step] = .running(nil)
    case .progress(let step, let fraction): states[step] = .running(fraction)
    case .finished(let step): states[step] = .done
    case .skipped(let step, _): states[step] = .skipped
    case .log(let line):
      log.append(line)
      if log.count > 500 { log.removeFirst(log.count - 500) }
    }
  }

  // MARK: Preferences

  private static func load(_ key: String) -> URL? {
    UserDefaults.standard.string(forKey: key).map { URL(fileURLWithPath: $0) }
  }
  private func save(_ key: String, _ url: URL?) {
    UserDefaults.standard.set(url?.path, forKey: key)
  }
}
