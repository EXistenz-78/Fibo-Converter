import Foundation

public enum Step: String, CaseIterable, Identifiable {
  case merge, importModel, textEncoder, quantizeModel, quantizeTextEncoder, register, cleanup
  public var id: String { rawValue }
  public var title: String {
    switch self {
    case .merge: return "Merge the two transformer files"
    case .importModel: return "Convert the model"
    case .textEncoder: return "Convert the text encoder"
    case .quantizeModel: return "Quantize the model to 8-bit S"
    case .quantizeTextEncoder: return "Quantize the text encoder to 8-bit S"
    case .register: return "Register in custom.json"
    case .cleanup: return "Remove temporary files"
    }
  }
}

public struct Job {
  public var transformerShards: [URL]
  public var textEncoderFile: URL
  public var work: URL
  public var models: URL
  public var precision: Precision
  public var helpers: URL

  public init(
    transformerShards: [URL], textEncoderFile: URL, work: URL, models: URL, precision: Precision,
    helpers: URL
  ) {
    self.transformerShards = transformerShards
    self.textEncoderFile = textEncoderFile
    self.work = work
    self.models = models
    self.precision = precision
    self.helpers = helpers
  }

  public var steps: [Step] {
    var steps: [Step] = [.merge, .importModel, .textEncoder]
    if precision.makesI8x { steps += [.quantizeModel, .quantizeTextEncoder] }
    return steps + [.register, .cleanup]
  }

  var mergedFile: URL { work.appendingPathComponent("\(FiboOutputs.modelBase).safetensors") }
}

public enum Event {
  case started(Step)
  case progress(Step, Double)
  case log(String)
  case finished(Step)
  case skipped(Step, String)
}

/// Runs the conversion. Steps already completed in an earlier run (recorded in the work folder)
/// are skipped, so a stopped run can simply be started again.
public final class Pipeline: @unchecked Sendable {
  private let lock = NSLock()
  private var process: Process?
  private var cancelled = false

  public init() {}

  public func cancel() {
    lock.lock()
    cancelled = true
    process?.terminate()
    lock.unlock()
  }
  /// False when the run was already cancelled (the process must not start).
  private func register(_ process: Process) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    if cancelled { return false }
    self.process = process
    return true
  }
  /// Forgets the process; returns whether the run was cancelled meanwhile.
  private func unregister() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    process = nil
    return cancelled
  }
  private var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  public func run(_ job: Job, emit: @escaping @Sendable (Event) -> Void) async throws {
    let manager = FileManager.default
    try manager.createDirectory(at: job.work, withIntermediateDirectories: true)
    let statePath = job.work.appendingPathComponent("state.json")
    var done = Set(
      ((try? JSONSerialization.jsonObject(with: Data(contentsOf: statePath))) as? [String])
        ?? [])
    func markDone(_ step: Step) throws {
      done.insert(step.rawValue)
      try JSONSerialization.data(withJSONObject: done.sorted()).write(to: statePath)
    }
    let converter = job.helpers.appendingPathComponent("ModelConverter")
    let quantizer = job.helpers.appendingPathComponent("QuantI8x")
    let modelF16 = job.models.appendingPathComponent(FiboOutputs.model("f16"))
    let encoderF16 = job.models.appendingPathComponent(FiboOutputs.textEncoder("f16"))
    let modelI8x = job.models.appendingPathComponent(FiboOutputs.model("i8x"))
    let encoderI8x = job.models.appendingPathComponent(FiboOutputs.textEncoder("i8x"))

    for step in job.steps {
      if isCancelled { throw CancellationError() }
      if done.contains(step.rawValue) && step != .register && step != .cleanup {
        emit(.skipped(step, "Already done in an earlier run."))
        continue
      }
      emit(.started(step))
      switch step {
      case .merge:
        try Self.removeModelFile(job.mergedFile)
        try Safetensors.merge(
          job.transformerShards, to: job.mergedFile, isCancelled: { self.isCancelled },
          progress: { written, total in emit(.progress(.merge, Double(written) / Double(total))) })
      case .importModel:
        try Self.removeModelFile(modelF16)
        try await exec(
          converter,
          ["-f", job.mergedFile.path, "-n", FiboOutputs.modelBase, "-o", job.models.path], emit)
        guard manager.fileExists(atPath: modelF16.path) else {
          throw FiboError.message("The converter did not write \(modelF16.lastPathComponent).")
        }
      case .textEncoder:
        try Self.removeModelFile(encoderF16)
        // The converter reads every .safetensors in the folder it is given, and the download
        // folder may hold others: give it a folder of its own with a link to the one file.
        let stage = job.work.appendingPathComponent("text_encoder")
        try? manager.removeItem(at: stage)
        try manager.createDirectory(at: stage, withIntermediateDirectories: true)
        try manager.createSymbolicLink(
          at: stage.appendingPathComponent("model.safetensors"), withDestinationURL: job.textEncoderFile)
        try await exec(
          converter,
          [
            "--fibo2-text-encoder", "-f", stage.path,
            "-n", FiboOutputs.textEncoderBase, "-o", job.models.path,
          ], emit)
        guard manager.fileExists(atPath: encoderF16.path) else {
          throw FiboError.message("The converter did not write \(encoderF16.lastPathComponent).")
        }
      case .quantizeModel:
        try Self.removeModelFile(modelI8x)
        try await exec(
          quantizer, ["-i", modelF16.path, "-m", "fibo2", "-o", modelI8x.path], emit)
        guard manager.fileExists(atPath: modelI8x.path) else {
          throw FiboError.message("The quantizer did not write \(modelI8x.lastPathComponent).")
        }
      case .quantizeTextEncoder:
        try Self.removeModelFile(encoderI8x)
        try await exec(
          quantizer,
          ["-i", encoderF16.path, "-m", "fibo2", "-o", encoderI8x.path, "--text-encoder"], emit)
        guard manager.fileExists(atPath: encoderI8x.path) else {
          throw FiboError.message("The quantizer did not write \(encoderI8x.lastPathComponent).")
        }
      case .register:
        var entries = [[String: Any]]()
        if job.precision.keepsF16 {
          entries.append(
            try CustomJSON.entry(
              name: "Fibo 2.0 Edit Turbo", file: FiboOutputs.model("f16"),
              textEncoder: FiboOutputs.textEncoder("f16")))
        }
        if job.precision.makesI8x {
          entries.append(
            try CustomJSON.entry(
              name: "Fibo 2.0 Edit Turbo (8-bit S)", file: FiboOutputs.model("i8x"),
              textEncoder: FiboOutputs.textEncoder("i8x")))
        }
        if let backup = try CustomJSON.register(entries, in: job.models) {
          emit(.log("Backup of custom.json: \(backup.lastPathComponent)"))
        }
      case .cleanup:
        try? manager.removeItem(at: job.mergedFile)
        if !job.precision.keepsF16 {
          // The f16 files were only inputs of the quantizer: the Trash keeps them recoverable.
          for file in [modelF16, encoderF16] {
            for url in [file, URL(fileURLWithPath: file.path + "-tensordata")]
            where manager.fileExists(atPath: url.path) {
              try manager.trashItem(at: url, resultingItemURL: nil)
            }
          }
        }
      }
      try markDone(step)
      emit(.finished(step))
    }
  }

  /// Removes a model file and the external tensor data Draw Things' engine keeps beside it.
  static func removeModelFile(_ url: URL) throws {
    let manager = FileManager.default
    for target in [url, URL(fileURLWithPath: url.path + "-tensordata")]
    where manager.fileExists(atPath: target.path) {
      try manager.removeItem(at: target)
    }
  }

  private func exec(_ executable: URL, _ arguments: [String], _ emit: @escaping @Sendable (Event) -> Void)
    async throws
  {
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
      throw FiboError.message("Missing helper tool: \(executable.lastPathComponent).")
    }
    let process = Process()
    process.executableURL = executable
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    let tail = TailBuffer()
    pipe.fileHandleForReading.readabilityHandler = { handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      for line in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline) {
        let text = String(line)
        emit(.log(text))
        tail.append(text)
      }
    }
    if !register(process) { throw CancellationError() }
    let status: Int32 = try await withCheckedThrowingContinuation { continuation in
      process.terminationHandler = { finished in continuation.resume(returning: finished.terminationStatus) }
      do { try process.run() } catch { continuation.resume(throwing: error) }
    }
    pipe.fileHandleForReading.readabilityHandler = nil
    if unregister() { throw CancellationError() }
    guard status == 0 else {
      let detail = tail.text
      throw FiboError.message("\(executable.lastPathComponent) failed (code \(status)).\n\(detail)")
    }
  }
}

/// The last few output lines of a helper, kept for the error message.
private final class TailBuffer: @unchecked Sendable {
  private let lock = NSLock()
  private var lines = [String]()
  func append(_ line: String) { lock.withLock { lines = Array((lines + [line]).suffix(6)) } }
  var text: String { lock.withLock { lines.joined(separator: "\n") } }
}
