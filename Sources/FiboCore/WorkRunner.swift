import Foundation

public enum WorkEvent: Sendable {
  case started(String)
  case progress(String, Double)
  case log(String)
  case finished(String)
  case skipped(String)
}

/// One step of a conversion. Steps already done in an earlier run (recorded in the work folder)
/// are skipped, except those marked `alwaysRun`.
public struct WorkStep: Identifiable, Sendable {
  public let id: String
  public let title: String
  public let alwaysRun: Bool
  public let action: @Sendable (WorkContext) async throws -> Void
  public init(
    _ id: String, _ title: String, alwaysRun: Bool = false,
    action: @escaping @Sendable (WorkContext) async throws -> Void
  ) {
    self.id = id
    self.title = title
    self.alwaysRun = alwaysRun
    self.action = action
  }
}

public struct WorkContext: Sendable {
  let runner: WorkRunner
  let emit: @Sendable (WorkEvent) -> Void
  let stepID: String

  public var isCancelled: Bool { runner.isCancelled }
  public func log(_ line: String) { emit(.log(line)) }
  public func progress(_ fraction: Double) { emit(.progress(stepID, fraction)) }
  /// Runs a helper tool; returns what it printed. Throws with its last lines when it fails.
  @discardableResult
  public func exec(_ executable: URL, _ arguments: [String]) async throws -> String {
    try await runner.exec(executable, arguments, emit)
  }
}

public final class WorkRunner: @unchecked Sendable {
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
  var isCancelled: Bool { lock.withLock { cancelled } }
  private func register(_ process: Process) -> Bool {
    lock.withLock {
      if cancelled { return false }
      self.process = process
      return true
    }
  }
  private func unregister() -> Bool {
    lock.withLock {
      process = nil
      return cancelled
    }
  }

  public func run(
    _ steps: [WorkStep], work: URL, emit: @escaping @Sendable (WorkEvent) -> Void
  ) async throws {
    try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
    let statePath = work.appendingPathComponent("state.json")
    var done = Set(
      ((try? JSONSerialization.jsonObject(with: Data(contentsOf: statePath))) as? [String]) ?? [])
    for step in steps {
      if isCancelled { throw CancellationError() }
      if done.contains(step.id) && !step.alwaysRun {
        emit(.skipped(step.id))
        continue
      }
      emit(.started(step.id))
      try await step.action(WorkContext(runner: self, emit: emit, stepID: step.id))
      done.insert(step.id)
      try JSONSerialization.data(withJSONObject: done.sorted()).write(to: statePath)
      emit(.finished(step.id))
    }
  }

  fileprivate func exec(
    _ executable: URL, _ arguments: [String], _ emit: @escaping @Sendable (WorkEvent) -> Void
  ) async throws -> String {
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
      throw FiboError.message("Missing tool: \(executable.lastPathComponent).")
    }
    let process = Process()
    process.executableURL = executable
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    let output = OutputBuffer()
    pipe.fileHandleForReading.readabilityHandler = { handle in
      let data = handle.availableData
      guard !data.isEmpty else { return }
      let text = String(decoding: data, as: UTF8.self)
      output.append(text)
      for line in text.split(whereSeparator: \.isNewline) { emit(.log(String(line))) }
    }
    if !register(process) { throw CancellationError() }
    let status: Int32 = try await withCheckedThrowingContinuation { continuation in
      process.terminationHandler = { finished in continuation.resume(returning: finished.terminationStatus) }
      do { try process.run() } catch { continuation.resume(throwing: error) }
    }
    pipe.fileHandleForReading.readabilityHandler = nil
    if unregister() { throw CancellationError() }
    guard status == 0 else {
      throw FiboError.message(
        "\(executable.lastPathComponent) failed (code \(status)).\n\(output.tail(6))")
    }
    return output.text
  }
}

private final class OutputBuffer: @unchecked Sendable {
  private let lock = NSLock()
  private var value = ""
  func append(_ text: String) {
    lock.withLock {
      value += text
      if value.count > 2_000_000 { value = String(value.suffix(1_000_000)) }
    }
  }
  var text: String { lock.withLock { value } }
  func tail(_ lines: Int) -> String {
    lock.withLock {
      value.split(whereSeparator: \.isNewline).suffix(lines).joined(separator: "\n")
    }
  }
}
