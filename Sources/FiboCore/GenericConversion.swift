import Foundation

/// Converts any safetensors file Draw Things' engine can import: optionally prepares it first
/// with prep-model.py (ComfyUI-quantized files), converts it to a ckpt, and registers it.
public struct GenericJob: Sendable {
  public var input: URL
  public var name: String
  public var models: URL
  public var work: URL
  public var helpers: URL
  /// prep-model.py; nil skips the preparation.
  public var prepScript: URL?
  /// Python interpreters to try for prep-model.py (the first with numpy is used).
  public var pythonCandidates: [URL]
  public var convertTextEncoders: Bool
  /// Lets prep-model.py download the few tensors it may need (pruned MiniMax H3 only).
  public var allowFetch: Bool

  public init(
    input: URL, name: String, models: URL, work: URL, helpers: URL, prepScript: URL? = nil,
    pythonCandidates: [URL] = GenericConversion.defaultPythons, convertTextEncoders: Bool = false,
    allowFetch: Bool = false
  ) {
    self.input = input
    self.name = name
    self.models = models
    self.work = work
    self.helpers = helpers
    self.prepScript = prepScript
    self.pythonCandidates = pythonCandidates
    self.convertTextEncoders = convertTextEncoders
    self.allowFetch = allowFetch
  }
}

public enum GenericConversion {
  public static let defaultPythons = [
    "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3",
  ].map { URL(fileURLWithPath: $0) }

  /// The Python to run prep-model.py with: the first that starts and has numpy.
  public static func findPython(_ candidates: [URL]) -> URL? {
    for candidate in candidates where FileManager.default.isExecutableFile(atPath: candidate.path) {
      let process = Process()
      process.executableURL = candidate
      process.arguments = ["-c", "import numpy"]
      process.standardOutput = FileHandle.nullDevice
      process.standardError = FileHandle.nullDevice
      guard (try? process.run()) != nil else { continue }
      process.waitUntilExit()
      if process.terminationStatus == 0 { return candidate }
    }
    return nil
  }

  /// The specification the converter prints after an import: the last JSON object in its output.
  public static func parseSpecification(_ output: String) throws -> [String: Any] {
    guard let range = output.range(of: "\n{\n", options: .backwards)
      ?? (output.hasPrefix("{\n") ? output.startIndex..<output.index(output.startIndex, offsetBy: 2) : nil)
    else { throw FiboError.message("The converter did not print the model specification.") }
    let start = output[range].hasPrefix("\n") ? output.index(after: range.lowerBound) : range.lowerBound
    let text = String(output[start...])
    guard let object = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
      object["file"] is String
    else { throw FiboError.message("The converter's specification could not be read.") }
    return object
  }

  public static func steps(_ job: GenericJob) -> [WorkStep] {
    let preparedDirectory = job.work.appendingPathComponent("prepared")
    let specificationFile = job.work.appendingPathComponent("specification.json")
    let converter = job.helpers.appendingPathComponent("ModelConverter")
    var steps = [WorkStep]()

    if let script = job.prepScript {
      steps.append(
        WorkStep("prepare", "Prepare the file with prep-model.py") { context in
          guard let python = findPython(job.pythonCandidates) else {
            throw FiboError.message(
              "prep-model.py needs Python 3 with numpy. In Terminal run: python3 -m pip install numpy")
          }
          try? FileManager.default.removeItem(at: preparedDirectory)
          try FileManager.default.createDirectory(at: preparedDirectory, withIntermediateDirectories: true)
          var arguments = [script.path, job.input.path, preparedDirectory.path + "/"]
          if !job.allowFetch { arguments.append("--no-fetch") }
          try await context.exec(python, arguments)
        })
    }
    steps.append(
      WorkStep("import", "Convert to a Draw Things model") { context in
        // prep-model.py leaves a file that is already in shape alone; then the original is used.
        let prepared = (try? FileManager.default.contentsOfDirectory(atPath: preparedDirectory.path))?
          .filter { $0.hasSuffix(".safetensors") }.sorted().first
        let source = prepared.map { preparedDirectory.appendingPathComponent($0) } ?? job.input
        var arguments = ["-f", source.path, "-n", job.name, "-o", job.models.path]
        if job.convertTextEncoders { arguments.append("--text-encoders") }
        let output = try await context.exec(converter, arguments)
        let specification = try parseSpecification(output)
        let file = specification["file"] as! String
        guard FileManager.default.fileExists(atPath: job.models.appendingPathComponent(file).path) else {
          throw FiboError.message("The converter did not write \(file).")
        }
        try JSONSerialization.data(withJSONObject: specification).write(to: specificationFile)
      })
    steps.append(
      WorkStep("register", "Add to the library (custom.json)", alwaysRun: true) { context in
        guard
          var entry = (try? JSONSerialization.jsonObject(with: Data(contentsOf: specificationFile)))
            as? [String: Any]
        else { throw FiboError.message("The model specification is missing.") }
        entry["name"] = job.name
        if let backup = try CustomJSON.register([entry], in: job.models) {
          context.log("Backup of custom.json: \(backup.lastPathComponent)")
        }
      })
    return steps
  }
}
