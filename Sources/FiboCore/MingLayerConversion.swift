import Foundation

/// Ming-Image 0.1 Design-Layer (inclusionAI, MIT): splits a finished design into transparent layers. Its DiT is
/// Design's with learned padding tokens; its text encoder is Design's with a different connector, projections and
/// query tokens (the Ling language model and the vision tower are the same), so the Layer text encoder is built from
/// the Design one already in the models folder, taking only those parts from the Layer file.
public enum MingLayerSources {
  public static let repositoryURL = URL(string: "https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer")!
  public static let licenseURL = MingSources.licenseURL

  public static let model = RequiredFile(
    repository: "Comfy-Org/Ming-Image", path: "diffusion_models/ming_image_0.1_design_layer_bf16.safetensors",
    size: 12_309_874_192)
  public static let textEncoder = RequiredFile(
    repository: "Comfy-Org/Ming-Image", path: "text_encoders/ming_image_0.1_ling_mini_2.0_layer_bf16.safetensors",
    size: 36_677_768_245)
  public static let all = [model, textEncoder]
}

/// The files the Layer conversion writes, and the Design files it needs.
public enum MingLayerOutputs {
  public static let modelBase = "ming_image_0.1_design_layer"
  public static func model(_ suffix: String) -> String { "\(modelBase)_\(suffix).ckpt" }
  public static let textEncoder = "ming_image_ling_mini_2.0_layer_i8x.ckpt"
  static let connectorBase = "ming_image_ling_mini_2.0_layer_connector"
  /// Design's files the Layer model uses as they are (vision tower, VAE, tokenizer) or builds on (text encoder).
  public static let designFiles = [
    MingOutputs.textEncoder, MingOutputs.textEncoder + "-tensordata", MingOutputs.tokenizer, MingOutputs.vision,
    MingOutputs.vae,
  ]
}

public struct MingLayerJob: Sendable {
  public var modelFile: URL
  public var textEncoderFile: URL
  public var models: URL
  public var work: URL
  public var helpers: URL
  public var precision: Precision
  /// The sqlite3 program that moves the Layer rows into the copy of Design's text encoder.
  public var sqlite: URL

  public init(
    modelFile: URL, textEncoderFile: URL, models: URL, work: URL, helpers: URL, precision: Precision,
    sqlite: URL = URL(fileURLWithPath: "/usr/bin/sqlite3")
  ) {
    self.modelFile = modelFile
    self.textEncoderFile = textEncoderFile
    self.models = models
    self.work = work
    self.helpers = helpers
    self.precision = precision
    self.sqlite = sqlite
  }
}

public enum MingLayerConversion {
  /// The Design files missing from `models`: the Layer model needs Ming-Image Design converted first.
  public static func missingDesignFiles(in models: URL) -> [String] {
    MingLayerOutputs.designFiles.filter {
      !FileManager.default.fileExists(atPath: models.appendingPathComponent($0).path)
    }
  }

  public static func entry(name: String, file: String) throws -> [String: Any] {
    var entry = try MingConversion.entry(name: name, file: file)
    entry["is_consistency_model"] = false
    entry["text_encoder"] = MingLayerOutputs.textEncoder
    entry["note"] =
      "Ming-Image 0.1 Design-Layer by inclusionAI (MIT): splits the start image into transparent layers. Batch size = layers + 1 (the first picture is the design put back together). Prompt: 'Decompose this image into N layers with the following specifications:', then 'Number of layers: N' and 'Layer 1: <front>' to 'Layer N: <background>'. 12 steps, guidance 3, shift 3.86, strength 1, 1024 px."
    return entry
  }

  public static func steps(_ job: MingLayerJob) -> [WorkStep] {
    let converter = job.helpers.appendingPathComponent("ModelConverter")
    let quantizer = job.helpers.appendingPathComponent("QuantI8x")
    let folder = job.models
    let modelF16 = folder.appendingPathComponent(MingLayerOutputs.model("f16"))
    let modelI8x = folder.appendingPathComponent(MingLayerOutputs.model("i8x"))
    let connectorF16 = job.work.appendingPathComponent(MingLayerOutputs.connectorBase + "_f16.ckpt")
    let connector = job.work.appendingPathComponent(MingLayerOutputs.connectorBase + "_i8x.ckpt")
    let designEncoder = folder.appendingPathComponent(MingOutputs.textEncoder)
    let encoder = folder.appendingPathComponent(MingLayerOutputs.textEncoder)

    @Sendable func expect(_ url: URL, from tool: String) throws {
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw FiboError.message("The \(tool) did not write \(url.lastPathComponent).")
      }
    }

    var steps = [WorkStep]()
    steps.append(
      WorkStep("design", "Check the Ming-Image Design files") { _ in
        let missing = missingDesignFiles(in: folder)
        guard missing.isEmpty else {
          throw FiboError.message(
            "Convert Ming-Image Design first (tab Ming-Image): the Layer model uses its text encoder, vision tower and VAE. Missing: \(missing.joined(separator: ", ")).")
        }
      })
    steps.append(
      WorkStep("connector", "Convert the Layer parts of the text encoder") { context in
        try FileManager.default.createDirectory(at: job.work, withIntermediateDirectories: true)
        try Pipeline.removeModelFile(connectorF16)
        try await context.exec(
          converter,
          [
            "--ming-text-encoder", "--ming-connector-only", "-f", job.textEncoderFile.path, "-n",
            MingLayerOutputs.connectorBase, "-o", job.work.path,
          ])
        try expect(connectorF16, from: "converter")
      })
    steps.append(
      WorkStep("quantizeConnector", "Quantize them to 8-bit S") { context in
        try Pipeline.removeModelFile(connector)
        try await context.exec(
          quantizer, ["-i", connectorF16.path, "-m", "ming_image", "-o", connector.path, "--text-encoder"])
        try expect(connector, from: "quantizer")
      })
    steps.append(
      WorkStep("encoder", "Build the Layer text encoder from Design's") { context in
        // A copy of Design's file with the Layer rows in place of Design's. Its tensor data, the language model
        // (18 GB), is a copy too: on APFS a clone that takes no space.
        let manager = FileManager.default
        try Pipeline.removeModelFile(encoder)
        try manager.copyItem(at: designEncoder, to: encoder)
        try manager.copyItem(
          at: URL(fileURLWithPath: designEncoder.path + "-tensordata"),
          to: URL(fileURLWithPath: encoder.path + "-tensordata"))
        let quoted = connector.path.replacingOccurrences(of: "'", with: "''")
        try await context.exec(
          job.sqlite,
          [
            encoder.path,
            "attach '\(quoted)' as layer; begin; delete from main.tensors where name in (select name from layer.tensors); insert into main.tensors select * from layer.tensors; commit; vacuum;",
          ])
      })
    steps.append(
      WorkStep("model", "Convert the model") { context in
        try Pipeline.removeModelFile(modelF16)
        try await context.exec(
          converter, ["-f", job.modelFile.path, "-n", MingLayerOutputs.modelBase, "-o", folder.path])
        try expect(modelF16, from: "converter")
      })
    if job.precision.makesI8x {
      steps.append(
        WorkStep("quantize", "Quantize the model to 8-bit S") { context in
          try Pipeline.removeModelFile(modelI8x)
          try await context.exec(quantizer, ["-i", modelF16.path, "-m", "ming_image", "-o", modelI8x.path])
          try expect(modelI8x, from: "quantizer")
        })
    }
    steps.append(
      WorkStep("register", "Add to the library (custom.json)", alwaysRun: true) { context in
        var entries = [[String: Any]]()
        if job.precision.keepsF16 {
          entries.append(try entry(name: "Ming Image 0.1 Design-Layer", file: MingLayerOutputs.model("f16")))
        }
        if job.precision.makesI8x {
          entries.append(try entry(name: "Ming Image 0.1 Design-Layer (8-bit S)", file: MingLayerOutputs.model("i8x")))
        }
        if let backup = try CustomJSON.register(entries, in: folder) {
          context.log("Backup of custom.json: \(backup.lastPathComponent)")
        }
      })
    steps.append(
      WorkStep("cleanup", "Remove temporary files", alwaysRun: true) { _ in
        try Pipeline.removeModelFile(connectorF16)
        try Pipeline.removeModelFile(connector)
        if !job.precision.keepsF16 {
          for url in [modelF16, URL(fileURLWithPath: modelF16.path + "-tensordata")]
          where FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
          }
        }
      })
    return steps
  }
}
