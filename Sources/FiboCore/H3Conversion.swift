import Foundation

/// MiniMax H3 Image (MiniMax-H3-Image Community License): the three files the conversion needs.
public enum H3Sources {
  public static let repositoryURL = URL(string: "https://huggingface.co/Comfy-Org/MiniMax-H3-Image")!
  public static let licenseURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE")!

  public static let model = RequiredFile(
    repository: "Comfy-Org/MiniMax-H3-Image", path: "diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
    size: 40_234_657_162)
  public static let vae = RequiredFile(
    repository: "Comfy-Org/MiniMax-H3-Image", path: "vae/minimax_h3_image_vae_fp16.safetensors",
    size: 1_168_505_312)
  /// Draw Things' own 8-bit S copy of the Qwen3-VL 32B text encoder (already converted).
  public static let textEncoder = RequiredFile(
    repository: "drawthingsai/MiniMax-H3", path: "qwen_3_vl_32b_50_i8x.ckpt", size: 26_547_669_248)
  public static let all = [model, vae, textEncoder]
}

public enum H3Outputs {
  public static let modelBase = "minimax_h3_image"
  public static let vaeBase = "minimax_h3_image_vae"
  public static func model(_ suffix: String) -> String { "\(modelBase)_\(suffix).ckpt" }
  public static let vae = "minimax_h3_image_vae_f16.ckpt"
  public static let textEncoder = "qwen_3_vl_32b_50_i8x.ckpt"
}

public struct H3Job: Sendable {
  public var modelFile: URL
  public var vaeFile: URL
  public var textEncoderFile: URL
  public var models: URL
  public var work: URL
  public var helpers: URL
  public var precision: Precision

  public init(
    modelFile: URL, vaeFile: URL, textEncoderFile: URL, models: URL, work: URL, helpers: URL,
    precision: Precision
  ) {
    self.modelFile = modelFile
    self.vaeFile = vaeFile
    self.textEncoderFile = textEncoderFile
    self.models = models
    self.work = work
    self.helpers = helpers
    self.precision = precision
  }
}

public enum H3Conversion {
  private static let template = """
    {
      "version": "minimax_h3_image", "modifier": "kontext", "is_consistency_model": true,
      "default_scale": 32, "prefix": "",
      "text_encoder": "qwen_3_vl_32b_50_i8x.ckpt", "autoencoder": "minimax_h3_image_vae_f16.ckpt",
      "objective": {"u": {"condition_scale": 1000}},
      "noise_discretization": {"rf": {"_0": {"sigma_min": 0, "sigma_max": 1, "condition_scale": 1000}}}
    }
    """

  public static func entry(name: String, file: String) throws -> [String: Any] {
    var entry = try JSONSerialization.jsonObject(with: Data(template.utf8)) as? [String: Any] ?? [:]
    entry["name"] = name
    entry["file"] = file
    return entry
  }

  public static func steps(_ job: H3Job) -> [WorkStep] {
    let converter = job.helpers.appendingPathComponent("ModelConverter")
    let quantizer = job.helpers.appendingPathComponent("QuantI8x")
    let modelF16 = job.models.appendingPathComponent(H3Outputs.model("f16"))
    let modelI8x = job.models.appendingPathComponent(H3Outputs.model("i8x"))
    let vae = job.models.appendingPathComponent(H3Outputs.vae)
    let encoder = job.models.appendingPathComponent(H3Outputs.textEncoder)
    var steps = [WorkStep]()

    steps.append(
      WorkStep("model", "Convert the model") { context in
        try Pipeline.removeModelFile(modelF16)
        try await context.exec(
          converter,
          ["-f", job.modelFile.path, "-n", H3Outputs.modelBase, "-o", job.models.path])
        guard FileManager.default.fileExists(atPath: modelF16.path) else {
          throw FiboError.message("The converter did not write \(modelF16.lastPathComponent).")
        }
      })
    steps.append(
      WorkStep("vae", "Convert the VAE") { context in
        try Pipeline.removeModelFile(vae)
        try await context.exec(
          converter,
          ["--h3-image-vae", "-f", job.vaeFile.path, "-n", H3Outputs.vaeBase, "-o", job.models.path])
        guard FileManager.default.fileExists(atPath: vae.path) else {
          throw FiboError.message("The converter did not write \(vae.lastPathComponent).")
        }
      })
    steps.append(
      WorkStep("encoder", "Put the text encoder in the models folder") { context in
        if job.textEncoderFile.standardizedFileURL.path == encoder.standardizedFileURL.path { return }
        try? FileManager.default.removeItem(at: encoder)
        context.log("Copying \(job.textEncoderFile.lastPathComponent) (26.5 GB)…")
        try FileManager.default.copyItem(at: job.textEncoderFile, to: encoder)
      })
    if job.precision.makesI8x {
      steps.append(
        WorkStep("quantize", "Quantize the model to 8-bit S") { context in
          try Pipeline.removeModelFile(modelI8x)
          try await context.exec(
            quantizer,
            ["-i", modelF16.path, "-m", "minimax_h3_image", "-o", modelI8x.path])
          guard FileManager.default.fileExists(atPath: modelI8x.path) else {
            throw FiboError.message("The quantizer did not write \(modelI8x.lastPathComponent).")
          }
        })
    }
    steps.append(
      WorkStep("register", "Add to the library (custom.json)", alwaysRun: true) { context in
        var entries = [[String: Any]]()
        if job.precision.keepsF16 {
          entries.append(try entry(name: "MiniMax H3 Image", file: H3Outputs.model("f16")))
        }
        if job.precision.makesI8x {
          entries.append(try entry(name: "MiniMax H3 Image (8-bit S)", file: H3Outputs.model("i8x")))
        }
        if let backup = try CustomJSON.register(entries, in: job.models) {
          context.log("Backup of custom.json: \(backup.lastPathComponent)")
        }
      })
    steps.append(
      WorkStep("cleanup", "Remove temporary files", alwaysRun: true) { _ in
        if !job.precision.keepsF16 {
          // Only an input of the quantizer: the Trash keeps it recoverable.
          for url in [modelF16, URL(fileURLWithPath: modelF16.path + "-tensordata")]
          where FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
          }
        }
      })
    return steps
  }
}
