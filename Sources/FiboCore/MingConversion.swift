import Foundation

/// Ming-Image 0.1 Design (inclusionAI, MIT): the three ComfyUI files of Comfy-Org/Ming-Image the conversion needs.
public enum MingSources {
  public static let repositoryURL = URL(string: "https://huggingface.co/inclusionAI/Ming-Image-0.1-Design")!
  public static let licenseURL = URL(string: "https://opensource.org/license/mit")!

  public static let model = RequiredFile(
    repository: "Comfy-Org/Ming-Image", path: "diffusion_models/ming_image_0.1_design_bf16.safetensors",
    size: 12_309_858_664)
  /// Ling MoE language model, connector, projections, vision tower and tokenizer, all in one file.
  public static let textEncoder = RequiredFile(
    repository: "Comfy-Org/Ming-Image", path: "text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors",
    size: 36_677_768_245)
  public static let vae = RequiredFile(
    repository: "Comfy-Org/Ming-Image", path: "vae/ming_image_vae_bf16.safetensors", size: 253_816_696)
  public static let all = [model, textEncoder, vae]
}

/// The files the conversion writes in the models folder: the names the engine fork looks for.
public enum MingOutputs {
  public static let modelBase = "ming_image_0.1_design"
  public static func model(_ suffix: String) -> String { "\(modelBase)_\(suffix).ckpt" }
  public static let textEncoderBase = "ming_image_ling_mini_2.0"
  public static let textEncoderF16 = "ming_image_ling_mini_2.0_f16.ckpt"
  public static let textEncoder = "ming_image_ling_mini_2.0_i8x.ckpt"
  /// Written by the converter next to the text encoder; the engine reads it by this name.
  public static let tokenizer = "ming_image_ling_mini_2.0_tokenizer.json"
  public static let visionBase = "ming_image_ling_mini_2.0_vision"
  public static let vision = "ming_image_ling_mini_2.0_vision_f16.ckpt"
  public static let vaeBase = "ming_image_vae"
  public static let vae = "ming_image_vae_f16.ckpt"
}

public struct MingJob: Sendable {
  public var modelFile: URL
  public var textEncoderFile: URL
  public var vaeFile: URL
  public var models: URL
  public var work: URL
  public var helpers: URL
  public var precision: Precision

  public init(
    modelFile: URL, textEncoderFile: URL, vaeFile: URL, models: URL, work: URL, helpers: URL,
    precision: Precision
  ) {
    self.modelFile = modelFile
    self.textEncoderFile = textEncoderFile
    self.vaeFile = vaeFile
    self.models = models
    self.work = work
    self.helpers = helpers
    self.precision = precision
  }
}

public enum MingConversion {
  private static let template = """
    {
      "version": "ming_image", "modifier": "kontext", "is_consistency_model": true,
      "default_scale": 16, "prefix": "", "hires_fix_scale": 24,
      "high_precision_autoencoder": false, "upcast_attention": false,
      "text_encoder": "ming_image_ling_mini_2.0_i8x.ckpt", "autoencoder": "ming_image_vae_f16.ckpt",
      "clip_encoder": "ming_image_ling_mini_2.0_vision_f16.ckpt",
      "objective": {"u": {"condition_scale": 1000}},
      "noise_discretization": {"rf": {"_0": {"sigma_min": 0, "sigma_max": 1, "condition_scale": 1000}}},
      "note": "Ming-Image 0.1 Design by inclusionAI (MIT): graphics, posters, UI and text, transparent (RGBA) output, image editing. 12 steps, no guidance."
    }
    """

  /// FP16 scaling of the activations, as Draw Things' Z-Image Base: 34 blocks (2 context refiners, 2 noise refiners,
  /// 30 layers). Without it the DiT overflows to NaN (a black image) once image and text tokens grow, e.g. above
  /// 1216 px with a long prompt.
  static func activationScaling(first: Int, rest: Int) -> [String: Int] {
    Dictionary(uniqueKeysWithValues: (0..<34).map { (String($0), $0 < 2 ? first : rest) })
  }

  public static func entry(name: String, file: String) throws -> [String: Any] {
    var entry = try JSONSerialization.jsonObject(with: Data(template.utf8)) as? [String: Any] ?? [:]
    entry["name"] = name
    entry["file"] = file
    entry["mmdit"] = [
      "qk_norm": true, "dual_attention_layers": [Int](),
      "activation_qk_scaling": activationScaling(first: 1, rest: 32),
      "activation_proj_scaling": activationScaling(first: 2, rest: 1),
      "activation_ffn_proj_up_scaling": activationScaling(first: 1, rest: 32),
      "activation_ffn_scaling": activationScaling(first: 2, rest: 1),
    ] as [String: Any]
    return entry
  }

  public static func steps(_ job: MingJob) -> [WorkStep] {
    let converter = job.helpers.appendingPathComponent("ModelConverter")
    let quantizer = job.helpers.appendingPathComponent("QuantI8x")
    let folder = job.models
    let modelF16 = folder.appendingPathComponent(MingOutputs.model("f16"))
    let modelI8x = folder.appendingPathComponent(MingOutputs.model("i8x"))
    let encoderF16 = folder.appendingPathComponent(MingOutputs.textEncoderF16)
    let encoder = folder.appendingPathComponent(MingOutputs.textEncoder)
    let tokenizer = folder.appendingPathComponent(MingOutputs.tokenizer)
    let vision = folder.appendingPathComponent(MingOutputs.vision)
    let vae = folder.appendingPathComponent(MingOutputs.vae)

    @Sendable func expect(_ url: URL, from tool: String) throws {
      guard FileManager.default.fileExists(atPath: url.path) else {
        throw FiboError.message("The \(tool) did not write \(url.lastPathComponent).")
      }
    }

    var steps = [WorkStep]()
    // The text encoder first: its full-precision copy (35 GB) is removed as soon as the 8-bit S one is written,
    // before the model adds its own files.
    steps.append(
      WorkStep("encoder", "Convert the text encoder") { context in
        try Pipeline.removeModelFile(encoderF16)
        try await context.exec(
          converter,
          ["--ming-text-encoder", "-f", job.textEncoderFile.path, "-n", MingOutputs.textEncoderBase, "-o", folder.path])
        try expect(encoderF16, from: "converter")
        try expect(tokenizer, from: "converter")
      })
    steps.append(
      WorkStep("quantizeEncoder", "Quantize the text encoder to 8-bit S") { context in
        try Pipeline.removeModelFile(encoder)
        try await context.exec(
          quantizer, ["-i", encoderF16.path, "-m", "ming_image", "-o", encoder.path, "--text-encoder"])
        try expect(encoder, from: "quantizer")
        // Only an input of the quantizer, and too big to keep in the Trash on the models drive.
        try Pipeline.removeModelFile(encoderF16)
      })
    steps.append(
      WorkStep("vision", "Convert the vision tower (for editing)") { context in
        try Pipeline.removeModelFile(vision)
        try await context.exec(
          converter, ["--ming-vision", "-f", job.textEncoderFile.path, "-n", MingOutputs.visionBase, "-o", folder.path])
        try expect(vision, from: "converter")
      })
    steps.append(
      WorkStep("vae", "Convert the VAE") { context in
        try Pipeline.removeModelFile(vae)
        try await context.exec(
          converter, ["--ming-vae", "-f", job.vaeFile.path, "-n", MingOutputs.vaeBase, "-o", folder.path])
        try expect(vae, from: "converter")
      })
    steps.append(
      WorkStep("model", "Convert the model") { context in
        try Pipeline.removeModelFile(modelF16)
        try await context.exec(converter, ["-f", job.modelFile.path, "-n", MingOutputs.modelBase, "-o", folder.path])
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
          entries.append(try entry(name: "Ming Image 0.1 Design", file: MingOutputs.model("f16")))
        }
        if job.precision.makesI8x {
          entries.append(try entry(name: "Ming Image 0.1 Design (8-bit S)", file: MingOutputs.model("i8x")))
        }
        if let backup = try CustomJSON.register(entries, in: folder) {
          context.log("Backup of custom.json: \(backup.lastPathComponent)")
        }
      })
    steps.append(
      WorkStep("cleanup", "Remove temporary files", alwaysRun: true) { _ in
        try Pipeline.removeModelFile(encoderF16)
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
