import XCTest

@testable import FiboCore

private final class Events: @unchecked Sendable {
  private let lock = NSLock()
  private var all = [WorkEvent]()
  func add(_ event: WorkEvent) { lock.withLock { all.append(event) } }
  var finished: [String] { lock.withLock { all.compactMap { if case .finished(let s) = $0 { s } else { nil } } } }
  var skipped: [String] { lock.withLock { all.compactMap { if case .skipped(let s) = $0 { s } else { nil } } } }
}

final class ConversionTests: XCTestCase {
  var root: URL!
  var helpers: URL!
  var models: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory.appendingPathComponent("conv-\(UUID().uuidString)")
    helpers = root.appendingPathComponent("helpers")
    models = root.appendingPathComponent("models")
    for url in [helpers!, models!] { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) }
  }
  override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

  func script(_ name: String, _ body: String, in folder: URL? = nil) throws -> URL {
    let url = (folder ?? helpers).appendingPathComponent(name)
    try ("#!/bin/bash\n" + body).write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    return url
  }
  func input(_ name: String) throws -> URL {
    let url = root.appendingPathComponent(name)
    try Data([1, 2, 3]).write(to: url)
    return url
  }

  // MARK: Specification

  func testParseSpecificationTakesTheLastObject() throws {
    let output = "progress 10%\nprogress {x}\n{\n  \"file\" : \"a_f16.ckpt\",\n  \"version\" : \"flux1\"\n}\n"
    let spec = try GenericConversion.parseSpecification(output)
    XCTAssertEqual(spec["file"] as? String, "a_f16.ckpt")
    XCTAssertThrowsError(try GenericConversion.parseSpecification("nothing here"))
  }

  // MARK: Generic

  private func stubConverter() throws {
    _ = try script(
      "ModelConverter",
      """
      while [ $# -gt 0 ]; do case "$1" in -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; -f) f="$2"; shift 2;; *) shift;; esac; done
      [ -e "$f" ] || { echo "missing input $f"; exit 3; }
      echo "reading $f"; echo data > "$o/${n}_f16.ckpt"
      printf '{\\n  "file" : "%s_f16.ckpt",\\n  "version" : "flux1",\\n  "name" : "%s"\\n}\\n' "$n" "$n"
      """)
  }

  func testGenericConvertsAndRegisters() async throws {
    try stubConverter()
    let job = GenericJob(
      input: try input("my.safetensors"), name: "my_model", models: models,
      work: models.appendingPathComponent("_w"), helpers: helpers)
    let events = Events()
    try await WorkRunner().run(GenericConversion.steps(job), work: job.work) { events.add($0) }
    XCTAssertEqual(events.finished, ["import", "register"])
    XCTAssertTrue(FileManager.default.fileExists(atPath: models.appendingPathComponent("my_model_f16.ckpt").path))
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list.first?["file"] as? String, "my_model_f16.ckpt")
    XCTAssertEqual(list.first?["version"] as? String, "flux1")
    // Again: the import is skipped (its specification was saved), registering runs again.
    let again = Events()
    try await WorkRunner().run(GenericConversion.steps(job), work: job.work) { again.add($0) }
    XCTAssertEqual(again.skipped, ["import"])
    let list2 = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list2.count, 1)
  }

  func testGenericPreparesWithPythonScript() async throws {
    try stubConverter()
    // A stand-in "python": succeeds at `-c import numpy`, and otherwise writes <dir>/prepared-DT.safetensors.
    let python = try script(
      "python3",
      """
      if [ "$1" = "-c" ]; then exit 0; fi
      echo prepping "$2"; echo data > "$3/prepared-DT.safetensors"
      """, in: root)
    let prep = root.appendingPathComponent("prep-model.py")
    try "x".write(to: prep, atomically: true, encoding: .utf8)
    var job = GenericJob(
      input: try input("my.safetensors"), name: "m", models: models,
      work: models.appendingPathComponent("_w"), helpers: helpers, prepScript: prep,
      pythonCandidates: [python])
    job.convertTextEncoders = true
    let events = Events()
    try await WorkRunner().run(GenericConversion.steps(job), work: job.work) { events.add($0) }
    XCTAssertEqual(events.finished, ["prepare", "import", "register"])
  }

  func testGenericWithoutNumpyExplainsWhat() async throws {
    try stubConverter()
    let prep = root.appendingPathComponent("prep-model.py")
    try "x".write(to: prep, atomically: true, encoding: .utf8)
    let job = GenericJob(
      input: try input("my.safetensors"), name: "m", models: models,
      work: models.appendingPathComponent("_w"), helpers: helpers, prepScript: prep,
      pythonCandidates: [root.appendingPathComponent("no-python")])
    do { try await WorkRunner().run(GenericConversion.steps(job), work: job.work) { _ in }; XCTFail() } catch {
      XCTAssertTrue("\(error)".contains("numpy"))
    }
  }

  func testConverterFailureCarriesItsLastLines() async throws {
    _ = try script("ModelConverter", "echo boom; exit 4")
    let job = GenericJob(
      input: try input("my.safetensors"), name: "m", models: models,
      work: models.appendingPathComponent("_w"), helpers: helpers)
    do { try await WorkRunner().run(GenericConversion.steps(job), work: job.work) { _ in }; XCTFail() } catch {
      XCTAssertTrue("\(error)".contains("boom") && "\(error)".contains("code 4"))
    }
  }

  // MARK: H3

  func testH3ConvertsQuantizesAndRegisters() async throws {
    _ = try script(
      "ModelConverter",
      """
      v=0; while [ $# -gt 0 ]; do case "$1" in --h3-image-vae) v=1; shift;; -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; -f) f="$2"; shift 2;; *) shift;; esac; done
      [ -e "$f" ] || { echo "missing input $f"; exit 3; }
      echo data > "$o/${n}_f16.ckpt"
      """)
    _ = try script(
      "QuantI8x",
      """
      while [ $# -gt 0 ]; do case "$1" in -i) i="$2"; shift 2;; -m) m="$2"; shift 2;; -o) o="$2"; shift 2;; *) shift;; esac; done
      [ -e "$i" ] || { echo "missing input $i"; exit 3; }
      [ "$m" = "minimax_h3_image" ] || { echo "wrong version $m"; exit 5; }
      echo data > "$o"
      """)
    let downloads = root.appendingPathComponent("dl")
    try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
    let encoder = downloads.appendingPathComponent("qwen_3_vl_32b_50_i8x.ckpt")
    try Data([9]).write(to: encoder)
    let job = H3Job(
      modelFile: try input("model.safetensors"), vaeFile: try input("vae.safetensors"),
      textEncoderFile: encoder, models: models, work: models.appendingPathComponent("_w"),
      helpers: helpers, precision: .eightBitS)
    let events = Events()
    try await WorkRunner().run(H3Conversion.steps(job), work: job.work) { events.add($0) }
    XCTAssertEqual(events.finished, ["model", "vae", "encoder", "quantize", "register", "cleanup"])
    let fm = FileManager.default
    for file in ["minimax_h3_image_i8x.ckpt", "minimax_h3_image_vae_f16.ckpt", "qwen_3_vl_32b_50_i8x.ckpt"] {
      XCTAssertTrue(fm.fileExists(atPath: models.appendingPathComponent(file).path), file)
    }
    XCTAssertFalse(fm.fileExists(atPath: models.appendingPathComponent("minimax_h3_image_f16.ckpt").path))
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list.map { $0["file"] as? String }, ["minimax_h3_image_i8x.ckpt"])
    XCTAssertEqual(list.first?["version"] as? String, "minimax_h3_image")
    XCTAssertEqual(list.first?["name"] as? String, "MiniMax H3 Image (8-bit S)")
  }

  func testRequiredFilesFromOtherRepositories() {
    XCTAssertEqual(
      H3Sources.vae.url.absoluteString,
      "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors?download=true")
    XCTAssertEqual(H3Sources.textEncoder.name, "qwen_3_vl_32b_50_i8x.ckpt")
    XCTAssertEqual(Set(H3Sources.all.map(\.id)).count, 3)
  }

  // MARK: Ming-Image

  func testMingConvertsQuantizesAndRegisters() async throws {
    _ = try script(
      "ModelConverter",
      """
      t=0; while [ $# -gt 0 ]; do case "$1" in --ming-text-encoder) t=1; shift;; -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; -f) f="$2"; shift 2;; *) shift;; esac; done
      [ -e "$f" ] || { echo "missing input $f"; exit 3; }
      echo data > "$o/${n}_f16.ckpt"
      [ $t = 1 ] && echo '{}' > "$o/${n}_tokenizer.json"
      exit 0
      """)
    _ = try script(
      "QuantI8x",
      """
      while [ $# -gt 0 ]; do case "$1" in -i) i="$2"; shift 2;; -m) m="$2"; shift 2;; -o) o="$2"; shift 2;; *) shift;; esac; done
      [ -e "$i" ] || { echo "missing input $i"; exit 3; }
      [ "$m" = "ming_image" ] || { echo "wrong version $m"; exit 5; }
      echo data > "$o"
      """)
    let job = MingJob(
      modelFile: try input("model.safetensors"), textEncoderFile: try input("encoder.safetensors"),
      vaeFile: try input("vae.safetensors"), models: models, work: models.appendingPathComponent("_w"),
      helpers: helpers, precision: .eightBitS)
    let events = Events()
    try await WorkRunner().run(MingConversion.steps(job), work: job.work) { events.add($0) }
    XCTAssertEqual(
      events.finished, ["encoder", "quantizeEncoder", "vision", "vae", "model", "quantize", "register", "cleanup"])
    let fm = FileManager.default
    for file in [
      "ming_image_0.1_design_i8x.ckpt", "ming_image_ling_mini_2.0_i8x.ckpt", "ming_image_ling_mini_2.0_tokenizer.json",
      "ming_image_ling_mini_2.0_vision_f16.ckpt", "ming_image_vae_f16.ckpt",
    ] {
      XCTAssertTrue(fm.fileExists(atPath: models.appendingPathComponent(file).path), file)
    }
    for file in ["ming_image_0.1_design_f16.ckpt", "ming_image_ling_mini_2.0_f16.ckpt"] {
      XCTAssertFalse(fm.fileExists(atPath: models.appendingPathComponent(file).path), file)
    }
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list.map { $0["file"] as? String }, ["ming_image_0.1_design_i8x.ckpt"])
    let entry = try XCTUnwrap(list.first)
    XCTAssertEqual(entry["version"] as? String, "ming_image")
    XCTAssertEqual(entry["text_encoder"] as? String, "ming_image_ling_mini_2.0_i8x.ckpt")
    XCTAssertEqual(entry["clip_encoder"] as? String, "ming_image_ling_mini_2.0_vision_f16.ckpt")
    XCTAssertEqual(entry["autoencoder"] as? String, "ming_image_vae_f16.ckpt")
    // The activation scaling that keeps large images with long prompts from overflowing to NaN.
    let mmdit = try XCTUnwrap(entry["mmdit"] as? [String: Any])
    let qk = try XCTUnwrap(mmdit["activation_qk_scaling"] as? [String: Int])
    XCTAssertEqual(qk.count, 34)
    XCTAssertEqual(qk["0"], 1)
    XCTAssertEqual(qk["2"], 32)
    XCTAssertEqual((mmdit["activation_ffn_scaling"] as? [String: Int])?["1"], 2)
  }

  func testMingFilesComeFromComfyOrg() {
    XCTAssertEqual(
      MingSources.textEncoder.url.absoluteString,
      "https://huggingface.co/Comfy-Org/Ming-Image/resolve/main/text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors?download=true")
    XCTAssertEqual(MingSources.model.size, 12_309_858_664)
    XCTAssertEqual(Set(MingSources.all.map(\.id)).count, 3)
  }
}

extension ConversionTests {
  // MARK: Ming-Image Design-Layer

  func testMingLayerNeedsDesignFirst() async throws {
    XCTAssertEqual(MingLayerConversion.missingDesignFiles(in: models).count, 5)
    let job = MingLayerJob(
      modelFile: try input("layer.safetensors"), textEncoderFile: try input("layer-encoder.safetensors"),
      models: models, work: models.appendingPathComponent("_w"), helpers: helpers, precision: .eightBitS)
    do {
      try await WorkRunner().run(MingLayerConversion.steps(job), work: job.work) { _ in }
      XCTFail("expected a failure")
    } catch {
      XCTAssertTrue("\(error)".contains("Convert Ming-Image Design first"), "\(error)")
    }
  }

  func testMingLayerBuildsTheEncoderFromDesignsAndRegisters() async throws {
    for file in MingLayerOutputs.designFiles {
      try Data(file.utf8).write(to: models.appendingPathComponent(file))
    }
    _ = try script(
      "ModelConverter",
      """
      c=0; while [ $# -gt 0 ]; do case "$1" in --ming-connector-only) c=1; shift;; -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; -f) f="$2"; shift 2;; *) shift;; esac; done
      [ -e "$f" ] || { echo "missing input $f"; exit 3; }
      case "$n" in *_connector) [ $c = 1 ] || { echo "connector without --ming-connector-only"; exit 4; };; esac
      echo data > "$o/${n}_f16.ckpt"
      """)
    _ = try script(
      "QuantI8x",
      """
      while [ $# -gt 0 ]; do case "$1" in -i) i="$2"; shift 2;; -m) m="$2"; shift 2;; -o) o="$2"; shift 2;; *) shift;; esac; done
      [ -e "$i" ] || { echo "missing input $i"; exit 3; }
      [ "$m" = "ming_image" ] || { echo "wrong version $m"; exit 5; }
      echo data > "$o"
      """)
    let sqlite = try script("sqlite3", "echo \"$2\" > \"$1.sql\"\n", in: root)
    let job = MingLayerJob(
      modelFile: try input("layer.safetensors"), textEncoderFile: try input("layer-encoder.safetensors"),
      models: models, work: models.appendingPathComponent("_w"), helpers: helpers, precision: .eightBitS,
      sqlite: sqlite)
    let events = Events()
    try await WorkRunner().run(MingLayerConversion.steps(job), work: job.work) { events.add($0) }
    XCTAssertEqual(
      events.finished,
      ["design", "connector", "quantizeConnector", "encoder", "model", "quantize", "register", "cleanup"])
    let fm = FileManager.default
    let encoder = models.appendingPathComponent("ming_image_ling_mini_2.0_layer_i8x.ckpt")
    // A copy of Design's text encoder (and of its tensor data), with the Layer rows put in by sqlite3.
    XCTAssertEqual(try String(contentsOf: encoder, encoding: .utf8), "ming_image_ling_mini_2.0_i8x.ckpt")
    XCTAssertEqual(
      try String(contentsOf: URL(fileURLWithPath: encoder.path + "-tensordata"), encoding: .utf8),
      "ming_image_ling_mini_2.0_i8x.ckpt-tensordata")
    let sql = try String(contentsOf: URL(fileURLWithPath: encoder.path + ".sql"), encoding: .utf8)
    XCTAssertTrue(sql.contains("ming_image_ling_mini_2.0_layer_connector_i8x.ckpt' as layer"), sql)
    XCTAssertTrue(sql.contains("delete from main.tensors where name in (select name from layer.tensors)"), sql)
    XCTAssertTrue(fm.fileExists(atPath: models.appendingPathComponent("ming_image_0.1_design_layer_i8x.ckpt").path))
    XCTAssertFalse(fm.fileExists(atPath: models.appendingPathComponent("ming_image_0.1_design_layer_f16.ckpt").path))
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    let entry = try XCTUnwrap(list.first { $0["file"] as? String == "ming_image_0.1_design_layer_i8x.ckpt" })
    XCTAssertEqual(entry["name"] as? String, "Ming Image 0.1 Design-Layer (8-bit S)")
    XCTAssertEqual(entry["text_encoder"] as? String, "ming_image_ling_mini_2.0_layer_i8x.ckpt")
    XCTAssertEqual(entry["clip_encoder"] as? String, "ming_image_ling_mini_2.0_vision_f16.ckpt")
    XCTAssertEqual(entry["is_consistency_model"] as? Bool, false)
    let mmdit = try XCTUnwrap(entry["mmdit"] as? [String: Any])
    XCTAssertEqual((mmdit["activation_qk_scaling"] as? [String: Int])?["2"], 32)
    XCTAssertEqual((mmdit["activation_proj_scaling"] as? [String: Int])?["0"], 2)
  }

  func testMingLayerFilesComeFromComfyOrg() {
    XCTAssertEqual(
      MingLayerSources.textEncoder.url.absoluteString,
      "https://huggingface.co/Comfy-Org/Ming-Image/resolve/main/text_encoders/ming_image_0.1_ling_mini_2.0_layer_bf16.safetensors?download=true")
    XCTAssertEqual(MingLayerSources.model.size, 12_309_874_192)
  }
}

final class PrepFetchTests: XCTestCase {
  func testPrepIsToldNotToFetchUnlessAllowed() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("prep-\(UUID().uuidString)")
    let models = root.appendingPathComponent("models"), helpers = root.appendingPathComponent("h")
    for url in [models, helpers] { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) }
    defer { try? FileManager.default.removeItem(at: root) }
    func script(_ name: String, _ body: String, in folder: URL) throws -> URL {
      let url = folder.appendingPathComponent(name)
      try ("#!/bin/bash\n" + body).write(to: url, atomically: true, encoding: .utf8)
      try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
      return url
    }
    _ = try script("ModelConverter", """
      while [ $# -gt 0 ]; do case "$1" in -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; *) shift;; esac; done
      echo data > "$o/${n}_f16.ckpt"
      printf '{\\n  "file" : "%s_f16.ckpt"\\n}\\n' "$n"
      """, in: helpers)
    let python = try script("python3", """
      if [ "$1" = "-c" ]; then exit 0; fi
      echo "ARGS: $*"; echo data > "$3/x-DT.safetensors"
      """, in: root)
    let prep = root.appendingPathComponent("prep-model.py")
    try "x".write(to: prep, atomically: true, encoding: .utf8)
    let input = root.appendingPathComponent("in.safetensors"); try Data([1]).write(to: input)
    for allow in [false, true] {
      let log = LogBox()
      let job = GenericJob(
        input: input, name: "m\(allow)", models: models, work: root.appendingPathComponent("w\(allow)"),
        helpers: helpers, prepScript: prep, pythonCandidates: [python], allowFetch: allow)
      try await WorkRunner().run(GenericConversion.steps(job), work: job.work) {
        if case .log(let line) = $0 { log.add(line) }
      }
      XCTAssertEqual(log.lines.contains { $0.hasPrefix("ARGS:") && $0.contains("--no-fetch") }, !allow)
    }
  }
}

private final class LogBox: @unchecked Sendable {
  private let lock = NSLock()
  private var all = [String]()
  func add(_ line: String) { lock.withLock { all.append(line) } }
  var lines: [String] { lock.withLock { all } }
}

/// The Ming-Image conversion with the real tools and weights. Skipped unless MING_REAL_SOURCES (the folder with the
/// three Comfy-Org files), MING_REAL_HELPERS (ModelConverter and QuantI8x) and MING_REAL_MODELS (an empty folder) are set.
final class MingRealConversionTests: XCTestCase {
  func testRealConversion() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let sources = env["MING_REAL_SOURCES"], let helpers = env["MING_REAL_HELPERS"],
      let models = env["MING_REAL_MODELS"]
    else { throw XCTSkip("Set MING_REAL_SOURCES, MING_REAL_HELPERS and MING_REAL_MODELS to run it.") }
    let located = FiboSources.locate(MingSources.all, in: URL(fileURLWithPath: sources))
    let model = try XCTUnwrap(located[MingSources.model])
    let encoder = try XCTUnwrap(located[MingSources.textEncoder])
    let vae = try XCTUnwrap(located[MingSources.vae])
    let folder = URL(fileURLWithPath: models)
    let job = MingJob(
      modelFile: model, textEncoderFile: encoder, vaeFile: vae, models: folder,
      work: folder.appendingPathComponent("_ming-work"), helpers: URL(fileURLWithPath: helpers), precision: .eightBitS)
    try await WorkRunner().run(MingConversion.steps(job), work: job.work) { event in print(event) }
  }
}

/// The Ming-Image Design-Layer conversion with the real tools and weights. Skipped unless MING_LAYER_REAL_MODEL (the
/// Layer DiT file), MING_LAYER_REAL_ENCODER (the Layer text encoder, or a file with only the parts that differ),
/// MING_REAL_HELPERS and MING_LAYER_REAL_MODELS (a folder with Ming-Image Design already converted) are set.
final class MingLayerRealConversionTests: XCTestCase {
  func testRealLayerConversion() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let model = env["MING_LAYER_REAL_MODEL"], let encoder = env["MING_LAYER_REAL_ENCODER"],
      let helpers = env["MING_REAL_HELPERS"], let models = env["MING_LAYER_REAL_MODELS"]
    else { throw XCTSkip("Set MING_LAYER_REAL_MODEL, MING_LAYER_REAL_ENCODER, MING_REAL_HELPERS and MING_LAYER_REAL_MODELS.") }
    let folder = URL(fileURLWithPath: models)
    let job = MingLayerJob(
      modelFile: URL(fileURLWithPath: model), textEncoderFile: URL(fileURLWithPath: encoder), models: folder,
      work: folder.appendingPathComponent("_ming-layer-work"), helpers: URL(fileURLWithPath: helpers),
      precision: .eightBitS)
    try await WorkRunner().run(MingLayerConversion.steps(job), work: job.work) { event in print(event) }
  }
}
