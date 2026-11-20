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
