import Foundation
import XCTest

@testable import FiboCore

final class FiboCoreTests: XCTestCase {
  private var directory: URL!

  override func setUpWithError() throws {
    directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "fibo-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  }
  override func tearDownWithError() throws { try? FileManager.default.removeItem(at: directory) }

  /// Writes a safetensors file with the given tensors (name → bytes), as `U8` vectors.
  private func writeSafetensors(_ name: String, _ tensors: [(String, [UInt8])]) throws -> URL {
    var header = [String: Any]()
    var cursor = 0
    var body = Data()
    for (key, bytes) in tensors {
      header[key] = ["dtype": "U8", "shape": [bytes.count], "data_offsets": [cursor, cursor + bytes.count]]
      cursor += bytes.count
      body.append(contentsOf: bytes)
    }
    header["__metadata__"] = ["format": "pt"]
    var json = try JSONSerialization.data(withJSONObject: header)
    json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
    var length = UInt64(json.count).littleEndian
    var file = Data(bytes: &length, count: 8)
    file.append(json)
    file.append(body)
    let url = directory.appendingPathComponent(name)
    try file.write(to: url)
    return url
  }

  func testMergeKeepsEveryTensorByteForByte() throws {
    let a = try writeSafetensors("a.safetensors", [("x", [1, 2, 3]), ("y", [4, 5])])
    let b = try writeSafetensors("b.safetensors", [("z", [6, 7, 8, 9])])
    let output = directory.appendingPathComponent("merged.safetensors")
    var lastProgress: Int64 = 0
    try Safetensors.merge([a, b], to: output) { written, _ in lastProgress = written }
    XCTAssertEqual(lastProgress, 9)
    let header = try Safetensors.readHeader(output)
    XCTAssertEqual(Set(header.entries.map(\.name)), ["x", "y", "z"])
    let data = try Data(contentsOf: output)
    for (name, expected) in [("x", [1, 2, 3]), ("y", [4, 5]), ("z", [6, 7, 8, 9])] as [(String, [UInt8])] {
      let entry = header.entries.first { $0.name == name }!
      let bytes = [UInt8](data[(header.dataStart + entry.begin)..<(header.dataStart + entry.end)])
      XCTAssertEqual(bytes, expected, name)
    }
    XCTAssertEqual(header.dataStart % 8, 0)
  }

  func testMergeRefusesDuplicateTensors() throws {
    let a = try writeSafetensors("a.safetensors", [("x", [1])])
    let b = try writeSafetensors("b.safetensors", [("x", [2])])
    XCTAssertThrowsError(
      try Safetensors.merge([a, b], to: directory.appendingPathComponent("m.safetensors")))
  }

  func testTruncatedFileIsRefused() throws {
    let a = try writeSafetensors("a.safetensors", [("x", [1, 2, 3, 4])])
    let data = try Data(contentsOf: a)
    try data.dropLast(2).write(to: a)
    XCTAssertThrowsError(try Safetensors.readHeader(a))
  }

  func testCustomJSONAddsAndReplacesByFile() throws {
    let existing = #"[{"name":"Other","file":"other_f16.ckpt","version":"v1"}]"#
    try existing.write(to: directory.appendingPathComponent("custom.json"), atomically: true, encoding: .utf8)
    let entry = try CustomJSON.entry(name: "Fibo", file: "fibo_2_edit_turbo_i8x.ckpt", textEncoder: "te_i8x.ckpt")
    let backup = try CustomJSON.register([entry], in: directory)
    XCTAssertNotNil(backup)
    // Registering again replaces the entry instead of adding a second one.
    let renamed = try CustomJSON.entry(name: "Fibo v2", file: "fibo_2_edit_turbo_i8x.ckpt", textEncoder: "te_i8x.ckpt")
    try CustomJSON.register([renamed], in: directory)
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list.count, 2)
    XCTAssertEqual(list[0]["name"] as? String, "Other")
    XCTAssertEqual(list[1]["name"] as? String, "Fibo v2")
    XCTAssertEqual(list[1]["version"] as? String, "fibo2")
    XCTAssertEqual(list[1]["modifier"] as? String, "kontext")
  }

  func testCustomJSONCreatedWhenMissingAndNotListLeftAlone() throws {
    let entry = try CustomJSON.entry(name: "Fibo", file: "f.ckpt", textEncoder: "t.ckpt")
    XCTAssertNil(try CustomJSON.register([entry], in: directory))
    let url = directory.appendingPathComponent("custom.json")
    try "{}".write(to: url, atomically: true, encoding: .utf8)
    XCTAssertThrowsError(try CustomJSON.register([entry], in: directory))
    XCTAssertEqual(try String(contentsOf: url), "{}")
  }

  func testLocateRequiresExactSize() throws {
    let url = directory.appendingPathComponent("model.safetensors")
    try Data([0]).write(to: url)
    XCTAssertTrue(FiboSources.locate(in: directory).isEmpty)
  }

  func testPrecisionOutputs() {
    XCTAssertTrue(Precision.eightBitS.makesI8x && !Precision.eightBitS.keepsF16)
    XCTAssertTrue(Precision.full.keepsF16 && !Precision.full.makesI8x)
    XCTAssertTrue(Precision.both.keepsF16 && Precision.both.makesI8x)
  }
}

/// Runs the whole pipeline with stand-in helper tools that only write the files the real ones would.
final class PipelineTests: XCTestCase {
  func testPipelineRunsStepsRegistersAndCleansUp() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("fibo-pipe-\(UUID().uuidString)")
    let helpers = root.appendingPathComponent("helpers"), models = root.appendingPathComponent("models")
    let source = root.appendingPathComponent("source")
    for url in [helpers, models, source] { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) }
    defer { try? FileManager.default.removeItem(at: root) }
    // Stand-ins: ModelConverter writes the f16 file named by -n in -o; QuantI8x writes -o.
    try """
    #!/bin/bash
    while [ $# -gt 0 ]; do case "$1" in -n) n="$2"; shift 2;; -o) o="$2"; shift 2;; -f) f="$2"; shift 2;; *) shift;; esac; done
    [ -e "$f" ] || { echo "missing input $f"; exit 3; }
    echo converting; echo data > "$o/${n}_f16.ckpt"
    """.write(to: helpers.appendingPathComponent("ModelConverter"), atomically: true, encoding: .utf8)
    try """
    #!/bin/bash
    while [ $# -gt 0 ]; do case "$1" in -i) i="$2"; shift 2;; -o) o="$2"; shift 2;; *) shift;; esac; done
    [ -e "$i" ] || { echo "missing input $i"; exit 3; }
    echo data > "$o"
    """.write(to: helpers.appendingPathComponent("QuantI8x"), atomically: true, encoding: .utf8)
    for name in ["ModelConverter", "QuantI8x"] {
      try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helpers.appendingPathComponent(name).path)
    }
    func shard(_ name: String, _ key: String) throws -> URL {
      var json = Data(#"{"\#(key)":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#.utf8)
      json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
      var length = UInt64(json.count).littleEndian
      var data = Data(bytes: &length, count: 8); data.append(json); data.append(contentsOf: [1, 2])
      let url = source.appendingPathComponent(name); try data.write(to: url); return url
    }
    let job = Job(
      transformerShards: [try shard("a.safetensors", "x"), try shard("b.safetensors", "y")],
      textEncoderFile: try shard("model.safetensors", "t"), work: models.appendingPathComponent("_fibo-work"),
      models: models, precision: .eightBitS, helpers: helpers)
    let events = EventBox()
    try await Pipeline().run(job) { events.add($0) }
    XCTAssertEqual(events.finished, job.steps)
    let fm = FileManager.default
    XCTAssertTrue(fm.fileExists(atPath: models.appendingPathComponent("fibo_2_edit_turbo_i8x.ckpt").path))
    XCTAssertTrue(fm.fileExists(atPath: models.appendingPathComponent("fibo_2_qwen_3_vl_4b_i8x.ckpt").path))
    XCTAssertFalse(fm.fileExists(atPath: models.appendingPathComponent("fibo_2_edit_turbo_f16.ckpt").path), "f16 intermediates go to the Trash")
    XCTAssertFalse(fm.fileExists(atPath: job.work.appendingPathComponent("fibo_2_edit_turbo.safetensors").path))
    let list = try JSONSerialization.jsonObject(with: Data(contentsOf: models.appendingPathComponent("custom.json"))) as! [[String: Any]]
    XCTAssertEqual(list.map { $0["file"] as? String }, ["fibo_2_edit_turbo_i8x.ckpt"])
    // A second run skips what the state file records, except registering and cleaning.
    let again = EventBox()
    try await Pipeline().run(job) { again.add($0) }
    XCTAssertEqual(again.skipped.count, 5)
  }

  func testMissingHelperAndFailingHelperReportErrors() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("fibo-pipe-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var json = Data(#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#.utf8)
    json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
    var length = UInt64(json.count).littleEndian
    var data = Data(bytes: &length, count: 8); data.append(json); data.append(7)
    let file = root.appendingPathComponent("s.safetensors"); try data.write(to: file)
    let job = Job(transformerShards: [file], textEncoderFile: file, work: root.appendingPathComponent("w"),
      models: root, precision: .full, helpers: root.appendingPathComponent("nope"))
    do { try await Pipeline().run(job) { _ in }; XCTFail("should fail") } catch {
      XCTAssertTrue("\(error)".contains("Missing helper"))
    }
  }
}

private final class EventBox: @unchecked Sendable {
  private let lock = NSLock()
  private var events = [Event]()
  func add(_ event: Event) { lock.withLock { events.append(event) } }
  var finished: [Step] { lock.withLock { events.compactMap { if case .finished(let s) = $0 { s } else { nil } } } }
  var skipped: [Step] { lock.withLock { events.compactMap { if case .skipped(let s, _) = $0 { s } else { nil } } } }
}
