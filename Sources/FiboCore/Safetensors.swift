import Foundation

public enum FiboError: LocalizedError {
  case message(String)
  public var errorDescription: String? {
    switch self {
    case .message(let text): return text
    }
  }
}

/// One tensor of a safetensors file: where its bytes sit, relative to the start of the data block.
public struct SafetensorsEntry: Equatable {
  public var name: String
  public var dtype: String
  public var shape: [Int]
  public var begin: Int
  public var end: Int
  public var count: Int { end - begin }
}

public struct SafetensorsHeader {
  public var entries: [SafetensorsEntry]
  /// Offset of the data block from the start of the file.
  public var dataStart: Int
  public var dataLength: Int
}

public enum Safetensors {
  public static func readHeader(_ url: URL) throws -> SafetensorsHeader {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
      throw FiboError.message("\(url.lastPathComponent) is not a safetensors file.")
    }
    let length = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
    guard length > 1, length < 200_000_000 else {
      throw FiboError.message("\(url.lastPathComponent) has an invalid header.")
    }
    guard let json = try handle.read(upToCount: Int(length)), json.count == Int(length),
      let object = try JSONSerialization.jsonObject(with: json) as? [String: Any]
    else {
      throw FiboError.message("\(url.lastPathComponent) has an unreadable header.")
    }
    var entries = [SafetensorsEntry]()
    for (name, value) in object where name != "__metadata__" {
      guard let info = value as? [String: Any], let dtype = info["dtype"] as? String,
        let shape = info["shape"] as? [Int], let offsets = info["data_offsets"] as? [Int],
        offsets.count == 2, offsets[0] <= offsets[1]
      else {
        throw FiboError.message("Tensor \(name) in \(url.lastPathComponent) is malformed.")
      }
      entries.append(
        SafetensorsEntry(name: name, dtype: dtype, shape: shape, begin: offsets[0], end: offsets[1]))
    }
    entries.sort { $0.begin < $1.begin }
    let dataStart = 8 + Int(length)
    let fileSize =
      (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0
    let dataLength = entries.last?.end ?? 0
    guard dataStart + dataLength <= fileSize else {
      throw FiboError.message("\(url.lastPathComponent) is truncated: the download is incomplete.")
    }
    return SafetensorsHeader(entries: entries, dataStart: dataStart, dataLength: dataLength)
  }

  /// Joins the tensors of several safetensors files into one file, copying the bytes unchanged.
  /// `progress` receives (bytes written, total bytes). `isCancelled` is polled between chunks.
  public static func merge(
    _ inputs: [URL], to output: URL,
    isCancelled: () -> Bool = { false },
    progress: (Int64, Int64) -> Void = { _, _ in }
  ) throws {
    var sources = [(url: URL, header: SafetensorsHeader)]()
    var seen = Set<String>()
    for url in inputs {
      let header = try readHeader(url)
      for entry in header.entries {
        guard seen.insert(entry.name).inserted else {
          throw FiboError.message("Tensor \(entry.name) appears in more than one file.")
        }
      }
      sources.append((url, header))
    }
    // New layout: every tensor one after the other, in file order.
    var layout = [(source: Int, entry: SafetensorsEntry, begin: Int)]()
    var cursor = 0
    for (index, source) in sources.enumerated() {
      for entry in source.header.entries {
        layout.append((index, entry, cursor))
        cursor += entry.count
      }
    }
    let total = cursor
    var header = [String: Any]()
    header["__metadata__"] = ["format": "pt"]
    for item in layout {
      header[item.entry.name] = [
        "dtype": item.entry.dtype, "shape": item.entry.shape,
        "data_offsets": [item.begin, item.begin + item.entry.count],
      ] as [String: Any]
    }
    var json = try JSONSerialization.data(
      withJSONObject: header, options: [.withoutEscapingSlashes])
    json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))

    let fileManager = FileManager.default
    try? fileManager.removeItem(at: output)
    guard fileManager.createFile(atPath: output.path, contents: nil) else {
      throw FiboError.message("Cannot create \(output.path).")
    }
    let out = try FileHandle(forWritingTo: output)
    defer { try? out.close() }
    var length = UInt64(json.count).littleEndian
    try out.write(contentsOf: Data(bytes: &length, count: 8))
    try out.write(contentsOf: json)

    let handles = try sources.map { try FileHandle(forReadingFrom: $0.url) }
    defer { handles.forEach { try? $0.close() } }
    let chunk = 16 * 1024 * 1024
    var written: Int64 = 0
    for item in layout {
      let handle = handles[item.source]
      try handle.seek(toOffset: UInt64(sources[item.source].header.dataStart + item.entry.begin))
      var remaining = item.entry.count
      while remaining > 0 {
        if isCancelled() { throw CancellationError() }
        guard let data = try handle.read(upToCount: min(chunk, remaining)), !data.isEmpty else {
          throw FiboError.message("Unexpected end of \(sources[item.source].url.lastPathComponent).")
        }
        try out.write(contentsOf: data)
        remaining -= data.count
        written += Int64(data.count)
        progress(written, Int64(total))
      }
    }
    try out.synchronize()
    let size = (try fileManager.attributesOfItem(atPath: output.path)[.size] as? NSNumber)?.intValue
    guard size == 8 + json.count + total else {
      throw FiboError.message("The merged file has the wrong size.")
    }
  }
}
