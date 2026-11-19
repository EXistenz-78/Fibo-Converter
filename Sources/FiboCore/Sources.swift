import Foundation

/// The three files of briaai/Fibo-2.0-edit-turbo the converter needs. Browsers save them flat in
/// Downloads, so they are recognised by name and exact size, wherever they sit under the folder.
public struct RequiredFile: Identifiable, Hashable {
  public var id: String { path }
  /// Path inside the Hugging Face repository.
  public var path: String
  public var size: Int64
  public var name: String { (path as NSString).lastPathComponent }
  public var folder: String { (path as NSString).deletingLastPathComponent }
  public var url: URL {
    URL(string: "https://huggingface.co/briaai/Fibo-2.0-edit-turbo/resolve/main/\(path)?download=true")!
  }
}

public enum FiboSources {
  public static let repositoryURL = URL(string: "https://huggingface.co/briaai/Fibo-2.0-edit-turbo")!
  public static let licenseURL = URL(string: "https://creativecommons.org/licenses/by-nc/4.0/")!

  public static let shard1 = RequiredFile(
    path: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", size: 9_991_176_550)
  public static let shard2 = RequiredFile(
    path: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", size: 6_263_592_868)
  public static let textEncoder = RequiredFile(
    path: "text_encoder/model.safetensors", size: 8_875_719_408)
  public static let all = [shard1, shard2, textEncoder]

  /// Looks for each required file under `folder` (three levels deep). Missing files map to nil.
  public static func locate(in folder: URL) -> [RequiredFile: URL] {
    var found = [RequiredFile: URL]()
    let manager = FileManager.default
    guard
      let enumerator = manager.enumerator(
        at: folder, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants])
    else { return found }
    for case let url as URL in enumerator {
      if enumerator.level > 3 { enumerator.skipDescendants() }
      guard let file = all.first(where: { $0.name == url.lastPathComponent }), found[file] == nil,
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
        values.isRegularFile == true, Int64(values.fileSize ?? -1) == file.size
      else { continue }
      found[file] = url
    }
    return found
  }
}

public enum Precision: String, CaseIterable, Identifiable {
  case eightBitS, full, both
  public var id: String { rawValue }
  public var makesI8x: Bool { self != .full }
  public var keepsF16: Bool { self != .eightBitS }
}

/// Names of the files the converter writes in the models folder.
public enum FiboOutputs {
  public static let modelBase = "fibo_2_edit_turbo"
  public static let textEncoderBase = "fibo_2_qwen_3_vl_4b"
  public static func model(_ suffix: String) -> String { "\(modelBase)_\(suffix).ckpt" }
  public static func textEncoder(_ suffix: String) -> String {
    "\(textEncoderBase)_\(suffix).ckpt"
  }
}
