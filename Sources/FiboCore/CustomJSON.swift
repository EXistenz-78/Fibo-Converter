import Foundation

/// Registers the converted files in the models folder's custom.json, as Draw Things reads it.
public enum CustomJSON {
  private static let template = """
    {
      "default_scale": 16,
      "version": "fibo2",
      "objective": {"u": {"condition_scale": 1000}},
      "prefix": "",
      "modifier": "kontext",
      "noise_discretization": {"rf": {"_0": {"sigma_min": 0, "condition_scale": 1000, "sigma_max": 1}}},
      "upcast_attention": false,
      "autoencoder": "flux_2_vae_f16.ckpt",
      "hires_fix_scale": 24,
      "high_precision_autoencoder": true
    }
    """

  public static func entry(name: String, file: String, textEncoder: String) throws -> [String: Any]
  {
    var entry =
      try JSONSerialization.jsonObject(with: Data(template.utf8)) as? [String: Any] ?? [:]
    entry["name"] = name
    entry["file"] = file
    entry["text_encoder"] = textEncoder
    return entry
  }

  /// Adds or replaces entries (matched by `file`). Backs custom.json up first, once per call.
  /// Returns the backup URL when a file existed.
  @discardableResult
  public static func register(_ entries: [[String: Any]], in folder: URL) throws -> URL? {
    let url = folder.appendingPathComponent("custom.json")
    let manager = FileManager.default
    var existing = [[String: Any]]()
    var backup: URL?
    if manager.fileExists(atPath: url.path) {
      guard let list = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]]
      else { throw FiboError.message("custom.json is not a list of models; left untouched.") }
      existing = list
      let copy = folder.appendingPathComponent("custom.json.bak-fibo-converter")
      try? manager.removeItem(at: copy)
      try manager.copyItem(at: url, to: copy)
      backup = copy
    }
    for entry in entries {
      let file = entry["file"] as? String
      if let index = existing.firstIndex(where: { ($0["file"] as? String) == file }) {
        existing[index] = entry
      } else {
        existing.append(entry)
      }
    }
    let data = try JSONSerialization.data(
      withJSONObject: existing, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: url, options: .atomic)
    return backup
  }
}
