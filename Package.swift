// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "FiboConverter",
  platforms: [.macOS(.v14)],
  products: [.executable(name: "FiboConverter", targets: ["FiboConverter"])],
  targets: [
    .target(name: "FiboCore"),
    .executableTarget(name: "FiboConverter", dependencies: ["FiboCore"]),
    .testTarget(name: "FiboCoreTests", dependencies: ["FiboCore"]),
  ]
)
