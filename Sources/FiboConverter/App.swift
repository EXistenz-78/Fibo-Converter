import SwiftUI

struct FiboConverterApp: App {
  var body: some Scene {
    WindowGroup("Fibo Converter") {
      ContentView()
        .frame(minWidth: 720, minHeight: 640)
    }
    .defaultSize(width: 780, height: 940)
    .windowResizability(.contentMinSize)
  }
}
