#!/bin/bash
# Builds "Fibo Converter.app" in build/. The conversion tools come from the Draw Things fork
# (branch fibo-tools, see README): FORK=/path/to/fork builds them if they are missing.
set -euo pipefail
cd "$(dirname "$0")/.."
FORK="${FORK:-$HOME/Developer-krea2/fork}"
APP="build/Fibo Converter.app"

for tool in ModelConverter QuantI8x; do
  if [ ! -x "$FORK/.build/release/$tool" ]; then
    (cd "$FORK" && swift build -c release --product "$tool")
  fi
done
swift build -c release
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$APP/Contents/Resources"
cp .build/release/FiboConverter "$APP/Contents/MacOS/"
cp Resources/prep-model.py "$APP/Contents/Resources/"
cp "$FORK/.build/release/ModelConverter" "$FORK/.build/release/QuantI8x" "$APP/Contents/Helpers/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>FiboConverter</string>
<key>CFBundleIdentifier</key><string>com.exiztenz.fibo-converter</string>
<key>CFBundleName</key><string>Fibo Converter</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.2</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP/Contents/Helpers/ModelConverter" "$APP/Contents/Helpers/QuantI8x"
codesign --force --sign - "$APP"
echo "Built $APP"
