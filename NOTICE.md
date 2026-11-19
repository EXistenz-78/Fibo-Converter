# Third-party software

`Fibo Converter.app` bundles two command-line tools in `Contents/Helpers`:
`ModelConverter` and `QuantI8x`. Both are built from the Draw Things community engine
(<https://github.com/drawthingsai/draw-things-community>, GPL-3.0), using the
`krea2-edit` branch of <https://github.com/ianiv/draw-things-community> at commit
`a76e42b` plus the patch in `Third-party/0001-SwiftPM-targets-for-ModelConverter-and-an-i8x-8-bit-.patch`
(adds SwiftPM targets for `ModelConverter` and for `QuantI8x`, an i8x/8-bit S variant of the upstream quantizer).

That is the complete corresponding source for the bundled binaries: check out that commit,
apply the patch, and run `swift build -c release --product ModelConverter --product QuantI8x`.
The engine is licensed under the GNU GPL v3 (<https://www.gnu.org/licenses/gpl-3.0.html>).
The tools run as separate processes; the app's own code in `Sources/` is independent of them.
