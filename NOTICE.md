# Third-party software

`Fibo Converter.app` bundles two command-line tools in `Contents/Helpers`:
`ModelConverter` and `QuantI8x`. Both are built from the Draw Things community engine
(<https://github.com/drawthingsai/draw-things-community>, GPL-3.0), using the
`krea2-edit` branch of <https://github.com/ianiv/draw-things-community> at commit
`a76e42b` plus the two patches in `Third-party/` (apply in order): `0001` adds SwiftPM targets for `ModelConverter` and for
`QuantI8x`, an i8x/8-bit S variant of the upstream quantizer; `0002` adds the MiniMax H3 Image rule to it (following
Atelier's `MOSSImporter.quantize`, <https://github.com/ianiv/Atelier>, GPL-3.0).

That is the complete corresponding source for the bundled binaries: check out that commit,
apply the patches, and run `swift build -c release --product ModelConverter --product QuantI8x`.
The engine is licensed under the GNU GPL v3 (<https://www.gnu.org/licenses/gpl-3.0.html>).
The tools run as separate processes; the app's own code in `Sources/` is independent of them.

`Resources/prep-model.py` (mark VI, 2026-11-05) is by teaspan (<https://gist.github.com/teaspan/ea15f459934974f914c4771e4d9ff87e>),
included unchanged with the author's permission. It carries no licence text; its header credits the gists it derives from.
