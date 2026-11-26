# Third-party software

`Model Converter.app` bundles two command-line tools in `Contents/Helpers`:
`ModelConverter` and `QuantI8x`. Both are built from the Draw Things community engine
(<https://github.com/drawthingsai/draw-things-community>, GPL-3.0), using the
`atelier-engine` branch of <https://github.com/ianiv/draw-things-community> at commit
`f4d03b0` (the engine of DT Hub's server) plus the seventeen patches in `Third-party/` (apply in order with `git am`):
`0001` Krea 2 reference images and `0002` the per-request scheduler (DT Hub's server patch); `0003` SwiftPM targets for
`ModelConverter` and for `QuantI8x`, an i8x/8-bit S variant of the upstream quantizer; `0004` the MiniMax H3 Image rule
(following Atelier's `MOSSImporter.quantize`, <https://github.com/ianiv/Atelier>, GPL-3.0); `0005`-`0007` Ming-Image
(converters for the model, text encoder, vision tower and VAE, its 8-bit S rule, the engine code to run it, editing, and
Z-Image LoRAs on it); `0008`-`0010` Ming-Image Design-Layer (the connector-only conversion of its text encoder and the
engine code to run it); `0011`-`0017` seamless tiling (DT Hub's server patch). The tree they give is the engine of DT Hub
1.10.0's server.

That is the complete corresponding source for the bundled binaries: check out that commit,
apply the patches, and run `swift build -c release --product ModelConverter --product QuantI8x`.
The engine is licensed under the GNU GPL v3 (<https://www.gnu.org/licenses/gpl-3.0.html>).
The tools run as separate processes; the app's own code in `Sources/` is independent of them.

`Resources/prep-model.py` (mark VI, 2026-11-05) is by teaspan (<https://gist.github.com/teaspan/ea15f459934974f914c4771e4d9ff87e>),
included unchanged with the author's permission. It carries no licence text; its header credits the gists it derives from.
