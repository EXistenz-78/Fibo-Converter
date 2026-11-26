# Model Converter

A small macOS app with four tabs that turns model files into models Draw Things, Atelier and DT Hub can run, and
registers them in the models folder's `custom.json`:

- **Fibo 2.0 Edit Turbo**: merges the two transformer files of [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo),
  converts the model and the Qwen3-VL text encoder, optionally quantizes both to 8-bit S (i8x).
- **Any safetensors**: converts any safetensors checkpoint the Draw Things engine can import to a ckpt. For ComfyUI-quantized
  files (FP8, int8, MXFP8, NVFP4) it can first run `prep-model.py` by [teaspan](https://gist.github.com/teaspan/ea15f459934974f914c4771e4d9ff87e)
  (bundled with the author's permission; needs Python 3 and numpy).
- **MiniMax H3 Image**: converts the model and the VAE, puts the already-converted text encoder in place, and quantizes the
  model to 8-bit S. Read its licence: it excludes the EU, UK, South Korea and the US. H3 runs only on the Draw Things
  engine fork's server (DT Hub 1.8+ includes it).

- **Ming-Image**: converts the three ComfyUI files of [Comfy-Org/Ming-Image](https://huggingface.co/Comfy-Org/Ming-Image)
  ([inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), MIT): the model, the Ling
  text encoder (quantized to 8-bit S, with its tokenizer), the vision tower for editing and the 4-channel VAE, and quantizes
  the model to 8-bit S. The `custom.json` entry carries the activation scaling that keeps large images with long prompts from
  turning black. Ming-Image runs only on the Draw Things engine fork's server (DT Hub 1.8.3+ includes it); Z-Image LoRAs work on it.
  Needs about 55 GB free while it runs (26 GB at the end).

- **Ming Layer** (new in 1.4): converts [Ming-Image 0.1 Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)
  (inclusionAI, MIT), which splits a finished design into transparent layers. Convert Ming-Image Design first: the Layer
  model uses its vision tower, VAE and tokenizer, and its text encoder is built from Design's, taking only the parts that
  differ from the Layer file (connector, projections, query tokens; on APFS the shared 18 GB are a clone). Download the two
  Layer files of Comfy-Org/Ming-Image (about 49 GB); the result adds about 8 GB. It runs on the server of DT Hub 1.10.0+:
  the design as the start image, batch size = layers + 1, guidance 3.

The Any-safetensors and MiniMax H3 tabs are new in 1.1 and have only been tested with stand-in tools, not on real weights.

**Licence.** Fibo 2.0 is CC BY-NC 4.0 (non-commercial, credit to Bria). The app contains and downloads no weights: you log in
to Hugging Face, accept the terms and download three files yourself (the app lists them, with links, and recognises them by
name and exact size). Converted copies are adaptations under the same licence: do not share them.

## Use (Fibo tab; the other tabs follow the same steps)
1. Accept the licence, download the three files, choose the folder they are in.
2. Choose the models folder and the result (8-bit S, full precision, or both).
3. Convert. A stopped run continues from the step it was on. Needs about 55 GB free while it runs (41 GB for f16 only).
   Work files live in `_fibo-work` inside the models folder and are removed at the end.

With 8-bit S only, the full-precision intermediates are moved to the Trash. `custom.json` is backed up as
`custom.json.bak-fibo-converter` before it is changed.

## Build
```bash
Scripts/build-app.sh      # writes build/Model Converter.app
swift test                # merge, custom.json and pipeline (with stand-in tools)
```
The conversion tools are built from the Draw Things fork checkout (`FORK`, default `~/Software developement/DT Hub/Developer-krea2/fork-ming-layer`, branch
`dthub-engine`: SwiftPM targets `ModelConverter` and `QuantI8x`, an i8x variant of the fork's `ModelQuantizer` that also knows
the MiniMax H3 Image and Ming-Image rules). The patches are in `Third-party/`. The tools are copied into the app (`Contents/Helpers`).

## Credits and third-party tools

- `prep-model.py` (mark VI, 2026-11-05) is by [teaspan](https://github.com/teaspan), included unchanged with his permission:
  [gist](https://gist.github.com/teaspan/ea15f459934974f914c4771e4d9ff87e). It derives from gists by bemo-git and kernkraft235,
  as its header says. teaspan also maintains a [Homebrew tap](https://github.com/teaspan/homebrew-draw-things) with a recent
  Draw Things CLI build, a quantizer with H3 support and more import paths: worth a look.

The released app bundles `ModelConverter` and `QuantI8x`, built from the GPL-3.0 Draw Things engine.
Source, base commit and patch: see [NOTICE.md](NOTICE.md).

## Installing the release

The app is signed ad hoc, not notarized. After downloading the zip, unzip it, then right-click the app, choose Open, and confirm once.
