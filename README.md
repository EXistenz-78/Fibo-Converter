# Fibo Converter

A small macOS app that turns the files of [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
into models Draw Things, Atelier and DT Hub can run: it merges the two transformer files, converts the model and the
Qwen3-VL text encoder, optionally quantizes both to 8-bit S (i8x), and registers them in the models folder's `custom.json`.

**Licence.** Fibo 2.0 is CC BY-NC 4.0 (non-commercial, credit to Bria). The app contains and downloads no weights: you log in
to Hugging Face, accept the terms and download three files yourself (the app lists them, with links, and recognises them by
name and exact size). Converted copies are adaptations under the same licence: do not share them.

## Use
1. Accept the licence, download the three files, choose the folder they are in.
2. Choose the models folder and the result (8-bit S, full precision, or both).
3. Convert. A stopped run continues from the step it was on. Needs about 55 GB free while it runs (41 GB for f16 only).
   Work files live in `_fibo-work` inside the models folder and are removed at the end.

With 8-bit S only, the full-precision intermediates are moved to the Trash. `custom.json` is backed up as
`custom.json.bak-fibo-converter` before it is changed.

## Build
```bash
Scripts/build-app.sh      # writes build/Fibo Converter.app
swift test                # merge, custom.json and pipeline (with stand-in tools)
```
The conversion tools are built from the Draw Things fork checkout (`FORK`, default `~/Developer-krea2/fork`, branch
`fibo-tools`: SwiftPM targets `ModelConverter` and `QuantI8x`, an i8x variant of the fork's `ModelQuantizer`). The patch is
also in DT Hub's `Tools/atelier-fork-patches/`. The tools are copied into the app (`Contents/Helpers`).

## Third-party tools

The released app bundles `ModelConverter` and `QuantI8x`, built from the GPL-3.0 Draw Things engine.
Source, base commit and patch: see [NOTICE.md](NOTICE.md).

## Installing the release

The app is signed ad hoc, not notarized. After downloading the zip, unzip it, then right-click the app, choose Open, and confirm once.
