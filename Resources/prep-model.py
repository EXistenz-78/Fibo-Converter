#!/usr/bin/env python3
"""
Prepare quantized or ComfyUI safetensors for Draw Things import.

    prep-model INPUT [OUTPUT | OUTPUT_DIR]
        [--donor UNPRUNED_FILE | EXCERPT | DIR] [--no-fetch]

The default output is <input>-DT.safetensors.

Requires Python 3.8 or later, and numpy for dequantization.

Four cases are handled:

1. LTX-2.3. Extract the DiT as BF16 and drop bundled companions, preserve
   tensor names and metadata, apply recognised scales, and widen bare FP8.
   Warn when a scaled file contains FP8 without a matching scale.

   NB: some kind of LTX-2.3 import patch for draw-things-cli is still necessary
   after this preparation.  If you have my latest LTX-2.3 import patchset,
   which can perform the fp8 dequant itself and slightly better, then you may
   not need prep-model at all.

2. ComfyUI quantizations.  Dequantize to BF16 and remove scale and descriptor
   sidecars.  Supported schemes are scaled FP8, tensorwise or rowwise int8,
   int8 convrot, mxfp8 with E8M0 block scales, and nvfp4 with block and
   tensor scales.  Read .comfy_quant first, then _quantization_metadata.
   Preserve other tensors.  Reduce bundles to model.diffusion_model.
   Apply Flux key renames and strip Krea 2's prefix where needed.

3. Unscaled weights.  Draw Things reads bare FP8, BF16, and F16.  Strip the
   Flux model prefix and rename key_norm/query_norm .weight to .scale.
   Reduce bundles and strip Krea 2's prefix as above, and other files pass through.

4. Pruned MiniMax H3, restoring adaLN projections from their eight-dimensional
   basis and adaln_t_table.  Copies the missing time embedder from an official
   fl2va or ref2va file, or from a saved excerpt.  If the donor is missing, fetch
   four tensors (63 MB) from Comfy-Org/MiniMax-H3 and cache them.
   --donor selects the file or cache directory. The default cache is
   ~/.cache/prep-model, and the release defaults to fl2va unless filenames
   identify ref2va.  --no-fetch refuses a download.  Pruned LoRAs are refused.

Unsupported quantization is rejected, the DT stock importers can silently skip
unknown datatypes, which is how we end up with empty 4096-byte .ckpt files after
a UI import that didn't complain but also didn't work.

Version

  This is prep-model.py mark VI, released 2026-11-05.

Provenance

  This script is a derivative of "Bemo's Draw Things Klein Fix",
  https://gist.github.com/bemo-git/9c6013af9c54d44b58cb79d85c2e038a,
  which is itself a derivative of kernkraft235's fix-safetensors-header.py,
  https://gist.github.com/kernkraft235/2c74dbb690f2957c955ed9d578d7e653.
  Thank you Bemo & kernkraft for showing the way.
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

F16_MAX = 65504.0
PREFIX = "model.diffusion_model."
# Divisible by 8, so aligned for every dtype we touch.
# The env override is a test knob that forces row-split chunks on toy tensors.
CHUNK = int(os.environ.get("PREP_MODEL_CHUNK", 64 * 1024 * 1024))

ap = argparse.ArgumentParser(
    usage="prep-model <input.safetensors> [output.safetensors | output-dir/] "
          "[--donor FILE | DIR] [--no-fetch]")
ap.add_argument("input")
ap.add_argument("output", nargs="?")
ap.add_argument("--donor", help="MiniMax H3 time embedder source: an official unpruned file, "
                                "or the excerpt prep-model fetches into this path if it is missing")
ap.add_argument("--no-fetch", action="store_true",
                help="never download the MiniMax H3 time embedder")
args = ap.parse_args()

src = Path(args.input)
if not src.exists():
    print(f"error: no such file: {src}")
    sys.exit(1)

dst = src.with_name(src.stem + "-DT.safetensors")
if args.output:
    out = Path(args.output)
    dst = out / dst.name if out.is_dir() else out


# ── Machinery ──────────────────────────────────────────────────────────────────

def f8_e4m3_table():
    """256-entry decode table for float8 e4m3fn. Bias 7, no inf, 0x7F/0xFF NaN."""
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if (i >> 7) else 1.0
        exp, man = (i >> 3) & 0xF, i & 0x7
        if exp == 0xF and man == 0x7:
            table[i] = float("nan")
        elif exp == 0:
            table[i] = sign * (man / 8.0) * (2.0 ** -6)
        else:
            table[i] = sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    return table


E4M3 = f8_e4m3_table() if np is not None else None
# In fp4 e2m1, bit 3 is the sign, the low three bits index the magnitudes.
E2M1 = (np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                 dtype=np.float32) if np is not None else None)

_HADAMARD = {}


def hadamard(group):
    """comfy-kitchen's ConvRot matrix, a Kronecker power of a fixed 4x4 divided
    by sqrt(G).  It is symmetric and orthogonal, so it is its own inverse."""
    if group not in _HADAMARD:
        h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=np.float32)
        h = h4
        while h.shape[0] < group:
            h = np.kron(h, h4)
        _HADAMARD[group] = h / np.float32(group ** 0.5)
    return _HADAMARD[group]


def power_of_four(n):
    return n >= 4 and n & (n - 1) == 0 and (n.bit_length() - 1) % 2 == 0


def from_blocked(m, rows, cols):
    """Undo the cuBLAS tiled layout that comfy-kitchen stores block scales in.
    The layout pads to 128 rows by 4 cols of blocks, so the size check uses the
    padded count, not rows times cols."""
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    if m.size != nrb * 128 * ncb * 4:
        raise Refusal(f"block scale has {m.size} entries, expected "
                      f"{nrb * 128 * ncb * 4} for the swizzled layout")
    s = m.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(0, 2, 1, 3)
    s = s.reshape(nrb, ncb, 4, 32, 4).reshape(nrb, ncb, 128, 4).transpose(0, 2, 1, 3)
    return s.reshape(nrb * 128, ncb * 4)[:rows, :cols]


def to_bf16(floats):
    """f32 -> raw bf16 bytes, round-to-nearest-even."""
    u = floats.astype("<f4").view("<u4")
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()


class ClipAudit:
    """Tracks max |value| across kept tensors.  The importer writes most
    tensors as f16, and anything past 65504 clips there."""

    def __init__(self):
        self.worst, self.where, self.count = 0.0, "", 0

    def floats(self, name, arr):
        if not arr.size:
            return
        self.count += 1
        m = float(np.nanmax(np.abs(arr)))
        if m > self.worst:
            self.worst, self.where = m, name

    def raw(self, name, dtype, buf):
        if np is None or not buf:
            return
        if dtype == "BF16":
            self.floats(name, (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).view("<f4"))
        elif dtype in ("F32", "F64"):
            self.floats(name, np.frombuffer(buf, dtype="<f4" if dtype == "F32" else "<f8"))

    def report(self):
        if np is None:
            print("Range check skipped — numpy not installed")
        elif not self.count:
            print("Range check: nothing to check (f16/fp8 values are all in f16 range)")
        elif self.worst > F16_MAX:
            print(f"Range check: max |value| {self.worst:.1f} ({self.where}) — "
                  f"WARNING: over the f16 max of {F16_MAX:.0f}, conversion will clip")
        else:
            print(f"Range check: max |value| {self.worst:.1f} ({self.where}) — "
                  f"within f16 range ({F16_MAX:.0f})")


audit = ClipAudit()


class Refusal(Exception):
    """A layer this tool won't write.  The reason is the message."""


NP_DTYPE = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2",
            "I8": "i1", "U8": "u1", "F8_E4M3": "u1", "F8_E5M2": "u1"}


def read_tensor(k):
    """A whole (small) tensor, as stored.  Sidecars only."""
    info = header[k]
    start, end = info["data_offsets"]
    with src.open("rb") as f:
        f.seek(data_start + start)
        buf = f.read(end - start)
    return np.frombuffer(buf, dtype=NP_DTYPE[info["dtype"]]).reshape(info["shape"])


def read_floats(k):
    """A float sidecar as f32, whatever width it was stored at."""
    arr, dtype = read_tensor(k), header[k]["dtype"]
    if dtype == "BF16":
        return (arr.astype("<u4") << 16).view("<f4")
    if dtype in ("F16", "F64"):
        return arr.astype("<f4")
    if dtype == "F32":
        return arr
    raise Refusal(f"{k} is {dtype}, not a float scale")


# Decoders, one per scheme.
#   unit        smallest input byte run decodable alone (1 for scalar-scaled
#               fp8, one packed row otherwise)
#   out_shape   shape the BF16 output reports
#   scheme      name for the report
#   tag         progress token
#   decode(raw, row0) turns whole units into f32

class F8Scaled:
    unit = 1

    def __init__(self, k, scale):
        self.out_shape = header[k]["shape"]
        self.scale = None if scale is None else float(scale)
        self.scheme = "bare F8 widened" if scale is None else "F8*scale"
        self.tag = "F8->BF16" if scale is None else "F8*scale->BF16"

    def decode(self, raw, row0):
        floats = E4M3[np.frombuffer(raw, dtype=np.uint8)]
        return floats if self.scale is None else floats * self.scale


class Int8Rows:
    def __init__(self, k, scale, group=None):
        shape = header[k]["shape"]
        rows = shape[0] if shape else 1
        cols = int(np.prod(shape)) // rows if shape else 1
        self.out_shape, self.unit, self.cols = shape, cols, cols
        if scale.size == rows:
            self.scale, granularity = scale.reshape(rows, 1), "per-row"
        elif scale.size == 1:
            self.scale, granularity = scale.reshape(1, 1), "tensorwise"
        else:
            raise Refusal(f"int8 scale has {scale.size} entries for {rows} rows")
        self.h = None
        self.scheme = f"int8 {granularity}"
        self.tag = "I8*row->BF16" if granularity == "per-row" else "I8*scale->BF16"
        if group is not None:
            if not power_of_four(group):
                raise Refusal(f"convrot group {group} is not a power of 4")
            if cols % group:
                raise Refusal(f"in_features {cols} not divisible by convrot group {group}")
            self.h, self.group = hadamard(group), group
            self.scheme = f"int8 convrot g={group}"
            self.tag = "I8*row@H->BF16"

    def decode(self, raw, row0):
        q = np.frombuffer(raw, dtype=np.int8).reshape(-1, self.cols).astype(np.float32)
        w = q * (self.scale if self.scale.shape[0] == 1 else self.scale[row0:row0 + q.shape[0]])
        if self.h is not None:
            w = (w.reshape(w.shape[0], -1, self.group) @ self.h).reshape(w.shape)
        return w


class Mxfp8:
    scheme = "mxfp8"
    tag = "MXFP8->BF16"

    def __init__(self, k, e8m0):
        rows, cols = header[k]["shape"]
        if cols % 32:
            raise Refusal(f"in_features {cols} not a multiple of the mxfp8 block (32)")
        self.out_shape, self.unit, self.cols = [rows, cols], cols, cols
        exp = from_blocked(e8m0, rows, cols // 32).astype(np.int32)
        scales = np.ldexp(np.float32(1.0), exp - 127)
        self.scales = np.where(exp == 0, np.float32(0), scales)

    def decode(self, raw, row0):
        codes = E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(-1, self.cols // 32, 32)
        n = codes.shape[0]
        return codes * self.scales[row0:row0 + n, :, None]


class Nvfp4:
    scheme = "nvfp4"
    tag = "NVFP4->BF16"

    def __init__(self, k, block_scale, scale2):
        rows, packed = header[k]["shape"]
        cols = packed * 2
        if cols % 16:
            raise Refusal(f"in_features {cols} not a multiple of the nvfp4 block (16)")
        self.out_shape, self.unit, self.cols = [rows, cols], packed, cols
        self.scales = np.float32(scale2) * E4M3[from_blocked(block_scale, rows, cols // 16)]

    def decode(self, raw, row0):
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, self.cols // 2)
        codes = np.stack([b >> 4, b & 15], axis=-1).reshape(-1, self.cols // 16, 16)
        n = codes.shape[0]
        return E2M1[codes] * self.scales[row0:row0 + n, :, None]


def floats_of(raw, dtype):
    """Stored bytes of a float tensor -> f32."""
    arr = np.frombuffer(raw, dtype=NP_DTYPE[dtype])
    if dtype == "BF16":
        return (arr.astype("<u4") << 16).view("<f4")
    return arr.astype("<f4")


class CurveUnprune:
    """Rows in the curve basis -> rows in the full timestep-embedding
    space: W = W_p @ basis."""
    scheme = "adaLN curve-basis"
    tag = "Wp@basis->BF16"

    def __init__(self, k, basis):
        rows, small = header[k]["shape"]
        self.dtype = header[k]["dtype"]
        self.basis = basis
        self.out_shape = [rows, basis.shape[1]]
        self.unit = small * (4 if self.dtype == "F32" else 2)
        self.small = small

    def decode(self, raw, row0):
        return floats_of(raw, self.dtype).reshape(-1, self.small) @ self.basis


class CurveBias:
    """The adaLN bias with the curve offset the pruning folded in taken back
    out: b = b_p + W_p @ d, so the full weight is never built."""
    scheme = "adaLN curve-basis bias"
    tag = "bp+Wp@d->BF16"

    def __init__(self, k, weight_key, offset):
        self.dtype = header[k]["dtype"]
        self.out_shape = header[k]["shape"]
        self.unit = header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
        self.lift = read_floats(weight_key) @ offset

    def decode(self, raw, row0):
        return floats_of(raw, self.dtype) + self.lift


class Synth:
    """A tensor that does not come from the input file: bytes decided up
    front, written verbatim."""

    def __init__(self, dtype, shape, payload, tag):
        self.dtype, self.out_shape, self.payload, self.tag = dtype, list(shape), payload, tag
        self.scheme = tag


# Every other dtype is silently skipped by Draw Things reader.
DT_READABLE = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
# A layer's .comfy_attention.config names ComfyUI's attention kernel, as JSON,
# and holds no weights.
SIDECAR_SUFFIXES = ("weight_scale", "weight_scale_2", "scale_weight", ".comfy_quant",
                    ".comfy_attention.config")


def sidecars_of(k):
    """leaf -> key, for the sidecars present beside k.  `<key>_scale` and
    `<key>_scale_2` are looked up for every key.  `<module>.scale_weight` and
    `<module>.comfy_quant` are looked up only when k is the module's weight.
    A quantized Linear's bias shares the stem but is not quantized, so those
    two sidecars must not be matched to it."""
    stem, _, leaf = k.rpartition(".")
    found = {"scale": k + "_scale", "scale_2": k + "_scale_2"}
    if leaf == "weight":
        found.setdefault("quant", stem + ".comfy_quant")
        if found["scale"] not in header:
            found["scale"] = stem + ".scale_weight"
    return {leaf: key for leaf, key in found.items() if key in header}


def quant_metadata():
    """The header's _quantization_metadata layers, keyed by module name both
    as written and under the model.diffusion_model. prefix, which comfy-kitchen
    omits from the names while keeping it on the tensors."""
    qm = (metadata or {}).get("_quantization_metadata")
    if isinstance(qm, str):
        try:
            qm = json.loads(qm)
        except ValueError:
            qm = None
    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
    index = {}
    for name, entry in layers.items():
        if isinstance(entry, dict):
            index[name] = entry
            index[PREFIX + name] = entry
    return index


def classify(keys):
    """Pair every key with the decoder its sidecars call for.

    Returns (decoders, refusals, bare, schemes).
      decoders   key -> decoder, for what gets dequantized
      refusals   key -> reason, for what must not be written
      bare       unscaled F8_E4M3 tensors, for the caller's policy
      schemes    decoder count by scheme, for the report"""
    meta = quant_metadata()
    decoders, refusals, bare, schemes = {}, {}, [], Counter()
    for k in keys:
        info, side = header[k], sidecars_of(k)
        dtype, stem = info["dtype"], k.rpartition(".")[0]
        quant = {}
        if "quant" in side:
            try:
                quant = json.loads(read_tensor(side["quant"]).tobytes())
            except ValueError:
                quant = {}  # a marker with no JSON in it says nothing
        if not isinstance(quant, dict):
            quant = {}
        entry = meta.get(stem, {})
        fmt = quant.get("format") or entry.get("format")
        try:
            if dtype == "I8":
                if fmt is None:
                    raise Refusal("I8 payload with no comfy_quant or metadata entry")
                if fmt != "int8_tensorwise":
                    raise Refusal(f"I8 payload with format {fmt!r}, an unknown scheme")
                if "scale" not in side:
                    raise Refusal("int8 layer has no weight_scale")
                group = None
                if quant.get("convrot", entry.get("convrot", False)):
                    group = int(quant.get("convrot_groupsize",
                                          entry.get("convrot_groupsize", 256)))
                decoders[k] = Int8Rows(k, read_floats(side["scale"]), group)
            elif dtype == "F8_E4M3" and (fmt == "mxfp8" or (
                    "scale" in side and header[side["scale"]]["dtype"] == "U8")):
                if "scale" not in side:
                    raise Refusal("mxfp8 layer has no weight_scale")
                decoders[k] = Mxfp8(k, read_tensor(side["scale"]))
            elif dtype == "F8_E4M3" and "scale" in side:
                scale = read_floats(side["scale"])
                if scale.size != 1:
                    raise Refusal(f"fp8 scale has {scale.size} entries, expected 1")
                decoders[k] = F8Scaled(k, scale.reshape(()))
            elif dtype == "F8_E4M3":
                bare.append(k)
            elif dtype == "U8" and (fmt == "nvfp4" or "scale" in side or "scale_2" in side):
                if "scale" not in side or header[side["scale"]]["dtype"] != "F8_E4M3":
                    raise Refusal("nvfp4 layer has no F8_E4M3 weight_scale")
                if "scale_2" not in side:
                    raise Refusal("nvfp4 layer has no weight_scale_2")
                decoders[k] = Nvfp4(k, read_tensor(side["scale"]),
                                    read_floats(side["scale_2"]).reshape(()))
        except Refusal as why:
            refusals[k] = str(why)
        if k in decoders:
            schemes[decoders[k].scheme] += 1
    return decoders, refusals, bare, schemes


def describe(schemes):
    return ", ".join(f"{s} x{n}" for s, n in sorted(schemes.items()))


def layers(n):
    return f"{n} layer{'' if n == 1 else 's'}"


def build_plan(keys, rename, decoders, refusals):
    """One pass over offset-sorted keys.  It returns the output header with
    offsets rebuilt, plus a per-tensor plan of (key, dtype, decoder|None,
    start, end).  A tensor with no decoder is copied verbatim.

    Every strategy funnels through here, so this is also where the refusal
    lives.  Never copy a tensor we did not actually handle."""
    plan, new_header, cursor, unhandled = [], {}, 0, []
    for k in keys:
        dec = decoders.get(k)
        if isinstance(dec, Synth):
            new_header[rename(k)] = {
                "dtype": dec.dtype,
                "shape": dec.out_shape,
                "data_offsets": [cursor, cursor + len(dec.payload)],
            }
            plan.append((k, dec.dtype, dec, 0, 0))
            cursor += len(dec.payload)
            continue
        info = header[k]
        start, end = info["data_offsets"]
        if dec is None:
            if k in refusals:
                unhandled.append((k, refusals[k]))
            elif info["dtype"] not in DT_READABLE:
                unhandled.append(
                    (k, f"{info['dtype']} payload — Draw Things skips this "
                        f"dtype silently"))
            elif (side := sidecars_of(k)):
                leaf = next(iter(side.values())).rsplit(".", 1)[-1]
                unhandled.append((k, f"live {leaf} sidecar, not applied"))
        shape = info["shape"] if dec is None else dec.out_shape
        nbytes = end - start if dec is None else int(np.prod(shape)) * 2
        new_header[rename(k)] = {
            "dtype": info["dtype"] if dec is None else "BF16",
            "shape": list(shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        plan.append((k, info["dtype"], dec, start, end))
        cursor += nbytes
    if unhandled:
        n = len(unhandled)
        print(f"error: {n} tensor{'s' if n > 1 else ''} "
              f"use{'' if n > 1 else 's'} a quantization scheme this tool does "
              "not implement:")
        for k, why in unhandled[:5]:
            print(f"  {k}  ({why})")
        if len(unhandled) > 5:
            print(f"  ... and {len(unhandled) - 5} more")
        print("Copying them through would convert 'successfully' into a model "
              "missing those weights. Try a bf16 source, or a pre-converted "
              ".ckpt.")
        sys.exit(1)
    return plan, new_header, cursor


def write_output(plan, new_header, total, keep_metadata=False):
    """Stream the plan into dst, auditing values on the way through.  Decoded
    tensors are read in whole units so a chunk never splits a row."""
    out_header = new_header
    if keep_metadata and metadata is not None:
        out_header = {"__metadata__": metadata, **new_header}
    header_bytes = json.dumps(
        out_header, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    print(f"Writing {(8 + len(header_bytes) + total) / 1e9:.1f} GB -> {dst.name}")
    written = 0
    stale = 0  # width of the previous status line, so shorter lines blank its tail
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for i, (k, dtype, dec, start, end) in enumerate(plan):
            if isinstance(dec, Synth):
                audit.raw(k, dtype, dec.payload)
                fout.write(dec.payload)
                written += len(dec.payload)
                remaining, row0, step = 0, 0, CHUNK
            else:
                fin.seek(data_start + start)
                remaining, row0 = end - start, 0
                step = CHUNK if dec is None else max(dec.unit, CHUNK // dec.unit * dec.unit)
            while remaining:
                raw = fin.read(min(remaining, step))
                if not raw:
                    print(f"\nerror: unexpected EOF inside {k} — truncated file?")
                    sys.exit(1)
                remaining -= len(raw)
                if dec is None:
                    audit.raw(k, dtype, raw)
                    out = raw
                else:
                    floats = dec.decode(raw, row0)
                    row0 += len(raw) // dec.unit
                    audit.floats(k, floats)
                    out = to_bf16(floats)
                fout.write(out)
                written += len(out)
            tag = dtype if dec is None else dec.tag
            line = (f"  [{i + 1:4d}/{len(plan)}] {written / max(total, 1) * 100:5.1f}%  "
                    f"{tag}  {k.split('.')[-1]}")
            print(line.ljust(stale), end="\r")
            stale = len(line)
    print()


def need_numpy(why):
    if np is None:
        print(f"error: numpy is required to {why} (pip install numpy)")
        sys.exit(1)


# ── Read header ────────────────────────────────────────────────────────────────

try:
    with src.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
except (struct.error, ValueError, UnicodeDecodeError):
    print(f"error: {src.name} is not a safetensors file (bad header)")
    sys.exit(1)

metadata = header.get("__metadata__")
all_keys = [k for k in header if k != "__metadata__"]
data_start = 8 + header_len
payload = max((header[k]["data_offsets"][1] for k in all_keys), default=0)
dtypes = {header[k].get("dtype") for k in all_keys}

print(f"Reading {src.name}: {len(all_keys)} tensors, {payload / 1e9:.1f} GB, "
      + "dtypes " + "/".join(sorted(d for d in dtypes if d))
      + (", metadata" if metadata is not None else ""))

if "F8_E5M2" in dtypes and any(
        k.endswith(("weight_scale", "scale_weight")) for k in all_keys):
    print("error: F8_E5M2 tensors alongside quantization scales — scaled-E5M2 "
          "is not supported (E4M3 only)")
    sys.exit(1)


def has(sub):
    return any(sub in k for k in all_keys)


if has("to_gate_compress"):
    print("warning: Draw Things can't use FastVideo VSA attention gate-compress projections")

H3_ADALN = "adaln_proj.linear"
H3_EMBEDDER = ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")
# The env override is a test knob that serves the official files locally.
H3_HF = os.environ.get("PREP_MODEL_HF", "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/"
                                        "e5eb578a89295337b8ff433a035929ce0279e0b6/")
H3_OFFICIAL = "diffusion_models/minimax_h3_{}_bf16.safetensors"


def h3_release(*names):
    """The release the first name to mention one says, else fl2va."""
    return next((r for n in names for r in ("fl2va", "ref2va") if r in n), "fl2va")


def h3_donor(release):
    """The time embedder's file: --donor, the default name inside a --donor
    directory, or the default name in the user's cache."""
    name = f"minimax_h3_{release}_time_embedder.safetensors"
    if args.donor is None:
        return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "prep-model" / name
    donor = Path(args.donor)
    return donor / name if donor.is_dir() else donor


def fetch_embedder(release, path):
    """Range-read the time embedder out of the official file into a
    safetensors of its own at `path`.  Only a 206 is accepted, because a
    server ignoring Range would start sending the whole 66 GB."""
    url = H3_HF + H3_OFFICIAL.format(release)

    def ranged(start, end):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}",
                                                   "User-Agent": "prep-model"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise OSError(f"HTTP {r.status} to a range request")
            body = r.read()
        if len(body) != end - start:
            raise OSError(f"{len(body)} bytes where {end - start} were asked for")
        return body

    # Per process, because conversions running at once share the default cache.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        (n,) = struct.unpack("<Q", ranged(0, 8))
        remote = json.loads(ranged(8, 8 + n))
        keys = [next((k for k in (name, PREFIX + name) if k in remote), name)
                for name in ("time_embedder." + leaf for leaf in H3_EMBEDDER)]
        missing = [k for k in keys if k not in remote]
        if missing:
            raise OSError(f"no {missing[0]}")
        keys.sort(key=lambda k: remote[k]["data_offsets"][0])
        size = sum(e - s for s, e in (remote[k]["data_offsets"] for k in keys))
        print(f"  fetching the {release} time embedder ({size / 1e6:.0f} MB) -> {path}")
        excerpt, blobs, cursor = {}, [], 0
        for k in keys:
            s, e = remote[k]["data_offsets"]
            blobs.append(ranged(8 + n + s, 8 + n + e))
            excerpt[k] = {**remote[k], "data_offsets": [cursor, cursor + e - s]}
            cursor += e - s
    except (OSError, ValueError) as e:
        print(f"error: could not fetch the time embedder from {url}: {e}")
        print("  download an official unpruned MiniMax H3 and pass it as --donor")
        sys.exit(1)
    header_bytes = json.dumps({"__metadata__": {"source": url.replace("/resolve/", "/blob/")},
                               **excerpt}, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as f:
        f.write(struct.pack("<Q", len(header_bytes)) + header_bytes)
        for blob in blobs:
            f.write(blob)
    os.replace(partial, path)


def h3_curve(w1, b1, w2, b2, rows):
    """The adaLN input at `rows` timesteps evenly spaced over [0, 1]: the time
    embedder (UNet-style sinusoid, cos then sin, period 10000, then linear,
    swish, linear) and the swish MiniMaxH3Fixed applies to its output."""
    t = np.linspace(0, 1, rows)
    half = w1.shape[1] // 2
    freq = np.exp(-np.log(10000.0) * np.arange(half) / half)[None, :] * t[:, None]
    sinusoid = np.concatenate([np.cos(freq), np.sin(freq)], axis=1)

    def swish(x):
        return x / (1 + np.exp(-x))

    return swish(swish(sinusoid @ w1.T + b1) @ w2.T + b2)


def fit_curve(x, table):
    """The affine map c = B x + d the pruning applied, by least squares over
    the centred samples.  (B, d, residual relative to the table's largest entry)."""
    xm, cm = x.mean(axis=0), table.mean(axis=0)
    bt = np.linalg.lstsq(x - xm, table - cm, rcond=1e-6)[0]
    d = cm - xm @ bt
    residual = np.abs(x @ bt + d - table).max() / np.abs(table).max()
    return bt.T, d, residual


# The curve basis is as wide as the table: 8 for Comfy's pruned files, 16 for P-H3-Image.
h3_table_key = next((k for k in all_keys if k.endswith("adaln_t_table")), None)
h3_curve_width = header[h3_table_key]["shape"][-1] if h3_table_key else 8
h3_modules = [k[: -len(".weight")] for k in all_keys
              if k.endswith(H3_ADALN + ".weight") and header[k]["shape"][-1] == h3_curve_width]
is_h3_pruned = bool(h3_modules)
if any(H3_ADALN + ".lora" in k and header[k]["shape"][-1] == h3_curve_width for k in all_keys):
    print(f"warning: pruned MiniMax H3 LoRA (adaLN lora_A of in-dim {h3_curve_width}); "
          "prep-model restores checkpoints only, so those pairs stay pruned")
if has("adaln_t_table") and not h3_modules:
    print(f"warning: adaln_t_table present but no adaLN projection has in-dim {h3_curve_width}; "
          "nothing to unprune")

LTX_MARKERS = ("aggregate_embed", "bwe_generator", "prompt_scale_shift_table",
               "audio_vae.", "vocoder.")
ltx_hits = [m for m in LTX_MARKERS if has(m)]
is_ltx = bool(ltx_hits) and not has("audio_cross_attn.kv_linear.")  # not longcat

if is_h3_pruned:
    # ── Strategy 4. Pruned MiniMax H3 checkpoint ──────────────────────────────
    need_numpy("unprune a MiniMax H3 checkpoint")
    print(f"Strategy: MiniMax H3 pruned checkpoint ({layers(len(h3_modules))} of adaLN in the curve basis)")
    t_key = h3_table_key
    if t_key is None:
        print("error: pruned adaLN but no adaln_t_table, so the compression cannot be recovered")
        sys.exit(1)
    release = h3_release(Path(args.donor or "").name, src.name)
    donor = h3_donor(release)
    if not donor.exists():
        if args.no_fetch:
            print(f"error: no time embedder at {donor}, and --no-fetch is set.  "
                  f"Pass --donor an official unpruned MiniMax H3 ({H3_OFFICIAL.format(release)}).")
            sys.exit(1)
        fetch_embedder(release, donor)
    with donor.open("rb") as f:
        dn = struct.unpack("<Q", f.read(8))[0]
        donor_header = json.loads(f.read(dn))
        donor_header.pop("__metadata__", None)
    prefixed = any(k.startswith(PREFIX) for k in h3_modules)
    synth, embedder = {}, []
    for leaf in H3_EMBEDDER:
        name = "time_embedder." + leaf
        d_key = next((k for k in (PREFIX + name, name) if k in donor_header), None)
        if d_key is None:
            print(f"error: donor {donor.name} has no {name}")
            sys.exit(1)
        info = donor_header[d_key]
        if info["dtype"] not in ("F32", "F16", "BF16"):
            print(f"error: donor {d_key} is {info['dtype']}, not a plain float tensor")
            sys.exit(1)
        s, e = info["data_offsets"]
        with donor.open("rb") as f:
            f.seek(8 + dn + s)
            payload = f.read(e - s)
        synth[(PREFIX + name) if prefixed else name] = Synth(
            info["dtype"], info["shape"], payload, "donor")
        embedder.append(floats_of(payload, info["dtype"]).reshape(info["shape"]).astype("f8"))
    table = read_floats(t_key).astype("f8")
    basis, offset, residual = fit_curve(h3_curve(*embedder, rows=table.shape[0]), table)
    print(f"  curve from {donor.name}'s time embedder fits adaln_t_table to {residual:.1e}")
    if residual > 1e-2:
        print(f"  warning: adaln_t_table does not lie on {donor.name}'s timestep curve; "
              "the restored adaLN will not reproduce the pruned one")
    basis, offset = basis.astype("<f4"), offset.astype("<f4")

    drop = {k for k in all_keys
            if k.endswith(("adaln_t_table", "adaln_basis", "adaln_mean"))
            or k.endswith(SIDECAR_SUFFIXES)}
    keep = sorted((k for k in all_keys if k not in drop and k not in synth),
                  key=lambda k: header[k]["data_offsets"][0])
    decoders, refusals, bare, schemes = classify(keep)
    if bare and decoders:
        print(f"  warning: {len(bare)} F8_E4M3 tensors have no matching scale (e.g. {bare[0]}); "
              "widened as bare fp8, which is wrong if they were scaled")
    for k in bare:
        decoders[k] = F8Scaled(k, None)
    for stem in h3_modules:
        decoders[stem + ".weight"] = CurveUnprune(stem + ".weight", basis)
        if stem + ".bias" in header:
            decoders[stem + ".bias"] = CurveBias(stem + ".bias", stem + ".weight", offset)
    decoders.update(synth)
    keep += list(synth)
    print(f"  unprune {len(h3_modules)} adaLN modules -> [out, {basis.shape[1]}] and their biases, "
          f"time embedder from {donor.name}"
          + (f", dequantize {layers(sum(schemes.values()))} ({describe(schemes)})" if schemes else "")
          + f", drop {len(drop)} tensors (t_table, sidecars)")
    plan, new_header, total = build_plan(keep, lambda k: k, decoders, refusals)
    write_output(plan, new_header, total)

elif is_ltx:
    # ── Strategy 1. LTX, DiT-only BF16 ────────────────────────────────────────
    print(f"Strategy: LTX-2.x DiT-only BF16 prep ({', '.join(ltx_hits)})")

    sidecar_keys = {k for k in all_keys if k.endswith(SIDECAR_SUFFIXES)}
    # A blocklist, not a keep-list, so an unfamiliar finetune's extra tensors
    # survive rather than vanish.
    COMPANIONS = ("vae.", "audio_vae.", "vocoder.", "text_encoders.")
    companion_keys = {k for k in all_keys if k.startswith(COMPANIONS)}
    keep = sorted((k for k in all_keys if k not in sidecar_keys | companion_keys),
                  key=lambda k: header[k]["data_offsets"][0])
    if not keep:
        print("error: nothing left to keep — this isn't an LTX checkpoint after all")
        sys.exit(1)

    if companion_keys:
        counts = Counter(next(p for p in COMPANIONS if k.startswith(p))
                         for k in companion_keys)
        detail = ", ".join(f"{p}* {n}" for p, n in sorted(counts.items()))
        print(f"  drop {len(companion_keys)} bundled companions ({detail})")
    if sidecar_keys:
        print(f"  drop {len(sidecar_keys)} sidecars (weight_scale / comfy_quant)")

    kept_tops = Counter(k.split(".", 1)[0] for k in keep)
    print("  keep " + ", ".join(f"{t}.* ({n})" for t, n in sorted(kept_tops.items()))
          + ", names and __metadata__ unchanged")

    quantized = sidecar_keys or {"I8", "U8", "F8_E4M3"} & dtypes
    decoders, refusals, bare, schemes = {}, {}, [], Counter()
    if quantized:
        need_numpy("dequantize")
        decoders, refusals, bare, schemes = classify(keep)
        if decoders:
            print(f"  dequantize {layers(len(decoders))} -> BF16 ({describe(schemes)})")
        if bare and decoders:
            print(f"  warning: {len(bare)} F8_E4M3 tensors have no matching scale "
                  f"(e.g. {bare[0]}); widened as bare fp8, which is wrong if they were scaled")
        if bare:
            print(f"  widen {len(bare)} bare F8_E4M3 -> BF16 (exact)")
            for k in bare:
                decoders[k] = F8Scaled(k, None)

    if not decoders and not refusals and not sidecar_keys and not companion_keys:
        print("  nothing to change — already importable; not writing a copy")
        sys.exit(0)

    if not dst.name.startswith("ltx_"):
        print(f"  note: '{dst.name}' is outside the ltx_ family, so the client "
              "will match recommended settings by version alone")

    plan, new_header, total = build_plan(keep, lambda k: k, decoders, refusals)
    write_output(plan, new_header, total, keep_metadata=True)

else:
    # Bundled text-encoder and VAE internals confuse Draw Things architecture
    # detection, and it can't use them anyway, its companion models are separate
    # ckpts.  SD-era bundles such as cond_stage_model. and first_stage_model.
    # are left alone because those companions are imported on purpose.
    BUNDLE_PREFIXES = ("text_encoders.", "vae.", "audio_encoders.")
    dropped_bundle = 0
    if any(k.startswith(PREFIX) for k in all_keys) and \
            any(k.startswith(BUNDLE_PREFIXES) for k in all_keys):
        keep_bundle = [k for k in all_keys if k.startswith(PREFIX)]
        dropped_bundle = len(all_keys) - len(keep_bundle)
        counts = Counter(next(p for p in BUNDLE_PREFIXES if k.startswith(p))
                         for k in all_keys if k.startswith(BUNDLE_PREFIXES))
        detail = ", ".join(f"{p}* {n}" for p, n in sorted(counts.items()))
        other = dropped_bundle - sum(counts.values())
        if other:
            detail += f", other {other}"
        print(f"  all-in-one bundle: keep {len(keep_bundle)} {PREFIX}*, "
              f"drop {dropped_bundle} bundled companions ({detail})")
        all_keys = keep_bundle
        dtypes = {header[k].get("dtype") for k in all_keys}

    sidecar_keys = {k for k in all_keys if k.endswith(SIDECAR_SUFFIXES)}
    is_comfy_quant = bool(sidecar_keys) or bool(quant_metadata())

    # The Flux & Krea importers get key surgery, other arches currently appear to
    # import fine under their original names.
    is_flux_family = any(
        marker in k
        for k in all_keys
        for marker in ("double_blocks.", "single_blocks.", "img_attn",
                       "single_transformer_blocks.", "txt_mod", "img_mod")
    )
    is_krea_family = not is_flux_family and any(
        marker in k
        for k in all_keys
        for marker in ("txtfusion.projector.weight", "text_fusion.projector.weight")
    )
    SURGERY = ("strip " + PREFIX + ", rename key_norm/query_norm .weight "
               "-> .scale (Flux family)" if is_flux_family
               else "strip " + PREFIX + " (Krea 2)" if is_krea_family
               else "names unchanged")

    def fix_key(k):
        if (is_flux_family or is_krea_family) and k.startswith(PREFIX):
            k = k[len(PREFIX):]
        if is_flux_family and k.endswith(("key_norm.weight", "query_norm.weight")):
            k = k[: -len("weight")] + "scale"
        return k

    if is_comfy_quant:
        # ── Strategy 2. ComfyUI quantized, any arch ──────────────────────────
        need_numpy("dequantize ComfyUI quantized models")

        keep = sorted((k for k in all_keys if k not in sidecar_keys),
                      key=lambda k: header[k]["data_offsets"][0])
        decoders, refusals, bare, schemes = classify(keep)

        print(f"Strategy: ComfyUI quantized ({describe(schemes) or 'no decodable layers'})")
        print(f"  dequantize {layers(len(decoders))} -> BF16")
        if bare:
            print(f"  keep {len(bare)} F8_E4M3 raw — no matching scale "
                  "(Draw Things decodes bare fp8 natively)")
        print(f"  drop {len(sidecar_keys)} sidecars (weight_scale / weight_scale_2 / "
              "comfy_quant / comfy_attention.config)"
              + (", drop __metadata__" if metadata is not None else ""))
        print(f"  pass {len(keep) - len(decoders)} other tensors through; " + SURGERY)

        plan, new_header, total = build_plan(keep, fix_key, decoders, refusals)
        write_output(plan, new_header, total)

    else:
        # ── Strategy 3. Key surgery only, no scales to apply ─────────────────
        mode = ("F8_E4M3" if "F8_E4M3" in dtypes
                else "F8_E5M2" if "F8_E5M2" in dtypes else "BF16/F16")
        if not (is_flux_family or dropped_bundle):
            print(f"Strategy: none needed — {mode}, no scales, no key surgery; "
                  "Draw Things imports this file as-is")
            sys.exit(0)

        print(f"Strategy: {'key surgery only' if is_flux_family else 'unbundle'} "
              f"({mode}, no scales to apply)")
        print("  " + SURGERY
              + (", drop __metadata__" if metadata is not None else ""))

        keep = sorted(all_keys, key=lambda k: header[k]["data_offsets"][0])
        plan, new_header, total = build_plan(keep, fix_key, {}, {})
        write_output(plan, new_header, total)

audit.report()
print(f"Done -> {dst}")

