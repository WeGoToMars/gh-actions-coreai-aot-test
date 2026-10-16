# SwiftUI Demo

A standalone macOS application demonstrating `tinkerbox-ane` on Apple Silicon.

Requires macOS 27 on Apple Silicon.

## Prebuilt App

1. Download and extract the latest `Tinkerbox-ANE-*-macos-arm64.zip` from the GitHub releases page.
2. Launch **Tinkerbox.app**. The app is ad-hoc signed, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

The app bundles a standalone `uv` binary. On first run, it configures an isolated Python 3.12 environment, installs the precompiled `tinkerbox-ane` wheel, and fetches the required model weights from Hugging Face. Subsequent launches reuse the cached environment.

The default model bundle targets `h17s` (M5-family) Neural Engine cores. Setup progress and error logs are written to `~/Library/Logs/tinkerbox-ane/demo.log`.

## Building from Source

Building requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py --build-only
```

This generates `artifacts/swiftui/Tinkerbox.app` and a distributable ZIP archive. Omit `--build-only` to launch the application immediately after building.

To test against a local model bundle instead of downloading from Hugging Face:

```sh
TINKERBOX_ASSET="$PWD/artifacts/d1_fp16_fused128_ctx32768" uv run --no-project --python 3.12 python examples/swiftui/run.py
```

## Supported Tasks

- `noul`: Binary decisions with normalized probabilities.
- `choice`: Selection across two or more discrete options.
- `score`: Ranked evaluation across 2–10 ordered levels.

Displayed decision latencies measure synchronous inference time (prompt formatting, tokenization, embeddings, and ANE execution). They exclude one-time model initialization and UI IPC overhead, but the session median and p95 include first-use warm-up of each context tier.
