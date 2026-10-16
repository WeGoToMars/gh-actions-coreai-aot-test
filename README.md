# Tinkerbox for Apple Neural Engine

Run [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) on Apple's Neural Engine through Core AI. FP16 backbone, 128-token chunks, 32,768-token context. Tokenization, embeddings and answer readout run on the CPU.

Requires Apple Silicon, macOS 27, and Python 3.12 or 3.13.

## Quickstart

Install the wheel from the GitHub release (Xcode is not required):

```sh
python -m pip install ./tinkerbox_ane-*.whl
```

```python
from tinkerbox_ane import d1

with d1.load() as model:
    result = model.system_one(
        "I was charged twice this month.",
        {"refund": {"type": "noul", "instructions": "Is this a refund request?"}},
    )
    print(result)
```

`load()` downloads the matching Core AI architecture from [wegotomars/d1-3B-CoreAI-ANE](https://huggingface.co/wegotomars/d1-3B-CoreAI-ANE) (`v1.0.0`) into `~/Library/Caches/tinkerbox-ane/d1/models/`. Pass a directory path to `load(path)` to select a local AOT bundle.

The API exposes `system_one`, `system_one_batch`, `probabilities`, and `probabilities_batch` for binary, choice, and score queries. Requests run sequentially. Prompts and images exceeding 32,768 rendered tokens are rejected.

## SwiftUI Demo

A desktop demo is available in [examples/swiftui](examples/swiftui). To build and launch it from source:

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py
```

Pass `--build-only` to create `artifacts/swiftui/Tinkerbox.app` and the release ZIP without launching. The prebuilt app manages its own Python environment and dependencies via an embedded `uv` binary.

## Architecture & Staging

The Swift host manages recurrent state and allocates attention cache buffers dynamically. Initial execution compiles and specializes the AOT asset for the local ANE hardware, after which Core AI reuses its persistent cache.

Weight staging uses an [APFS copy-on-write workaround](docs/cow-staging.md) to eliminate redundant disk writes during specialization.

## Building from Source

Building requires Xcode 27 / Swift 6.4 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv build
```

The wheel in `dist/` bundles the native host and links against macOS system frameworks. Attach it to the GitHub release. Source and editable installs compile the host during installation.

To export and compile model weights locally:

```sh
uv run python -m tinkerbox_ane.d1.export
uv run python -m tinkerbox_ane.d1.compile
```

Export writes `artifacts/build/d1_fp16_fused128_ctx32768`; compilation produces the final bundle in `artifacts/d1_fp16_fused128_ctx32768`. Load the local result with `d1.load(d1.DEFAULT_ASSET)`. Rebuilds require fresh `--out` directories.

Compilation verifies source identity, all nine functions, and ANE placement. Without cached sudo authentication, worker cancellation is best effort and emits a warning.

## Validation

```sh
uv run python scripts/build_reference.py
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/check -- uv run python scripts/validate.py
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/smoke -- uv run pytest -q
```

Reference generation downloads benchmark inputs and runs the official FP32 CPU model. Validation checks prompt tokens and probabilities across all nine cache capacities. The two public API smoke tests require a local bundle and cached checkpoint.

Weights and upstream code are pinned to `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432` under the [LFM Open License v1.0](LICENSES/LFM-Open-License-v1.0.txt). See [NOTICE](NOTICE).
