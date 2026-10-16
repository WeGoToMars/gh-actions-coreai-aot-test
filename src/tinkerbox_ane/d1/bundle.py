"""Portable fused bundle manifests and source identity."""

from __future__ import annotations

import json
from pathlib import Path
import plistlib
import re


def download_bundle() -> Path:
    from huggingface_hub import snapshot_download

    from tinkerbox_ane.d1 import ARTIFACT_REPO, ARTIFACT_REVISION
    from tinkerbox_ane.d1.host import BUILD_DIR, device_architecture

    architecture = device_architecture()
    folder = f"macos27/{architecture}"
    root = BUILD_DIR / "models" / ARTIFACT_REVISION
    snapshot_download(ARTIFACT_REPO, revision=ARTIFACT_REVISION,
                      allow_patterns=f"{folder}/*", local_dir=root)
    bundle = root / folder
    if not (bundle / "model.json").is_file():
        raise ValueError(f"no AOT bundle for {architecture} in {ARTIFACT_REPO}@{ARTIFACT_REVISION}")
    return bundle


def source_hash(asset: Path) -> str:
    if asset.suffix == ".aimodelc":
        return json.loads((asset / "metadata.json").read_text())["sourceHash"].lower()
    return (asset / "main.hash").read_bytes().hex()


def verify_aot(asset: Path, expected: str) -> None:
    if asset.suffix != ".aimodelc" or source_hash(asset) != expected:
        raise ValueError(f"{asset}: AOT asset does not match the source identity")
    manifests = list(asset.rglob("manifest.plist"))
    if len(manifests) != 1:
        raise ValueError(f"{asset}: expected one compiled manifest")
    manifest = plistlib.loads(manifests[0].read_bytes())
    modules = next(iter(manifest["Package Version"].values()))["Optimized Modules"]
    if not modules:
        raise ValueError(f"{asset}: missing compiled modules")
    for module in modules.values():
        entries = module.get("Entry Function Attributes")
        if not isinstance(entries, dict) or not entries or not all(
            "mps.fullyPlacedOnANE" in attributes for attributes in entries.values()
        ):
            raise ValueError(f"{asset}: compiled entries are not fully placed on ANE")


def read_bundle(root: Path) -> dict:
    root = Path(root)
    spec = json.loads((root / "model.json").read_text())
    if spec.get("kind") != "d1ane-fused" or spec.get("version") != 1:
        raise ValueError("unsupported fused bundle manifest")
    width, context, tiers = spec["width"], spec["context"], spec["tiers"]
    if (width <= 0 or context <= width or context % width or not tiers
            or tiers != sorted(set(tiers)) or tiers[-1] != context
            or any(t <= width or t > context or t % width for t in tiers)):
        raise ValueError("invalid fused attention capacities")
    for name in ("hidden", "heads", "kv_heads", "head_dim", "conv_taps"):
        if not isinstance(spec[name], int) or spec[name] <= 0:
            raise ValueError(f"invalid {name}")
    if (spec["hidden"] != spec["heads"] * spec["head_dim"]
            or spec["heads"] % spec["kv_heads"] or spec["conv_taps"] < 2):
        raise ValueError("inconsistent model dimensions")
    for name in ("conv_layers", "attention_layers"):
        layers = spec[name]
        if not layers or layers != sorted(set(layers)) or any(i < 0 for i in layers):
            raise ValueError(f"invalid {name}")
    if set(spec["conv_layers"]) & set(spec["attention_layers"]):
        raise ValueError("overlapping convolution and attention layers")
    name = spec["asset"]
    if (not name or name != Path(name).name or "\\" in name or name in {".", ".."}
            or Path(name).suffix not in {".aimodel", ".aimodelc"}):
        raise ValueError(f"unsafe or unsupported asset path: {name!r}")
    expected = spec["source_hash"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or source_hash(root / name) != expected:
        raise ValueError("bundle source identity mismatch")
    if Path(name).suffix == ".aimodelc":
        verify_aot(root / name, expected)
    return spec
