"""Compile one fused AOT deployment with a deadline and SSD-write watchdog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from tinkerbox_ane.d1 import DEFAULT_ASSET, SOURCE
from tinkerbox_ane.d1.bundle import read_bundle, verify_aot
from tinkerbox_ane.d1.guard import run
from tinkerbox_ane.d1.host import device_architecture


def compile_model(source: Path = SOURCE, out: Path = DEFAULT_ASSET) -> Path:
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("keep source exports and deployment bundles in separate directories")
    spec = read_bundle(source)
    src = source / spec["asset"]
    if src.suffix != ".aimodel":
        raise ValueError("compile requires a source .aimodel bundle")
    architecture = device_architecture()
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"fused.{architecture}.aimodelc"
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix=".compile-", dir=out) as temporary:
            produced = Path(temporary) / target.name
            command = ["xcrun", "coreai-build", "compile", str(src), "--architecture", architecture,
                       "--platform", "macOS", "--min-deployment-version", "27.0",
                       "--preferred-compute", "neural-engine", "--output", str(produced)]
            check = run(command, out / "compile", timeout=600)
            if not check["passed"]:
                raise RuntimeError("fused AOT compilation failed; see compile.json and compile.log")
            verify_aot(produced, spec["source_hash"])
            produced.rename(target)
    verify_aot(target, spec["source_hash"])
    inspected = subprocess.run(["xcrun", "coreai-build", "inspect", str(target), "--json"],
                               capture_output=True, text=True, check=True, timeout=60)
    info = json.loads(inspected.stdout)
    expected = {f"c{capacity}" for capacity in spec["tiers"]}
    functions = {function["name"] for function in info["summary"]["functions"]}
    if functions != expected:
        raise ValueError(f"compiled function set differs: {functions}")
    spec.update(asset=target.name, architecture=architecture, min_os="27.0", deployment="aot-fused")
    manifest = out / ".model.json"
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    manifest.replace(out / "model.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_ASSET)
    args = parser.parse_args()
    print(compile_model(args.source, args.out))


if __name__ == "__main__":
    main()
