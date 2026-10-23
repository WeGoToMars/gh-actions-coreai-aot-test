#!/usr/bin/env python3
"""Offline Core AI disk cleanup. Preview by default; --apply deletes the listed paths."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
GB = 1_000_000_000


def checked(path: Path) -> Path:
    """Only operate on owned directories with no symlinked ancestors or mount crossings."""
    if path.resolve() != path:
        raise ValueError(f"refusing symlinked target: {path}")
    root = path.lstat()
    if not stat.S_ISDIR(root.st_mode) or root.st_uid != os.getuid():
        raise ValueError(f"not an owned directory: {path}")

    def failed(error: OSError) -> None:
        raise error

    for parent, directories, files in os.walk(path, followlinks=False, onerror=failed):
        for name in directories + files:
            entry = (Path(parent) / name).lstat()
            if entry.st_uid != os.getuid() or entry.st_dev != root.st_dev:
                raise ValueError(f"foreign owner or mounted filesystem: {Path(parent) / name}")
    return path


def active_clients(names: set[str]) -> list[str]:
    # Keep comm separate: executable paths may contain spaces in ps output.
    processes = {}
    for column in ("comm", "args"):
        result = subprocess.run(["ps", "-axo", f"pid=,{column}="],
                                capture_output=True, text=True, check=True)
        processes[column] = dict(line.strip().split(maxsplit=1)
                                 for line in result.stdout.splitlines() if line.strip())
    active = []
    for pid, executable in processes["comm"].items():
        if int(pid) == os.getpid():
            continue
        arguments = processes["args"].get(pid, executable)
        name = Path(executable).name
        if (name in names | {"coreai-build", "TinkerboxDemo", "ANECompilerService",
                             "ANELargeModelCompilerService"}
                or name.startswith("tinkerbox-ane-host")
                or (name.startswith("python") and (
                    re.search(r"\btinkerbox_ane\..*(?:guard|compile|export|validation)", arguments)
                    or re.search(r"(?:^|/)worker\.py(?:\s|$)", arguments)))):
            active.append(f"PID {pid}: {arguments}")
    return active


def staging(root: Path) -> list[Path]:
    found = []
    if not root.exists():
        return found
    if root.resolve() != root:
        raise ValueError(f"refusing symlinked search root: {root}")
    for parent, directories, _ in os.walk(root, followlinks=False):
        for name in directories[:]:
            path = Path(parent) / name
            if name.startswith((".host-", ".compile-")):
                found.append(path)
                directories.remove(name)
            elif path.is_symlink() or path.suffix in {".aimodel", ".aimodelc", ".app"}:
                directories.remove(name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="delete after preflight; never kills clients")
    mode.add_argument("--dry-run", action="store_true", help="preview only (the default)")
    parser.add_argument("--include-build", action="store_true", help="also remove this repo's artifacts/build")
    parser.add_argument("--include-models", action="store_true",
                        help="also remove Tinkerbox's model cache, including locally installed bundles")
    parser.add_argument("--target-free-gb", type=float, default=400,
                        help="report the gap to this much free disk space, in decimal GB (default: 400)")
    args = parser.parse_args()
    if not math.isfinite(args.target_free_gb) or args.target_free_gb <= 0:
        parser.error("--target-free-gb must be finite and positive")
    if sys.platform != "darwin" or os.getuid() == 0 or "SUDO_USER" in os.environ:
        parser.error("run as your normal macOS user, without sudo")

    home = Path.home().resolve()
    persistent = home / "Library/Application Support/coreai-cache"
    runtime = home / "Library/Caches/coreai-cache"
    models = home / "Library/Caches/tinkerbox-ane"
    temporary = Path(tempfile.gettempdir()).resolve()
    if not temporary.is_dir() or temporary.stat().st_uid != os.getuid():
        raise ValueError(f"invalid user temporary directory: {temporary}")

    targets: list[tuple[str, Path]] = []
    names = {"tinkerbox-ane-host"}
    for root, label in ((persistent, "persistent specialization"), (runtime, "Core AI locks/temp")):
        if root.exists() or root.is_symlink():
            checked(root)
            targets.append((label, root))
    if persistent.exists():
        for version in persistent.iterdir():
            if version.is_dir() and not version.is_symlink():
                names.update(client.name for client in version.iterdir() if client.is_dir())

    # Only MPSGraph scratch directories whose creating PID has exited.
    graph = temporary / "com.apple.MetalPerformanceShadersGraphHost"
    if graph.exists() or graph.is_symlink():
        checked(graph)
        for path in sorted(graph.iterdir()):
            match = re.fullmatch(r"mpsgraph-(\d+)-.+", path.name)
            if match and path.is_dir():
                try:
                    os.kill(int(match[1]), 0)
                except ProcessLookupError:
                    targets.append(("orphan MPSGraph scratch", path))
                except PermissionError:
                    pass  # Treat inaccessible/reused PIDs as alive.

    if args.include_build:
        targets.append(("build outputs", REPO / "artifacts/build"))
    if args.include_models:
        targets.append(("downloaded/installed models", models))
    for root in (REPO / "artifacts", models):
        targets.extend(("abandoned host/compiler staging", path) for path in staging(root))

    # Parents subsume staging children; do not count or delete a child twice.
    selected: list[tuple[str, Path]] = []
    for label, path in sorted(targets, key=lambda item: len(item[1].parts)):
        if (path.exists() or path.is_symlink()) and not any(
            path == parent or parent in path.parents for _, parent in selected
        ):
            selected.append((label, checked(path)))
    before = shutil.disk_usage(home).free
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — free now: {before / GB:.1f} GB")
    total = 0
    if selected:
        # One du invocation avoids counting hard-linked files multiple times.
        usage = subprocess.run(["du", "-k", "-s", *[str(p) for _, p in selected]],
                               capture_output=True, text=True, check=True)
        sizes = dict((line.split("\t", 1)[1], int(line.split("\t", 1)[0]) * 1024)
                     for line in usage.stdout.splitlines())
        for label, path in selected:
            size = sizes[str(path)]
            total += size
            print(f"  {size / GB:7.2f} GB  {label}: {path}")
    print(f"Selected allocated size: {total / GB:.1f} GB")
    print(f"Estimated free after cleanup: {(before + total) / GB:.1f} GB; "
          f"gap to {args.target_free_gb:g} GB: "
          f"{max(0, args.target_free_gb - (before + total) / GB):.1f} GB")
    print("APFS clones, snapshots, open files, and concurrent writes can change actual recovery.")
    print("Preserved: other artifacts/research, Hugging Face checkpoints, uv/pip, and system ANE caches.")

    clients = active_clients(names)
    if clients:
        print("\nClose Core AI clients and stop benchmarks/exports/compiles before cleanup:")
        print("\n".join(clients))
        if args.apply:
            print("Refusing cleanup; nothing deleted.", file=sys.stderr)
            return 1
    if not args.apply:
        print("\nNothing deleted. Add --apply to perform this plan.")
        return 0

    for _, path in selected:
        # Recheck clients and path safety immediately before each removal.
        if active_clients(names):
            raise RuntimeError("a Core AI client started; cleanup stopped")
        shutil.rmtree(checked(path))
        print(f"Removed {path}", flush=True)
    after = shutil.disk_usage(home).free
    print(f"Free now: {after / GB:.1f} GB (change: {(after - before) / GB:+.1f} GB)")
    print(f"Remaining gap to {args.target_free_gb:g} GB: "
          f"{max(0, args.target_free_gb - after / GB):.1f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Cleanup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
