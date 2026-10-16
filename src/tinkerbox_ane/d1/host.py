"""Streaming pipe client for the Core AI host."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

BINARY = Path(__file__).with_name("fused-host")
BUILD_DIR = Path.home() / "Library/Caches/tinkerbox-ane/d1"


@lru_cache(maxsize=1)
def device_architecture() -> str:
    result = subprocess.run([str(BINARY), "--architecture"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


class Host:
    def __init__(self, asset: str | Path):
        from tinkerbox_ane.d1.bundle import read_bundle

        asset = Path(asset).resolve()
        spec = read_bundle(asset)
        if Path(spec["asset"]).suffix != ".aimodelc":
            raise ValueError("load requires a fused AOT bundle; run python -m tinkerbox_ane.d1.compile first")
        if spec.get("architecture") and spec["architecture"] != device_architecture():
            raise ValueError(f"bundle targets {spec['architecture']}, not this Mac")
        # Stage on the asset volume: APFS clones must stay on the same filesystem.
        self._directory = tempfile.TemporaryDirectory(prefix=".d1host-", dir=asset.parent)
        clones = Path(self._directory.name) / "clones"
        clones.mkdir()
        environment = dict(os.environ, COW_CLIENT_DEST=str(clones),
                           COW_ASSET_PREFIX=str((asset / spec["asset"]).resolve()))
        try:
            self._process = subprocess.Popen([str(BINARY), str(asset)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment)
            ready = json.loads(self._read(self._u32()))
            if "error" in ready:
                raise RuntimeError(f"the Core AI host could not load {asset}: {ready['error']}")
            self.hidden_size = ready["hidden"]
            self.width, self.context = ready["width"], ready["context"]
        except BaseException:
            self.close()
            raise

    def _read(self, count: int) -> bytes:
        data = self._process.stdout.read(count)
        if len(data) != count:
            raise RuntimeError(f"the Core AI host exited (status {self._process.poll()})")
        return data

    def _u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _reply(self) -> None:
        if self._read(1) != b"\0":
            raise RuntimeError(f"Core AI host: {self._read(self._u32()).decode()}")

    def _send(self, *payloads) -> None:
        try:
            for payload in payloads:
                self._process.stdin.write(payload)
            self._process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"the Core AI host exited (status {self._process.poll()})") from error

    def set_rope(self, cos, sin) -> None:
        import numpy as np
        cos, sin = (np.ascontiguousarray(t, dtype=np.float16) for t in (cos, sin))
        if cos.shape != sin.shape or cos.ndim != 2:
            raise ValueError("cos and sin must be matching (head_dim, positions) tables")
        self._send(b"R" + struct.pack("<II", *cos.shape), memoryview(cos).cast("B"), memoryview(sin).cast("B"))
        self._reply()

    def hidden_chunks(self, chunks, tokens: int):
        import numpy as np
        if not 0 < tokens <= self.context:
            raise ValueError(f"prompt needs {tokens} tokens; context limit is {self.context}")
        length = -(-tokens // self.width) * self.width
        producer = iter(chunks)
        try:
            self._send(b"S" + struct.pack("<I", tokens))
            for _ in range(length // self.width):
                self._reply()
                chunk = np.ascontiguousarray(next(producer), dtype=np.float16)
                if chunk.shape != (self.hidden_size, self.width):
                    raise ValueError(f"embedding chunk must be ({self.hidden_size}, {self.width})")
                self._send(memoryview(chunk).cast("B"))
            self._reply()
            hidden = np.frombuffer(self._read(2 * self.hidden_size), dtype=np.float16).copy()
            return hidden
        except BaseException:
            # A failed producer cannot leave the native host waiting for a partial chunk.
            if self._process.poll() is None:
                self._process.kill()
            self.close()
            raise

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write(b"Q")
                process.stdin.close()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        if process is not None:
            process.wait()
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        self._directory.cleanup()
