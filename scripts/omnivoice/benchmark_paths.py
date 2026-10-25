"""Measure fused versus blockwise OmniVoice dispatch on the same FP16 inputs."""

import argparse
import ctypes
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.host import TensorHost
from tinkerbox_ane.omnivoice import KIND


class Usage(ctypes.Structure):
    # macOS sys/resource.h, rusage_info_v0.
    _fields_ = [('uuid', ctypes.c_uint8*16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime')]


class Memory:
    """Sample the native client's resident bytes and physical footprint."""
    def __init__(self, pid):
        self.pid = pid
        self.library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.peak_rss = self.peak_footprint = 0
        self.done = threading.Event()
        self.sample()
        self.baseline = dict(resident_bytes=self.peak_rss, footprint_bytes=self.peak_footprint)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        usage = Usage()
        if self.library.proc_pid_rusage(self.pid, 0, ctypes.byref(usage)):
            raise OSError(ctypes.get_errno(), 'proc_pid_rusage failed')
        self.peak_rss = max(self.peak_rss, usage.resident_size)
        self.peak_footprint = max(self.peak_footprint, usage.phys_footprint)

    def run(self):
        while not self.done.wait(.005):
            self.sample()

    def start(self):
        self.thread.start()

    def finish(self):
        self.sample()
        self.done.set()
        self.thread.join()
        return dict(baseline=self.baseline, sampled_peak_resident_bytes=self.peak_rss,
                    sampled_peak_footprint_bytes=self.peak_footprint, sample_interval_ms=5)


def compare(actual, expected):
    a, b = torch.from_numpy(actual).double().flatten(), torch.from_numpy(expected).double().flatten()
    cosine = float(F.cosine_similarity(a, b, dim=0))
    relative = float((a-b).norm()/b.norm())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(cosine=cosine, relative_l2=relative, finite=finite,
                passed=finite and cosine >= .95 and relative <= .35)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--reference', type=Path, default=Path('artifacts/omnivoice/chunked-reference/1100.pt'))
parser.add_argument('--lengths', type=int, nargs='+', default=[1024, 1100, 1536, 2048, 2200, 3072, 4096])
parser.add_argument('--repeats', type=int, default=7)
parser.add_argument('--visible-fraction', type=float, default=1.0,
                    help='Visible prefix fraction; smaller values model padded CFG rows')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if not 0 < args.visible_fraction <= 1:
    parser.error('--visible-fraction must be above zero and at most one')
torch.set_num_threads(4)
spec = read_bundle(args.asset, KIND)
assets = {row['role']: row for row in spec['assets']}
config = dict(hidden=spec['hidden'], heads=spec['heads'], kvHeads=spec['kv_heads'],
              headDim=spec['head_dim'], width=spec['attention_width'],
              logits=spec['codebooks']*spec['audio_vocab'], projections=spec['projections'],
              finishes=spec['finishes'], attention=spec['attention'], head=spec['head_function'])
reference = torch.load(args.reference, weights_only=True)['embeddings'].half()
report = dict(device=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
              os=platform.mac_ver()[0], architecture=spec['architecture'], asset=str(args.asset),
              asset_hashes={role: row['source_hash'] for role, row in assets.items()},
              visible_fraction=args.visible_fraction,
              method='Identical repeated upstream-captured embeddings; one sequence with the configured visible prefix per call. '
                     'Fresh native process per path/length. First call excluded from steady-state latency. '
                     'Wall time includes Python/native transport and returned bucket logits. '
                     '5 ms native-client memory samples include first-call setup and steady state; '
                     'exclude Python, CPU codec, and separate system ANE services. '
                     'Path order alternates by length; sequential ANE execution.',
              cases={})


def save():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


with torch.inference_mode():
    for index, length in enumerate(args.lengths):
        capacity = next(n for n in spec['short_capacities'] if n >= length)
        visible = max(1, int(length*args.visible_fraction))
        hidden = reference.repeat(1, length//reference.shape[1]+1, 1)[:, :length]
        hidden = hidden.transpose(1, 2).unsqueeze(2).numpy()
        inverse = spec['rope_theta']**(-np.arange(0, spec['head_dim'], 2, dtype=np.float32)/spec['head_dim'])
        angles = np.arange(capacity, dtype=np.float32)[:, None]*inverse
        angles = np.concatenate([angles, angles], axis=1).T[None, None]
        cos, sin = np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)
        values = dict(hidden=hidden, cos=cos[..., :length], sin=sin[..., :length])
        fused_hidden = np.zeros((1, spec['hidden'], 1, capacity), np.float16)
        fused_hidden[..., :length] = hidden
        mask = np.full((1, 1, capacity, capacity), -30000, np.float16)
        np.fill_diagonal(mask[0, 0], 0)
        mask[..., :visible, :visible] = 0
        fused = dict(hidden=fused_hidden, cos=cos, sin=sin, mask=mask)
        row = report['cases'][str(length)] = dict(fused_capacity=capacity, visible_positions=visible)
        outputs = {}
        for path in (['fused', 'blockwise'] if index % 2 == 0 else ['blockwise', 'fused']):
            gc.collect()
            started = time.perf_counter()
            host = TensorHost(args.asset/assets['short' if path == 'fused' else 'diffusion']['asset'],
                              bidirectional=config if path == 'blockwise' else None)
            setup = time.perf_counter()-started
            memory = Memory(host._process.pid)
            memory.start()
            try:
                def call():
                    if path == 'fused':
                        return host.run(f'omnivoice_s{capacity}', fused)
                    return host.full_attention(values, length, visible)
                started = time.perf_counter()
                result = call()
                first_call = time.perf_counter()-started
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    result = call()
                    times.append(time.perf_counter()-started)
                outputs[path] = result[..., :visible].copy()
                row[path] = dict(setup_seconds=setup, first_call_seconds=first_call,
                                 seconds=times, median_seconds=statistics.median(times),
                                 min_seconds=min(times), max_seconds=max(times), memory=memory.finish())
                print(length, path, row[path]['median_seconds'], 'seconds',
                      row[path]['memory']['sampled_peak_footprint_bytes']/1024**2, 'MiB footprint', flush=True)
            finally:
                memory.done.set()
                memory.thread.join()
                host.close()
            del result
            save()
        row['agreement'] = compare(outputs['fused'], outputs['blockwise'])
        row['fused_speedup'] = row['blockwise']['median_seconds']/row['fused']['median_seconds']
        del outputs
        save()
report['passed'] = all(row['agreement']['passed'] for row in report['cases'].values())
save()
raise SystemExit(0 if report['passed'] else 1)
