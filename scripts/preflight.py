"""Record bounded, public diagnostics without dumping the runner environment."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

phase = sys.argv[1]
out = Path('diagnostics') / phase
out.mkdir(parents=True, exist_ok=True)


def record(name, command, timeout=120):
    print(f'[{name}] {command}', flush=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        value = dict(command=command, returncode=result.returncode,
                     stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as error:
        value = dict(command=command, timeout=timeout,
                     stdout=str(error.stdout), stderr=str(error.stderr))
    (out / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2), flush=True)
    return value


summary = {'phase': phase, 'runner': {name: os.environ.get(name) for name in
    ('ImageOS', 'ImageVersion', 'RUNNER_OS', 'RUNNER_ARCH', 'RUNNER_NAME')}}
summary['disk'] = shutil.disk_usage('.')._asdict()
record('os', ['sw_vers'])
record('hardware', ['sysctl', 'hw.model', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'])
record('hypervisor', ['sysctl', 'kern.hv_vmm_present'])
record('xcode', ['xcodebuild', '-version'])
record('sdk', ['xcrun', '--sdk', 'macosx', '--show-sdk-version'])
record('disk', ['df', '-h'])
record('memory', ['vm_stat'])
io = record('io-storage', ['ioreg', '-r', '-c', 'IOBlockStorageDriver', '-w0', '-l'])
summary['ssd_write_counters'] = re.findall(r'"Bytes \(Write\)"=(\d+)', io['stdout'])
summary['watchdog_disk_headroom'] = summary['disk']['free'] >= 15 * 1024**3
record('compiler-location', ['xcrun', '--find', 'coreai-build'])
record('compiler-help', ['xcrun', 'coreai-build', 'compile', '--help'])
record('compiler-workers', ['pgrep', '-fl', 'ANE.*CompilerService.xpc/Contents/MacOS/'])
record('sudo', ['sudo', '-n', '/usr/bin/true'])
probe = out / 'probe.swift'
probe.write_text('import CoreAI\nprint("deviceArchitectureName=\\(AIModel.deviceArchitectureName)")\n')
compiled = record('probe-build', ['xcrun', 'swiftc', '-target', 'arm64-apple-macos27.0',
                                 str(probe), '-o', str(out / 'probe')], timeout=180)
if compiled.get('returncode') == 0:
    summary['device_probe'] = record('probe-run', [str(out / 'probe')], timeout=30)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
