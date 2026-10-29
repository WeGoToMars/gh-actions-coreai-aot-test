"""Collect compiler reports and relevant OS diagnostics after success or failure."""

import json
from pathlib import Path
import shutil
import subprocess

out = Path('diagnostics/final')
out.mkdir(parents=True, exist_ok=True)
for path in Path('source/artifacts').rglob('*'):
    if path.is_file() and path.suffix in ('.json', '.log', '.plist'):
        # Preserve build manifests/reports, not model weights or checkpoint code.
        if 'checkpoint' in path.parts or '.aimodelc' in ''.join(path.parts) or '.aimodel' in ''.join(path.parts):
            continue
        destination = out / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
for label, command in {
    'disk': ['df', '-h'],
    'swap': ['sysctl', 'vm.swapusage'],
    'compiler-workers': ['pgrep', '-fl', 'ANE.*CompilerService.xpc/Contents/MacOS/'],
    'compiler-events': ['log', 'show', '--last', '30m', '--style', 'compact', '--predicate',
                        'process CONTAINS "ANE" OR process == "coreai-build"'],
}.items():
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (out / f'{label}.log').write_text(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        (out / f'{label}.log').write_text('diagnostic collection timed out\n')
for root in [Path('/Library/Logs/DiagnosticReports'), Path.home() / 'Library/Logs/DiagnosticReports']:
    for path in root.glob('*'):
        if path.is_file() and ('ANE' in path.name or 'coreai' in path.name.lower()):
            shutil.copy2(path, out / path.name)
sizes = {}
for name in ('d1-source', 'd1-h16g'):
    root = Path('source/artifacts') / name
    sizes[name] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
(out / 'artifact-sizes.json').write_text(json.dumps(sizes, indent=2) + '\n')
print(json.dumps(sizes, indent=2))
