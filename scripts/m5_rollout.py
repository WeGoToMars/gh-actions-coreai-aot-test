import json
import os
from pathlib import Path
import sys
import urllib.request

model, stage = sys.argv[1:]
source = Path('artifacts/rollout/source')
out = Path('artifacts/rollout/compiled')
if stage == 'export':
    if model == 'd1-3b':
        from tinkerbox_ane.d1.export import export
        export(source)
    elif model in ('d3-nano', 'd3-mini'):
        from tinkerbox_ane.d3.export import export
        export(model, source)
    elif model == 'omnivoice':
        from tinkerbox_ane.omnivoice.export import export
        export(source)
    elif model == 'voxcpm2':
        from tinkerbox_ane.voxcpm2.export import export
        export(source)
elif stage == 'compile':
    if model == 'omnivoice':
        from tinkerbox_ane.omnivoice.export import compile
        compile(source, out, 'h17s')
    else:
        from tinkerbox_ane.bundle import compile_bundle
        from importlib import import_module
        module = 'd1' if model == 'd1-3b' else 'd3' if model.startswith('d3-') else 'voxcpm2'
        compile_bundle(source, out, import_module('tinkerbox_ane.' + module).KIND, 'h17s')
elif stage == 'compare':
    repos = {'d1-3b':('d1-3B','v2.1.0'), 'd3-nano':('d3-nano','v2.2.0'), 'd3-mini':('d3-mini','v2.2.0'), 'omnivoice':('OmniVoice','v2.2.0'), 'voxcpm2':('VoxCPM2','v2.2.0')}
    repo, rev = repos[model]
    url = f'https://huggingface.co/wegotomars/{repo}-CoreAI-ANE/resolve/{rev}/macos27/h16g/model.json'
    with urllib.request.urlopen(url, timeout=60) as response:
        published = json.load(response)
    built = json.loads((out / 'model.json').read_text())
    def hashes(spec):
        result = {k:spec[k] for k in ('source_hash','vision_source_hash','checkpoint_hash') if k in spec}
        for i, asset in enumerate(spec.get('assets', [])):
            result['asset:'+str(asset.get('role',i))] = asset['source_hash']
        return result
    actual, expected = hashes(built), hashes(published)
    rows = {k:{'built':actual.get(k),'huggingface':expected.get(k),'match':actual.get(k)==expected.get(k)} for k in sorted(actual.keys() | expected.keys())}
    report = {'model':model,'architecture':'h17s','published_manifest':url,'hashes':rows,'all_match':all(v['match'] for v in rows.values())}
    dest=Path('../diagnostics/hash-comparison.json'); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['all_match']: raise SystemExit(1)
