"""Compare compiled resource files with the published Hugging Face LFS SHA-256s."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

model=sys.argv[1]
repos={'d1-3b':('d1-3B','v2.1.0'),'d3-nano':('d3-nano','v2.2.0'),'d3-mini':('d3-mini','v2.2.0'),'omnivoice':('OmniVoice','v2.2.0'),'voxcpm2':('VoxCPM2','v2.2.0')}
repo,revision=repos[model]
prefix='macos27/h16g/'
url=f'https://huggingface.co/api/models/wegotomars/{repo}-CoreAI-ANE/tree/{revision}/{prefix.rstrip("/")}?recursive=1&expand=1&limit=100'
entries=[]
while url:
    with urllib.request.urlopen(url,timeout=60) as response:
        entries.extend(json.load(response))
        link=response.headers.get('Link','')
        url=link.split(';')[0].strip('<>') if 'rel="next"' in link else None
published={x['path'][len(prefix):]:x['lfs']['oid'] for x in entries if x.get('type')=='file' and x['path'].endswith('/resources.bin') and x.get('lfs')}
root=Path('artifacts/rollout/compiled')
local={str(p.relative_to(root)):p for p in root.rglob('resources.bin') if '.aimodelc' in str(p)}
rows={}
for name in sorted(local.keys() | published.keys()):
    digest=None
    if name in local:
        h=hashlib.sha256()
        with local[name].open('rb') as f:
            while chunk:=f.read(8*1024*1024): h.update(chunk)
        digest=h.hexdigest()
    expected=published.get(name)
    rows[name]={'built_sha256':digest,'huggingface_sha256':expected,'match':digest is not None and digest==expected}
report={'model':model,'architecture':'h16g','repo':f'wegotomars/{repo}-CoreAI-ANE','revision':revision,'files':rows,'all_match':bool(rows) and all(v['match'] for v in rows.values())}
out=Path('../diagnostics/resource-hashes.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)
