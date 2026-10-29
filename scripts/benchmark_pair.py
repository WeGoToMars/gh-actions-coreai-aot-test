"""Compare two h17s bundles on the same Mac using public task APIs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from PIL import Image
import tinkerbox_ane

parser=argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('local',type=Path)
parser.add_argument('hosted',type=Path)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
image=Image.new('RGB',(512,512),'blue')
question={'type':'noul','instructions':'Is this a refund request?'}
results={}
for label,path in [('local',args.local),('hosted',args.hosted)]:
    cases=[]
    started=time.perf_counter()
    with tinkerbox_ane.load(args.model,path) as model:
        load_seconds=time.perf_counter()-started
        if args.model.startswith('d'):
            for state,images in [('Please refund the duplicate charge.',None),('A blue picture.',[image])]:
                model.probabilities(state,[question],images)
                times=[]; outputs=[]
                for _ in range(3):
                    t=time.perf_counter(); outputs.append(model.probabilities(state,[question],images)[0]); times.append(time.perf_counter()-t)
                cases.append({'state':state,'image':bool(images),'seconds':times,'median_seconds':statistics.median(times),'probabilities':outputs})
        else:
            model.generate('Hello there.',seed=42)
            for seed in (42,43,44):
                t=time.perf_counter(); audio=np.asarray(model.generate('Hello there.',seed=seed)[0],dtype=np.float32); elapsed=time.perf_counter()-t
                out=args.out.parent / (args.model+'-'+label+'-'+str(seed)+'.npy'); out.parent.mkdir(parents=True,exist_ok=True); np.save(out,audio)
                cases.append({'seed':seed,'seconds':elapsed,'samples':int(audio.size),'audio':str(out)})
    spec=json.loads((path/'model.json').read_text())
    results[label]={'asset':str(path),'source':spec.get('source'),'checkpoint_hash':spec.get('checkpoint_hash'),
                    'source_hash':spec.get('source_hash'),'vision_source_hash':spec.get('vision_source_hash'),
                    'stage_hashes':[a.get('source_hash') for a in spec.get('assets',[])],
                    'load_seconds':load_seconds,'cases':cases}
comparisons=[]
for local,hosted in zip(results['local']['cases'],results['hosted']['cases']):
    row={'local_median_or_seconds':local.get('median_seconds',local.get('seconds')),
         'hosted_median_or_seconds':hosted.get('median_seconds',hosted.get('seconds'))}
    if 'probabilities' in local:
        a=np.asarray(local['probabilities'][0]); b=np.asarray(hosted['probabilities'][0])
        row.update(max_absolute_probability_difference=float(np.max(np.abs(a-b))), same_top_answer=bool(np.argmax(a)==np.argmax(b)))
    else:
        a=np.load(local['audio']); b=np.load(hosted['audio']); n=min(a.size,b.size)
        row.update(local_samples=int(a.size),hosted_samples=int(b.size),
                   rms_difference=float(np.sqrt(np.mean((a[:n]-b[:n])**2))) if n else None)
    comparisons.append(row)
results['comparisons']=comparisons
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
