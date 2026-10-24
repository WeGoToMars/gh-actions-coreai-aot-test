"""Exercise combined conditioning with a roughly 30-second neutral reference."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from tinkerbox_ane import load
from tinkerbox_ane.transcription import transcribe
from gate import character_error

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--reference', type=Path, default=Path('artifacts/voxcpm2/reference/reference-16k.wav'))
parser.add_argument('--out', type=Path, default=Path('artifacts/voxcpm2/long-reference'))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
waveform, rate = sf.read(args.reference)
reference = args.out/'reference.wav'
sf.write(reference, np.tile(waveform, 8), rate, subtype='FLOAT')
text = ('Every morning, the neighborhood slowly comes to life. '
        'People open their windows, birds gather in the trees, '
        'and a familiar voice welcomes everyone to a new day.')
with load('voxcpm2', args.asset) as model:
    started = time.monotonic()
    report = dict(asset=str(args.asset), context_positions=model.capabilities['context_positions'],
                  reference_seconds=8*len(waveform)/rate, text=text, passed=False)
    try:
        result = model.generate(text, ref_audio=str(reference),
            ref_text='Hello! Welcome to Tinkerbox. '*8, mode='clone_continuation', seed=23)[0]
    except ValueError as error:
        report.update(error=str(error), elapsed=time.monotonic()-started)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(1)
    audio = args.out/'speech.wav'
    sf.write(audio, result, model.sampling_rate, subtype='FLOAT')
    report.update(seconds=len(result)/model.sampling_rate, elapsed=time.monotonic()-started,
                  finite=bool(np.isfinite(result).all()), transcript=transcribe(audio, locale='en-US'))
    report['character_error'] = character_error(report['transcript'], text)
    report['passed'] = report['finite'] and report['seconds'] > 6 and report['character_error'] <= 0.20
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['passed'] else 1)
