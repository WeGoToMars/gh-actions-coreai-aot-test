"""Generate the 60-second speech case with upstream OmniVoice on CPU FP16, without importing the ANE port.

Each seed takes about 21 minutes on Apple M5 Pro; run one seed per watchdog run."""

import argparse
import json
from pathlib import Path
import time

import soundfile as sf
import torch
from omnivoice import OmniVoice

from tinkerbox_ane.transcription import transcribe
from speech_case import SETTINGS, TEXT, character_error

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/omnivoice/checkpoint'))
parser.add_argument('--out', type=Path, default=Path('artifacts/omnivoice/speech-reference'))
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(6)
model = OmniVoice.from_pretrained(args.checkpoint, dtype=torch.float16).eval()
torch.manual_seed(args.seed)
started = time.monotonic()
audio = model.generate(text=TEXT, **SETTINGS)[0]
path = args.out/f'{args.seed}.wav'
sf.write(path, audio.squeeze().float().numpy() if torch.is_tensor(audio) else audio, 24000, subtype='FLOAT')
transcript = transcribe(path, locale='en-US')
row = dict(seed=args.seed, elapsed=time.monotonic()-started, transcript=transcript,
           character_error=character_error(transcript, TEXT))
(args.out/f'{args.seed}.json').write_text(json.dumps(row, indent=2)+'\n')
print(json.dumps(row), flush=True)
