"""Compile d1's real vision tower using the unchanged shared compiler/watchdog."""

import json
from pathlib import Path

from tinkerbox_ane.coreai import compile_asset

source = Path('artifacts/d1-source')
manifest = json.loads((source / 'model.json').read_text())
compile_asset(source / manifest['vision_asset'], Path('artifacts/d1-vision/vision.h16g.aimodelc'),
              architecture='h16g', min_deployment_version='27.0', expected_functions=['main'])
