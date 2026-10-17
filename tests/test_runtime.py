import math
import subprocess
import sys

import pytest
from huggingface_hub import try_to_load_from_cache

from tinkerbox_ane.d1 import DEFAULT_ASSET, REPO, REVISION, load
from tinkerbox_ane.d1.host import (
    BINARY,
    SUPPORTED_ARCHITECTURES,
    UNSUPPORTED_ARCHITECTURES,
    device_architecture,
)


@pytest.mark.skipif(sys.platform != 'darwin', reason='fused-host is only built for macOS')
def test_host_binary_architecture_handshake():
    """Verify native fused-host binary exists, executes, and returns a recognized architecture."""
    assert BINARY.is_file(), f"native host binary missing: {BINARY}"
    result = subprocess.run([str(BINARY), "--architecture"], capture_output=True, text=True, check=True)
    arch = result.stdout.strip()
    assert arch, "host binary returned empty architecture string"
    assert arch in SUPPORTED_ARCHITECTURES or arch in UNSUPPORTED_ARCHITECTURES, f"unexpected architecture: {arch}"
    assert device_architecture() == arch


GREETING = {'type': 'noul', 'instructions': 'Is this a greeting?'}


@pytest.fixture
def model(monkeypatch):
    if sys.platform != 'darwin' or not (DEFAULT_ASSET / 'model.json').is_file():
        pytest.skip('requires the local Core AI bundle')
    if not isinstance(try_to_load_from_cache(REPO, 'model.safetensors', revision=REVISION), str):
        pytest.skip('requires the pinned checkpoint in the local cache')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    with load(DEFAULT_ASSET) as model:
        yield model


def test_public_decisions_batches_and_request_reset(model):
    questions = {
        'greeting': GREETING,
        'topic': {'type': 'choice', 'instructions': 'Choose a topic.',
                  'criteria': {'hello': 'A greeting', 'billing': 'A payment dispute'}},
        'tone': {'type': 'score', 'instructions': 'How friendly is this?',
                 'criteria': ['Unfriendly', 'Neutral', 'Friendly']},
    }
    state = 'Hi'
    result = model.system_one(state, questions)
    assert result['answers']['greeting']['noul'] == pytest.approx(0.97966643, abs=0.01)
    assert result['answers']['topic']['choice'] == 'hello'
    assert 0 <= result['answers']['tone']['score'] <= 2
    assert result['usage'] == {
        'input_tokens': sum(len(model.tokens(state, q)) for q in questions.values()),
        'output_tokens': 0,
    }
    assert model.system_one_batch([(state, questions), (state, {})]) == [
        result, {'answers': {}, 'usage': {'input_tokens': 0, 'output_tokens': 0}},
    ]
    expected = model.probabilities(state, [GREETING])
    assert model.probabilities_batch([(state, [GREETING]), (state, [GREETING])]) == [expected, expected]
    long = model.probabilities('Hello. ' * 150, [GREETING])[0]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in long)
    assert sum(long) == pytest.approx(1)
    assert model.probabilities(state, [GREETING]) == expected


def test_public_input_limits_and_close(model):
    with pytest.raises(ValueError, match='image'):
        model.probabilities('Hi', [GREETING], images=[object()])
    with pytest.raises(ValueError, match='context limit'):
        model.probabilities('hello ' * model.context, [GREETING])
    assert model.probabilities('Hi', [GREETING])[0][0] > 0.9
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match='closed'):
        model.probabilities('Hi', [GREETING])
