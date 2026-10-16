"""Synchronous System One API backed by the native Core AI host."""

import importlib
import json
from pathlib import Path
import sys
import types
import threading

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from tinkerbox_ane.d1 import CHUNK, RESIDUAL_SCALE, checkpoint
from tinkerbox_ane.d1.host import Host


def d1_code(checkpoint: Path) -> tuple[types.ModuleType, types.ModuleType]:
    pkg = "d1_remote"
    if pkg not in sys.modules:
        module = types.ModuleType(pkg)
        module.__path__ = [str(checkpoint)]
        sys.modules[pkg] = module
    return importlib.import_module(f"{pkg}.prompt"), importlib.import_module(f"{pkg}.api")


class Embeddings:
    def __init__(self, checkpoint: str | Path):
        with safe_open(Path(checkpoint) / "model.safetensors", "pt") as f:
            self.table = f.get_tensor("model.language_model.embed_tokens.weight")

    @torch.inference_mode()
    def gather(self, ids: list[int]) -> torch.Tensor:
        if len(ids) > CHUNK:
            raise ValueError(f"{len(ids)} tokens do not fit a {CHUNK}-token chunk")
        x = torch.zeros(1, self.table.shape[1], 1, CHUNK, dtype=torch.float16)
        rows = self.table[ids].float() * RESIDUAL_SCALE
        x[0, :, 0, :len(ids)] = rows.T
        return x

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
        return self.table[token_ids].float() @ hidden.float()


def load(asset):
    path = checkpoint()
    prompt, api = d1_code(path)

    class Model(api.SystemOneApi):
        def __init__(self):
            self._lock = threading.Lock()
            self._closed = False
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._embeddings = Embeddings(path)
            config = json.loads((path / "config.json").read_text())
            self._render_args = (self._tokenizer.bos_token or "", prompt.default_lead(config["model_type"]),
                                 prompt.DEFAULT_STATE_STYLE, prompt.DEFAULT_SYSTEM, "desc")
            self._host = Host(asset)
            self.context, self.chunk_width = self._host.context, self._host.width
            try:
                cfg = config["text_config"]
                dim = cfg["hidden_size"] // cfg["num_attention_heads"]
                theta = cfg["rope_parameters"]["rope_theta"]
                inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                angles = torch.arange(self.context, dtype=torch.float32)[:, None] * inv[None]
                angles = torch.cat((angles, angles), dim=-1).T
                cos, sin = angles.cos(), angles.sin()
                self._host.set_rope(cos.half().numpy(), sin.half().numpy())
            except BaseException:
                self.close()
                raise

        def tokens(self, state, question):
            text = prompt.render(self._tokenizer, state, prompt.as_question(question), *self._render_args)
            return self._tokenizer.encode(text, add_special_tokens=False)

        @torch.inference_mode()
        def _probabilities(self, ids, question):
            chunks = (self._embeddings.gather(ids[start:start + self.chunk_width])[0, :, 0].numpy()
                      for start in range(0, len(ids), self.chunk_width))
            hidden = self._host.hidden_chunks(chunks, len(ids))
            # The vocabulary normalizer cancels in the option softmax.
            option_ids = sorted({i for group in prompt.readout_ids(self._tokenizer, question) for i in group})
            logits = self._embeddings.logits(torch.from_numpy(hidden), option_ids)
            return prompt.readout(self._tokenizer, question, dict(zip(option_ids, logits.tolist())))

        def run(self, requests):
            with self._lock:
                if self._closed:
                    raise RuntimeError("the model is closed")
                prepared = []
                for state, questions, images in requests:
                    if images:
                        raise ValueError("image inputs are not supported")
                    rows = []
                    for question in questions:
                        question = prompt.as_question(question)
                        ids = self.tokens(state, question)
                        if not 0 < len(ids) <= self.context:
                            raise ValueError(f"prompt needs {len(ids)} tokens; context limit is {self.context}")
                        rows.append((ids, question))
                    prepared.append(rows)
                return [([self._probabilities(ids, q) for ids, q in rows], sum(len(ids) for ids, _ in rows))
                        for rows in prepared]

        def close(self):
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._host.close()

        def __enter__(self):
            if self._closed:
                raise RuntimeError("the model is closed")
            return self

        def __exit__(self, *_exc):
            self.close()

    return Model()
