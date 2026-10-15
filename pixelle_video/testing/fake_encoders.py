"""
Deterministic encoder for tests: maps images to their mean colour and colour
words in text to the same RGB space. Lets retrieval tests assert exact ranks
without downloading a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pixelle_video.editing.encoders import l2_normalise

_COLOUR_WORDS = {
    "红": (1.0, 0.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "绿": (0.0, 0.5, 0.0),
    "green": (0.0, 0.5, 0.0),
    "蓝": (0.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0),
    "白": (1.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "黄": (1.0, 1.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
}


class FakeColourEncoder:
    name = "fake-colour"
    dim = 3

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            hits = [np.array(v) for w, v in _COLOUR_WORDS.items() if w in t.lower()]
            vecs.append(np.mean(hits, axis=0) if hits else np.full(3, 1e-3))
        return l2_normalise(np.array(vecs, dtype=np.float32).reshape(-1, 3))

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        from PIL import Image

        vecs = [
            np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) / 255.0
            for p in paths
        ]
        return l2_normalise(np.array(vecs, dtype=np.float32).reshape(-1, 3))
