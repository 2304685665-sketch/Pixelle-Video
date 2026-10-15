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


class FakeDescriber:
    """Describes an image by its dominant colour name (Chinese), counting calls."""

    name = "fake-describer"

    def __init__(self, fail_on: set[str] | None = None):
        self.calls = 0
        self.fail_on = fail_on or set()

    def _colour(self, path) -> str:
        from PIL import Image

        r, g, b = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
        names = {"红色": (255, 0, 0), "绿色": (0, 128, 0), "蓝色": (0, 0, 255), "白色": (255, 255, 255),
                 "黄色": (255, 255, 0), "品红色": (255, 0, 255)}
        return min(names, key=lambda n: sum((a - b_) ** 2 for a, b_ in zip((r, g, b), names[n])))

    async def describe_image(self, path) -> str:
        self.calls += 1
        if any(f in str(path) for f in self.fail_on):
            raise RuntimeError("vlm error")
        return f"画面是一片{self._colour(path)}"

    async def describe_video(self, path) -> str:
        self.calls += 1
        return f"视频素材 {Path(path).stem}"


class FakeAssignLLM:
    """Picks assets whose description shares a colour word with the query; counts calls."""

    def __init__(self, descriptions: dict[str, str], hallucinate: bool = False):
        self.descriptions, self.hallucinate, self.calls = descriptions, hallucinate, 0

    async def __call__(self, prompt: str, response_type=None, **_):
        self.calls += 1
        query = prompt.split("旁白：", 1)[1].split("\n", 1)[0]
        aliases = {"红": ("红", "red"), "绿": ("绿", "green"), "蓝": ("蓝", "blue"),
                   "白": ("白", "white"), "黄": ("黄", "yellow")}
        q = query.lower()
        hits = [a for a, d in self.descriptions.items()
                if any(w in d and any(x in q for x in xs) for w, xs in aliases.items())]
        paths = (["made/up/path.mp4"] if self.hallucinate else []) + hits
        return response_type(paths=paths)
