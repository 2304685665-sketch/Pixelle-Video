"""
Text/image encoders sharing one embedding space.

``ChineseClipEncoder`` wraps Chinese-CLIP (OFA-Sys) through ``transformers``.
It loads weights from a local directory only: the model is downloaded once by
the user (see docs/editing/README.md) so indexing never depends on network
access to a model hub at runtime.

All encoders return L2-normalised float32 arrays so cosine similarity is a dot
product.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    name: str
    dim: int

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray: ...


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class ChineseClipEncoder:
    """
    Chinese-CLIP encoder loaded from a local ``model_dir``.

    Expected layout: the output of
    ``huggingface-cli download OFA-Sys/chinese-clip-vit-base-patch16 --local-dir <model_dir>``.
    Requires the ``editing-models`` extra (torch, transformers).
    """

    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 16):
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "ChineseClipEncoder needs the 'editing-models' extra: "
                "uv sync --extra editing --extra editing-models"
            ) from exc

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"no model found at {model_dir} (config.json missing)")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = ChineseCLIPModel.from_pretrained(model_dir).to(device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_dir)
        self.name = f"chinese-clip:{model_dir.name}"
        self.dim = int(self.model.config.projection_dim)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        with self._torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.processor(
                    text=list(texts[i : i + self.batch_size]),
                    padding=True, truncation=True, max_length=52, return_tensors="pt",
                ).to(self.device)
                out.append(self.model.get_text_features(**batch).float().cpu().numpy())
        return l2_normalise(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        from PIL import Image

        out: List[np.ndarray] = []
        with self._torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                images = [Image.open(p).convert("RGB") for p in paths[i : i + self.batch_size]]
                batch = self.processor(images=images, return_tensors="pt").to(self.device)
                out.append(self.model.get_image_features(**batch).float().cpu().numpy())
        return l2_normalise(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)
