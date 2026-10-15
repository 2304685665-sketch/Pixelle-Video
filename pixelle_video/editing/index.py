"""
Persistent segment index over a footage folder.

Layout of an index directory::

    meta.json          encoder name, segmentation params, footage root
    segments.json      list of Segment dicts
    keyframes/*.jpg    extracted keyframes (images are referenced in place)
    keyframe_emb.npy   (n_keyframes, dim) L2-normalised image embeddings
    keyframe_owner.npy (n_keyframes,) index into segments
    texts.json         {field: {segment_id: text}} for BM25 channels

Text fields are free-form and optional. ``filename`` is always present (it is
what the upstream LLM assignment effectively sees); ``caption`` and
``transcript`` can be added later by a captioner / ASR step via ``set_text``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from pixelle_video.editing.encoders import Encoder
from pixelle_video.editing.segments import Segment, extract_keyframes, list_assets, segment_asset


def _filename_text(rel_path: str) -> str:
    stem = Path(rel_path).stem
    parent = Path(rel_path).parent.as_posix()
    words = stem.replace("_", " ").replace("-", " ")
    return f"{parent} {words}".strip() if parent != "." else words


@dataclass
class SegmentIndex:
    root: str
    encoder_name: str
    segments: List[Segment]
    keyframe_emb: np.ndarray
    keyframe_owner: np.ndarray
    texts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        footage_root: str | Path,
        out_dir: str | Path,
        encoder: Encoder,
        threshold: float = 0.4,
        min_len: float = 1.0,
        max_len: Optional[float] = 6.0,
        keyframes_per_segment: int = 3,
        method: str = "hist",
    ) -> "SegmentIndex":
        root = Path(footage_root).resolve()
        out_dir = Path(out_dir)
        kf_dir = out_dir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)

        segments: List[Segment] = []
        kf_paths: List[Path] = []
        owners: List[int] = []
        assets = list_assets(root)
        if not assets:
            raise FileNotFoundError(f"no video or image assets under {root}")

        for asset in assets:
            before = len(segments)
            for seg in segment_asset(asset, root, threshold, min_len, max_len, method=method):
                idx = len(segments)
                if seg.kind == "image":
                    frames = [asset]
                    seg.keyframes = [str(asset)]
                else:
                    frames = extract_keyframes(asset, seg, kf_dir, n=keyframes_per_segment)
                    seg.keyframes = [p.relative_to(out_dir).as_posix() for p in frames]
                segments.append(seg)
                kf_paths.extend(frames)
                owners.extend([idx] * len(frames))
            logger.info(f"indexed {asset.name}: {len(segments) - before} segments")

        emb = encoder.encode_images([str(p) for p in kf_paths])
        index = cls(
            root=str(root),
            encoder_name=encoder.name,
            segments=segments,
            keyframe_emb=emb,
            keyframe_owner=np.asarray(owners, dtype=np.int64),
            texts={"filename": {s.segment_id: _filename_text(s.asset_path) for s in segments}},
            params=dict(
                threshold=threshold, min_len=min_len, max_len=max_len,
                keyframes_per_segment=keyframes_per_segment, method=method,
            ),
        )
        index.save(out_dir)
        return index

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(
            json.dumps({"root": self.root, "encoder": self.encoder_name, "params": self.params}, indent=2)
        )
        (out_dir / "segments.json").write_text(
            json.dumps([s.to_dict() for s in self.segments], ensure_ascii=False, indent=1)
        )
        (out_dir / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False, indent=1))
        np.save(out_dir / "keyframe_emb.npy", self.keyframe_emb)
        np.save(out_dir / "keyframe_owner.npy", self.keyframe_owner)

    @classmethod
    def load(cls, index_dir: str | Path) -> "SegmentIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(
            root=meta["root"],
            encoder_name=meta["encoder"],
            params=meta.get("params", {}),
            segments=[Segment.from_dict(d) for d in json.loads((index_dir / "segments.json").read_text())],
            keyframe_emb=np.load(index_dir / "keyframe_emb.npy"),
            keyframe_owner=np.load(index_dir / "keyframe_owner.npy"),
            texts=json.loads((index_dir / "texts.json").read_text()),
        )

    # ------------------------------------------------------------------ #

    def set_text(self, field_name: str, values: Dict[str, str]) -> None:
        unknown = set(values) - {s.segment_id for s in self.segments}
        if unknown:
            raise KeyError(f"unknown segment ids: {sorted(unknown)[:5]}")
        self.texts.setdefault(field_name, {}).update(values)

    def segment(self, segment_id: str) -> Segment:
        for s in self.segments:
            if s.segment_id == segment_id:
                return s
        raise KeyError(segment_id)

    def assets(self) -> List[str]:
        return sorted({s.asset_path for s in self.segments})
