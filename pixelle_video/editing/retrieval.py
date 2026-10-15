"""
Narration -> footage segment retrieval.

Channels each produce a ranked list of segment ids for a query:

* ``VisualChannel``  – text embedding vs keyframe embeddings (max over keyframes)
* ``BM25Channel``    – BM25 over one text field of the index (filename, caption,
  transcript), tokenised with jieba for Chinese

``SegmentRetriever`` fuses channels with weighted reciprocal rank fusion (RRF),
which only needs ranks, so channels with incomparable score scales can be mixed.

``AssetLevelBaseline`` reproduces the granularity of upstream
``AssetBasedPipeline``: it picks a whole asset per narration and returns the
full file time range. It is the baseline the segment-level methods are
compared against.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from pixelle_video.editing.encoders import Encoder
from pixelle_video.editing.index import SegmentIndex


@dataclass
class Prediction:
    segment_id: str
    asset_path: str
    start: float
    end: float
    score: float


class Method(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> List[Prediction]: ...


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_jieba = None


def tokenize(text: str) -> List[str]:
    """jieba search-mode tokens for CJK, regex word tokens for everything else, lowercased."""
    global _jieba
    if _jieba is None:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    tokens: List[str] = []
    for chunk in _TOKEN_RE.findall(text.lower()):
        if re.search(r"[一-鿿]", chunk):
            tokens.extend(t for t in _jieba.lcut_for_search(chunk) if t.strip())
        else:
            tokens.append(chunk)
    return tokens


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


class VisualChannel:
    def __init__(self, index: SegmentIndex, encoder: Encoder, aggregate: str = "max"):
        if index.encoder_name != encoder.name:
            raise ValueError(
                f"index built with {index.encoder_name!r} but query encoder is {encoder.name!r}"
            )
        if aggregate not in {"max", "mean"}:
            raise ValueError("aggregate must be 'max' or 'mean'")
        self.index, self.encoder, self.aggregate = index, encoder, aggregate
        self.name = f"visual-{aggregate}"

    def segment_scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode_text([query])[0]
        sims = self.index.keyframe_emb @ q
        n = len(self.index.segments)
        owner = self.index.keyframe_owner
        if self.aggregate == "max":
            scores = np.full(n, -np.inf)
            np.maximum.at(scores, owner, sims)
        else:
            sums = np.bincount(owner, weights=sims, minlength=n)
            counts = np.bincount(owner, minlength=n)
            scores = sums / np.clip(counts, 1, None)
        return scores

    def rank(self, query: str, k: int) -> List[tuple[str, float]]:
        scores = self.segment_scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.index.segments[i].segment_id, float(scores[i])) for i in order]


class BM25Channel:
    def __init__(self, index: SegmentIndex, field: str, k1: float = 1.5, b: float = 0.75):
        if field not in index.texts:
            raise KeyError(f"index has no text field {field!r}; available: {sorted(index.texts)}")
        self.name = f"bm25-{field}"
        self.k1, self.b = k1, b
        self.ids = [s.segment_id for s in index.segments if index.texts[field].get(s.segment_id)]
        docs = [tokenize(index.texts[field][i]) for i in self.ids]
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.ids))
        for j, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[j] / (self.avgdl or 1.0))
            out[j] = sum(
                self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf
            )
        return out

    def rank(self, query: str, k: int) -> List[tuple[str, float]]:
        s = self.scores(query)
        order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0][:k]
        return [(self.ids[i], float(s[i])) for i in order]


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Optional[Sequence[float]] = None
) -> List[tuple[str, float]]:
    """Weighted reciprocal rank fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank from 1."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for w, ranking in zip(weights, rankings):
        for r, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + r)
            first_seen.setdefault(doc, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


class SegmentRetriever:
    """Segment-level retrieval over one or more channels (RRF when more than one)."""

    def __init__(
        self,
        index: SegmentIndex,
        channels: Sequence,
        weights: Optional[Sequence[float]] = None,
        rrf_k: int = 60,
        depth: int = 50,
        name: Optional[str] = None,
    ):
        if not channels:
            raise ValueError("at least one channel")
        self.index, self.channels, self.weights = index, list(channels), weights
        self.rrf_k, self.depth = rrf_k, depth
        self.name = name or "+".join(c.name for c in self.channels)
        self._by_id = {s.segment_id: s for s in index.segments}

    def search(self, query: str, k: int = 10) -> List[Prediction]:
        if len(self.channels) == 1:
            ranked = self.channels[0].rank(query, k)
        else:
            lists = [[sid for sid, _ in c.rank(query, self.depth)] for c in self.channels]
            ranked = rrf_fuse(lists, k=self.rrf_k, weights=self.weights)[:k]
        out = []
        for sid, score in ranked:
            s = self._by_id[sid]
            out.append(Prediction(sid, s.asset_path, s.start, s.end, score))
        return out


class AssetLevelBaseline:
    """
    Whole-asset assignment, matching upstream ``AssetBasedPipeline`` granularity.

    ``scorer`` ranks assets: ``"filename"`` uses BM25 over file names (the main
    signal an LLM gets from a file listing); ``"visual"`` scores an asset by its
    best segment (mean keyframe similarity) but still returns the whole file, so
    comparing it with ``SegmentRetriever([VisualChannel])`` isolates the cost of
    file-level granularity alone.
    """

    def __init__(self, index: SegmentIndex, scorer: str = "filename", encoder: Optional[Encoder] = None):
        self.index = index
        self.name = f"asset-{scorer}"
        self._spans: Dict[str, tuple[float, float]] = {}
        for s in index.segments:
            a, b = self._spans.get(s.asset_path, (s.start, s.end))
            self._spans[s.asset_path] = (min(a, s.start), max(b, s.end))
        if scorer == "filename":
            self._channel = BM25Channel(index, "filename")
        elif scorer == "visual":
            if encoder is None:
                raise ValueError("visual scorer needs an encoder")
            self._channel = VisualChannel(index, encoder, aggregate="mean")
        else:
            raise ValueError(f"unknown scorer {scorer!r}")
        self._asset_of = {s.segment_id: s.asset_path for s in index.segments}

    def search(self, query: str, k: int = 10) -> List[Prediction]:
        best: Dict[str, float] = {}
        for sid, score in self._channel.rank(query, len(self.index.segments)):
            asset = self._asset_of[sid]
            best[asset] = max(best.get(asset, -math.inf), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
        return [
            Prediction(f"{a}#asset", a, self._spans[a][0], self._spans[a][1], s) for a, s in ranked
        ]
