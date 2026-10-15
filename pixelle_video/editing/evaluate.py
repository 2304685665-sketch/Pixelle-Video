"""
Evaluation of narration -> footage retrieval.

Label file (JSONL), one narration line per row::

    {"id": "q01", "query": "我还记得守在电话旁边", "asset": "home/phone.mp4", "start": 12.0, "end": 16.5}

``asset`` is relative to the footage root used for indexing. Images use
``start = end = 0``. Several acceptable shots can be given with ``"alts": [
{"asset": ..., "start": ..., "end": ...}, ...]``.

Hit rule (fixed before running experiments; change only with a new protocol
version): a prediction hits a gold span if it is in the same asset and either

* the temporal IoU is >= ``iou_threshold`` (default 0.3), or
* the prediction lies entirely inside the gold span and covers at least
  ``min_cover`` (default 0.5) of it.

The second clause accepts a sub-shot of a long labelled take; the IoU clause
rejects whole-file predictions for short gold spans, which is exactly the
granularity difference under test. For images, the asset match alone is a hit.

Paired comparison uses the exact two-sided McNemar test on Hit@1.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from pixelle_video.editing.retrieval import Method, Prediction

PROTOCOL_VERSION = "retrieval-v1"


@dataclass
class Span:
    asset: str
    start: float
    end: float


@dataclass
class Label:
    id: str
    query: str
    gold: List[Span]

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        spans = [Span(d["asset"], float(d.get("start", 0.0)), float(d.get("end", 0.0)))]
        spans += [Span(a["asset"], float(a.get("start", 0.0)), float(a.get("end", 0.0))) for a in d.get("alts", [])]
        return cls(id=str(d["id"]), query=d["query"], gold=spans)


def load_labels(path: str | Path) -> List[Label]:
    labels = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            labels.append(Label.from_dict(json.loads(line)))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{n}: bad label: {exc}") from exc
    ids = [lb.id for lb in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate label ids")
    return labels


def temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def is_hit(pred: Prediction, gold: Span, iou_threshold: float = 0.3, min_cover: float = 0.5) -> bool:
    if pred.asset_path != gold.asset:
        return False
    if gold.end <= gold.start:  # image or point label: asset match is enough
        return True
    if temporal_iou(pred.start, pred.end, gold.start, gold.end) >= iou_threshold:
        return True
    inside = pred.start >= gold.start - 1e-6 and pred.end <= gold.end + 1e-6
    cover = (pred.end - pred.start) / (gold.end - gold.start)
    return inside and cover >= min_cover


@dataclass
class QueryResult:
    id: str
    first_hit_rank: int | None  # 1-based, None if no hit in top-k
    first_asset_rank: int | None
    top1: dict | None


@dataclass
class MethodReport:
    method: str
    n: int
    hit_at: Dict[int, float]
    mrr: float
    asset_acc_at1: float
    per_query: List[QueryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hit_at"] = {str(k): v for k, v in self.hit_at.items()}
        return d


def evaluate_method(
    method: Method,
    labels: Sequence[Label],
    ks: Sequence[int] = (1, 3, 5),
    iou_threshold: float = 0.3,
    min_cover: float = 0.5,
) -> MethodReport:
    depth = max(ks)
    results: List[QueryResult] = []
    for lb in labels:
        preds = method.search(lb.query, k=depth)
        hit_rank = asset_rank = None
        for r, p in enumerate(preds, start=1):
            if asset_rank is None and any(p.asset_path == g.asset for g in lb.gold):
                asset_rank = r
            if hit_rank is None and any(is_hit(p, g, iou_threshold, min_cover) for g in lb.gold):
                hit_rank = r
        results.append(
            QueryResult(lb.id, hit_rank, asset_rank, asdict(preds[0]) if preds else None)
        )
    n = len(results) or 1
    return MethodReport(
        method=method.name,
        n=len(results),
        hit_at={k: sum(1 for q in results if q.first_hit_rank and q.first_hit_rank <= k) / n for k in ks},
        mrr=sum(1.0 / q.first_hit_rank for q in results if q.first_hit_rank) / n,
        asset_acc_at1=sum(1 for q in results if q.first_asset_rank == 1) / n,
        per_query=results,
    )


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value from discordant counts.
    ``b``: queries only method A got right; ``c``: only method B got right.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_hit1(a: MethodReport, b: MethodReport) -> dict:
    """Discordant pairs and exact McNemar p on Hit@1 (same label order required)."""
    if [q.id for q in a.per_query] != [q.id for q in b.per_query]:
        raise ValueError("reports must cover the same labels in the same order")
    a_only = sum(1 for qa, qb in zip(a.per_query, b.per_query) if qa.first_hit_rank == 1 and qb.first_hit_rank != 1)
    b_only = sum(1 for qa, qb in zip(a.per_query, b.per_query) if qb.first_hit_rank == 1 and qa.first_hit_rank != 1)
    return {"a": a.method, "b": b.method, "a_only": a_only, "b_only": b_only, "p": mcnemar_exact(a_only, b_only)}
