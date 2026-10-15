"""
VLM descriptions for retrieval, and the upstream-faithful LLM assignment baseline.

* ``caption_segments`` writes one VLM caption per segment (middle keyframe) into
  the index text field ``caption``, so ``bm25-caption`` / ``visual+bm25-caption``
  can match abstract narration through words rather than pixels.
* ``describe_assets`` writes one description per asset file, exactly what
  upstream ``AssetBasedPipeline.setup_environment`` feeds its LLM.
* ``LLMAssetBaseline`` reproduces upstream assignment: the LLM sees the list of
  "Path + Description" and picks assets. Difference, stated in the docs: upstream
  writes the narration and assigns assets in one call; here the narration is the
  fixed label, so the comparison isolates the assignment step.

Every paid call is cached on disk (JSON, keyed by input), so re-running an
experiment or resuming after a failure never pays for the same call twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Protocol

from loguru import logger
from pydantic import BaseModel, Field

from pixelle_video.editing.index import SegmentIndex
from pixelle_video.editing.retrieval import Prediction


class Describer(Protocol):
    name: str

    async def describe_image(self, path: str) -> str: ...

    async def describe_video(self, path: str) -> str: ...


class UpstreamDescriber:
    """Upstream APIAssetAnalysisService (DashScope VLM, keys from config.yaml)."""

    def __init__(self, model: str):
        from pixelle_video.config import config_manager
        from pixelle_video.services.api_asset_analysis import APIAssetAnalysisService

        self.svc = APIAssetAnalysisService(config_manager.config.to_dict())
        self.model = model
        self.name = f"upstream-vlm:{model}"

    async def describe_image(self, path: str) -> str:
        return await self.svc.analyze_image(path, model=self.model)

    async def describe_video(self, path: str) -> str:
        return await self.svc.analyze_video(path, model=self.model)


class _JsonCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        self.data[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written cache


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


async def _bounded(items, fn, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(x):
        async with sem:
            return await fn(x)

    return await asyncio.gather(*(one(x) for x in items), return_exceptions=True)


async def caption_segments(
    index: SegmentIndex, index_dir: str | Path, describer: Describer, concurrency: int = 4
) -> Dict[str, int]:
    """Caption every segment's middle keyframe; results go to texts['caption']."""
    index_dir = Path(index_dir)
    cache = _JsonCache(index_dir / f"captions_cache.{_key(describer.name)}.json")
    stats = {"cached": 0, "called": 0, "failed": 0}

    async def one(seg):
        kf = seg.keyframes[len(seg.keyframes) // 2]
        path = kf if Path(kf).is_absolute() else str(index_dir / kf)
        k = _key(describer.name, seg.segment_id, path)
        hit = cache.get(k)
        if hit is not None:
            stats["cached"] += 1
            return seg.segment_id, hit
        text = await describer.describe_image(path)
        cache.put(k, text)
        stats["called"] += 1
        return seg.segment_id, text

    results = await _bounded(index.segments, one, concurrency)
    captions = {}
    for r in results:
        if isinstance(r, BaseException):
            stats["failed"] += 1
            logger.warning(f"caption failed: {r}")
        else:
            captions[r[0]] = r[1]
    index.set_text("caption", captions)
    index.save(index_dir)
    return stats


async def describe_assets(
    index: SegmentIndex, index_dir: str | Path, describer: Describer, concurrency: int = 2
) -> Dict[str, str]:
    """One description per asset (upstream's input to LLM assignment); cached in asset_descriptions.json."""
    index_dir = Path(index_dir)
    cache = _JsonCache(index_dir / f"asset_descriptions.{_key(describer.name)}.json")
    kinds = {s.asset_path: s.kind for s in index.segments}

    async def one(asset):
        k = _key(describer.name, asset)
        hit = cache.get(k)
        if hit is not None:
            return asset, hit
        full = str(Path(index.root) / asset)
        text = await (describer.describe_image(full) if kinds[asset] == "image" else describer.describe_video(full))
        cache.put(k, text)
        return asset, text

    results = await _bounded(index.assets(), one, concurrency)
    out = {}
    for r in results:
        if isinstance(r, BaseException):
            logger.warning(f"asset description failed: {r}")
        else:
            out[r[0]] = r[1]
    return out


# --------------------------------------------------------------------------- #
# Upstream-style LLM assignment baseline
# --------------------------------------------------------------------------- #


class AssetRanking(BaseModel):
    paths: List[str] = Field(description="Asset paths from the list, best match first")


ASSIGN_PROMPT = """你是短视频剪辑师。下面是可用素材列表（路径 + 内容描述），以及一句旁白。
请为这句旁白选择最合适的素材，按匹配程度从高到低给出最多 {k} 个素材路径。
只能使用列表中出现的路径，原样输出，不要修改。

旁白：{query}

可用素材：
{assets_text}
"""

LLMCall = Callable[..., Awaitable[AssetRanking]]


class LLMAssetBaseline:
    """
    Upstream-faithful file-level assignment by an LLM.

    ``llm`` is an async callable compatible with upstream ``LLMService.__call__``
    (prompt=..., response_type=AssetRanking). Answers are cached per
    (llm name, query, asset list) so evaluation re-runs are free.
    """

    def __init__(self, index: SegmentIndex, descriptions: Dict[str, str], llm: LLMCall,
                 cache_path: str | Path, llm_name: str = "llm", k: int = 5):
        missing = set(index.assets()) - set(descriptions)
        if missing:
            raise ValueError(f"missing descriptions for {len(missing)} assets, e.g. {sorted(missing)[:3]}")
        self.index, self.llm, self.k = index, llm, k
        self.name = "llm-asset"
        self.assets_text = "\n".join(
            f"- Path: {a}\n  Description: {descriptions[a]}" for a in sorted(descriptions)
        )
        self.cache = _JsonCache(Path(cache_path))
        self.llm_name = llm_name
        self.invalid_paths = 0
        self._spans: Dict[str, tuple[float, float]] = {}
        for s in index.segments:
            a, b = self._spans.get(s.asset_path, (s.start, s.end))
            self._spans[s.asset_path] = (min(a, s.start), max(b, s.end))

    async def asearch(self, query: str, k: int = 10) -> List[Prediction]:
        key = _key(self.llm_name, query, self.assets_text)
        cached = self.cache.get(key)
        if cached is None:
            ranking = await self.llm(
                prompt=ASSIGN_PROMPT.format(k=self.k, query=query, assets_text=self.assets_text),
                response_type=AssetRanking, temperature=0.0,
            )
            cached = json.dumps(ranking.paths, ensure_ascii=False)
            self.cache.put(key, cached)
        paths = json.loads(cached)
        out, seen = [], set()
        for p in paths:
            if p not in self._spans:  # hallucinated / modified path
                self.invalid_paths += 1
                continue
            if p in seen:
                continue
            seen.add(p)
            a, b = self._spans[p]
            out.append(Prediction(f"{p}#asset", p, a, b, 1.0 / (len(out) + 1)))
        return out[:k]

    def search(self, query: str, k: int = 10) -> List[Prediction]:
        """Sync wrapper so the baseline plugs into evaluate_method; call outside a running loop."""
        return asyncio.run(self.asearch(query, k))


def upstream_llm(model_override: Optional[str] = None) -> LLMCall:
    """Upstream LLMService (keys/model from config.yaml)."""
    from pixelle_video.config import config_manager
    from pixelle_video.services.llm_service import LLMService

    svc = LLMService(config_manager.config.to_dict())

    async def call(**kw):
        if model_override:
            kw["model"] = model_override
        return await svc(**kw)

    return call
