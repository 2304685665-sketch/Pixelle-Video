"""
Shot segmentation and keyframe extraction with ffmpeg.

Hard cuts are detected from colour-histogram changes between frames sampled by
ffmpeg (see ``detect_cuts`` for why ffmpeg's own luma-driven scene score is not
the default). Raw cuts are then post-processed for retrieval:

* shots shorter than ``min_len`` are merged into their predecessor (flash cuts,
  camera bumps), and
* shots longer than ``max_len`` are split into equal windows, because handheld
  vlog footage often has long continuous takes whose content drifts; a single
  60 s "shot" is too coarse to align with one narration line.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


@dataclass
class Segment:
    """A contiguous time range of one asset. Images are a single 0-length segment."""

    segment_id: str
    asset_path: str  # relative to the footage root
    start: float
    end: float
    kind: str = "video"  # "video" | "image"
    keyframes: List[str] = field(default_factory=list)  # relative to the index dir

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(**data)


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_duration(path: str | Path) -> float:
    """Container duration in seconds (ffprobe)."""
    proc = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return float(json.loads(proc.stdout)["format"]["duration"])


def detect_cuts(
    path: str | Path,
    threshold: float = 0.4,
    method: str = "hist",
    sample_fps: float = 5.0,
) -> List[float]:
    """
    Timestamps (seconds) of hard cuts.

    ``method="hist"`` (default): sample small RGB frames at ``sample_fps`` and
    cut where the total-variation distance between consecutive colour
    histograms exceeds ``threshold`` (in [0, 1]). Precision is ``1/sample_fps``.

    ``method="ffmpeg"``: ffmpeg's ``scene`` score. It is driven mostly by luma,
    so it misses cuts between shots of similar brightness but different colour
    (e.g. pure red -> pure green scores ~0). Kept for comparison.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if method == "hist":
        return _detect_cuts_hist(path, threshold, sample_fps)
    if method != "ffmpeg":
        raise ValueError(f"unknown method: {method}")
    proc = _run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an", "-f", "null", "-",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed for {path}: {proc.stderr[-500:]}")
    return sorted({round(float(m), 3) for m in _PTS_RE.findall(proc.stdout)})


_SAMPLE_W, _SAMPLE_H, _BINS = 64, 36, 4  # 4x4x4 RGB histogram


def _sample_frames(path: str | Path, fps: float):
    import numpy as np

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vf", f"fps={fps},scale={_SAMPLE_W}:{_SAMPLE_H}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame sampling failed for {path}: {proc.stderr[-500:]!r}")
    frame_bytes = _SAMPLE_W * _SAMPLE_H * 3
    n = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(
        n, _SAMPLE_H, _SAMPLE_W, 3
    )
    times = np.arange(n) / fps
    return times, frames


def colour_histograms(frames):
    """Normalised 4x4x4 RGB histograms, shape (n, 64)."""
    import numpy as np

    q = (frames // (256 // _BINS)).astype(np.int32)
    codes = (q[..., 0] * _BINS + q[..., 1]) * _BINS + q[..., 2]
    codes = codes.reshape(len(frames), -1)
    hists = np.stack([np.bincount(c, minlength=_BINS**3) for c in codes]).astype(np.float64)
    return hists / hists.sum(axis=1, keepdims=True)


def _detect_cuts_hist(path: str | Path, threshold: float, fps: float) -> List[float]:
    import numpy as np

    times, frames = _sample_frames(path, fps)
    if len(frames) < 2:
        return []
    hists = colour_histograms(frames)
    tv = 0.5 * np.abs(np.diff(hists, axis=0)).sum(axis=1)  # total variation in [0, 1]
    return [round(float(times[i + 1]), 3) for i in np.nonzero(tv > threshold)[0]]


def _boundaries(
    cuts: List[float], duration: float, min_len: float, max_len: Optional[float]
) -> List[tuple[float, float]]:
    points = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    spans = [(a, b) for a, b in zip(points, points[1:]) if b > a]

    merged: List[tuple[float, float]] = []
    for a, b in spans:
        if merged and (b - a) < min_len:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    # A too-short first span is merged forward instead.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_len:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    if not max_len:
        return merged
    out: List[tuple[float, float]] = []
    for a, b in merged:
        n = max(1, int(-(-(b - a) // max_len)))  # ceil
        step = (b - a) / n
        out.extend((round(a + i * step, 3), round(a + (i + 1) * step, 3)) for i in range(n))
    return out


def extract_keyframes(
    video_path: str | Path,
    segment: Segment,
    out_dir: str | Path,
    n: int = 3,
    width: int = 448,
) -> List[Path]:
    """Extract ``n`` evenly spaced frames inside the segment as JPEGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        t = segment.start + segment.duration * (i + 1) / (n + 1)
        safe_id = segment.segment_id.replace("/", "_").replace("#", "_")
        target = out_dir / f"{safe_id}_k{i}.jpg"
        proc = _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1",
                "-vf", f"scale={width}:-2", "-q:v", "3", str(target),
            ]
        )
        if proc.returncode == 0 and target.exists():
            frames.append(target)
    if not frames:
        raise RuntimeError(f"no keyframes extracted for {segment.segment_id}")
    return frames


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def segment_asset(
    asset_path: str | Path,
    root: str | Path,
    threshold: float = 0.4,
    min_len: float = 1.0,
    max_len: Optional[float] = 6.0,
    method: str = "hist",
) -> List[Segment]:
    """
    Split one asset into segments. ``segment_id`` is ``<relative path>#<index>``
    so it stays stable across re-indexing of the same footage folder.
    """
    asset_path = Path(asset_path)
    rel = asset_path.resolve().relative_to(Path(root).resolve()).as_posix()
    ext = asset_path.suffix.lower()

    if ext in IMAGE_EXTS:
        return [Segment(segment_id=f"{rel}#000", asset_path=rel, start=0.0, end=0.0, kind="image")]
    if ext not in VIDEO_EXTS:
        raise ValueError(f"unsupported asset type: {asset_path}")

    duration = probe_duration(asset_path)
    cuts = detect_cuts(asset_path, threshold, method=method)
    return [
        Segment(segment_id=f"{rel}#{i:03d}", asset_path=rel, start=a, end=b, kind="video")
        for i, (a, b) in enumerate(_boundaries(cuts, duration, min_len, max_len))
    ]


def list_assets(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS and not p.name.startswith(".")
    )
