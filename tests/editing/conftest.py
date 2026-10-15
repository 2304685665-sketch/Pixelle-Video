"""
Synthetic footage with known ground truth, generated with ffmpeg.

footage/
  clips/rgb.mp4    red 0-3s | green 3-6s | blue 6-9s | magenta 9-18s
  walk/white.mp4   white 0-14s (one long static take)
  stills/yellow.png
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SIZE = "160x120"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _colour_video(out: Path, spans: list[tuple[str, float]]) -> None:
    inputs: list[str] = []
    for colour, dur in spans:
        inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s={SIZE}:r=10:d={dur}"]
    n = len(spans)
    graph = "".join(f"[{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0"
    out.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(*inputs, "-filter_complex", graph, "-pix_fmt", "yuv420p", str(out))


@pytest.fixture(scope="session")
def footage(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    root = tmp_path_factory.mktemp("footage")
    _colour_video(root / "clips" / "rgb.mp4", [("red", 3), ("green", 3), ("blue", 3), ("magenta", 9)])
    _colour_video(root / "walk" / "white.mp4", [("white", 14)])
    from PIL import Image

    (root / "stills").mkdir()
    Image.new("RGB", (160, 120), (255, 255, 0)).save(root / "stills" / "yellow.png")
    return root


@pytest.fixture(scope="session")
def fake_index(footage, tmp_path_factory):
    from pixelle_video.editing.index import SegmentIndex
    from pixelle_video.testing.fake_encoders import FakeColourEncoder

    out = tmp_path_factory.mktemp("index")
    return SegmentIndex.build(footage, out, FakeColourEncoder(), max_len=6.0), out


LABELS = [
    {"id": "red", "query": "红色的电话画面", "asset": "clips/rgb.mp4", "start": 0, "end": 3},
    {"id": "green", "query": "绿色的窗台", "asset": "clips/rgb.mp4", "start": 3, "end": 6},
    {"id": "blue", "query": "blue night sky", "asset": "clips/rgb.mp4", "start": 6, "end": 9},
    {"id": "yellow", "query": "一张黄色的照片", "asset": "stills/yellow.png"},
    {"id": "white", "query": "白色的走廊", "asset": "walk/white.mp4", "start": 0, "end": 14},
]
