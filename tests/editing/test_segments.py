from __future__ import annotations

import pytest

from pixelle_video.editing.segments import (
    _boundaries,
    detect_cuts,
    extract_keyframes,
    probe_duration,
    segment_asset,
)


def _near(values, target, tol=0.21):
    return any(abs(v - target) <= tol for v in values)


def test_hist_detector_finds_all_colour_cuts(footage):
    cuts = detect_cuts(footage / "clips" / "rgb.mp4")
    assert len(cuts) == 3
    for t in (3.0, 6.0, 9.0):
        assert _near(cuts, t), (t, cuts)


def test_static_take_has_no_cuts(footage):
    assert detect_cuts(footage / "walk" / "white.mp4") == []


def test_probe_duration(footage):
    assert probe_duration(footage / "clips" / "rgb.mp4") == pytest.approx(18.0, abs=0.15)


class TestBoundaries:
    def test_plain_cuts(self):
        assert _boundaries([3.0, 6.0], 9.0, min_len=1.0, max_len=None) == [(0, 3), (3, 6), (6, 9)]

    def test_short_span_merged_into_previous(self):
        assert _boundaries([3.0, 3.4], 9.0, min_len=1.0, max_len=None) == [(0, 3.4), (3.4, 9)]

    def test_short_first_span_merged_forward(self):
        assert _boundaries([0.5], 9.0, min_len=1.0, max_len=None) == [(0.0, 9.0)]

    def test_long_span_split_evenly(self):
        assert _boundaries([], 14.0, min_len=1.0, max_len=6.0) == [
            (0.0, 4.667), (4.667, 9.333), (9.333, 14.0)
        ]

    def test_cuts_outside_duration_ignored(self):
        assert _boundaries([-1.0, 0.0, 9.0, 12.0], 9.0, 1.0, None) == [(0.0, 9.0)]


def test_segment_ids_are_relative_and_stable(footage):
    segs = segment_asset(footage / "clips" / "rgb.mp4", footage, max_len=6.0)
    assert [s.segment_id for s in segs] == [f"clips/rgb.mp4#{i:03d}" for i in range(len(segs))]
    # red, green, blue, then magenta (9 s) split into two windows
    assert len(segs) == 5
    assert segs[-1].end == pytest.approx(18.0, abs=0.15)
    assert segment_asset(footage / "clips" / "rgb.mp4", footage, max_len=6.0) == segs


def test_image_is_single_zero_length_segment(footage):
    (seg,) = segment_asset(footage / "stills" / "yellow.png", footage)
    assert (seg.kind, seg.start, seg.end, seg.segment_id) == ("image", 0.0, 0.0, "stills/yellow.png#000")


def test_unsupported_extension_rejected(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        segment_asset(f, tmp_path)


def test_keyframes_lie_inside_segment(footage, tmp_path):
    from PIL import Image

    segs = segment_asset(footage / "clips" / "rgb.mp4", footage, max_len=6.0)
    green = segs[1]
    frames = extract_keyframes(footage / "clips" / "rgb.mp4", green, tmp_path, n=3)
    assert len(frames) == 3
    for f in frames:
        r, g, b = Image.open(f).convert("RGB").getpixel((10, 10))
        assert g > r and g > b  # every keyframe of the green shot is green
