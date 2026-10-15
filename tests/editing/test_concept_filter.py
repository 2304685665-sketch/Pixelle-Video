import numpy as np

from benchmarks.concept_filter import merge_spans, score_segments
from pixelle_video.editing.segments import Segment


def test_merge_adjacent_kept_segments_per_file():
    segs = [Segment("a#0", "a.mp4", 0, 2), Segment("a#1", "a.mp4", 2, 4), Segment("a#2", "a.mp4", 4, 6),
            Segment("b#0", "b.mp4", 0, 3)]
    spans = merge_spans(segs, [True, True, False, True], [0.1, 0.3, -0.2, 0.05])
    assert [(s["asset"], s["start"], s["end"]) for s in spans] == [("a.mp4", 0, 4), ("b.mp4", 0, 3)]
    assert spans[0]["margin"] == 0.3 and spans[0]["segments"] == ["a#0", "a#1"]


def test_score_is_max_over_prompts():
    class Ch:
        def segment_scores(self, q):
            return {"p1": np.array([0.1, 0.5]), "p2": np.array([0.3, 0.2]), "n1": np.array([0.2, 0.6])}[q]

    pos, neg = score_segments(None, Ch(), ["p1", "p2"], ["n1"])
    assert pos.tolist() == [0.3, 0.5] and (pos - neg > 0).tolist() == [True, False]
