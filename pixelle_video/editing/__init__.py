"""
Shot-level footage retrieval for script-driven editing.

Upstream ``AssetBasedPipeline`` lets the LLM assign whole asset files to script
scenes (see ``pipelines/asset_based.py``, "LLM directly assigns assets to
scenes"). This package replaces that with an explicit retrieval step:

1. ``segments``  – split footage into shots (ffmpeg scene detection) and extract keyframes
2. ``encoders``  – pluggable text/image encoders (Chinese-CLIP or test fakes)
3. ``index``     – persist segments, keyframes, embeddings and text fields
4. ``retrieval`` – visual + BM25 channels fused with reciprocal rank fusion
5. ``evaluate``  – temporal hit rule, Hit@k / MRR, exact McNemar paired test
"""

from pixelle_video.editing.segments import Segment, detect_cuts, segment_asset

__all__ = ["Segment", "detect_cuts", "segment_asset"]
