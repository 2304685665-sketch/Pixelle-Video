"""
Content fingerprint of a shot's generation inputs.

Two shots with the same fingerprint produce interchangeable segments, so a
segment generated once can be reused after the script is edited, even when
shots are inserted or deleted and indices shift. Correctness hinges on the
fingerprint covering *every* input that changes the output:

* narration (drives TTS audio, subtitles and duration)
* image prompt (drives media generation)
* generation settings in StoryboardConfig (size, media workflow, template and
  its params, voice, TTS workflow/speed/mode, reference audio, API video params, fps)
* the shot index, but only if the frame template renders it (``{{index}}``,
  e.g. the "healing" templates hide the title after the first frame)

Anything outside this list (title, other shots, task id) does not affect a
shot's segment in upstream FrameProcessor.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

_INDEX_RE = re.compile(r"\{\{\s*index\s*\}\}")

GENERATION_FIELDS = (
    "media_width", "media_height", "media_workflow", "frame_template", "template_params",
    "voice_id", "tts_workflow", "tts_speed", "tts_inference_mode", "ref_audio",
    "api_video_params", "video_fps",
)


@lru_cache(maxsize=256)
def template_uses_index(frame_template: str) -> bool:
    """True if the template renders the shot index. Unknown templates are treated as index-sensitive."""
    from pathlib import Path

    candidates = []
    try:
        from pixelle_video.utils.template_util import resolve_template_path

        candidates.append(Path(resolve_template_path(frame_template)))
    except Exception:  # noqa: BLE001 - fall back to the templates shipped with the package
        pass
    candidates.append(Path(__file__).resolve().parents[2] / "templates" / frame_template)
    for path in candidates:
        if path.is_file():
            return bool(_INDEX_RE.search(path.read_text(encoding="utf-8")))
    return True  # unknown template: be conservative, never reuse across indices


def shot_fingerprint(frame, config) -> str:
    payload = {
        "narration": frame.narration,
        "image_prompt": frame.image_prompt,
        "config": {f: getattr(config, f, None) for f in GENERATION_FIELDS},
    }
    if template_uses_index(config.frame_template):
        payload["index"] = frame.index
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]
