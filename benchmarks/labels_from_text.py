"""
Turn a plain-text label sheet into dev / held-out JSONL label files.

Input (one narration per line, '#' starts a comment):

    旁白 | 文件名 | 开始秒 | 结束秒
    我还记得守在电话旁边等他们打来 | 大致_clip12.mp4 | 31 | 36
    那年冬天的第一场雪 | 大致_clip7.mp4 | 0 | 4 | 大致_clip9.mp4 | 10 | 14   (optional alternatives)

Times may be seconds (31.5) or m:ss (1:05). Every row is validated against the
index (file exists, 0 <= start < end <= duration). Rows are shuffled with a
fixed seed and split into dev / holdout; the holdout SHA-256 is printed for the
preregistration document.

    uv run python benchmarks/labels_from_text.py --text benchmarks/labels/raw.txt \
        --index runs/footage_idx --holdout 0.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.editing.index import SegmentIndex  # noqa: E402


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def parse(text: str, durations: dict[str, float]) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or (len(parts) - 1) % 3:
            errors.append(f"line {n}: expected '旁白 | 文件 | 开始 | 结束' (+ optional triples): {raw}")
            continue
        query, spans = parts[0], []
        for i in range(1, len(parts), 3):
            asset = parts[i]
            if asset not in durations:
                errors.append(f"line {n}: unknown file {asset!r}")
                continue
            try:
                a, b = parse_time(parts[i + 1]), parse_time(parts[i + 2])
            except ValueError:
                errors.append(f"line {n}: bad time in {parts[i + 1]!r}/{parts[i + 2]!r}")
                continue
            if not (0 <= a < b <= durations[asset] + 0.5):
                errors.append(f"line {n}: {asset} {a}-{b}s outside 0-{durations[asset]:.1f}s")
                continue
            spans.append({"asset": asset, "start": a, "end": b})
        if not query:
            errors.append(f"line {n}: empty narration")
        elif spans:
            rows.append({"query": query, **spans[0], **({"alts": spans[1:]} if spans[1:] else {})})
    return rows, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--holdout", type=float, default=0.6, help="fraction of rows in the frozen holdout set")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out-dir", default="benchmarks/labels")
    ap.add_argument("--pilot", action="store_true",
                    help="small pilot: every row goes to one frozen test file, no dev split, no tuning")
    ap.add_argument("--name", default="footage_pilot_v1", help="output file stem in --pilot mode")
    args = ap.parse_args()

    idx = SegmentIndex.load(args.index)
    durations: dict[str, float] = {}
    for s in idx.segments:
        durations[s.asset_path] = max(durations.get(s.asset_path, 0.0), s.end)

    rows, errors = parse(Path(args.text).read_text(encoding="utf-8"), durations)
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} problem(s); nothing written")
    dupes = {r["query"] for r in rows if sum(x["query"] == r["query"] for x in rows) > 1}
    if dupes:
        sys.exit(f"duplicate narrations: {sorted(dupes)[:5]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        path = out / f"{args.name}.jsonl"
        if path.exists():
            sys.exit(f"{path} already exists: pilot labels are frozen. Use a new --name for a new version.")
        lines = [json.dumps({"id": f"p{i:03d}", **r}, ensure_ascii=False) for i, r in enumerate(rows)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files = sorted({r["asset"] for r in rows})
        print(f"pilot labels: {len(rows)} -> {path}  (files covered: {len(files)}/{len(durations)})")
        print(f"SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
        return

    random.Random(args.seed).shuffle(rows)
    n_hold = round(len(rows) * args.holdout)
    hold, dev = rows[:n_hold], rows[n_hold:]
    paths = {}
    for name, part in (("dev", dev), ("holdout", hold)):
        path = out / f"footage_{name}.jsonl"
        if name == "holdout" and path.exists():
            sys.exit(f"{path} already exists: the holdout set is frozen. Delete it deliberately to re-split.")
        lines = [json.dumps({"id": f"{name[0]}{i:03d}", **r}, ensure_ascii=False) for i, r in enumerate(part)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[name] = path

    digest = hashlib.sha256(paths["holdout"].read_bytes()).hexdigest()
    files = sorted({r["asset"] for r in rows})
    print(f"labels: {len(rows)}  dev: {len(dev)} -> {paths['dev']}  holdout: {len(hold)} -> {paths['holdout']}")
    print(f"files covered: {len(files)}/{len(durations)}")
    print(f"holdout SHA-256 (put this in docs/preregistration_retrieval_v1.md): {digest}")


if __name__ == "__main__":
    main()
