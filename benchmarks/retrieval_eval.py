"""
Footage retrieval benchmark CLI.

    # 1. index a footage folder (real model, run on your machine)
    uv run python benchmarks/retrieval_eval.py index --footage ~/Movies/footage \
        --out runs/footage_idx --model-dir models/chinese-clip-vit-base-patch16

    # 2. contact sheet of all segments, to help write labels
    uv run python benchmarks/retrieval_eval.py sheet --index runs/footage_idx

    # 3. evaluate methods against labels
    uv run python benchmarks/retrieval_eval.py eval --index runs/footage_idx \
        --labels benchmarks/labels/footage_v1.jsonl \
        --model-dir models/chinese-clip-vit-base-patch16 \
        --methods asset-filename asset-visual visual \
        --baseline asset-filename --out benchmarks/results/retrieval_v1.json

Method specs: ``asset-filename``, ``asset-visual`` (upstream granularity
baselines), or segment-level channel combinations joined by ``+``:
``visual``, ``visual-mean``, ``bm25-<field>`` (e.g. ``visual+bm25-caption``).
Use ``--fake`` instead of ``--model-dir`` for the colour test encoder.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.editing.evaluate import (  # noqa: E402
    PROTOCOL_VERSION,
    evaluate_method,
    load_labels,
    paired_hit1,
)
from pixelle_video.editing.index import SegmentIndex  # noqa: E402
from pixelle_video.editing.retrieval import (  # noqa: E402
    AssetLevelBaseline,
    BM25Channel,
    SegmentRetriever,
    VisualChannel,
)


def make_encoder(args):
    if args.fake:
        from pixelle_video.testing.fake_encoders import FakeColourEncoder

        return FakeColourEncoder()
    if not args.model_dir:
        sys.exit("need --model-dir (or --fake)")
    from pixelle_video.editing.encoders import ChineseClipEncoder

    return ChineseClipEncoder(args.model_dir)


def build_method(spec: str, index: SegmentIndex, encoder):
    if spec == "asset-filename":
        return AssetLevelBaseline(index, "filename")
    if spec == "asset-visual":
        return AssetLevelBaseline(index, "visual", encoder)
    channels = []
    for part in spec.split("+"):
        if part == "visual":
            channels.append(VisualChannel(index, encoder, "max"))
        elif part == "visual-mean":
            channels.append(VisualChannel(index, encoder, "mean"))
        elif part.startswith("bm25-"):
            channels.append(BM25Channel(index, part[len("bm25-"):]))
        else:
            raise SystemExit(f"unknown method part: {part}")
    return SegmentRetriever(index, channels, name=spec)


# --------------------------------------------------------------------------- #


def cmd_index(args) -> None:
    idx = SegmentIndex.build(
        args.footage, args.out, make_encoder(args),
        threshold=args.threshold, min_len=args.min_len,
        max_len=None if args.max_len <= 0 else args.max_len,
        keyframes_per_segment=args.keyframes, method=args.cut_method,
    )
    print(f"indexed {len(idx.assets())} assets, {len(idx.segments)} segments -> {args.out}")


def cmd_sheet(args) -> None:
    idx = SegmentIndex.load(args.index)
    rows = []
    for s in idx.segments:
        # Video keyframes are relative to the index dir; image assets are absolute paths.
        imgs = "".join(f'<img src="{html.escape(k)}" loading="lazy">' for k in s.keyframes)
        span = "image" if s.kind == "image" else f"{s.start:.1f}–{s.end:.1f}s"
        rows.append(
            f"<tr><td><code>{html.escape(s.segment_id)}</code><br>{span}</td><td>{imgs}</td></tr>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Segments</title>"
        "<style>body{font:14px system-ui;margin:16px}img{height:96px;margin-right:4px}"
        "td{border-bottom:1px solid #ddd;padding:6px;vertical-align:top}</style>"
        f"<p>{len(idx.segments)} segments · footage root <code>{html.escape(idx.root)}</code></p>"
        "<table>" + "".join(rows) + "</table>"
    )
    out = Path(args.index) / "contact_sheet.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")


def cmd_eval(args) -> None:
    idx = SegmentIndex.load(args.index)
    labels = load_labels(args.labels)
    known = set(idx.assets())
    missing = sorted({g.asset for lb in labels for g in lb.gold} - known)
    if missing:
        sys.exit(f"labels reference assets not in the index: {missing[:5]}")

    encoder = make_encoder(args) if any("visual" in m for m in args.methods) else None
    reports = {}
    for spec in args.methods:
        reports[spec] = evaluate_method(
            build_method(spec, idx, encoder), labels, iou_threshold=args.iou, min_cover=args.min_cover
        )

    print(f"protocol {PROTOCOL_VERSION} · {len(labels)} queries · iou>={args.iou} · cover>={args.min_cover}")
    print(f"{'method':<28} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'asset@1':>8}")
    for spec, r in reports.items():
        print(
            f"{spec:<28} {r.hit_at[1]:>6.3f} {r.hit_at[3]:>6.3f} {r.hit_at[5]:>6.3f} "
            f"{r.mrr:>6.3f} {r.asset_acc_at1:>8.3f}"
        )

    comparisons = []
    if args.baseline:
        if args.baseline not in reports:
            sys.exit("--baseline must be one of --methods")
        for spec, r in reports.items():
            if spec != args.baseline:
                cmp = paired_hit1(r, reports[args.baseline])
                comparisons.append(cmp)
                print(
                    f"McNemar {spec} vs {args.baseline}: only-{spec}={cmp['a_only']} "
                    f"only-baseline={cmp['b_only']} p={cmp['p']:.4f}"
                )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "index": str(args.index),
                    "index_params": idx.params,
                    "encoder": idx.encoder_name,
                    "labels": str(args.labels),
                    "iou_threshold": args.iou,
                    "min_cover": args.min_cover,
                    "reports": {k: v.to_dict() for k, v in reports.items()},
                    "comparisons": comparisons,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_encoder_args(p):
        p.add_argument("--model-dir", default=None)
        p.add_argument("--fake", action="store_true", help="use the colour test encoder")

    p = sub.add_parser("index")
    p.add_argument("--footage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--min-len", type=float, default=1.0)
    p.add_argument("--max-len", type=float, default=6.0, help="<=0 disables splitting long shots")
    p.add_argument("--keyframes", type=int, default=3)
    p.add_argument("--cut-method", choices=["hist", "ffmpeg"], default="hist")
    add_encoder_args(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("sheet")
    p.add_argument("--index", required=True)
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("eval")
    p.add_argument("--index", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--methods", nargs="+", default=["asset-filename", "asset-visual", "visual"])
    p.add_argument("--baseline", default="asset-filename")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--min-cover", type=float, default=0.5)
    p.add_argument("--out", default=None)
    add_encoder_args(p)
    p.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args.func(args)


if __name__ == "__main__":
    main()
