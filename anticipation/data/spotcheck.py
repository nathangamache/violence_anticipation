"""Human spot-check sheets for HIVAU-70k train-split onsets (the VA30 ground truth nobody has validated yet).

HIVAU's test-split entries are copies of the official annotations, so the automatic cross-check
(setup/build_hivau_onsets.py) cannot tell whether HIVAU's own train-split onsets are right. This tool samples
eligible30 events whose onsets come from HIVAU and renders one contact sheet per event (frames at fixed offsets
around the onset) plus a CSV to fill in by hand.

Usage (from the repo root, CPU, about 1 s per sheet):

    .venv/bin/python -m anticipation.data.spotcheck --n 50 [--seed 0] [--onset_source hivau] \
        [--out_dir data/annotations/hivau/spotcheck]
    .venv/bin/python -m anticipation.data.spotcheck --summarize   # after filling spotcheck.csv

Outputs in ``--out_dir``:
    sheets/<k>_<stem>.jpg   7 frames at onset - 10, - 5, - 2, 0, + 2, + 5, + 10 s, each stamped with its time
    spotcheck.csv           k, video_id, source_kind, classes, event_index, onset_sec, end_sec, sheet,
                            verdict, corrected_onset_sec, notes   (verdict / corrected_onset_sec / notes are blank)

Verdicts to use: ``ok`` (violence visibly starts within about 2 s of the marked onset), ``early`` (marked before
anything violent happens; give corrected_onset_sec), ``late`` (violence already under way at the marked onset),
``wrong`` (no violence in the segment), ``unclear``. ``--summarize`` prints the verdict counts and the median
correction. A sample is ``--n`` events drawn with a fixed seed, stratified by source kind.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from anticipation.data.annotations import HIVAU_DIR
from anticipation.data.catalog import EVENTS_JSON, VIDEOS_CSV, load_catalog, load_events

OFFSETS = (-10, -5, -2, 0, 2, 5, 10)
OUT_DIR = os.path.join(HIVAU_DIR, "spotcheck")


def sample_events(df, events, n, seed=0, onset_source="hivau"):
    """``n`` eligible30 events with the given onset source, stratified by source kind (fixed seed)."""
    rows = []
    for vid in df.index[(df.role == "violent") & (df.onset_source == onset_source)]:
        for e in events.get(vid, {}).get("events", []):
            if e["eligible30"]:
                rows.append({"video_id": vid, "source_kind": df.loc[vid, "source_kind"],
                             "classes": df.loc[vid, "classes"], "event_index": e["index"],
                             "onset_sec": e["onset"], "end_sec": e["end"], "path": df.loc[vid, "path"]})
    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    rng = np.random.default_rng(seed)
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    share = pool.source_kind.value_counts(normalize=True)
    take = {k: max(1, int(round(n * p))) for k, p in share.items()}
    out = pd.concat([pool[pool.source_kind == k].head(m) for k, m in take.items()]).head(n)
    return out.sort_values(["source_kind", "video_id", "onset_sec"]).reset_index(drop=True)


def render_sheet(path, onset, out_path, offsets=OFFSETS, width=320):
    """Contact sheet of frames at ``onset + offsets`` (seconds), time-stamped; returns the output path."""
    import decord
    from PIL import Image, ImageDraw

    vr = decord.VideoReader(path, num_threads=1)
    fps = float(vr.get_avg_fps()) or 25.0
    n = len(vr)
    idx = [int(np.clip(round((onset + o) * fps), 0, n - 1)) for o in offsets]
    frames = vr.get_batch(idx).asnumpy()
    h, w = frames.shape[1:3]
    th = int(round(h * width / w))
    tiles = []
    for fr, o in zip(frames, offsets):
        im = Image.fromarray(fr).resize((width, th))
        d = ImageDraw.Draw(im)
        label = f"t0{o:+d}s ({onset + o:.1f}s)" if o else f"ONSET {onset:.1f}s"
        d.rectangle([0, 0, width, 14], fill=(0, 0, 0))
        d.text((3, 1), label, fill=(255, 255, 0) if o == 0 else (255, 255, 255))
        tiles.append(im)
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * width, rows * th), (40, 40, 40))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * width, (k // cols) * th))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=85)
    return out_path


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    done = df[df.verdict.fillna("").str.len() > 0]
    print(f"[spotcheck] {len(done)}/{len(df)} events reviewed")
    if len(done):
        print(done.verdict.value_counts().to_string())
        corr = done.corrected_onset_sec.dropna() - done.loc[done.corrected_onset_sec.notna(), "onset_sec"]
        if len(corr):
            print(f"[spotcheck] corrections: n {len(corr)}, median {corr.median():+.1f} s, "
                  f"|median| {corr.abs().median():.1f} s")
        for k, g in done.groupby("source_kind"):
            print(f"  {k}: " + ", ".join(f"{v} {c}" for v, c in g.verdict.value_counts().items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onset_source", default="hivau")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--videos", default=VIDEOS_CSV)
    ap.add_argument("--events", default=EVENTS_JSON)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv)
    csv_path = os.path.join(args.out_dir, "spotcheck.csv")
    if args.summarize:
        summarize(csv_path)
        return 0
    if os.path.exists(csv_path):
        print(f"[spotcheck] {csv_path} exists (it may hold human verdicts); refusing to overwrite. "
              f"Use another --out_dir.")
        return 2
    df, events = load_catalog(args.videos), load_events(args.events)
    s = sample_events(df, events, args.n, args.seed, args.onset_source)
    sheets = []
    for k, r in s.iterrows():
        stem = r.video_id.split("/", 1)[1][:60].replace("/", "_")
        out = os.path.join(args.out_dir, "sheets", f"{k:03d}_{stem}.jpg")
        try:
            sheets.append(render_sheet(r.path, float(r.onset_sec), out))
        except Exception as e:  # unreadable file
            sheets.append(f"ERROR {type(e).__name__}: {e}")
    s = s.drop(columns=["path"]).assign(sheet=sheets, verdict="", corrected_onset_sec="", notes="")
    s.insert(0, "k", range(len(s)))
    os.makedirs(args.out_dir, exist_ok=True)
    s.to_csv(csv_path, index=False)
    print(f"[spotcheck] wrote {len(s)} sheets and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
