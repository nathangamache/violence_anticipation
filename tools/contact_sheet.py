"""Render timestamped contact sheets (frame grids) so a reviewer can locate events in a video by eye.

Each tile shows its video time (seconds and mm:ss). Used to annotate onsets in the curated YouTube hold-out set.

Usage:
    # coarse pass: one frame every 2 s, 30 tiles (60 s) per sheet, whole video
    python tools/contact_sheet.py --video path.mp4 --out_dir sheets/vid --step 2
    # fine pass: one frame every 0.5 s between 80 s and 95 s
    python tools/contact_sheet.py --video path.mp4 --out_dir sheets/vid --start 80 --end 95 --step 0.5 --prefix fine
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anticipation.features.io import Video  # noqa: E402

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def render_sheets(video_path, out_dir, start=0.0, end=None, step=2.0, cols=6, rows=5, tile_w=256, prefix="coarse", num_threads=4):
    """Write sheets ``<prefix>_<first_t>-<last_t>.jpg`` covering [start, end) and return their paths."""
    v = Video(video_path, num_threads=num_threads)
    end = v.info.duration_sec if end is None else min(end, v.info.duration_sec)
    times = np.arange(start, end, step)
    if len(times) == 0:
        return []
    tile_h = max(1, int(round(tile_w * v.info.height / max(1, v.info.width))))
    font = _font(max(12, tile_w // 14))
    per = cols * rows
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k in range(0, len(times), per):
        chunk = times[k:k + per]
        frames = v.frames_at(chunk)
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)
        for i, (t, fr) in enumerate(zip(chunk, frames)):
            x, y = (i % cols) * tile_w, (i // cols) * tile_h
            sheet.paste(Image.fromarray(fr).resize((tile_w, tile_h)), (x, y))
            label = f"{t:.1f}s  {int(t // 60)}:{t % 60:04.1f}"
            box = draw.textbbox((x + 3, y + 3), label, font=font)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((x + 3, y + 3), label, fill=(255, 255, 0), font=font)
        used_rows = (len(chunk) + cols - 1) // cols
        sheet = sheet.crop((0, 0, cols * tile_w, used_rows * tile_h))
        p = os.path.join(out_dir, f"{prefix}_{chunk[0]:07.1f}-{chunk[-1]:07.1f}.jpg")
        sheet.save(p, quality=85)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--tile_w", type=int, default=256)
    ap.add_argument("--prefix", default="coarse")
    ap.add_argument("--num_threads", type=int, default=4, choices=range(1, 5))
    a = ap.parse_args()
    v = Video(a.video, num_threads=a.num_threads)
    print(f"duration_sec={v.info.duration_sec:.2f} fps={v.info.fps:.3f} size={v.info.width}x{v.info.height}")
    for p in render_sheets(a.video, a.out_dir, a.start, a.end, a.step, a.cols, a.rows, a.tile_w, a.prefix, a.num_threads):
        print(p)


if __name__ == "__main__":
    main()
