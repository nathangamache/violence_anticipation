"""
xd_violence_investigation.py
=============================
Deep investigation into the XD-Violence dataset structure to understand
what we are actually training on and why generalization is hard.

Key questions:
  1. How many videos have genuine pre-violence windows (10+ frames before first violent frame)?
  2. What is the class balance across videos?
  3. How long are violent vs. non-violent segments?
  4. Are there systematic temporal patterns (violence always at the end)?
  5. What does the embedding distribution look like in violent vs. pre-violent vs. neutral?
  6. Which videos would be best for anticipation training?

Usage:
    python xd_violence_investigation.py \
        --embeddings data/processed/vlm_emb \
        --labels data/processed/labels_multih \
        --output_dir results/xd_investigation
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_all_videos(emb_dir: Path, labels_dir: Path, max_videos: int = 1000):
    videos = []
    for vf in sorted(emb_dir.rglob("*.npz"))[:max_videos]:
        vid = vf.stem
        for candidate in [
            labels_dir / f"{vid}.npy",
            labels_dir / "xd_violence" / f"{vid}.npy",
        ]:
            if candidate.exists():
                try:
                    emb = np.load(vf)["emb"].astype(np.float32)
                    lbl = np.load(candidate)
                    if lbl.ndim > 1:
                        lbl = lbl.max(axis=1)
                    n = min(len(emb), len(lbl))
                    binary_lbl = (lbl[:n] > 0).astype(int)
                    videos.append({
                        "name": vid,
                        "emb": emb[:n],
                        "labels": binary_lbl,
                        "n_frames": n,
                    })
                except Exception:
                    pass
                break
    return videos


# ──────────────────────────────────────────────
# Pre-violence window analysis
# ──────────────────────────────────────────────

def analyze_pre_violence_windows(video: Dict, min_previolence_frames: int = 5) -> Dict:
    """
    For a video, find the first violent frame and measure the pre-violence window.
    Also identifies any 'recovery' windows after violence ends.
    """
    labels = video["labels"]
    n = len(labels)

    # Find first and last violent frame
    violent_frames = np.where(labels == 1)[0]
    if len(violent_frames) == 0:
        return {
            "name": video["name"],
            "has_violence": False,
            "n_frames": n,
            "n_violent": 0,
            "n_previolent": 0,
            "n_neutral": n,
            "previolence_window": 0,
            "violence_fraction": 0.0,
            "violence_start": -1,
            "violence_end": -1,
            "usable_for_anticipation": False,
        }

    first_violent = int(violent_frames[0])
    last_violent = int(violent_frames[-1])

    # Check for gaps in violence (multiple violence segments)
    # Find transitions: 0→1 (violence starts) and 1→0 (violence ends)
    transitions_start = [i for i in range(1, n) if labels[i] == 1 and labels[i - 1] == 0]
    transitions_end = [i for i in range(1, n) if labels[i] == 0 and labels[i - 1] == 1]
    n_violence_segments = len(transitions_start) + (1 if labels[0] == 1 else 0)

    # Pre-violence window = number of frames before first violent frame
    previolence_window = first_violent  # frames 0..first_violent-1

    # Post-violence window = frames after last violent frame
    postviolence_window = n - last_violent - 1

    return {
        "name": video["name"],
        "has_violence": True,
        "n_frames": n,
        "n_violent": int(violent_frames.sum()),
        "n_previolent": int(previolence_window),
        "n_neutral": int(labels[labels == 0].sum()),
        "previolence_window": int(previolence_window),
        "postviolence_window": int(postviolence_window),
        "violence_fraction": float(len(violent_frames) / n),
        "violence_start": first_violent,
        "violence_end": last_violent,
        "n_violence_segments": n_violence_segments,
        "usable_for_anticipation": previolence_window >= min_previolence_frames,
    }


# ──────────────────────────────────────────────
# Embedding distribution analysis
# ──────────────────────────────────────────────

def analyze_embedding_distribution(videos: List[Dict]) -> Dict:
    """
    Separate embeddings into violent, pre-violent (window before first event),
    and neutral (no violence) and compute statistics on each.
    """
    neutral_embs = []      # from videos with no violence
    previolent_embs = []   # frames before first violent frame
    violent_embs = []      # frames labeled violent
    postviolent_embs = []  # frames after violence ends

    for v in videos:
        labels = v["labels"]
        emb = v["emb"]
        violent_frames = np.where(labels == 1)[0]

        if len(violent_frames) == 0:
            neutral_embs.append(emb)
        else:
            first = int(violent_frames[0])
            last = int(violent_frames[-1])
            if first > 0:
                previolent_embs.append(emb[:first])
            violent_embs.append(emb[violent_frames])
            if last < len(emb) - 1:
                postviolent_embs.append(emb[last + 1:])

    stats = {}
    for name, emb_list in [
        ("neutral", neutral_embs),
        ("previolent", previolent_embs),
        ("violent", violent_embs),
        ("postviolent", postviolent_embs),
    ]:
        if not emb_list:
            stats[name] = {"n_frames": 0}
            continue
        combined = np.vstack(emb_list)
        stats[name] = {
            "n_frames": len(combined),
            "mean_norm": float(np.linalg.norm(combined, axis=1).mean()),
            "std_norm": float(np.linalg.norm(combined, axis=1).std()),
            "dim_mean": combined.mean(axis=0),  # for pairwise comparisons
            "dim_std": combined.std(axis=0),
        }

    # Compute pairwise Cohen's d between conditions
    def cohens_d(a_mean, a_std, b_mean, b_std):
        pooled = np.sqrt((a_std ** 2 + b_std ** 2) / 2) + 1e-8
        return float(np.abs(a_mean - b_mean).mean() / pooled.mean())

    comparisons = {}
    pairs = [("neutral", "violent"), ("previolent", "violent"), ("neutral", "previolent")]
    for a, b in pairs:
        if stats[a].get("n_frames", 0) > 0 and stats[b].get("n_frames", 0) > 0:
            d = cohens_d(
                stats[a]["dim_mean"], stats[a]["dim_std"],
                stats[b]["dim_mean"], stats[b]["dim_std"],
            )
            comparisons[f"{a}_vs_{b}"] = round(d, 4)

    return stats, comparisons


# ──────────────────────────────────────────────
# Temporal pattern analysis
# ──────────────────────────────────────────────

def analyze_temporal_patterns(windows: List[Dict]) -> Dict:
    """
    Ask: is violence always at the end of videos? Beginning? Random position?
    """
    violent_videos = [w for w in windows if w["has_violence"]]
    if not violent_videos:
        return {}

    n_frames_list = [w["n_frames"] for w in violent_videos]
    start_positions = [w["violence_start"] / w["n_frames"] for w in violent_videos]  # normalized 0-1
    end_positions = [w["violence_end"] / w["n_frames"] for w in violent_videos]
    violence_fractions = [w["violence_fraction"] for w in violent_videos]
    previolence_fractions = [w["previolence_window"] / w["n_frames"] for w in violent_videos]

    return {
        "n_violent_videos": len(violent_videos),
        "mean_video_length": float(np.mean(n_frames_list)),
        "median_video_length": float(np.median(n_frames_list)),
        "violence_start_position": {
            "mean": float(np.mean(start_positions)),
            "std": float(np.std(start_positions)),
            "min": float(np.min(start_positions)),
            "max": float(np.max(start_positions)),
        },
        "violence_end_position": {
            "mean": float(np.mean(end_positions)),
            "std": float(np.std(end_positions)),
        },
        "violence_fraction": {
            "mean": float(np.mean(violence_fractions)),
            "median": float(np.median(violence_fractions)),
        },
        "previolence_fraction": {
            "mean": float(np.mean(previolence_fractions)),
            "median": float(np.median(previolence_fractions)),
        },
    }


# ──────────────────────────────────────────────
# Anticipation-ready video identification
# ──────────────────────────────────────────────

def identify_best_anticipation_videos(windows: List[Dict], min_previolence: int = 10) -> List[Dict]:
    """
    Find videos with the most useful pre-violence context for anticipation training.
    Ranked by pre-violence window length.
    """
    good = [w for w in windows if w.get("usable_for_anticipation") and w["previolence_window"] >= min_previolence]
    good.sort(key=lambda w: w["previolence_window"], reverse=True)
    return good


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--max_videos", type=int, default=1000)
    parser.add_argument("--min_previolence", type=int, default=5,
                        help="Minimum frames before first violent frame to count as 'usable'")
    parser.add_argument("--output_dir", type=Path, default=Path("results/xd_investigation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("XD-VIOLENCE DATA INVESTIGATION")
    print("=" * 65)

    print(f"\nLoading up to {args.max_videos} videos...")
    videos = load_all_videos(args.embeddings, args.labels, args.max_videos)
    print(f"Loaded {len(videos)} videos")

    if not videos:
        print("No videos loaded. Check paths.")
        return

    # ── Pre-violence window analysis ──────────
    print("\n[1] Pre-Violence Window Analysis")
    windows = [analyze_pre_violence_windows(v, args.min_previolence) for v in videos]

    n_violent = sum(1 for w in windows if w["has_violence"])
    n_nonviolent = len(windows) - n_violent
    print(f"  Total videos:     {len(windows)}")
    print(f"  With violence:    {n_violent} ({100*n_violent/len(windows):.1f}%)")
    print(f"  No violence:      {n_nonviolent} ({100*n_nonviolent/len(windows):.1f}%)")

    violent_windows = [w for w in windows if w["has_violence"]]
    if violent_windows:
        previolence_lengths = [w["previolence_window"] for w in violent_windows]
        print(f"\n  Pre-violence window lengths (in frames):")
        print(f"    Mean:   {np.mean(previolence_lengths):.1f}")
        print(f"    Median: {np.median(previolence_lengths):.1f}")
        print(f"    Min:    {np.min(previolence_lengths)}")
        print(f"    Max:    {np.max(previolence_lengths)}")

        thresholds = [0, 5, 10, 20, 30, 50]
        print(f"\n  Videos with pre-violence window >= N frames:")
        for t in thresholds:
            count = sum(1 for p in previolence_lengths if p >= t)
            print(f"    >= {t:3d} frames: {count:4d}  ({100*count/len(violent_windows):.1f}%)")

    # ── Videos best for anticipation ─────────
    print(f"\n[2] Best Videos for Anticipation Training (>= {args.min_previolence} pre-violence frames)")
    best = identify_best_anticipation_videos(windows, args.min_previolence)
    print(f"  Found {len(best)} usable videos")
    if best:
        print(f"  Top 20 by pre-violence window:")
        print(f"  {'Video':<40} {'Pre-viol frames':>16} {'Total frames':>13} {'Violence %':>11}")
        print("  " + "-" * 83)
        for w in best[:20]:
            print(f"  {w['name']:<40} {w['previolence_window']:>16}   {w['n_frames']:>12}  {w['violence_fraction']*100:>10.1f}%")

    # Save best video list
    best_list_path = args.output_dir / "best_anticipation_videos.txt"
    with open(best_list_path, "w") as f:
        f.write("# Videos with sufficient pre-violence context for anticipation training\n")
        f.write(f"# Minimum pre-violence window: {args.min_previolence} frames\n")
        f.write(f"# Format: video_name  previolence_frames  total_frames  violence_pct\n\n")
        for w in best:
            f.write(f"{w['name']}\t{w['previolence_window']}\t{w['n_frames']}\t{w['violence_fraction']*100:.1f}\n")
    print(f"\n  Best video list saved to {best_list_path}")

    # ── Temporal patterns ────────────────────
    print("\n[3] Temporal Patterns")
    temporal = analyze_temporal_patterns(windows)
    if temporal:
        print(f"  Violence start position (normalized 0=start, 1=end of video):")
        vsp = temporal["violence_start_position"]
        print(f"    Mean: {vsp['mean']:.2f}  Std: {vsp['std']:.2f}  Range: [{vsp['min']:.2f}, {vsp['max']:.2f}]")
        print(f"  Violence fraction of video:")
        vf = temporal["violence_fraction"]
        print(f"    Mean: {vf['mean']:.2f}  Median: {vf['median']:.2f}")
        print(f"  Pre-violence fraction of video:")
        pvf = temporal["previolence_fraction"]
        print(f"    Mean: {pvf['mean']:.2f}  Median: {pvf['median']:.2f}")

        # Interpretation
        if vsp["mean"] > 0.7:
            print(f"\n  FINDING: Violence is strongly end-biased (mean position {vsp['mean']:.2f})")
            print(f"  This means models can learn 'predict violence toward end of video'")
            print(f"  as a shortcut — which would explain good in-domain but poor real-world performance.")
        elif vsp["mean"] < 0.3:
            print(f"\n  FINDING: Violence is start-biased (mean position {vsp['mean']:.2f})")
        else:
            print(f"\n  FINDING: Violence is roughly uniform (mean position {vsp['mean']:.2f})")

    # ── Embedding distribution ────────────────
    print("\n[4] Embedding Distribution by Temporal Context")
    emb_stats, comparisons = analyze_embedding_distribution(videos)
    for name, s in emb_stats.items():
        if s.get("n_frames", 0) > 0:
            print(f"  {name:12s}: {s['n_frames']:6d} frames  |norm|={s['mean_norm']:.4f} ± {s['std_norm']:.4f}")

    if comparisons:
        print(f"\n  Pairwise mean Cohen's d (embedding separation):")
        for pair, d in comparisons.items():
            a, b = pair.split("_vs_")
            interpretation = "small" if d < 0.2 else "medium" if d < 0.5 else "large"
            print(f"    {pair:<30} d={d:.4f}  ({interpretation})")

        # Key finding
        prv_vio = comparisons.get("previolent_vs_violent", None)
        neu_vio = comparisons.get("neutral_vs_violent", None)
        if prv_vio and neu_vio:
            if prv_vio < neu_vio * 0.5:
                print(f"\n  FINDING: Pre-violent embeddings are much closer to violent than neutral is.")
                print(f"  This suggests the VLM IS capturing some pre-violence signal in embeddings.")
            else:
                print(f"\n  FINDING: Pre-violent embeddings are similar to neutral.")
                print(f"  This suggests limited pre-violence signal in current embeddings — prompting may help.")

    # ── Class balance per video ───────────────
    print("\n[5] Class Balance Distribution")
    viol_fracs = [w["violence_fraction"] for w in windows if w["has_violence"]]
    if viol_fracs:
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        print(f"  Violence fraction distribution (across violent videos):")
        for i in range(len(bins) - 1):
            count = sum(1 for f in viol_fracs if bins[i] <= f < bins[i + 1])
            bar = "█" * (count // max(1, len(viol_fracs) // 20))
            print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {count:4d} videos  {bar}")

    # ── Summary and recommendations ──────────
    print("\n" + "=" * 65)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 65)
    total = len(windows)
    n_usable = len(best)
    print(f"  Total videos loaded:                  {total}")
    print(f"  Videos with violence:                 {n_violent}")
    print(f"  Usable for anticipation (≥{args.min_previolence} frames): {n_usable}")
    print(f"  Proportion usable:                    {100*n_usable/max(1,n_violent):.1f}% of violent videos")
    print()
    if n_usable < 30:
        print("  WARNING: Very few videos with sufficient pre-violence context.")
        print("  Recommendation: Consider synthetic augmentation or alternative datasets")
        print("  (e.g., Hockey Fight, VIRAT, or CCTV-Fight which have more pre-event frames).")
    elif n_usable < 100:
        print("  CAUTION: Limited anticipation training data.")
        print("  Recommendation: Use these videos with high weight, augment with EMA smoothing.")
    else:
        print("  OK: Sufficient anticipation data. Focus on preventing shortcut learning.")

    # Save full stats
    import json
    stats_out = {
        "n_videos": total,
        "n_violent_videos": n_violent,
        "n_nonviolent_videos": n_nonviolent,
        "n_anticipation_usable": n_usable,
        "temporal_patterns": temporal,
        "embedding_comparisons": comparisons,
        "top_anticipation_videos": [
            {k: v for k, v in w.items() if k != "name" or True}
            for w in best[:50]
        ],
    }
    with open(args.output_dir / "investigation_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\n  Full stats saved to {args.output_dir}/investigation_stats.json")


if __name__ == "__main__":
    main()