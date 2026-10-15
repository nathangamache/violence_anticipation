"""Pre-registration of the screen: set definition with video-list hashes, probe grids, thresholds, decision rule."""

import datetime
import hashlib
import json
import os

from anticipation.data.catalog import REPO_ROOT, file_sha256
from anticipation.screen import decision as D
from anticipation.screen import features as F
from anticipation.screen import probes as P
from anticipation.screen import sets as S

PREREG_PATH = os.path.join(REPO_ROOT, "benchmark", "screens", "screen_set_v1.json")


def build(ss, git_commit=None):
    lists = ss.video_lists()
    counts = ss.counts()
    ev = ss.events()
    sets = {k: {"n": len(v), "sha256": S.list_sha256(v), "video_ids": v} for k, v in lists.items()}
    for name in ("s_real", "s_long60", "s_ramp", "s_offubi", "s_audio"):
        ids = sorted(set(ev.loc[ev[name], "video_id"]))
        sets[name] = {"n_events": int(ev[name].sum()), "n_videos": len(ids), "sha256": S.list_sha256(ids)}
    events_key = sorted(f"{r.video_id}|{r.index}" for r in ev.itertuples())
    sets["events"] = {"n": len(events_key), "sha256": S.list_sha256(events_key)}
    bench = ss.bench
    return {
        "name": "screen_set_v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "plan": "docs/MODALITY_PLAN.md section 6",
        "git_commit": git_commit,
        "inputs": {"folds": bench.provenance.get("folds_path"), "folds_content_sha256": bench.provenance.get("folds_content_sha256"),
                   "videos_csv_sha256": bench.provenance.get("videos_csv_sha256"),
                   "events_json_sha256": bench.provenance.get("events_json_sha256"),
                   "horizons": bench.provenance.get("video_horizons"),
                   "h1_manifest": "benchmark/extract_manifest_v2.csv",
                   "h1_manifest_sha256": file_sha256(S.MANIFEST_V2) if os.path.isfile(S.MANIFEST_V2) else None},
        "folds": {"screening": list(ss.screen_folds), "frozen_confirmation": ss.frozen_fold,
                  "rule": "no feature, score or label of a fold-1 video is read during the screen; fold-1 videos are "
                          "also excluded from every training and validation set (screen_set.ScreenSet.guard)"},
        "soft_masks": ss.soft_mask_info,
        "counts": counts,
        "plan_counts": {"positive_videos": 373, "events": 446, "first_segment_events": 253, "s_real": 165,
                        "s_long60": 100, "s_ramp": 61, "s_offubi": 74, "s_audio": 86, "clean_negative_videos": 3200,
                        "clean_negative_hours": 125, "h1_videos": 320, "h1p_videos": 102},
        "sets": sets,
        "tick_definitions": {
            "positive_ticks": "first-segment pre-violence unmasked ticks 1..(last eligible onset + 10 s) of eligible30 "
                              "videos of the screening test folds; masked ticks follow Benchmark.labels plus soft masks",
            "band_positive": "tte in (30, 120] (M1; also (30, 60], (60, 120] on S_long60, (0, 10] near)",
            "band_negative": "unmasked clean ticks with >= 60 s of footage left (first 300 s of capped clean videos)",
            "time_matching": "source kind x TIME_MATCH_EDGES bin weights (va30.time_match_weights)",
            "h1_ticks": "ticks 1..min(first anomaly, 300 s) of H1 videos; tte = first anomaly - t",
        },
        "blocks": {
            "phase_A": [b.name for b in F.BLOCKS.values() if b.phase == "A"],
            "phase_B": [b.name for b in F.BLOCKS.values() if b.phase == "B"],
            "phase_C": list(F.PHASE_C),
            "external_references": list(F.EXTERNAL_REFERENCES),
            "streams": {b.name: list(b.streams) for b in F.BLOCKS.values()},
            "excluded_columns": {b.name: list(b.exclude) for b in F.BLOCKS.values() if b.exclude},
            "included_columns": {b.name: list(b.include) for b in F.BLOCKS.values() if b.include},
            "p0_lower_is_risk": {b.name: list(b.lower_is_risk) for b in F.BLOCKS.values() if b.lower_is_risk},
            "audio_blocks_use_diegetic_s_audio": [b.name for b in F.BLOCKS.values() if b.audio],
        },
        "probes": {
            "P0": "per column AUROC on the pooled screening ticks, direction pre-declared (default higher = riskier)",
            "P1": {"windows_sec": list(F.WINDOWS), "statistics": ["mean", "max", "fraction above clean-training p95"],
                   "cut_at_clip_start_and_shot_cuts": f"{F.CUT_STREAM}:{F.CUT_COLUMN}", "divided_by_available_history": True},
            "P1s_extra_slopes_sec": list(F.SLOPE_WINDOWS),
            "P1b": f"per-video causal early baseline subtracted (running mean over the first {F.BASELINE_SEC} s)",
            "wide_block_columns_over": F.WIDE_COLUMNS,
            "pls": {"components": F.PLS_COMPONENTS, "target": "band", "weights": "time-match", "applied": "per tick, before the windows"},
            "model": "L2 logistic regression", "C_grid": list(P.C_GRID), "C_selected_on": "fold validation slice, time-matched AUROC",
            "target": "band (30,120] vs clean >= 60 s left, time-match weights normalised to equal class mass",
            "training_row_caps": {"positives": P.MAX_TRAIN_POS, "negatives": P.MAX_TRAIN_NEG, "validation_negatives": P.MAX_VAL_NEG},
            "feature_clip_z": P.CLIP_Z,
            "side_probes": ["P1-near: tte (0, 10]", "P1-later: later-segment lead-ins, tte (30, 120]"],
            "P2": "GRU 64 units over 180 s, 3 seeds, only for blocks passing G1 (stub; not part of this run)",
            "seeds": [0],
        },
        "controls": ["b_time_source_fit (source x time bin rate)", "nuisance N = source x time-bin + scene0p PCA32",
                     "shuffled-video", "B-detect", "first-frame broadcast on S_ramp"],
        "metrics": {"bootstrap": {"n": 1000, "cluster": "split_group", "seed": 0},
                    "tail_percentile_on_clean_validation": 99.5, "tail_min_ticks": 3, "m1_sparse_fpr_max": 0.01},
        "thresholds": {"G1_point": D.G1_POINT, "G1_sparse_point": D.G1_SPARSE_POINT,
                       "G1_sparse_tail_useful": D.G1_SPARSE_TAIL_USEFUL, "G2_fail_upper": D.G2_FAIL_UPPER,
                       "G2_ratio_pass": D.G2_RATIO_PASS, "G2_ratio_fail": D.G2_RATIO_FAIL, "G3_point": D.G3_POINT,
                       "near_lower": D.NEAR_LOWER, "long_point": D.LONG_POINT, "alpha": D.ALPHA,
                       "soundtrack_margin": D.SOUNDTRACK_MARGIN},
        "decision_rule": {
            "G1": "PASS: M1 point >= 0.58 and Holm-adjusted one-sided bootstrap p < 0.05 across the blocks of the "
                  "phase (or the sparse route); FAIL: 95% CI upper < 0.58 (and sparse route FAIL or absent)",
            "G2": "PASS: M3p CI lower > 0.50 and ratio >= 0.5; FAIL: CI upper < 0.55 or ratio < 0.25",
            "G3": "PASS: M4n >= +0.02 with CI lower > 0; FAIL: CI upper < +0.02",
            "classes": {"ANTICIPATION": "G1 PASS, G3 PASS, G2 not FAIL", "PROVISIONAL": "G1 PASS, no FAIL in G2/G3, not ANTICIPATION",
                        "CONTEXT/SHORTCUT": "G1 PASS with G2 FAIL or G3 FAIL", "SHORT-HORIZON": "G1 not PASS and M1-near CI lower > 0.60",
                        "INCONCLUSIVE": "G1 INCONCLUSIVE", "DROP": "G1 FAIL and not SHORT-HORIZON"},
            "tags": {"LONG-LEAD": "M1-long point >= 0.58 and CI lower > 0.50", "RAMP": "M5 Wilcoxon p < 0.05 and most videos rising"},
            "holm_families": {"G1": "M1 one-sided p-values across the blocks of one phase (A, B)"},
        },
        "confirmation": {"fold": ss.frozen_fold, "conditions": ["M1 CI lower > 0.50", "positive M8 point gain over N"],
                         "use": "once, for the final fused model"},
    }


def write(ss, path=PREREG_PATH, git_commit=None):
    d = build(ss, git_commit)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(d, indent=1, sort_keys=False) + "\n"
    with open(path, "w") as f:
        f.write(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(path)}\n")
    return sha


def verify(ss, path=PREREG_PATH):
    """Compare the live set definition with the pre-registered video-list hashes; returns a list of mismatches."""
    with open(path) as f:
        reg = json.load(f)
    live = build(ss)
    bad = []
    for k, v in reg["sets"].items():
        if live["sets"].get(k, {}).get("sha256") != v["sha256"]:
            bad.append(k)
    if live["inputs"]["folds_content_sha256"] != reg["inputs"]["folds_content_sha256"]:
        bad.append("folds")
    return bad
