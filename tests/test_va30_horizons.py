"""Per-video evaluation horizons (docs/VA30.md "Capped clean negatives"): clean negatives whose footage was cut
for feature extraction are evaluated only up to their cap.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_va30_horizons.py -q
"""

import json
import zlib

import numpy as np
import pandas as pd
import pytest

from anticipation.data.catalog import file_sha256
from anticipation.eval import va30
from anticipation.eval.va30 import (AlarmPolicy, Benchmark, EvalConfig, EventEval, ScoreTable, TickSet, evaluate,
                                    read_video_horizons)
from tests.test_va30_fixes import make_bench

CAP = 300.083
LONG = 3000.0
N_CLEAN = 8


def specs(clean_dur, n_violent=16, n_clean=N_CLEAN, role="violent"):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_violent):
        on = float(rng.uniform(70, 150))
        fold = i % 2
        out.append({"vid": f"v{i}", "duration": on + 30, "segments": [[on, on + 25]], "role": role, "fold": fold,
                    "val": [1 - fold] if i % 4 < 2 else []})
    for i in range(n_clean):
        fold = i % 2
        out.append({"vid": f"n{i}", "duration": clean_dur, "role": "clean_negative", "fold": fold,
                    "val": [1 - fold] if i % 4 < 2 else []})
    return out


def capped_bench(clean_dur=LONG, cap=CAP):
    return make_bench(specs(clean_dur)).with_horizons({f"n{i}": cap for i in range(N_CLEAN)})


def scores(bench, extra=False):
    """Deterministic scores per (video, tick) so benches with different catalog lengths agree on shared ticks."""
    parts = []
    for vid in bench.folds.video_ids:
        lab = bench.labels(vid)
        base = np.random.default_rng(zlib.crc32(vid.encode())).random(int(LONG))
        parts.append(pd.DataFrame({"video_id": vid, "t_sec": lab.t_sec,
                                   "score": base[:lab.T] + 0.5 * ((lab.tte <= 60) & ~lab.masked),
                                   "fold": -1, "seed": 0}))
        if extra and vid in bench.video_horizons:
            t = np.arange(lab.T + 1, int(LONG) + 1, dtype=float)
            junk = pd.DataFrame({"video_id": vid, "t_sec": t, "score": 1e9, "fold": -1, "seed": 0})
            junk.loc[junk.index[0], "score"] = np.nan
            parts += [junk, junk.iloc[:3]]
    return pd.concat(parts, ignore_index=True)


def cfg(**kw):
    kw.setdefault("n_boot", 30)
    kw.setdefault("breakdowns", False)
    kw.setdefault("amoc_points", 8)
    kw.setdefault("position_null", False)
    return EvalConfig(**kw)


def comparable(res):
    seed = json.loads(json.dumps(va30._clean_nans(res["seeds"]["0"]), default=va30._json_default))
    seed.pop("_seconds")
    seed["counts"].pop("video_horizons", None)
    return seed


def test_capped_negative_scored_to_the_cap_evaluates():
    bench = capped_bench()
    assert bench.labels("n0").T == 300
    df = scores(bench)
    assert df[df.video_id == "n0"].t_sec.max() == 300
    res = evaluate(df, bench, cfg())
    assert res["counts"]["clean_negative_videos"] == N_CLEAN
    assert res["seeds"]["0"]["far_window_auroc"]["clean"]["value"] is not None
    with pytest.raises(ValueError, match="without scores|not exactly present"):
        evaluate(df, make_bench(specs(LONG)), cfg())


def test_fa_per_hour_denominator_uses_the_cap():
    bench = capped_bench()
    sc = {}
    for vid in bench.folds.video_ids:
        lab = bench.labels(vid)
        s = np.zeros(lab.T)
        if vid.startswith("n"):
            s[99:105] = 1.0
        sc[vid] = s
    vids = [v for v in bench.folds.video_ids if v.startswith("n")]
    ts = TickSet(bench, vids, sc)
    ee = EventEval(ts, AlarmPolicy())
    hours = N_CLEAN * 300.083 / 3600.0
    assert ts.duration_v == pytest.approx(np.full(N_CLEAN, CAP))
    assert ee.clean_hours == pytest.approx(hours)
    assert ee.fa_clean(np.full(N_CLEAN, 0.5)) == pytest.approx(N_CLEAN / hours)
    res = evaluate(scores(bench), bench, cfg())
    m = res["seeds"]["0"]
    assert m["operating_points"]["fa2"]["fa_per_hour_clean"]["hours"] == pytest.approx(hours, abs=1e-3)
    assert m["counts"]["clean_negative_hours"] == pytest.approx(hours, abs=0.006)
    ch = m["counts"]["video_horizons"]
    assert ch["clean_negative_videos_shortened"] == N_CLEAN
    assert ch["clean_negative_hours_used"] == pytest.approx(hours, abs=0.006)
    assert ch["clean_negative_hours_catalog"] == pytest.approx(N_CLEAN * LONG / 3600.0, abs=0.006)


def test_right_censoring_applies_at_the_new_end():
    bench = capped_bench()
    lab = bench.labels("n1")
    assert lab.T == 300
    assert int(lab.censored[60].sum()) == 60 and not lab.censored[60][:240].any()
    res = evaluate(scores(bench), bench, cfg())
    assert res["seeds"]["0"]["far_window_auroc"]["clean"]["n_neg_videos"] == N_CLEAN
    far_neg = sum(int(((bench.labels(f"n{i}").remaining >= 60) & ~bench.labels(f"n{i}").masked).sum())
                  for i in range(N_CLEAN))
    assert far_neg == N_CLEAN * 240
    assert res["seeds"]["0"]["far_window_auroc"]["clean"]["n_neg"] == far_neg


def test_scores_past_the_cap_are_ignored():
    bench = capped_bench()
    base = evaluate(scores(bench), bench, cfg(position_null=True))
    extra = scores(bench, extra=True)
    n_extra = int((extra.t_sec > CAP).sum())
    assert n_extra > 0 and extra.score.isna().any()
    with_extra = evaluate(extra, bench, cfg(position_null=True))
    assert comparable(with_extra) == comparable(base)
    assert with_extra["coverage"]["score_rows_beyond_horizon_ignored"] == n_extra
    assert with_extra["coverage"]["scores_are_probabilities"] == base["coverage"]["scores_are_probabilities"]
    with pytest.raises(ValueError, match="NaN"):
        ScoreTable(extra)
    assert ScoreTable(extra, bench.video_horizons).n_rows == len(scores(bench))


def test_capped_evaluation_equals_natively_short_videos():
    capped, native = capped_bench(), make_bench(specs(CAP))
    r_cap = evaluate(scores(capped), capped, cfg(position_null=True, breakdowns=True))
    r_nat = evaluate(scores(native), native, cfg(position_null=True, breakdowns=True))
    assert comparable(r_cap) == comparable(r_nat)
    assert r_cap["summary_over_seeds"] == r_nat["summary_over_seeds"]
    assert "video_horizons" not in r_nat["benchmark"]


@pytest.mark.parametrize("role", ["violent", "holdout"])
def test_shortening_a_non_clean_negative_raises(role, tmp_path):
    bench = make_bench(specs(LONG, role=role))
    with pytest.raises(ValueError, match=r"not clean negatives.*v3 \(role " + role):
        bench.with_horizons({"v3": 20.0, "n0": CAP})
    with pytest.raises(ValueError, match="not clean negatives"):
        evaluate(scores(bench), bench, cfg(horizons_csv=write_csv(tmp_path, {"v0": 5.0})))


def test_violent_listed_at_or_beyond_its_catalog_duration_is_untouched():
    bench = make_bench(specs(LONG))
    dur = bench.videos["duration_sec"].astype(float).to_dict()
    listed = {v: d for v, d in dur.items() if v.startswith("v")}
    listed["v1"] = dur["v1"] + 500.0
    listed["v2"] = dur["v2"] - 5e-4
    listed["n0"] = CAP
    listed["not_in_catalog"] = 10.0
    b2 = bench.with_horizons(listed)
    assert set(b2.video_horizons) == {"n0"}
    info = b2.provenance["video_horizons"]
    assert (info["n_shortened"], info["n_not_in_catalog"], info["n_not_shorter_than_catalog"]) == (1, 1, 16)
    assert b2.duration("v1") == dur["v1"] and b2.duration("n0") == CAP and b2.duration("n1") == LONG


def test_without_horizons_results_and_metadata_are_unchanged():
    bench = make_bench(specs(CAP))
    df = scores(bench)
    ref = evaluate(df, bench, cfg(position_null=True))
    listed = {v: float(d) + (10.0 if v.startswith("n") else 0.0)
              for v, d in bench.videos["duration_sec"].items()}
    res = evaluate(df, bench.with_horizons(listed), cfg(position_null=True))
    assert comparable(res) == comparable(ref)
    assert res["benchmark"]["video_horizons"]["n_shortened"] == 0
    assert "video_horizons" not in ref["benchmark"] and "video_horizons" not in ref["counts"]
    assert "horizons_csv" not in ref["config"]
    assert "score_rows_beyond_horizon_ignored" not in ref["coverage"]
    assert bench.video_horizons == {} and bench.duration("n0") == CAP
    with pytest.raises(ValueError, match="already"):
        bench.with_horizons(listed).with_horizons(listed)


def write_csv(tmp_path, mapping):
    df = pd.DataFrame({"video_id": list(mapping), "role": "clean_negative", "duration_sec": list(mapping.values()),
                       "orig_duration_sec": 9999.0})
    path = str(tmp_path / "horizons.csv")
    df.to_csv(path, index=False)
    return path


def test_horizon_file_is_recorded_with_its_hash_and_matches_the_dict_route(tmp_path):
    path = write_csv(tmp_path, {f"n{i}": CAP for i in range(N_CLEAN)})
    mapping, src = read_video_horizons(path)
    assert mapping == {f"n{i}": CAP for i in range(N_CLEAN)} and src["sha256"] == file_sha256(path)
    base = make_bench(specs(LONG))
    df = scores(capped_bench())
    via_cfg = evaluate(df, base, cfg(horizons_csv=path))
    via_dict = evaluate(df, capped_bench(), cfg())
    rec = via_cfg["benchmark"]["video_horizons"]
    assert rec["file"] == path and rec["sha256"] == file_sha256(path) and rec["n_shortened"] == N_CLEAN
    assert via_cfg["config"]["horizons_csv"] == path
    assert comparable(via_cfg) == comparable(via_dict)
    assert via_dict["benchmark"]["video_horizons"]["file"] is None
    assert base.video_horizons == {}
    with pytest.raises(ValueError, match="already"):
        evaluate(df, base.with_horizons(path), cfg(horizons_csv=path))


def test_horizon_file_validation(tmp_path):
    with pytest.raises(ValueError, match="lacks columns"):
        read_video_horizons(_raw(tmp_path, "video_id,length\nn0,5\n"))
    with pytest.raises(ValueError, match="repeat"):
        read_video_horizons(_raw(tmp_path, "video_id,duration_sec\nn0,5\nn0,6\n"))
    with pytest.raises(ValueError, match="finite duration_sec > 0"):
        read_video_horizons(_raw(tmp_path, "video_id,duration_sec\nn0,\n"))
    with pytest.raises(ValueError, match="> 0"):
        make_bench(specs(LONG)).with_horizons({"n0": 0.0})


def _raw(tmp_path, text):
    p = tmp_path / "raw.csv"
    p.write_text(text)
    return str(p)


def test_cli_horizons_flag_end_to_end(tmp_path):
    bench = capped_bench()
    full = make_bench(specs(LONG))
    vpath, epath, fpath = tmp_path / "videos.csv", tmp_path / "events.json", tmp_path / "folds.json"
    full.videos.drop(columns=["video_id"]).rename_axis("video_id").to_csv(vpath)
    epath.write_text(json.dumps({"videos": full.events}))
    fpath.write_text(json.dumps(full.folds.d))
    spath = tmp_path / "scores.csv"
    scores(bench, extra=True).dropna().to_csv(spath, index=False)
    hpath = write_csv(tmp_path, {f"n{i}": CAP for i in range(N_CLEAN)})
    common = ["--scores", str(spath), "--folds", str(fpath), "--videos", str(vpath), "--events", str(epath),
              "--n_boot", "20", "--no_breakdowns", "--no_position_null"]
    assert va30.main(common + ["--out_dir", str(tmp_path / "with"), "--horizons", hpath]) == 0
    m = json.loads((tmp_path / "with" / "metrics.json").read_text())
    assert m["benchmark"]["video_horizons"]["sha256"] == file_sha256(hpath)
    assert m["benchmark"]["video_horizons"]["file"] == hpath
    assert m["counts"]["video_horizons"]["clean_negative_videos_shortened"] == N_CLEAN
    assert "Capped clean negatives" in (tmp_path / "with" / "report.md").read_text()
    scores(bench).to_csv(spath, index=False)
    with pytest.raises(ValueError, match="without scores|not exactly present"):
        va30.main(common + ["--out_dir", str(tmp_path / "without")])
