"""Synthetic checks of the fusion forward selection, the frozen spec and the one-shot fold-1 confirmation.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_fusion.py -q
"""

import json
import os
import types
import zlib

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from anticipation.screen import confirm as C
from anticipation.screen import features as F
from anticipation.screen import fusion as FU
from anticipation.screen import metrics as M
from anticipation.screen import run as R
from anticipation.screen import sets as S
from tests.test_screen import build_world

BAND = lambda t, onset: 2.5 * ((onset - t > 30) & (onset - t <= 120))  # noqa: E731
STREAMS = ("fx_useful", "fx_redundant", "fx_noise", "fx_near")
NEAR = lambda t, onset: 2.5 * ((onset - t > 0) & (onset - t <= 10))  # noqa: E731


@pytest.fixture(autouse=True)
def caps(monkeypatch):
    torch.set_num_threads(2)
    for name in STREAMS:
        monkeypatch.setitem(F.BLOCKS, name, F.Block(name, "A", (name,)))
    with threadpool_limits(limits=2):
        yield


def make_world(seen=None):
    ss, _ = build_world()
    bench = ss.bench

    def loader(stream, vid, probe=False, scene=False):
        if seen is not None:
            seen.append(vid)
        if scene:
            return np.random.default_rng(zlib.crc32(vid.encode())).normal(0, 1, 16).astype(np.float32)
        if stream not in STREAMS:
            if probe:
                return False
            raise FileNotFoundError(stream)
        if probe:
            return True
        row = bench.videos.loc[vid]
        T = int(np.floor(row["duration_sec"]))
        t = np.arange(1, T + 1, dtype=np.float32)
        base = np.random.default_rng(zlib.crc32(vid.encode()) + 1).normal(0, 1, (T, 4)).astype(np.float32)
        signal = np.zeros(T, np.float32)
        if row["role"] == "violent":
            signal = BAND(t, float(row["first_anomaly_sec"])).astype(np.float32)
        near = NEAR(t, float(row["first_anomaly_sec"])).astype(np.float32) if row["role"] == "violent" else np.zeros(T, np.float32)
        own = np.random.default_rng(zlib.crc32((vid + stream).encode())).normal(0, 1, (T, 4)).astype(np.float32)
        if stream == "fx_useful":
            x = base
            x[:, 0] += signal
        elif stream == "fx_redundant":
            x = 0.02 * own
            x[:, 0] += base[:, 0] + signal
        elif stream == "fx_near":
            x = own
            x[:, 0] += near
        else:
            x = own
        return t, x, np.ones(T, bool), {"columns": ["a", "b", "c", "d"]}

    return ss, F.FeatureSource(ss, loader=loader), loader


PR_INFO = {"path": "synthetic", "sha256": "synthetic"}


def select(tmp_path, ss, source, start=("fx_useful",), more=("fx_redundant", "fx_noise"), imminent=None, **kw):
    result, head = FU.run_selection(ss, source, list(start), list(more), str(tmp_path), PR_INFO, n_boot=200,
                                    require_cut=False, events=False, skip_p2=True, **kw)
    spec, path, sha = FU.assemble_config(ss, PR_INFO, {"early": head, "imminent": imminent}, str(tmp_path))
    return result, spec, sha


def test_forward_selection_keeps_the_useful_block_and_rejects_redundant_and_noise(tmp_path):
    ss, source, _ = make_world()
    result, spec, sha = select(tmp_path, ss, source)
    assert result["selected"] == ["fx_useful"]
    assert [s["action"] for s in result["steps"]] == ["start", "stop"]
    last = result["steps"][-1]
    assert set(last["candidates"]) == {"fx_redundant", "fx_noise"}
    for c in last["candidates"].values():
        lo = c["m8_gain"]["ci"][0]
        assert lo <= 0
    assert result["p1"]["stats"]["m1"]["value"] > 0.8
    assert result["p1"]["stats"]["m8"]["value"] > 0.8
    assert result["p1"]["vs_nuisance"]["m4n"]["value"] > 0.02
    assert spec["heads"]["early"]["blocks"][0]["name"] == "fx_useful" and len(spec["heads"]["early"]["blocks"]) == 1
    assert spec["heads"]["early"]["probe_type"] == "p1" and spec["heads"]["imminent"] is None
    assert sha == C.sha_bytes(open(tmp_path / "confirmation_config.json", "rb").read())


def test_noise_block_is_never_added_after_a_redundant_one_and_the_gain_is_paired(tmp_path):
    ss, source, _ = make_world()
    data = FU.FusionData(ss, source, ["fx_useful", "fx_noise"], require_cut=False)
    data.build()
    ev = FU.Evaluator(data, 200)
    a, b = ev.evaluate(["fx_useful"]), ev.evaluate(["fx_useful", "fx_noise"])
    g = M.diff(b["stats"]["m8"], a["stats"]["m8"])
    assert abs(g.value) < 0.05
    assert a["stats"]["m8"].boot.shape == b["stats"]["m8"].boot.shape


def test_selection_never_touches_fold_one(tmp_path):
    seen = []
    ss, source, _ = make_world(seen)
    select(tmp_path, ss, source)
    assert seen
    assert not [v for v in seen if ss.fold_of(v) == 1]
    with pytest.raises(S.FrozenFoldError):
        source.load("fx_useful", "synthetic/f1v0")


def test_p2_selection_rule_and_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(FU, "P2_MAX_POS", 150)
    monkeypatch.setattr(FU, "P2_MAX_NEG", 150)
    monkeypatch.setattr(FU, "P2_MAX_VAL_NEG", 100)
    monkeypatch.setattr(FU, "P2_BAG_TICKS", 4)
    ss, source, _ = make_world()
    pr_info = {"path": "synthetic", "sha256": "synthetic"}
    result, spec = FU.run_selection(ss, source, ["fx_useful"], [], str(tmp_path), pr_info, n_boot=100, require_cut=False,
                                         events=False, p2_seeds=(0, 1), p2_epochs=1)
    p2 = result["p2"]
    assert p2["seeds"] == [0, 1] and set(p2["seed_stats"]) == {"0", "1"}
    lower = p2["m8_gain_over_p1"]["ci"][0]
    assert result["final"]["probe"] == ("p2" if lower > 0 else "p1")
    assert spec["probe_type"] == result["final"]["probe"]
    assert (spec["p2"] is not None) == (spec["probe_type"] == "p2")


def test_forced_p2_spec_carries_epochs_and_hyperparameters(tmp_path):
    ss, source, _ = make_world()
    sel = {"final": {"probe": "p2", "blocks": ["fx_useful"], "p1_C": 0.1, "p1_C_per_fold": {}}, "nuisance_C": 0.1,
           "p2": {"seeds": [0, 1, 2]}}
    head = FU.freeze_head("imminent", sel, 0, 100, 100, p2_epochs_fixed=7)
    assert head["p2"]["fixed_epochs"] == 7 and head["p2"]["hidden"] == 64 and head["p2"]["layers"] == 1
    assert head["p2"]["context_sec"] == 180 and head["p2"]["seeds"] == [0, 1, 2]
    assert head["confirmation_rule"]["m1_near_ci_lower_gt"] == 0.6
    spec = FU.common_spec(ss, PR_INFO, {"early": None, "imminent": head})
    assert spec["training"]["folds"] == [0, 2, 3, 4] and 1 not in spec["training"]["folds"]


def test_gru_fixed_epochs_and_streaming_scores_match_window_scores(tmp_path):
    ss, source, _ = make_world()
    data = FU.FusionData(ss, source, ["fx_useful"], require_cut=False)
    c = data.build_fold(0)
    z, dims = FU.gru_inputs(data.runners, ["fx_useful"], c.vt, c.cohort["train"], c.cohort["train"] + c.cohort["val"] + c.cohort["test"])
    plan = FU.P.Plan([c.vt[v] for v in c.cohort["train"]], "band", np.random.default_rng(0), 100, 100, ss.source_kind)
    rows = FU.gru_rows(plan, c.vt)
    model, hist = FU.train_gru(z, dims, rows, rows[:50], [], 0, ctx=60, epochs=2, fixed_epochs=True)
    assert len(hist) == 2
    v = c.cohort["test"][0]
    full = FU.score_video(model, z[v], model.context)
    pick = [(v, 5, 0, 1), (v, 100, 0, 1), (v, len(full) - 1, 0, 1)]
    rowwise = FU.score_rows(model, z, pick, model.context)
    assert np.allclose(full[[5, 100, len(full) - 1]], rowwise, atol=1e-4)


def test_config_hash_gate(tmp_path):
    ss, source, _ = make_world()
    result, spec, sha = select(tmp_path, ss, source)
    path = str(tmp_path / "confirmation_config.json")
    assert C.read_config(path, sha, require_committed=False)["heads"]["early"]["probe_type"] == "p1"
    with pytest.raises(C.ConfirmError):
        C.read_config(path, "0" * 64, require_committed=False)
    with pytest.raises(C.ConfirmError):
        C.read_config(path, sha)
    with open(path, "ab") as f:
        f.write(b" ")
    with pytest.raises(C.ConfirmError):
        C.read_config(path, sha, require_committed=False)


def test_confirm_refuses_without_the_flag_before_reading_anything(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fold-1 machinery was reached")

    monkeypatch.setattr(R, "prepare", boom)
    monkeypatch.setattr(C, "run_confirmation", boom)
    monkeypatch.setattr(C, "ConfirmSet", boom)
    with pytest.raises(SystemExit):
        C.main(["--config_sha256", "0" * 64, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        C.main(["--confirm_once", "--config_sha256", "0" * 64, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit):
        C.main(["--confirm_once", "--config_sha256", "0" * 64, "--out", str(tmp_path / "taken")])


def test_real_fold_one_is_still_frozen_for_a_screening_set():
    ss, source, _ = make_world()
    with pytest.raises(S.FrozenFoldError):
        ss.ticks("synthetic/f1v0")
    cset = C.ConfirmSet(ss)
    assert cset.ticks("synthetic/f1v0").n > 0
    with pytest.raises(S.FrozenFoldError):
        ss.ticks("synthetic/f1v0")
    assert all(ss.fold_of(v) == 1 for v in cset.confirmation_test_ids())
    assert cset.confirmation_test_ids()


def test_verdict_rule():
    spec = {"confirmation_rule": {"m1_band_ci_lower_gt": 0.5, "m8_gain_over_n_point_gt": 0.0, "text": "t"}}
    boot = lambda v, s=0.01: np.random.default_rng(0).normal(v, s, 300)  # noqa: E731
    good = {"m1": M.Stat(0.7, boot(0.7))}
    assert C.verdict(spec, good, None, M.Stat(0.05, boot(0.05)))["confirmed"]
    assert not C.verdict(spec, good, None, M.Stat(-0.01, boot(-0.01)))["confirmed"]
    weak = {"m1": M.Stat(0.51, boot(0.51, 0.03))}
    assert not C.verdict(spec, weak, None, M.Stat(0.05, boot(0.05)))["confirmed"]


def two_heads(tmp_path, ss, source, early_start, imminent_blocks):
    """Run gates and both head selections on the synthetic world; returns the assembled spec and its sha256."""
    gates = FU.run_gates(ss, source, ["fx_useful", "fx_near", "fx_noise"], str(tmp_path / "imminent"), n_boot=200,
                         require_cut=False)
    passing = sorted(n for n, r in gates["blocks"].items() if r["pass"])
    head_i = None
    if imminent_blocks and passing:
        _, head_i = FU.run_selection(ss, source, passing, [], str(tmp_path / "imminent"), PR_INFO, n_boot=200,
                                     require_cut=False, events=False, skip_p2=True, head="imminent", root=str(tmp_path))
    _, head_e = FU.run_selection(ss, source, [early_start], [], str(tmp_path), PR_INFO, n_boot=200, require_cut=False,
                                 events=False, skip_p2=True)
    spec, path, sha = FU.assemble_config(ss, PR_INFO, {"early": head_e, "imminent": head_i}, str(tmp_path))
    return gates, spec, sha


def patch_confirm(monkeypatch, ss, loader):
    pr = types.SimpleNamespace(sha256="synthetic", path="synthetic", data={})
    monkeypatch.setattr(R, "prepare", lambda args: (pr, ss, {}))
    monkeypatch.setattr(C, "committed_bytes", lambda p: open(p, "rb").read())
    monkeypatch.setattr(C, "REQUIRE_CUT", False)
    monkeypatch.setattr(C.F, "FeatureSource", lambda s, root=None, _cls=F.FeatureSource: _cls(s, loader=loader))


def test_imminent_gates_pass_only_the_near_block(tmp_path):
    ss, source, _ = make_world()
    gates = FU.run_gates(ss, source, ["fx_useful", "fx_near", "fx_noise"], str(tmp_path), n_boot=200, require_cut=False)
    b = gates["blocks"]
    assert b["fx_near"]["pass"] and b["fx_near"]["N1"] and b["fx_near"]["N2"] and b["fx_near"]["N3"]
    assert not b["fx_noise"]["pass"] and not b["fx_noise"]["N1"]
    again = FU.run_gates(ss, source, ["fx_near"], str(tmp_path), n_boot=200, require_cut=False)
    assert again["blocks"]["fx_near"] == b["fx_near"]


def test_imminent_selection_reports_near_metrics(tmp_path):
    ss, source, _ = make_world()
    result, head = FU.run_selection(ss, source, ["fx_near"], ["fx_noise"], str(tmp_path), PR_INFO, n_boot=200, require_cut=False,
                                    events=False, skip_p2=True, head="imminent")
    assert result["selected"] == ["fx_near"]
    assert result["p1"]["stats"]["m1"]["value"] > 0.8 and result["p1"]["stats"]["m8"]["value"] > 0.8
    assert result["p1"]["stats"]["m3p"]["ci"] is not None
    assert result["b_detect_m1_near"]["value"] is not None
    assert head["head"] == "imminent" and head["confirmation_rule"]["m1_near_ci_lower_gt"] == 0.6
    assert os.path.isfile(tmp_path / "report.md")


def test_end_to_end_one_shot_confirmation_of_both_heads(tmp_path, monkeypatch):
    sel_dir = tmp_path / "fusion"
    ss, source, loader = make_world()
    gates, spec, sha = two_heads(sel_dir, ss, source, "fx_useful", True)
    assert spec["heads"]["imminent"]["blocks"][0]["name"] in ("fx_near", "fx_useful")
    cfg = str(sel_dir / "confirmation_config.json")
    patch_confirm(monkeypatch, ss, loader)
    out = str(tmp_path / "confirmation")
    args = ["--confirm_once", "--config", cfg, "--config_sha256", sha, "--out", out]
    assert C.main(args) == 0
    res = json.load(open(os.path.join(out, "confirmation.json")))
    assert res["spec_sha256"] == sha
    e, i = res["heads"]["early"], res["heads"]["imminent"]
    assert e["verdict"]["confirmed"] is True
    assert e["model"]["m1"]["value"] > 0.8 and e["m8_gain_over_n"]["value"] > 0
    assert e["coverage"]["scored"] == e["coverage"]["fold1_test_videos"] > 0
    assert "operating_points" in e["events"]
    assert i["model"]["m1"]["value"] > 0.8 and i["verdict"]["m1_near_lower_gt_0.6"] is True
    op = i["events"]["operating_points"]["fa2"]
    assert op["success"]["value"] is not None and "threshold_aware_ci" in op and "position_null" in op
    assert i["verdict"]["success10_threshold_aware_ci"] is not None
    assert os.path.isfile(os.path.join(out, "report.md"))
    with pytest.raises(SystemExit):
        C.main(args)
    with pytest.raises(SystemExit):
        C.main(["--confirm_once", "--config", cfg, "--config_sha256", "1" * 64, "--out", str(tmp_path / "other")])


def test_confirmation_does_not_confirm_noise_heads(tmp_path, monkeypatch):
    out_dir = tmp_path / "fusion"
    ss, source, loader = make_world()
    _, head_n = FU.run_selection(ss, source, ["fx_noise"], [], str(out_dir / "imminent"), PR_INFO, n_boot=200, require_cut=False,
                                 events=False, skip_p2=True, head="imminent", root=str(out_dir))
    _, head_e = FU.run_selection(ss, source, ["fx_noise"], [], str(out_dir), PR_INFO, n_boot=200, require_cut=False,
                                 events=False, skip_p2=True)
    spec, path, sha = FU.assemble_config(ss, PR_INFO, {"early": head_e, "imminent": head_n}, str(out_dir))
    patch_confirm(monkeypatch, ss, loader)
    C.main(["--confirm_once", "--config", path, "--config_sha256", sha, "--out", str(tmp_path / "c")])
    res = json.load(open(tmp_path / "c" / "confirmation.json"))
    assert res["confirmed"] == {"early": False, "imminent": False}


def test_missing_imminent_head_is_reported_unconfirmed(tmp_path, monkeypatch):
    out_dir = tmp_path / "fusion"
    ss, source, loader = make_world()
    _, head_e = FU.run_selection(ss, source, ["fx_useful"], [], str(out_dir), PR_INFO, n_boot=200, require_cut=False,
                                 events=False, skip_p2=True)
    spec, path, sha = FU.assemble_config(ss, PR_INFO, {"early": head_e, "imminent": None}, str(out_dir))
    patch_confirm(monkeypatch, ss, loader)
    C.main(["--confirm_once", "--config", path, "--config_sha256", sha, "--out", str(tmp_path / "c")])
    res = json.load(open(tmp_path / "c" / "confirmation.json"))
    assert res["confirmed"]["imminent"] is False and res["confirmed"]["early"] is True


def test_lead_report_matches_va30_success10_and_adds_threshold_aware_ci(tmp_path):
    from anticipation.eval import va30
    ss, source, _ = make_world()
    result, head = FU.run_selection(ss, source, ["fx_near"], [], str(tmp_path), PR_INFO, n_boot=100, require_cut=False,
                                    events=False, skip_p2=True, head="imminent")
    df = FU.pd.read_parquet(tmp_path / "oof_scores.parquet")
    df = df[df["video_id"].map(ss.kind) != "h1"]
    assign = {v: i for v, i in ss.bench.folds.assignment.items() if i != 1}
    fm = {0: 0, 2: 1, 3: 2, 4: 3}
    val = {str(fm[f]): [v for v in vs if v in assign] for f, vs in ss.bench.folds.val.items() if f in fm}
    folds = C.Folds({"name": "x", "n_folds": 4, "assignment": {v: fm[i] for v, i in assign.items()}, "val": val,
                     "content_sha256": "x"})
    bench = va30.Benchmark(ss.bench.videos, ss.bench.events, folds)
    d = df.assign(fold=df["fold"].map(fm))[["video_id", "t_sec", "score", "fold", "seed"]]
    rep = FU.lead_report(bench, d, 10.0, (2,), 100, 20)
    cfg = va30.EvalConfig(n_boot=100, cluster="split_group", fa_targets=(2.0,), restrict_to_scored=True, allow_missing=True)
    ref = va30.evaluate(d, bench, cfg)
    ref_op = ref["seeds"]["0"]["operating_points"]["fa2"]
    op = rep["operating_points"]["fa2"]
    assert op["success"]["value"] == pytest.approx(ref_op["eligible30"]["success10"]["value"])
    assert op["test_fa_per_hour_clean"] == pytest.approx(ref_op["fa_per_hour_clean"]["value"])
    lo, hi = op["threshold_aware_ci"]["success_ci"]
    assert lo <= op["success"]["value"] <= hi
    assert op["position_null"] is not None
