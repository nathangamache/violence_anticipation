import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from anticipation.data.folds import Folds, content_hash
from anticipation.eval.learned_baselines import (
    StudentMLP, catalog_mapping, fit_scene, fit_student, load_features,
    read_features, score_fold, tag_vector, training_arrays,
)
from anticipation.eval.baselines import sample_and_hold
from anticipation.eval.va30 import Benchmark, ScoreTable


def make_bench():
    rows = []
    for i in range(8):
        vid = f"xd/v{i}"
        rows.append(dict(video_id=vid, source="xd", path=f"/videos/v{i}.mp4", aliases="",
                         duration_sec=180, fps=10, role="violent" if i % 2 else "clean_negative",
                         onset_source="official", source_kind="xd_movie", interpersonal=True,
                         has_audio=False, group=vid))
    videos = pd.DataFrame(rows).set_index("video_id", drop=False)
    assignment = {f"xd/v{i}": i // 2 for i in range(8)}
    val = {str(f): [f"xd/v{2*((f+1)%4)}", f"xd/v{2*((f+1)%4)+1}"] for f in range(4)}
    folds = Folds(dict(n_folds=4, assignment=assignment, val=val,
                       content_sha256=content_hash(assignment, val)))
    return Benchmark(videos, {f"xd/v{i}": {"segments": [[60, 70], [120, 130]]} for i in (1, 3, 5, 7)}, folds)


def write_features(root, name, raw=(5, 15, 350, 650, 750, 850, 1000, 1600)):
    path = root / "xd_violence" / (name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.zeros((len(raw), 3584), np.float32)
    emb[:, 0] = np.arange(len(raw))
    np.savez(path, emb=emb, raw_frame_idx=raw, t=np.arange(len(raw)) * 5)
    return path


def test_clock_tag_alignment_and_no_future_frame(tmp_path):
    path = write_features(tmp_path, "a", raw=(30, 5, 15))
    tag = tmp_path / "a.jsonl"
    records = [dict(t=0, raw_frame_idx=30, actions=["future"]),
               dict(t=5, raw_frame_idx=5, actions=["walking"]),
               dict(t=5, raw_frame_idx=5, actions=["resumed"]),
               dict(t=10, raw_frame_idx=30, actions=["wrong clock"])]
    tag.write_text("\n".join(json.dumps(r) for r in records))
    f = read_features(path, tag, 10, 180)
    np.testing.assert_allclose(f["times"], [0.5, 1.5, 3])
    np.testing.assert_array_equal(f["x"][0, 3584:], tag_vector(records[1]))
    assert not f["x"][1, 3584:].any()
    assert f["scene"][0] == 1.5
    np.testing.assert_allclose(sample_and_hold(f["times"], np.array([0.1, 0.2, 0.9]), np.array([0, 1, 2, 3])), [0, .1, .2, .9])


def test_hash_matches_legacy_counts_and_empty_fields():
    rec = {"actions": [" WALK ", "walk", "Run"], "weapons": [], "scene": "Indoor"}
    v = tag_vector(rec)
    expected = np.zeros(160)
    for token in ("walk", "walk", "run"):
        expected[int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1 / 3
    expected[96 + int(hashlib.md5(b"indoor").hexdigest(), 16) % 64] = 1
    np.testing.assert_allclose(v, expected)
    assert not tag_vector({}).any()


def test_scene_missing_first_two_seconds_never_uses_later_frame(tmp_path):
    path = write_features(tmp_path, "a", raw=(30, 40))
    f = read_features(path, tmp_path / "missing.jsonl", 10, 180)
    assert f["scene_rows"] == 0
    assert not f["scene"].any()
    with pytest.raises(ValueError, match="fps"):
        read_features(path, tmp_path / "missing.jsonl", 0, 180)


def test_alias_mapping_refuses_ambiguity():
    bench = make_bench()
    bench.videos.loc["xd/v0", "aliases"] = "/renamed/old_name.mp4"
    assert catalog_mapping(bench.videos)[("xd", "old_name")] == "xd/v0"
    bench.videos.loc["xd/v1", "aliases"] = "/other/old_name.mp4"
    with pytest.raises(ValueError, match="Ambiguous"):
        catalog_mapping(bench.videos)


def test_seconds_masks_censoring_roles_and_fold_scores(tmp_path):
    bench = make_bench()
    for i in range(8):
        write_features(tmp_path / "emb", f"v{i}")
    bench.videos.loc["xd/v6", "role"] = "other_anomaly"
    features, cov = load_features(bench, tmp_path / "emb", tmp_path / "tags")
    assert "xd/v6" not in features
    f = features["xd/v1"]
    np.testing.assert_array_equal(f["y"], [0, 0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(f["valid"], [True, True, True, False, False, True, True, False])
    train = training_arrays(features, bench.folds.train_ids(0))
    assert set(bench.folds.train_ids(0)).isdisjoint(bench.folds.val_ids(0) + bench.folds.test_ids(0))
    assert len(train[1]) > 0 and cov["no_frame_in_first_2s"] == 0
    scores = score_fold(features, bench.folds.val_ids(0) + bench.folds.test_ids(0), bench, 0, 2,
                        lambda f, t: sample_and_hold(f["times"], f["y"], t))
    assert set(scores.video_id) == set(bench.folds.val_ids(0) + bench.folds.test_ids(0))
    assert set(scores.fold) == {0} and set(scores.seed) == {2}
    ScoreTable(scores)


def test_student_small_fit_is_seeded_and_selects_validation_checkpoint():
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 8)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    args = SimpleNamespace(device="cpu", dropout=.2, lr=.001, weight_decay=.0001,
                           batch_size=16, epochs=3, patience=2)
    a, ai = fit_student((x, y), (x, y), args, 7)
    b, bi = fit_student((x, y), (x, y), args, 7)
    assert ai == bi
    assert ai["val_auroc30"] == max(h["val_auroc30"] for h in ai["history"])
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k])
    assert StudentMLP()(torch.zeros(2, 3744)).shape == (2,)


def test_scene_scaler_fits_training_videos_only():
    bench = make_bench()
    features = {v: {"scene": np.full(3584, i, np.float32)} for i, v in enumerate(bench.videos.index)}
    ids = bench.folds.train_ids(0)
    scaler, model, info = fit_scene(features, ids, bench, 0)
    expected = np.mean([features[v]["scene"] for v in ids], axis=0)
    np.testing.assert_allclose(scaler.mean_, expected)
    assert info["training_videos"] == len(ids)
    assert np.isfinite(model.predict_proba(scaler.transform(features["xd/v0"]["scene"][None]))).all()
