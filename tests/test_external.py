import numpy as np

from anticipation.screen import external as X


def test_align_trims_and_holds():
    s, k = X.align(np.arange(5.0), 3)
    assert list(s) == [0, 1, 2] and k == 0
    s, k = X.align(np.arange(3.0), 5)
    assert list(s) == [0, 1, 2, 2, 2] and k == 2


def test_auc_boot_matches_rank_auc():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, 200)
    neg = rng.normal(0.0, 1.0, 300)
    score = np.concatenate([pos, neg])
    y = np.r_[np.ones(200, bool), np.zeros(300, bool)]
    vidx = np.arange(500)
    W = rng.multinomial(500, np.full(500, 1 / 500), size=20).astype(float)
    r = X.auc_boot(score, y, ~y, vidx, W)
    exact = (pos[:, None] > neg[None, :]).mean()
    assert abs(r["value"] - exact) < 1e-9
    assert r["ci"][0] < r["value"] < r["ci"][1]


def test_auc_boot_ties_and_empty():
    r = X.auc_boot(np.zeros(4), np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool), np.arange(4), np.ones((2, 4)))
    assert r["value"] == 0.5
    assert X.auc_boot(np.zeros(2), np.zeros(2, bool), np.ones(2, bool), np.arange(2), np.ones((2, 2)))["value"] is None
