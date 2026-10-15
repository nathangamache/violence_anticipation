"""Pre-registered gates, classes and tags (MODALITY_PLAN 6.6) applied to the statistics of metrics.json."""

from anticipation.screen.metrics import holm

G1_POINT = 0.58
G1_SPARSE_POINT = 0.55
G1_SPARSE_TAIL_USEFUL = 0.05
G2_FAIL_UPPER = 0.55
G2_RATIO_PASS = 0.5
G2_RATIO_FAIL = 0.25
G3_POINT = 0.02
NEAR_LOWER = 0.60
LONG_POINT = 0.58
ALPHA = 0.05
SOUNDTRACK_MARGIN = 0.08


def _v(d, k="value"):
    return None if d is None else d.get(k)


def _lo(d):
    return None if not d or not d.get("ci") else d["ci"][0]


def _hi(d):
    return None if not d or not d.get("ci") else d["ci"][1]


def gate_g1_main(m1, holm_p):
    """PASS: point >= 0.58 and the Holm-adjusted one-sided p-value < 0.05; FAIL: CI upper < 0.58."""
    if _v(m1) is None:
        return "INCONCLUSIVE"
    if _v(m1) >= G1_POINT and holm_p is not None and holm_p < ALPHA:
        return "PASS"
    if _hi(m1) is not None and _hi(m1) < G1_POINT:
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g1_sparse(pauc, tail):
    """PASS: standardized pAUC >= 0.55 with CI lower > 0.50, or tail-hit difference CI lower > 0.
    FAIL: both CIs exclude a useful effect (pAUC upper < 0.55 and tail-hit difference upper < 0.05)."""
    if _v(pauc) is None and _v(tail) is None:
        return "NA"
    if (_v(pauc) is not None and _v(pauc) >= G1_SPARSE_POINT and _lo(pauc) is not None and _lo(pauc) > 0.5) or \
            (_lo(tail) is not None and _lo(tail) > 0):
        return "PASS"
    if _hi(pauc) is not None and _hi(pauc) < G1_SPARSE_POINT and _hi(tail) is not None and _hi(tail) < G1_SPARSE_TAIL_USEFUL:
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g1(main, sparse):
    if main == "PASS" or sparse == "PASS":
        return "PASS"
    if main == "FAIL" and sparse in ("FAIL", "NA"):
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g2(m3p, ratio, exempt=False):
    """PASS: CI lower > 0.50 and ratio >= 0.5; FAIL: CI upper < 0.55 or ratio < 0.25; NA without H1p data."""
    if exempt:
        return "EXEMPT"
    if m3p is None or _v(m3p) is None:
        return "NA"
    r = _v(ratio)
    if _lo(m3p) is not None and _lo(m3p) > 0.5 and r is not None and r >= G2_RATIO_PASS:
        return "PASS"
    if (_hi(m3p) is not None and _hi(m3p) < G2_FAIL_UPPER) or (r is not None and r < G2_RATIO_FAIL):
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g3(m4n):
    """PASS: gain >= +0.02 with CI lower > 0; FAIL: CI upper < +0.02."""
    if m4n is None or _v(m4n) is None:
        return "NA"
    if _v(m4n) >= G3_POINT and _lo(m4n) is not None and _lo(m4n) > 0:
        return "PASS"
    if _hi(m4n) is not None and _hi(m4n) < G3_POINT:
        return "FAIL"
    return "INCONCLUSIVE"


def classify(g1, g2, g3, short_horizon):
    not_fail = lambda g: g != "FAIL"  # noqa: E731
    if g1 == "PASS":
        if g3 == "PASS" and not_fail(g2):
            return "ANTICIPATION"
        if not_fail(g2) and not_fail(g3):
            return "PROVISIONAL"
        return "CONTEXT/SHORTCUT"
    if short_horizon:
        return "SHORT-HORIZON"
    return "INCONCLUSIVE" if g1 == "INCONCLUSIVE" else "DROP"


def tags(m1_long, m5_ramp):
    out = []
    if _v(m1_long) is not None and _v(m1_long) >= LONG_POINT and _lo(m1_long) is not None and _lo(m1_long) > 0.5:
        out.append("LONG-LEAD")
    if m5_ramp and m5_ramp.get("wilcoxon_p") is not None and m5_ramp["wilcoxon_p"] < 0.05 \
            and (m5_ramp.get("share_rising") or 0) > 0.5:
        out.append("RAMP")
    return out


def decide_block(m, holm_p1, g2_exempt=False):
    """``m``: the block's stat dicts (keys m1, m1_near, m1_long, m1_sparse_pauc, tail, m3p, m3p_ratio, m4n, m5_ramp)."""
    main = gate_g1_main(m.get("m1"), holm_p1)
    sparse = gate_g1_sparse(m.get("m1_sparse_pauc"), m.get("tail"))
    g1 = gate_g1(main, sparse)
    g2 = gate_g2(m.get("m3p"), m.get("m3p_ratio"), g2_exempt)
    g3 = gate_g3(m.get("m4n"))
    short = g1 != "PASS" and _lo(m.get("m1_near")) is not None and _lo(m.get("m1_near")) > NEAR_LOWER
    return {"G1": g1, "G1_main": main, "G1_sparse": sparse, "G2": g2, "G3": g3, "short_horizon": bool(short),
            "class": classify(g1, g2, g3, short), "tags": tags(m.get("m1_long"), m.get("m5_ramp")),
            "holm_p_g1": holm_p1}


def decide_phase(blocks, exempt=()):
    """``blocks``: {name: stat dict}; Holm adjustment of the G1 p-values across the blocks of the phase."""
    adj = holm({k: _v(m.get("m1"), "p_le_null") for k, m in blocks.items()})
    return {k: decide_block(m, adj[k], k in exempt) for k, m in blocks.items()}


def soundtrack_flag(m1_xd_movie, m1_audio):
    """Audio blocks: flag a soundtrack shortcut when xd_movie M1 exceeds the S_audio M1 by more than 0.08."""
    if m1_xd_movie is None or m1_audio is None:
        return None
    return bool(m1_xd_movie - m1_audio > SOUNDTRACK_MARGIN)
