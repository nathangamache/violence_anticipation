"""Frozen gates, classes and tags (benchmark/screens/screen_set_v1.json) applied to the statistics of metrics.json."""

import re

from anticipation.screen import metrics as MT
from anticipation.screen.metrics import holm

G1_POINT = 0.58
G1_SPARSE_POINT = 0.55
G2_FAIL_UPPER = 0.55
G2_RATIO_PASS = 0.5
G2_RATIO_FAIL = 0.25
G3_POINT = 0.02
NEAR_LOWER = 0.60
LONG_POINT = 0.58
ALPHA = 0.05
SOUNDTRACK_MARGIN = 0.08
EPS = 1e-9
M4B_BROADCAST_LOWER = 0.55


def _v(d, k="value"):
    return None if d is None else d.get(k)


def _lo(d):
    return None if not d or not d.get("ci") else d["ci"][0]


def _hi(d):
    return None if not d or not d.get("ci") else d["ci"][1]


def _ci_ok(d):
    return bool(d) and bool(d.get("ci"))


def gate_g1_main(m1, holm_p):
    """PASS: point >= 0.58 and the Holm-adjusted one-sided p-value < 0.05 (adjusted CI lower > 0.50); FAIL: CI upper < 0.58."""
    if _v(m1) is None:
        return "INCONCLUSIVE"
    if _v(m1) >= G1_POINT - EPS and holm_p is not None and holm_p < ALPHA and _ci_ok(m1):
        return "PASS"
    if _hi(m1) is not None and _hi(m1) < G1_POINT - EPS:
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g1_sparse(pauc, tail, holm_pauc=None, holm_tail=None):
    """PASS: standardized pAUC >= 0.55 with Holm-adjusted one-sided p < 0.05 (null 0.5), or a positive tail-hit
    difference with adjusted one-sided p < 0.05 (null 0); a CI is required for either; otherwise NA / not PASS."""
    if _v(pauc) is None and _v(tail) is None:
        return "NA"
    if (_v(pauc) is not None and _v(pauc) >= G1_SPARSE_POINT - EPS and _ci_ok(pauc)
            and holm_pauc is not None and holm_pauc < ALPHA) or \
            (_v(tail) is not None and _v(tail) > 0 and _ci_ok(tail) and holm_tail is not None and holm_tail < ALPHA):
        return "PASS"
    return "INCONCLUSIVE"


def gate_g1(main, sparse):
    """PASS via either route; FAIL when the main route fails (CI upper < 0.58) and the sparse route did not pass."""
    if main == "PASS" or sparse == "PASS":
        return "PASS"
    if main == "FAIL":
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g2(m3p, ratio, exempt=False):
    """PASS: CI lower > 0.50 and a defined ratio >= 0.5; FAIL: CI upper < 0.55 (also when the ratio is undefined) or
    ratio < 0.25; an undefined ratio (UCF band AUROC at or below 0.5) otherwise leaves INCONCLUSIVE; NA without H1p data."""
    if exempt:
        return "EXEMPT"
    if m3p is None or _v(m3p) is None:
        return "NA"
    r = _v(ratio)
    if r is not None and _lo(m3p) is not None and _lo(m3p) > 0.5 and r >= G2_RATIO_PASS - EPS:
        return "PASS"
    if (_hi(m3p) is not None and _hi(m3p) < G2_FAIL_UPPER - EPS) or (r is not None and r < G2_RATIO_FAIL - EPS):
        return "FAIL"
    return "INCONCLUSIVE"


def gate_g3(m4n):
    """PASS: gain >= +0.02 with CI lower > 0; FAIL: CI upper < +0.02."""
    if m4n is None or _v(m4n) is None:
        return "NA"
    if _v(m4n) >= G3_POINT - EPS and _lo(m4n) is not None and _lo(m4n) > 0:
        return "PASS"
    if _hi(m4n) is not None and _hi(m4n) < G3_POINT - EPS:
        return "FAIL"
    return "INCONCLUSIVE"


def classify(g1, g2, g3, short_horizon):
    """Frozen class table. An exempt G2 (unused) caps the class at PROVISIONAL."""
    not_fail = lambda g: g != "FAIL"  # noqa: E731
    if g1 == "PASS":
        if g3 == "PASS" and not_fail(g2) and g2 != "EXEMPT":
            return "ANTICIPATION"
        if not_fail(g2) and not_fail(g3):
            return "PROVISIONAL"
        return "CONTEXT_SHORTCUT"
    if short_horizon:
        return "SHORT_HORIZON"
    return "INCONCLUSIVE" if g1 == "INCONCLUSIVE" else "DROP"


def tags(m1_long, m5_ramp):
    out = []
    if _v(m1_long) is not None and _v(m1_long) >= LONG_POINT and _lo(m1_long) is not None and _lo(m1_long) > 0.5:
        out.append("LONG_LEAD")
    if m5_ramp and m5_ramp.get("wilcoxon_p") is not None and m5_ramp["wilcoxon_p"] < 0.05 \
            and (m5_ramp.get("share_rising") or 0) > 0.5:
        out.append("RAMP")
    return out


def first_frame_warning(m):
    """Non-gating descriptive fields: the first-frame broadcast AUROC (M4b) on S_ramp and M1 minus M4b on the same
    ticks; flagged when the difference CI includes 0 or the broadcast alone separates the classes."""
    b, g = m.get("m4b_first_frame"), m.get("m1_ramp_gain_over_m4b")
    if _v(b) is None:
        return {"m4b": None, "m1_ramp_gain_over_m4b": None, "first_frame_flag": None}
    flag = (not _ci_ok(g)) or _lo(g) <= 0 or (_lo(b) is not None and _lo(b) > M4B_BROADCAST_LOWER)
    return {"m4b": _v(b), "m1_ramp_gain_over_m4b": {"value": _v(g), "ci": (g or {}).get("ci")},
            "first_frame_flag": bool(flag)}


def decide_block(m, holm_p1, g2_exempt=False, holm_pauc=None, holm_tail=None):
    """``m``: the block's stat dicts (keys m1, m1_near, m1_long, m1_sparse_pauc, tail, m3p, m3p_ratio, m4n, m5_ramp)."""
    main = gate_g1_main(m.get("m1"), holm_p1)
    sparse = gate_g1_sparse(m.get("m1_sparse_pauc"), m.get("tail"), holm_pauc, holm_tail)
    g1 = gate_g1(main, sparse)
    g2 = gate_g2(m.get("m3p"), m.get("m3p_ratio"), g2_exempt)
    g3 = gate_g3(m.get("m4n"))
    short = g1 != "PASS" and _lo(m.get("m1_near")) is not None and _lo(m.get("m1_near")) > NEAR_LOWER
    return {"G1": g1, "G1_main": main, "G1_sparse": sparse, "G2": g2, "G3": g3, "short_horizon": bool(short),
            "class": classify(g1, g2, g3, short), "tags": tags(m.get("m1_long"), m.get("m5_ramp")),
            "unevaluated_gates": [n for n, g in (("G2", g2), ("G3", g3)) if g == "NA"],
            "holm_p_g1": holm_p1, "holm_p_sparse_pauc": holm_pauc, "holm_p_sparse_tail": holm_tail,
            "soundtrack_flag": m.get("soundtrack_flag"), **first_frame_warning(m)}


def decide_phase(blocks, exempt=()):
    """``blocks``: {name: stat dict}. One Holm family over every G1 test of the phase: the M1 test, the pAUC test
    and the first-segment tail-hit test of each block (one-sided bootstrap p-values)."""
    ps = {}
    for k, m in blocks.items():
        ps[(k, "main")] = _v(m.get("m1"), "p_le_null")
        ps[(k, "pauc")] = _v(m.get("m1_sparse_pauc"), "p_le_null")
        ps[(k, "tail")] = _v(m.get("tail"), "p_le_null")
    adj = holm(ps)
    return {k: decide_block(m, adj[(k, "main")], k in exempt, adj[(k, "pauc")], adj[(k, "tail")])
            for k, m in blocks.items()}


def soundtrack_flag(m1_xd_movie, m1_audio):
    """Audio blocks: flag a soundtrack shortcut when xd_movie M1 exceeds the S_audio M1 by more than 0.08."""
    if m1_xd_movie is None or m1_audio is None:
        return None
    return bool(m1_xd_movie - m1_audio > SOUNDTRACK_MARGIN + EPS)


def _num(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def check_constants(pr):
    """Compare the numeric thresholds in this module with the numbers in the frozen gates and metrics; returns the
    list of mismatches (the run refuses to start when it is not empty)."""
    g, m, c, t = pr.data["gates"], pr.data["metrics"], pr.data["classes"], pr.data["tags"]
    ex = pr.data["exceptions"]
    want = [
        ("G1_POINT", G1_POINT, _num(g["G1"]["PASS"], r"point>=([\d.]+)")),
        ("G1 fail upper", G1_POINT, _num(g["G1"]["FAIL"], r"upper<([\d.]+)")),
        ("G1_SPARSE_POINT", G1_SPARSE_POINT, _num(g["G1"]["sparse_PASS"], r"pAUC>=([\d.]+)")),
        ("G2_RATIO_PASS", G2_RATIO_PASS, _num(g["G2"]["PASS"], r"ratio>=([\d.]+)")),
        ("G2_FAIL_UPPER", G2_FAIL_UPPER, _num(g["G2"]["FAIL"], r"upper<([\d.]+)")),
        ("G2_RATIO_FAIL", G2_RATIO_FAIL, _num(g["G2"]["FAIL"], r"ratio<([\d.]+)")),
        ("G3_POINT", G3_POINT, _num(g["G3"]["PASS"], r">=([\d.]+)")),
        ("G3 fail upper", G3_POINT, _num(g["G3"]["FAIL"], r"upper<([\d.]+)")),
        ("ALPHA", ALPHA, _num(g["G1"]["multiplicity"], r"alpha([\d.]+)")),
        ("NEAR_LOWER", NEAR_LOWER, _num(c["SHORT_HORIZON"], r"lower>([\d.]+)")),
        ("LONG_POINT", LONG_POINT, _num(t["LONG_LEAD"], r"point>=([\d.]+)")),
        ("SOUNDTRACK_MARGIN", SOUNDTRACK_MARGIN, _num(ex["soundtrack_flag"], r">([\d.]+)")),
        ("TAIL_PCT", MT.TAIL_PCT / 100.0, m["M1_sparse"]["tail_train_quantile"]),
        ("TAIL_MIN_TICKS", MT.TAIL_MIN_TICKS, m["M1_sparse"]["min_ticks"]),
        ("sparse FPR", MT.SPARSE_FPR, m["M1_sparse"]["clean_FPR"]),
        ("band", (30.0, 120.0), tuple(float(x) for x in m["M1_band_sec"])),
    ]
    return [f"{n}: code {a} frozen {b}" for n, a, b in want if b is None or a != b]
