"""Loader and verifier of the frozen pre-registration benchmark/screens/screen_set_v1.json (never written here)."""

import hashlib
import json
import os

import numpy as np

from anticipation.data.catalog import REPO_ROOT, file_sha256

PREREG_PATH = os.path.join(REPO_ROOT, "benchmark", "screens", "screen_set_v1.json")
PINNED_SHA256 = {"screen_set_v1.json": "8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b"}
REQUIRED_KEYS = ("version", "status", "screen_folds", "confirmation_fold", "sets", "scored_intervals_sec", "audio_gate",
                 "label_policy", "blocks", "controls", "probes", "nuisance", "metrics", "gates", "classes", "tags",
                 "forward_selection", "fold_membership", "provenance")
REQUIRED_SETS = ("positive_videos", "eligible30_events", "clean_cpu_videos", "clean_vlm_videos", "h1_videos", "h1p_videos",
                 "first_events")


class PreregError(RuntimeError):
    """The pre-registration is missing, altered or inconsistent; the screen must not run."""


class Prereg:
    def __init__(self, data, path, sha256):
        self.data, self.path, self.sha256 = data, path, sha256
        self.screen_folds = tuple(int(f) for f in data["screen_folds"])
        self.confirmation_fold = int(data["confirmation_fold"])
        self._members = {}
        self._sets = {k: v for k, v in data["sets"].items()}

    def ids(self, name):
        return list(self._sets[name])

    def events(self, name):
        return list(self._sets[name])

    def interval(self, vid):
        lo, hi = self.data["scored_intervals_sec"][vid]
        return float(lo), float(hi)

    def membership(self, fold, split):
        if int(fold) == self.confirmation_fold:
            raise PreregError(f"fold {fold} is the confirmation fold and has no screening membership")
        return self.data["fold_membership"][str(int(fold))][split]

    @property
    def audio_blocked(self):
        return str(self.data["audio_gate"]).startswith("Phase B blocked")

    def block_phases(self):
        return {k: list(v) for k, v in self.data["blocks"].items()}

    def all_listed_videos(self):
        out = set(self._sets["positive_videos"]) | set(self._sets["clean_cpu_videos"]) | set(self._sets["h1_videos"])
        return out


def load(path=PREREG_PATH, pinned=None):
    """Read the frozen file; raise PreregError unless its sha256 equals the one in ``<path>.sha256`` (and the pinned
    value for the registered file name)."""
    pinned = PINNED_SHA256 if pinned is None else pinned
    if not os.path.isfile(path):
        raise PreregError(f"no pre-registration at {path}")
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    sidecar = next((c for c in (os.path.splitext(path)[0] + ".sha256", path + ".sha256") if os.path.isfile(c)), None)
    if sidecar is None:
        raise PreregError(f"missing checksum file next to {path}")
    declared = open(sidecar).read().split()
    if len(declared) > 1 and declared[1].lstrip("*") != os.path.basename(path):
        raise PreregError(f"{sidecar} names {declared[1]}, not {os.path.basename(path)}")
    if not declared or declared[0] != sha:
        raise PreregError(f"sha256 mismatch for {path}: file {sha}, declared {declared[0] if declared else None}")
    pin = pinned.get(os.path.basename(path))
    if pin is not None and pin != sha:
        raise PreregError(f"sha256 of {path} is {sha}, the registered value is {pin}")
    data = json.loads(raw)
    missing = [k for k in REQUIRED_KEYS if k not in data] + [f"sets.{k}" for k in REQUIRED_SETS if k not in data.get("sets", {})]
    if missing:
        raise PreregError(f"pre-registration lacks {missing}")
    if data["confirmation_fold"] in data["screen_folds"]:
        raise PreregError("the confirmation fold is listed among the screening folds")
    if str(data["confirmation_fold"]) in data["fold_membership"]:
        raise PreregError("fold_membership contains the confirmation fold")
    return Prereg(data, path, sha)


def verify_inputs(pr, bench, plan_path=os.path.join(REPO_ROOT, "docs", "MODALITY_PLAN.md")):
    """Hash checks of the inputs the registration pins; returns ``(fatal, notes)`` lists of strings."""
    prov, got = pr.data["provenance"], bench.provenance
    fatal, notes = [], []
    for key in ("folds_content_sha256", "videos_csv_sha256", "events_json_sha256"):
        if prov.get(key) != got.get(key):
            fatal.append(f"{key}: registered {prov.get(key)} live {got.get(key)}")
    if os.path.isfile(plan_path) and prov.get("plan_sha256") != file_sha256(plan_path):
        notes.append("plan_sha256 differs from the live docs/MODALITY_PLAN.md (the plan was edited after registration)")
    for f in pr.screen_folds:
        for split in ("train", "val", "test"):
            wrong = [v for v in pr.membership(f, split) if v in bench.folds.assignment and bench.folds.assignment[v] == pr.confirmation_fold]
            if wrong:
                fatal.append(f"fold {f} {split} contains {len(wrong)} videos of the confirmation fold")
    return fatal, notes


def _diff(name, frozen, computed, limit=8):
    fz, cp = set(frozen), set(computed)
    return {"name": name, "frozen": len(fz), "computed": len(cp), "agree": fz == cp,
            "only_frozen": sorted(fz - cp)[:limit], "n_only_frozen": len(fz - cp),
            "only_computed": sorted(cp - fz)[:limit], "n_only_computed": len(cp - fz)}


def agreement(pr, ss, ref, bench):
    """Compare the frozen sets with what the previous in-code rules compute (``ref``: a ScreenSet without prereg).

    Returns ``{"disagreements": [...], "checks": [...]}``; every check has the frozen and computed sizes and the first
    differing members."""
    checks = []
    ev_ref = ref.events()
    lists = ref.video_lists()
    pos_ref = lists["positive_videos"]
    checks.append(_diff("positive_videos", pr.ids("positive_videos"), pos_ref))
    key = lambda df, col=None: {(r.video_id, int(r.index)) for r in df.itertuples()}  # noqa: E731
    fz_ev = {(e["video_id"], int(e["index"])) for e in pr.events("eligible30_events")}
    checks.append(_diff("eligible30_events", fz_ev, key(ev_ref)))
    for name, col in (("first_events", "first"), ("real_first_events", "s_real"), ("long60_first_events", "s_long60"),
                      ("ramp90_first_events", "s_ramp"), ("official_ubi_first_events", "s_offubi"),
                      ("audio_first_event_candidates", "s_audio")):
        fz = {(e["video_id"], int(e["index"])) for e in pr.events(name)}
        checks.append(_diff(name, fz, key(ev_ref[ev_ref[col]]) if len(ev_ref) else set()))
    checks.append(_diff("clean_cpu_videos", pr.ids("clean_cpu_videos"), lists["clean_negative_videos"]))
    checks.append(_diff("h1_videos", pr.ids("h1_videos"), lists["h1_videos"]))
    checks.append(_diff("h1p_videos", pr.ids("h1p_videos"), lists["h1p_videos"]))
    clean_fz = set(pr.ids("clean_cpu_videos"))
    vlm = pr.ids("clean_vlm_videos")
    checks.append({"name": "clean_vlm_videos_subset_of_clean_cpu", "frozen": len(vlm), "computed": None,
                   "agree": set(vlm) <= clean_fz, "note": "no previous in-code rule; checked as a subset"})
    for f in pr.screen_folds:
        for split, old in (("test", lambda: ref.test_ids(f, kinds=("violent", "clean", "h1"))),
                           ("train", lambda: ref.train_ids(f)), ("val", lambda: ref.val_ids(f))):
            fz = [v for v in pr.membership(f, split) if ss._kind_of.get(v) is not None]
            if split != "test":
                fz = [v for v in fz if ss._kind_of[v] != "h1"]
            checks.append(_diff(f"fold{f}_{split}", fz, old()))
    tick_bad = []
    for v in sorted(pr.all_listed_videos()):
        if bench.folds.assignment.get(v) == pr.confirmation_fold:
            continue
        a = ss.n_ticks_frozen(v)
        try:
            b = ref.ticks_computed(v)
        except (KeyError, ValueError):
            b = None
        if b is None or a != b:
            tick_bad.append((v, a, b))
    checks.append({"name": "scored_ticks_per_video", "frozen": len(pr.all_listed_videos()), "computed": None,
                   "agree": not tick_bad, "n_differ": len(tick_bad), "examples": tick_bad[:8],
                   "max_abs_diff": int(max([abs(a - (b or 0)) for _, a, b in tick_bad], default=0))})
    return {"disagreements": [c for c in checks if not c["agree"]], "checks": checks}
