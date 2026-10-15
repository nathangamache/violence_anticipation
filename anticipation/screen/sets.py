"""Screening set S (MODALITY_PLAN 6.3), tick labels for violent, clean and H1 videos, and the fold-1 freeze."""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anticipation.data.catalog import REPO_ROOT, file_sha256
from anticipation.eval import va30

SCREEN_FOLDS = (0, 2, 3, 4)
FROZEN_FOLD = 1
H1P_CLASSES = ("stealing", "shoplifting", "burglary", "vandalism", "arson")
H1_MAX_SEC = 300.0
POST_ONSET_SEC = 10.0
SOFT_MASKS_PATH = os.path.join(REPO_ROOT, "benchmark", "audits", "soft_masks_v1.json")
MANIFEST_V1 = os.path.join(REPO_ROOT, "benchmark", "extract_manifest_v1.csv")
MANIFEST_V2 = os.path.join(REPO_ROOT, "benchmark", "extract_manifest_v2.csv")


class FrozenFoldError(RuntimeError):
    """Raised on any attempt to read the confirmation fold during the screen."""


@dataclass
class VideoTicks:
    vid: str
    kind: str
    t: np.ndarray
    tte: np.ndarray
    pre: np.ndarray
    masked: np.ndarray
    in_violence: np.ndarray
    remaining: np.ndarray
    next_event: np.ndarray

    @property
    def n(self):
        return len(self.t)

    def cut(self, n):
        if n >= self.n:
            return self
        sl = slice(0, max(n, 0))
        return VideoTicks(self.vid, self.kind, self.t[sl], self.tte[sl], self.pre[sl], self.masked[sl],
                          self.in_violence[sl], self.remaining[sl], self.next_event[sl])


def load_soft_masks(path=SOFT_MASKS_PATH):
    """``(masks, info)``; masks is ``{video_id: [[start, end, reason], ...]}`` or empty when the file is absent."""
    if path and os.path.isfile(path):
        with open(path) as f:
            masks = json.load(f)
        return masks, {"applied": True, "path": os.path.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT) else path,
                       "sha256": file_sha256(path), "videos": len(masks)}
    return {}, {"applied": False, "path": None, "sha256": None, "videos": 0}


class ScreenSet:
    """Videos, ticks and events of the screening folds; fold ``frozen_fold`` is unreachable."""

    def __init__(self, bench, h1=None, screen_folds=SCREEN_FOLDS, frozen_fold=FROZEN_FOLD, soft_masks=None,
                 soft_mask_info=None, prereg=None):
        if prereg is not None:
            screen_folds, frozen_fold = tuple(prereg.screen_folds), prereg.confirmation_fold
            soft_masks, soft_mask_info = {}, {"applied": False, "path": None, "sha256": None, "videos": 0,
                                              "reason": "label_policy: proposed masks are not applied in v2"}
        if frozen_fold in screen_folds:
            raise ValueError("the frozen fold cannot be a screening fold")
        self.prereg = prereg
        self._kind_of = {}
        if prereg is not None:
            for name, kind in (("positive_videos", "violent"), ("clean_cpu_videos", "clean"), ("h1_videos", "h1")):
                for v in prereg.ids(name):
                    self._kind_of[v] = kind
        self.bench = bench
        self.screen_folds = tuple(screen_folds)
        self.frozen_fold = frozen_fold
        self.soft_masks = soft_masks or {}
        self.soft_mask_info = soft_mask_info or {"applied": bool(self.soft_masks), "path": None, "sha256": None,
                                                 "videos": len(self.soft_masks)}
        self.h1 = h1 if h1 is not None else pd.DataFrame(
            columns=["video_id", "source_kind", "classes", "first_anomaly_sec", "duration_sec", "split_group"])
        self.h1 = self.h1.set_index("video_id", drop=False)
        self._ticks, self._events = {}, None
        v = bench.videos
        self.split_group = (v["split_group"].where(v["split_group"].astype(str) != "", v["group"]) if "split_group" in v
                            else v["group"]).to_dict()
        for vid in self.h1.index:
            if vid not in self.split_group:
                self.split_group[vid] = self.h1.loc[vid, "split_group"] if "split_group" in self.h1 else vid

    def fold_of(self, vid):
        return self.bench.folds.fold_of(vid)

    def guard(self, vid):
        if self.fold_of(vid) == self.frozen_fold:
            raise FrozenFoldError(f"{vid} belongs to fold {self.frozen_fold}, which is frozen during the screen")

    def guard_fold(self, f):
        if f == self.frozen_fold:
            raise FrozenFoldError(f"fold {self.frozen_fold} is frozen during the screen")
        if f not in self.screen_folds:
            raise ValueError(f"fold {f} is not a screening fold {self.screen_folds}")

    def assert_no_frozen(self, df):
        """Raise if a score table has rows of the frozen fold (by model fold or by video)."""
        bad = pd.Series(False, index=df.index)
        if "fold" in df:
            bad |= df["fold"] == self.frozen_fold
        folds = df["video_id"].map(self.bench.folds.assignment)
        bad |= folds == self.frozen_fold
        if bad.any():
            raise FrozenFoldError(f"score table has {int(bad.sum())} rows of frozen fold {self.frozen_fold}")

    def _in_screen(self, ids):
        return [v for v in ids if self.fold_of(v) in self.screen_folds]

    def eval_ids(self):
        if self.prereg is not None:
            return set(self._kind_of)
        return set(self._in_screen(self.bench.eval_ids()))

    def test_ids(self, f, kinds=("violent", "clean")):
        self.guard_fold(f)
        if self.prereg is not None:
            return sorted(v for v in self.prereg.membership(f, "test") if self._kind_of.get(v) in kinds)
        ids = self.eval_ids()
        out = [v for v in self.bench.folds.test_ids(f) if v in ids]
        out = [v for v in out if self.kind(v) in kinds]
        if "h1" in kinds:
            out += [v for v in self.bench.folds.test_ids(f) if v in self.h1.index]
        return sorted(out)

    def train_ids(self, f):
        self.guard_fold(f)
        if self.prereg is not None:
            return [v for v in self.prereg.membership(f, "train") if self._kind_of.get(v) in ("violent", "clean")]
        ids = self.eval_ids()
        return [v for v in self.bench.folds.train_ids(f) if v in ids]

    def val_ids(self, f):
        self.guard_fold(f)
        if self.prereg is not None:
            return [v for v in self.prereg.membership(f, "val") if self._kind_of.get(v) in ("violent", "clean")]
        ids = self.eval_ids()
        return [v for v in self.bench.folds.val_ids(f) if v in ids]

    def kind(self, vid):
        if vid in self._kind_of:
            return self._kind_of[vid]
        if vid in self.h1.index and vid not in self.bench.videos.index:
            return "h1"
        role = self.bench.videos.loc[vid, "role"]
        if role == "violent":
            return "violent"
        if role == "clean_negative":
            return "clean"
        return "h1" if vid in self.h1.index else "other"

    def source_kind(self, vid):
        if vid in self.bench.videos.index:
            return str(self.bench.videos.loc[vid, "source_kind"])
        return str(self.h1.loc[vid, "source_kind"])

    def violent_cut(self, vid):
        rows = self.bench.event_rows(vid)
        elig = [r["onset"] for r in rows if r["eligible30"]]
        last = max(elig) if elig else (max(r["onset"] for r in rows) if rows else 0.0)
        return int(np.floor(last + POST_ONSET_SEC))

    def n_ticks_frozen(self, vid):
        """Number of scored ticks of a video from the frozen scored interval (None without a pre-registration)."""
        if self.prereg is None:
            return None
        lo, hi = self.prereg.interval(vid)
        return int(np.floor(hi + 1e-9)) - int(np.ceil(lo - 1e-9)) + 1

    def ticks_computed(self, vid):
        """Tick count the previous in-code rules give (violent: last eligible onset + 10 s; clean: label length;
        H1: min(first anomaly, 300 s, duration))."""
        kind = self.kind(vid)
        if kind == "h1":
            r = self.h1.loc[vid]
            return max(int(np.floor(min(float(r["first_anomaly_sec"]) - 1e-6, H1_MAX_SEC, float(r["duration_sec"])))), 0)
        lab = self.bench.labels(vid)
        return lab.T if kind == "clean" else min(lab.T, self.violent_cut(vid))

    def ticks(self, vid):
        self.guard(vid)
        if vid in self._ticks:
            return self._ticks[vid]
        kind = self.kind(vid)
        nf = self.n_ticks_frozen(vid)
        if kind == "h1":
            r = self.h1.loc[vid]
            fa = float(r["first_anomaly_sec"])
            n = self.ticks_computed(vid) if nf is None else min(nf, int(np.floor(float(r["duration_sec"]))))
            t = np.arange(1, max(n, 0) + 1, dtype=np.float64)
            vt = VideoTicks(vid, kind, t, fa - t, np.ones(len(t), bool), np.zeros(len(t), bool),
                            np.zeros(len(t), bool), float(r["duration_sec"]) - t, np.zeros(len(t), np.int64))
        else:
            lab = self.bench.labels(vid)
            n = self.ticks_computed(vid) if nf is None else min(lab.T, nf)
            sl = slice(0, n)
            masked = lab.masked[sl].copy()
            for s, e, *_ in self.soft_masks.get(vid, []):
                masked |= (lab.t_sec[sl] >= s) & (lab.t_sec[sl] <= e)
            vt = VideoTicks(vid, kind, lab.t_sec[sl], lab.tte[sl], lab.pre_violence[sl], masked,
                            lab.in_violence[sl], lab.remaining[sl], lab.next_event[sl])
        self._ticks[vid] = vt
        return vt

    def events(self):
        """Eligible30 events of the screening-fold test videos, one row each (frozen list under a pre-registration)."""
        if self._events is not None:
            return self._events
        if self.prereg is not None:
            self._events = self._frozen_events()
            return self._events
        rows = []
        for f in self.screen_folds:
            for vid in self.test_ids(f, kinds=("violent",)):
                r = self.bench.videos.loc[vid]
                for e in self.bench.event_rows(vid):
                    if not e["eligible30"]:
                        continue
                    rows.append(self._event_row(vid, f, e, str(r["source_kind"]), str(r["onset_source"]), bool(r["has_audio"]),
                                                e["index"] == 0))
        df = pd.DataFrame(rows)
        for name, cond in (("s_real", lambda r: r.first and r.source_kind != "xd_movie"),
                           ("s_long60", lambda r: r.first and r.eligible60),
                           ("s_ramp", lambda r: r.first and r.max_lead >= 90),
                           ("s_offubi", lambda r: r.first and r.onset_source in ("official", "ubi")),
                           ("s_audio", lambda r: r.first and r.source_kind in ("ucf", "xd_youtube") and r.has_audio)):
            df[name] = [bool(cond(r)) for r in df.itertuples()] if len(df) else []
        self._events = df
        return self._events

    def _event_row(self, vid, f, e, src, onset_source, has_audio, first):
        return {"video_id": vid, "fold": f, "index": e["index"], "onset": e["onset"], "clear_start": e["clear_start"],
                "leadin": e["leadin"], "max_lead": e["max_lead"], "eligible60": e["eligible60"], "first": first,
                "source_kind": src, "onset_source": onset_source, "has_audio": has_audio,
                "split_group": self.split_group.get(vid, vid),
                "win_lo": max(e["onset"] - 120.0, e["clear_start"]), "win_hi": e["onset"] - 30.0}

    def _frozen_events(self):
        pr = self.prereg
        keyed = lambda name: {(e["video_id"], e["index"]) for e in pr.events(name)}  # noqa: E731
        first = keyed("first_events")
        rows = []
        for e in pr.events("eligible30_events"):
            vid = e["video_id"]
            r = self.bench.videos.loc[vid]
            rows.append(self._event_row(vid, self.fold_of(vid), e, e["source_kind"], e["onset_source"], bool(r["has_audio"]),
                                        (vid, e["index"]) in first))
        df = pd.DataFrame(rows)
        for col, name in (("s_real", "real_first_events"), ("s_long60", "long60_first_events"),
                          ("s_ramp", "ramp90_first_events"), ("s_offubi", "official_ubi_first_events"),
                          ("s_audio", "audio_first_event_candidates")):
            keys = keyed(name)
            df[col] = [(v, i) in keys for v, i in zip(df["video_id"], df["index"])]
        return df[df["fold"].isin(self.screen_folds)].reset_index(drop=True)

    def h1_ids(self, folds=None):
        folds = self.screen_folds if folds is None else folds
        if self.prereg is not None:
            return sorted(v for v in self.prereg.ids("h1_videos") if self.fold_of(v) in folds)
        return sorted(v for v in self.h1.index if self.fold_of(v) in folds)

    def h1p_ids(self):
        if self.prereg is not None:
            return sorted(self.prereg.ids("h1p_videos"))
        out = []
        for v in self.h1_ids():
            if self.source_kind(v) == "ucf" and any(c in str(self.h1.loc[v, "classes"]).split("|") for c in H1P_CLASSES):
                out.append(v)
        return out

    def counts(self):
        ev = self.events()
        clean = [v for f in self.screen_folds for v in self.test_ids(f, kinds=("clean",))]
        pos = sorted(set(ev["video_id"]))
        first_ev = int(ev["first"].sum())
        pos_ticks = sum(self.ticks(v).n for v in pos)
        out = {"positive_videos": len(pos), "events": int(len(ev)), "first_segment_events": first_ev,
               "s_real": int(ev["s_real"].sum()), "s_long60": int(ev["s_long60"].sum()),
               "s_ramp": int(ev["s_ramp"].sum()), "s_offubi": int(ev["s_offubi"].sum()),
               "s_audio_before_diegetic_filter": int(ev["s_audio"].sum()),
               "positive_hours": round(pos_ticks / 3600, 3),
               "clean_negative_videos": len(clean),
               "clean_negative_hours": round(sum(self.ticks(v).n for v in clean) / 3600, 3),
               "h1_videos": len(self.h1_ids()), "h1p_videos": len(self.h1p_ids()),
               "events_per_fold": {str(f): int((ev["fold"] == f).sum()) for f in self.screen_folds}}
        return out

    def video_lists(self):
        ev = self.events()
        clean = sorted(v for f in self.screen_folds for v in self.test_ids(f, kinds=("clean",)))
        return {"positive_videos": sorted(set(ev["video_id"])), "clean_negative_videos": clean,
                "h1_videos": self.h1_ids(), "h1p_videos": self.h1p_ids()}


def list_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_screen_set(prereg=None, folds=None, horizons=MANIFEST_V1, h1_manifest=MANIFEST_V2, soft_masks_path=SOFT_MASKS_PATH):
    """With ``prereg`` the cohorts, intervals and fold memberships come from the frozen file; without it the
    previous in-code rules are used (kept for the agreement check)."""
    bench = va30.Benchmark.load(folds, horizons=horizons)
    h1 = None
    if h1_manifest and os.path.isfile(h1_manifest):
        m = pd.read_csv(h1_manifest, dtype={"video_id": str})
        m = m[m["role"] == "other_anomaly"].copy()
        cat = bench.videos
        keep = m["video_id"].isin(cat.index)
        m = m[keep]
        m["classes"] = cat.loc[m["video_id"], "classes"].values
        m["first_anomaly_sec"] = cat.loc[m["video_id"], "first_anomaly_sec"].values
        m["split_group"] = cat.loc[m["video_id"], "split_group"].values
        h1 = m[["video_id", "source_kind", "classes", "first_anomaly_sec", "duration_sec", "split_group"]]
        h1 = h1[h1["first_anomaly_sec"].notna()]
    if prereg is not None:
        return ScreenSet(bench, h1, prereg=prereg)
    masks, info = load_soft_masks(soft_masks_path)
    return ScreenSet(bench, h1, soft_masks=masks, soft_mask_info=info)
