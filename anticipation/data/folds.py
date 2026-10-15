"""5-fold grouped cross-validation for VA30 (BASELINE.md section 6.3), frozen fold files in benchmark/.

Current file: benchmark/folds_v2.json (FOLDS_DEFAULT). benchmark/folds_v1.json is kept frozen for reproducing the
first baseline report only; it predates UBI-Fights, the near-duplicate grouping and the per-fold validation bins.

Usage (from the repo root):

    .venv/bin/python -m anticipation.data.folds --out benchmark/folds_v2.json --salt va30-v2 --group_key split_group
        # build a fold file from scratch (refuses to overwrite an existing file)
    .venv/bin/python -m anticipation.data.folds --base benchmark/folds_v2.json --out benchmark/folds_v3.json
        # EXTEND an existing fold file with videos added to the catalog later: every video of the base keeps its
        # fold and validation membership, new videos whose group already exists inherit that group's fold, new
        # groups are assigned to keep the strata balanced.
    .venv/bin/python -m anticipation.data.folds --check benchmark/folds_v2.json   # verify hash + invariants

Rules:
  * Groups never cross folds. v2 groups by the catalog's ``split_group``: catalog ``group`` (XD: source film /
    YouTube id before ``__#``; UCF and UBI: video id) merged across XD franchises (``annotations.XD_FRANCHISES``)
    and near-duplicate pairs (``near_dups``). v1 grouped by ``group`` only. The key is stored as ``group_key``.
  * Stratified by source (xd_movie, xd_youtube, ucf, ubi) and by class: has an eligible30 event / other violent
    / clean negative / other (other_anomaly, unlabelled, duplicate, contaminated). Greedy balanced assignment,
    ties broken by a salted sha256 of the group name, so the result is deterministic.
  * Every fold f also defines a group-disjoint validation slice ``val[f]`` inside its training folds (about 1/8
    of the training groups, stratified the same way) for checkpoint and threshold selection.
    train(f) = all fold videos - test(f) - val(f). v2: inside each stratum the groups are ordered by a per-fold
    salted hash (not by size) and the validation bin is ``sha256(salt, f) % 8``, so the five validation slices
    are representative and nearly independent (v1 took bin 0 of a size-ordered assignment every time: the same
    large, long groups in every fold's validation slice).
  * Hold-out videos (curated YouTube) are never placed in a fold.
  * The file stores a ``content_sha256`` over the assignment; ``Folds.load`` recomputes it and refuses a
    file that was edited. The CLI refuses to overwrite an existing file unless ``--force``.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict

from anticipation.data.catalog import BENCHMARK_DIR, EVENTS_JSON, VIDEOS_CSV, file_sha256, load_catalog

FOLDS_V1 = os.path.join(BENCHMARK_DIR, "folds_v1.json")
FOLDS_V2 = os.path.join(BENCHMARK_DIR, "folds_v2.json")
FOLDS_DEFAULT = FOLDS_V2 if os.path.exists(FOLDS_V2) else FOLDS_V1
N_FOLDS = 5
VAL_BINS = 8
SALT = "va30-v2"
GROUP_KEY = "split_group"
CLASS_PRIORITY = ("elig30", "violent", "other", "clean_negative")


def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def video_class(row):
    if row["role"] == "violent":
        return "elig30" if bool(row["eligible30"]) else "violent"
    if row["role"] == "clean_negative":
        return "clean_negative"
    return "other"


def video_stratum(row):
    return f"{row['source_kind']}:{video_class(row)}"


def group_table(df, group_key="group"):
    """``{group: {"videos": [...], "stratum": str, "weight": int}}``; a group's stratum is that of its most
    important video (elig30 > violent > other > clean_negative)."""
    groups = defaultdict(list)
    for vid, row in df.iterrows():
        groups[row[group_key]].append(vid)
    out = {}
    for g, vids in groups.items():
        rows = [df.loc[v] for v in vids]
        best = min(rows, key=lambda r: (CLASS_PRIORITY.index(video_class(r)), r["video_id"]))
        out[g] = {"videos": sorted(vids), "stratum": video_stratum(best), "weight": len(vids),
                  "hours": float(sum(float(r["duration_sec"]) for r in rows)) / 3600.0}
    return out


def balanced_assign(groups, n_bins, salt, init_stratum=None, init_total=None, order="size"):
    """Greedy stratified assignment of groups to ``n_bins`` bins.

    Strata are processed in sorted order; inside a stratum, groups by decreasing size, then decreasing hours
    of footage, then salted hash (``order="size"``, best balance; used for the test folds), or by salted hash
    alone (``order="hash"``, unbiased; used for the validation bins). Each group goes to the bin with the fewest
    videos of its stratum, then the fewest hours of its stratum (so long clean-negative CCTV videos spread
    evenly), then the fewest videos overall, then the lowest index.
    """
    cs = defaultdict(lambda: [0] * n_bins)
    hs = defaultdict(lambda: [0.0] * n_bins)
    if init_stratum:
        for s, v in init_stratum.items():
            cs[s] = list(v)
    tot = list(init_total) if init_total else [0] * n_bins
    by_s = defaultdict(list)
    for g, info in groups.items():
        by_s[info["stratum"]].append(g)
    out = {}
    def key(g):
        if order == "hash":
            return (_h(salt, g),)
        return (-groups[g]["weight"], -round(groups[g].get("hours", 0.0), 6), _h(salt, g))

    for s in sorted(by_s):
        for g in sorted(by_s[s], key=key):
            b = min(range(n_bins), key=lambda k: (cs[s][k], round(hs[s][k], 6), tot[k], k))
            out[g] = b
            cs[s][b] += groups[g]["weight"]
            hs[s][b] += groups[g].get("hours", 0.0)
            tot[b] += groups[g]["weight"]
    return out


def content_hash(assignment, val):
    payload = {"assignment": dict(sorted(assignment.items())),
               "val": {str(k): sorted(v) for k, v in sorted(val.items(), key=lambda kv: int(kv[0]))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fold_stats(df, assignment, val, n_folds):
    stats = {}
    for f in range(n_folds):
        for part, ids in (("test", [v for v, k in assignment.items() if k == f]), ("val", val[str(f)])):
            sub = df.loc[ids]
            viol = sub[sub.role == "violent"]
            stats[f"{f}_{part}"] = {
                "videos": int(len(sub)), "groups": int(sub.group.nunique()),
                "hours": round(float(sub.duration_sec.sum()) / 3600, 2),
                "violent": int(len(viol)), "eligible30_videos": int(viol.eligible30.sum()),
                "eligible30_events": int(viol.n_eligible30.sum()), "eligible60_videos": int(viol.eligible60.sum()),
                "clean_negative": int((sub.role == "clean_negative").sum()),
                "clean_negative_hours": round(float(sub[sub.role == "clean_negative"].duration_sec.sum()) / 3600, 2),
                "other": int((~sub.role.isin(["violent", "clean_negative"])).sum()),
                "mean_clean_negative_sec": round(float(sub[sub.role == "clean_negative"].duration_sec.mean()), 1)
                if (sub.role == "clean_negative").any() else None,
                "by_source_kind": {k: int(x) for k, x in sub.source_kind.value_counts().items()},
            }
    return stats


def val_bin(salt, f, val_bins=VAL_BINS):
    """Validation bin of fold ``f`` (v2 rule): a salted hash, so folds do not all take bin 0."""
    return int(_h(salt, "val-bin", f), 16) % val_bins


def make_folds(df, n_folds=N_FOLDS, val_bins=VAL_BINS, salt=SALT, base=None, group_key=GROUP_KEY,
               val_rule="v2"):
    """Return ``(assignment {video: fold}, val {str(fold): [videos]})`` for the non-hold-out catalog rows.

    ``val_rule="v2"``: per-fold hash order and hashed bin (see module docstring); ``"v1"``: size order, bin 0.
    """
    df = df[df.role != "holdout"]
    if group_key not in df:
        group_key = "group"
    groups = group_table(df, group_key)
    assignment = {}
    val = {str(f): [] for f in range(n_folds)}
    if base is None:
        g2f = balanced_assign(groups, n_folds, salt)
        base_val_groups = {str(f): set() for f in range(n_folds)}
    else:
        if base["n_folds"] != n_folds:
            raise ValueError("base fold file has a different n_folds")
        missing = set(base["assignment"]) - set(df.index)
        if missing:
            raise ValueError(f"{len(missing)} videos of the base fold file are no longer in the catalog, "
                             f"e.g. {sorted(missing)[:3]}; refusing to extend")
        g2f = {}
        for vid, f in base["assignment"].items():
            g = df.loc[vid, group_key]
            if g2f.setdefault(g, f) != f:
                raise ValueError(f"group {g} spans folds in the base file")
        new = {g: info for g, info in groups.items() if g not in g2f}
        cs, tot = defaultdict(lambda: [0] * n_folds), [0] * n_folds
        for g, f in g2f.items():
            cs[groups[g]["stratum"]][f] += groups[g]["weight"]
            tot[f] += groups[g]["weight"]
        g2f.update(balanced_assign(new, n_folds, salt, init_stratum=cs, init_total=tot))
        base_val_groups = {f: {df.loc[v, group_key] for v in vs} for f, vs in base["val"].items()}
    base_groups = {df.loc[v, group_key] for v in base["assignment"]} if base is not None else set()
    order = "hash" if val_rule == "v2" else "size"
    for g, info in groups.items():
        for v in info["videos"]:
            assignment[v] = int(g2f[g])
    for f in range(n_folds):
        pool = {g: info for g, info in groups.items() if g2f[g] != f}
        vb = val_bin(salt, f, val_bins) if val_rule == "v2" else 0
        if base is None:
            bins = balanced_assign(pool, val_bins, f"{salt}-val{f}", order=order)
            chosen = {g for g, b in bins.items() if b == vb}
        else:
            keep = base_val_groups[str(f)]
            new = {g: info for g, info in pool.items() if g not in base_groups}
            bins = balanced_assign(new, val_bins, f"{salt}-val{f}", order=order)
            chosen = keep | {g for g, b in bins.items() if b == vb}
        val[str(f)] = sorted(v for g in chosen for v in groups[g]["videos"])
    return assignment, val


class Folds:
    """Read-only view of a fold file. ``Folds.load(path)`` verifies the content hash."""

    def __init__(self, d, path=None):
        self.d = d
        self.path = path
        self.n_folds = int(d["n_folds"])
        self.assignment = {k: int(v) for k, v in d["assignment"].items()}
        self.val = {int(k): list(v) for k, v in d["val"].items()}
        self._val_sets = {k: set(v) for k, v in self.val.items()}
        self.sha256 = d["content_sha256"]

    @classmethod
    def load(cls, path=None, verify=True):
        path = path or FOLDS_DEFAULT
        with open(path) as f:
            d = json.load(f)
        if verify:
            h = content_hash(d["assignment"], d["val"])
            if h != d["content_sha256"]:
                raise ValueError(f"{path}: content hash mismatch (file edited?) {h} != {d['content_sha256']}")
        return cls(d, path)

    @property
    def video_ids(self):
        return sorted(self.assignment)

    def fold_of(self, vid):
        return self.assignment.get(vid)

    def test_ids(self, f):
        return sorted(v for v, k in self.assignment.items() if k == f)

    def val_ids(self, f):
        return list(self.val[f])

    def train_ids(self, f):
        vs = self._val_sets[f]
        return sorted(v for v, k in self.assignment.items() if k != f and v not in vs)

    def role_in_fold(self, vid, f):
        """'test', 'val' or 'train' for the model of fold f; None for a video that is not in the fold file
        (hold-out, newer catalog rows, typos): such videos are never training data. Trainers should build their
        training set from ``train_ids(f)`` (and filter catalog roles), not by asking every catalog id."""
        if vid not in self.assignment:
            return None
        if self.assignment[vid] == f:
            return "test"
        return "val" if vid in self._val_sets[f] else "train"

    @property
    def group_key(self):
        return self.d.get("group_key", "group")


def check(path, catalog_path=VIDEOS_CSV):
    """Verify the hash and the invariants. Raises on failure; returns a short report.

    Invariants: every fold video is in the catalog and none is a hold-out; groups (the file's ``group_key``)
    never cross folds; for every fold the validation slice is a subset of the assignment, contains no video of
    its own test fold, and shares no group with the fold's training videos.
    """
    fo = Folds.load(path)
    df = load_catalog(catalog_path)
    unknown = [v for v in fo.assignment if v not in df.index]
    if unknown:
        raise ValueError(f"{len(unknown)} fold videos not in catalog, e.g. {unknown[:3]}")
    hold = [v for v in fo.assignment if df.loc[v, "role"] == "holdout"]
    if hold:
        raise ValueError(f"{len(hold)} hold-out videos in the fold assignment, e.g. {hold[:3]}")
    key = fo.group_key if fo.group_key in df else "group"
    g2f = {}
    for v, f in fo.assignment.items():
        g = df.loc[v, key]
        if g2f.setdefault(g, f) != f:
            raise ValueError(f"group {g} ({key}) crosses folds")
    for f in range(fo.n_folds):
        val = set(fo.val_ids(f))
        outside = val - set(fo.assignment)
        if outside:
            raise ValueError(f"fold {f}: {len(outside)} validation videos are not in the assignment, "
                             f"e.g. {sorted(outside)[:3]}")
        if val & set(fo.test_ids(f)):
            raise ValueError(f"fold {f}: val overlaps test")
        vg = {df.loc[v, key] for v in val}
        tg = {df.loc[v, key] for v in fo.train_ids(f)}
        if vg & tg:
            raise ValueError(f"fold {f}: val groups overlap train groups")
    n_val = [len(fo.val_ids(f)) for f in range(fo.n_folds)]
    overlap = max((len(set(fo.val_ids(a)) & set(fo.val_ids(b))) / max(1, min(n_val[a], n_val[b])))
                  for a in range(fo.n_folds) for b in range(a + 1, fo.n_folds)) if fo.n_folds > 1 else 0.0
    return {"path": path, "content_sha256": fo.sha256, "videos": len(fo.assignment), "groups": len(g2f),
            "group_key": key, "val_videos_per_fold": n_val, "max_pairwise_val_overlap": round(overlap, 3), "ok": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=VIDEOS_CSV)
    ap.add_argument("--events", default=EVENTS_JSON)
    ap.add_argument("--out", default=FOLDS_V2)
    ap.add_argument("--name", default=None, help="default: basename of --out")
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--salt", default=SALT)
    ap.add_argument("--group_key", default=GROUP_KEY, help="catalog column that defines split groups")
    ap.add_argument("--val_rule", default="v2", choices=["v1", "v2"])
    ap.add_argument("--base", default=None, help="extend this fold file instead of building from scratch")
    ap.add_argument("--exclude_folder_regex", default=None,
                    help="drop catalog rows whose path matches this regex (e.g. in-flight downloads)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--check", default=None, help="only verify an existing fold file")
    args = ap.parse_args(argv)

    if args.check:
        print(json.dumps(check(args.check, args.catalog), indent=1))
        return 0
    if os.path.exists(args.out) and not args.force:
        print(f"[folds] REFUSING to overwrite {args.out} (fold files are frozen; pass --force only if you mean it, "
              f"or --base {args.out} --out <new file> to extend)")
        return 2
    df = load_catalog(args.catalog)
    if args.exclude_folder_regex:
        df = df[~df.path.str.contains(args.exclude_folder_regex, regex=True)]
    base = None
    if args.base:
        with open(args.base) as f:
            base = json.load(f)
        if content_hash(base["assignment"], base["val"]) != base["content_sha256"]:
            raise SystemExit(f"base file {args.base} fails its content hash")
    if base is not None and base.get("group_key", "group") != args.group_key:
        raise SystemExit(f"base file groups by {base.get('group_key', 'group')}; pass --group_key accordingly")
    assignment, val = make_folds(df, n_folds=args.n_folds, salt=args.salt, base=base, group_key=args.group_key,
                                 val_rule=args.val_rule)
    gk = args.group_key if args.group_key in df else "group"
    sub = df.loc[sorted(assignment)]
    d = {
        "name": args.name or os.path.splitext(os.path.basename(args.out))[0],
        "version": 2, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_folds": args.n_folds, "val_fraction_of_train_groups": 1.0 / VAL_BINS, "salt": args.salt,
        "group_key": gk, "val_rule": args.val_rule,
        "base": {"path": args.base, "content_sha256": base["content_sha256"]} if base else None,
        "catalog": {"videos_csv_sha256": file_sha256(args.catalog),
                    "events_json_sha256": file_sha256(args.events) if os.path.exists(args.events) else None,
                    "n_catalog_rows": int(len(df)), "n_fold_videos": len(assignment),
                    "excluded_holdout": int((df.role == "holdout").sum()),
                    "exclude_folder_regex": args.exclude_folder_regex},
        "strata": "source_kind x {elig30, violent, other, clean_negative}; group stratum = most important video",
        "group_of": {v: str(sub.loc[v, gk]) for v in sorted(assignment)},
        "assignment": dict(sorted(assignment.items())),
        "val": val,
        "stats": fold_stats(sub, assignment, val, args.n_folds),
        "content_sha256": content_hash(assignment, val),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, args.out)
    print(f"[folds] wrote {args.out}: {len(assignment)} videos, sha256 {d['content_sha256']}")
    for k, s in d["stats"].items():
        print(f"  {k:8s} videos {s['videos']:5d} violent {s['violent']:4d} elig30 {s['eligible30_videos']:3d} "
              f"(events {s['eligible30_events']:3d}) clean_neg {s['clean_negative']:4d} ({s['clean_negative_hours']} h)")
    print(json.dumps(check(args.out, args.catalog)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
