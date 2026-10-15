"""Demonstrate the catalog-duration denominator for a capped clean-negative score file."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anticipation.data.folds import Folds, content_hash
from anticipation.eval.va30 import AlarmPolicy, Benchmark, EventEval, ScoreTable, TickSet

rows = pd.DataFrame([{"video_id": "synthetic/negative", "duration_sec": 3600, "role": "clean_negative",
                      "source_kind": "ucf", "interpersonal": False, "has_audio": False, "group": "n"}]).set_index("video_id")
assignment, val = {"synthetic/negative": 0}, {str(f): [] for f in range(5)}
folds = Folds({"name": "synthetic_audit", "n_folds": 5, "assignment": assignment, "val": val,
               "content_sha256": content_hash(assignment, val)})
bench = Benchmark(rows, {}, folds)
scores = ScoreTable(pd.DataFrame({"video_id": "synthetic/negative", "t_sec": np.arange(1, 301), "score": 0.0}))
aligned, held, leading = scores.aligned(0, 0, "synthetic/negative", bench.labels("synthetic/negative").t_sec)
ts = TickSet(bench, ["synthetic/negative"], {"synthetic/negative": aligned})
exposure = EventEval(ts, AlarmPolicy()).clean_hours
result = {"purpose": "synthetic evaluator exposure audit; not a model result",
          "evaluator_sha256": hashlib.sha256(Path("anticipation/eval/va30.py").read_bytes()).hexdigest(),
          "catalog_duration_sec": 3600, "score_ticks": 300, "held_ticks": held, "leading_fill_ticks": leading,
          "expected_scored_hours": 300 / 3600, "actual_evaluator_clean_hours": exposure,
          "inflation_factor": exposure / (300 / 3600)}
Path(__file__).with_name("audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
