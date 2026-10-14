# Violence Anticipation — Experiments

Six experiment scripts, one per task, in priority order.

---

## Files

| File | Purpose |
|---|---|
| `ema_smoothing.py` | EMA smoothing at 3 levels: scores, embeddings, LSTM hidden states |
| `algorithm_variations.py` | Transformer encoder, LR-as-primary, sliding window aggregation |
| `prompt_engineering_stats.py` | 15-prompt library + offline/online embedding separation analysis |
| `ucf_crime_testing.py` | Cross-domain evaluation on UCF-Crime with distribution shift analysis |
| `xd_violence_investigation.py` | Pre-violence window inventory, temporal bias, embedding distribution |
| `baseline_comparisons.py` | Unified table: all old and new models with consistent metrics |
| `run_all_experiments.sh` | One-shot runner for all of the above |

---

## Quick start

```bash
# Run everything (from your project root)
bash experiments/run_all_experiments.sh

# Or run individually
python experiments/baseline_comparisons.py   # fast, good starting point
python experiments/ema_smoothing.py          # ~5 min
python experiments/algorithm_variations.py   # ~15 min (Transformer training)
python experiments/xd_violence_investigation.py  # fast, no GPU needed
python experiments/ucf_crime_testing.py      # needs UCF embeddings extracted first
python experiments/prompt_engineering_stats.py --mode offline  # fast
python experiments/prompt_engineering_stats.py --mode list     # just prints prompts
```

---

## Output files (all in `results/`)

| File | Contents |
|---|---|
| `ema_results.csv` | AUC at each alpha for all 3 EMA levels |
| `algorithm_comparison.csv` | Transformer / LR+temporal / sliding window AUC + AP |
| `prompt_engineering.csv` | Prompt ranking by LDA AUC and Cohen's d |
| `ucf_crime_results.csv` | Cross-domain AUC + false alarm rates |
| `xd_investigation/investigation_stats.json` | Full data stats JSON |
| `xd_investigation/best_anticipation_videos.txt` | Ranked video list for anticipation training |
| `baseline_comparison.csv` | All models: frame AUC, frame AP, per-video AUC, FA/10min |

---

## Key metrics used

- **Frame AUC** — frame-level ROC-AUC (the primary metric)
- **Frame AP** — average precision (better for imbalanced classes)
- **Per-video AUC** — mean AUC computed independently per video (catches per-video failures)
- **FA/10min** — false alarm events per 10 minutes of normal footage (event-level, not frame-level)

The FA/10min metric addresses the known issue where frame-level metrics masked near-constant
over-triggering. A model predicting violence on every frame will have FA/10min >> 0
even if it has low per-event false alarm count.

---

## Running prompt engineering in online mode (needs GPU0)

```bash
# First, collect probe frames:
# - data/probe_frames/violent/    (20-50 frames from known violent events)
# - data/probe_frames/previolent/ (20-50 frames from 5-30s before violence)
# - data/probe_frames/neutral/    (20-50 frames from calm scenes)

python experiments/prompt_engineering_stats.py \
    --mode online \
    --vlm_model /path/to/Qwen2.5-VL \
    --frames_dir data/probe_frames \
    --device cuda:0
```

The online mode runs all 15 prompts and ranks them by how well each prompt's
embeddings separate violent from non-violent frames in LDA space. Use the
top-ranked prompt for future embedding extraction.