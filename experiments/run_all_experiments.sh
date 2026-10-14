#!/usr/bin/env bash
# run_all_experiments.sh
# ======================
# Runs all experiments in priority order.
# Edit paths at the top to match your server layout.
# Run from your project root (e.g., ~/violence_anticipation/).

set -e

# ── Paths ────────────────────────────────────────────
EMB_DIR="data/processed/vlm_emb"
LABELS_DIR="data/processed/labels_multih"
LSTM_CKPT="outputs/outputs_temporal/best_temporal_model.pt"
UCF_EMB_DIR="data/processed/vlm_emb/ucf_crime"   # adjust if needed
RESULTS_DIR="results"
EXPERIMENTS_DIR="$(dirname "$0")"  # wherever this script lives

mkdir -p "$RESULTS_DIR"

# ── 1. EMA smoothing ─────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  1. EMA SMOOTHING                    ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/ema_smoothing.py" \
    --embeddings "$EMB_DIR" \
    --labels "$LABELS_DIR" \
    --checkpoint "$LSTM_CKPT" \
    --alphas 0.1 0.3 0.5 0.7 0.9 \
    --output "$RESULTS_DIR/ema_results.csv"

# ── 2. Algorithm variations ──────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  2. ALGORITHM VARIATIONS             ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/algorithm_variations.py" \
    --embeddings "$EMB_DIR" \
    --labels "$LABELS_DIR" \
    --checkpoint "$LSTM_CKPT" \
    --output "$RESULTS_DIR/algorithm_comparison.csv"

# ── 3. Prompt engineering stats (offline) ───────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  3. PROMPT ENGINEERING (offline)     ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/prompt_engineering_stats.py" \
    --mode offline \
    --embeddings "$EMB_DIR" \
    --labels "$LABELS_DIR" \
    --output "$RESULTS_DIR/prompt_engineering.csv"

# Optional: run online (requires GPU0 + Qwen2.5-VL):
# python "$EXPERIMENTS_DIR/prompt_engineering_stats.py" \
#     --mode online \
#     --vlm_model /path/to/Qwen2.5-VL \
#     --frames_dir data/probe_frames \
#     --output "$RESULTS_DIR/prompt_engineering_online.csv"

# ── 4. UCF-Crime cross-domain testing ───────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  4. UCF-CRIME TESTING                ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/ucf_crime_testing.py" \
    --ucf_embeddings "$UCF_EMB_DIR" \
    --xd_embeddings "$EMB_DIR" \
    --xd_labels "$LABELS_DIR" \
    --lstm_checkpoint "$LSTM_CKPT" \
    --output "$RESULTS_DIR/ucf_crime_results.csv"

# ── 5. XD-Violence data investigation ───────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  5. XD-VIOLENCE INVESTIGATION        ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/xd_violence_investigation.py" \
    --embeddings "$EMB_DIR" \
    --labels "$LABELS_DIR" \
    --min_previolence 5 \
    --output_dir "$RESULTS_DIR/xd_investigation"

# ── 6. Baseline comparisons (comprehensive) ─────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  6. BASELINE COMPARISONS             ║"
echo "╚══════════════════════════════════════╝"
python "$EXPERIMENTS_DIR/baseline_comparisons.py" \
    --embeddings "$EMB_DIR" \
    --labels "$LABELS_DIR" \
    --lstm_checkpoint "$LSTM_CKPT" \
    --ema_alpha 0.5 \
    --output "$RESULTS_DIR/baseline_comparison.csv"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"