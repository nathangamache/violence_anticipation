"""
prompt_engineering_stats.py
============================
Explores prompt variations for Qwen2.5-VL to extract richer pre-violence statistics.
The core hypothesis: current prompts describe violence/non-violence in binary terms,
but pre-violence cues (tension, body language, proxemics, verbal aggression) may
require targeted prompts to surface in embeddings.

This script:
  1. Defines a library of candidate prompt strategies
  2. Runs each prompt on a small set of reference frames (violent, pre-violent, neutral)
  3. Computes embedding separation statistics (Cohen's d, LDA separability, cosine similarity)
  4. Ranks prompts by how well they separate classes in embedding space

Usage:
    python prompt_engineering_stats.py \
        --frames_dir data/probe_frames \
        --vlm_model /path/to/Qwen2.5-VL \
        --output results/prompt_engineering.csv

Note: Requires GPU with Qwen2.5-VL loaded. For offline analysis on pre-extracted
embeddings, use --offline mode with --embeddings.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Prompt Library
# ──────────────────────────────────────────────

PROMPT_LIBRARY: Dict[str, str] = {
    # --- Baseline / current approach ---
    "binary_violence": (
        "Describe whether violence is occurring in this scene. "
        "Rate the severity: none, mild, moderate, severe."
    ),

    # --- Pre-violence behavioral cues ---
    "body_language": (
        "Analyze the body language of all people in this scene. "
        "Focus on: posture (tense/relaxed), personal space violations, "
        "aggressive stance indicators (clenched fists, squared shoulders), "
        "and movement patterns (approaching/retreating)."
    ),

    "proxemics": (
        "Describe the spatial relationships between people in this scene. "
        "How close are they? Is anyone invading personal space? "
        "Are people facing each other in a confrontational way or in a neutral way?"
    ),

    "facial_emotion": (
        "Describe the facial expressions and emotional states of all visible people. "
        "Focus on: anger, fear, distress, aggression, calm, or neutral affect. "
        "Note any asymmetry or unusual expressions."
    ),

    "object_threat": (
        "Identify all objects in the scene. Flag any that could be used as weapons "
        "or that indicate risk: bottles, knives, tools, vehicles moving toward people. "
        "Describe the context in which these objects appear."
    ),

    "group_dynamics": (
        "Describe the group dynamics in this scene. "
        "Is there a clear confrontation between groups? "
        "Are people clustering, dispersing, or forming a circle? "
        "Describe the overall social tension level."
    ),

    # --- Scene/environment context ---
    "environment_risk": (
        "Describe the environment and its risk factors. "
        "Location type, time indicators (night/day), crowd density, "
        "visibility of security features, and any environmental hazards."
    ),

    "attention_focus": (
        "Where is the attention of people in this scene directed? "
        "Are people watching something specific? "
        "Is there a focal point of tension? Describe what people are focused on."
    ),

    # --- Anticipatory / predictive prompts ---
    "pre_event_prediction": (
        "Based solely on what you observe in this frame — body language, spacing, "
        "emotional state, object positions — rate the probability that a physical "
        "altercation will occur in the next 30 seconds: very low, low, moderate, high, very high. "
        "Explain your reasoning."
    ),

    "behavioral_escalation": (
        "Describe any signs of behavioral escalation in this scene. "
        "Are there indicators of escalating tension such as raised voices (inferred), "
        "encroachment of personal space, finger-pointing, or crowd gathering? "
        "Rate the escalation level 0-10."
    ),

    "situational_awareness": (
        "Provide a situational awareness report for this scene. "
        "Identify: (1) number of people, (2) their roles/relationships, "
        "(3) current activity, (4) potential threat indicators, "
        "(5) exits/escape routes visible."
    ),

    # --- Structured output prompts ---
    "structured_risk": (
        "Analyze this surveillance frame and respond in this exact format:\n"
        "PEOPLE_COUNT: <number>\n"
        "TENSION_LEVEL: <0-10>\n"
        "AGGRESSION_INDICATORS: <list>\n"
        "SPATIAL_VIOLATIONS: <yes/no>\n"
        "WEAPON_VISIBLE: <yes/no>\n"
        "RISK_SCORE: <0-10>"
    ),

    "structured_cues": (
        "For each person visible, describe:\n"
        "- Posture: tense or relaxed\n"
        "- Movement: toward, away, stationary\n"
        "- Expression: aggressive, fearful, neutral, distressed\n"
        "- Hands: open, closed fists, holding object\n"
        "Then give an overall threat assessment."
    ),

    # --- Contrastive prompts ---
    "normal_vs_abnormal": (
        "Describe what would be normal vs. abnormal behavior in this context. "
        "Does what you see match normal behavior for this setting? "
        "Identify anything that deviates from expected norms."
    ),

    "minimal_description": (
        "Briefly describe what is happening in this scene in 2-3 sentences."
    ),

    "rich_narrative": (
        "Write a detailed narrative of exactly what is happening in this scene, "
        "including the actions, expressions, and apparent motivations of every person visible. "
        "Note any concerning elements."
    ),
}


# ──────────────────────────────────────────────
# Offline analysis (on pre-extracted embeddings)
# ──────────────────────────────────────────────

def cohens_d_overall(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Mean Cohen's d across all dimensions."""
    mu_a, mu_b = group_a.mean(axis=0), group_b.mean(axis=0)
    s_a, s_b = group_a.std(axis=0), group_b.std(axis=0)
    pooled = np.sqrt((s_a ** 2 + s_b ** 2) / 2) + 1e-8
    d = np.abs(mu_a - mu_b) / pooled
    return d.mean()


def lda_separation(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    LDA on (X, y). Returns:
      - Fisher ratio (between-class / within-class variance on LDA projection)
      - AUC of LDA score as a binary classifier
    """
    if len(np.unique(y)) < 2:
        return 0.0, 0.5
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, y).flatten()
    v_mean = X_lda[y == 1].mean()
    nv_mean = X_lda[y == 0].mean()
    v_std = X_lda[y == 1].std() + 1e-8
    nv_std = X_lda[y == 0].std() + 1e-8
    fisher = (v_mean - nv_mean) ** 2 / (v_std ** 2 + nv_std ** 2)
    auc = roc_auc_score(y, X_lda) if len(np.unique(y)) > 1 else 0.5
    return float(fisher), float(auc)


def analyze_embedding_set(embeddings: np.ndarray, labels: np.ndarray, prompt_name: str) -> Dict:
    """
    Given a set of embeddings with labels (0=non-violent, 1=violent/pre-violent),
    compute separation statistics.
    """
    v_idx = labels == 1
    nv_idx = labels == 0

    if v_idx.sum() == 0 or nv_idx.sum() == 0:
        return {"prompt": prompt_name, "cohen_d": float("nan"), "fisher": float("nan"), "lda_auc": float("nan")}

    v_emb = embeddings[v_idx]
    nv_emb = embeddings[nv_idx]

    cd = cohens_d_overall(v_emb, nv_emb)
    fisher, lda_auc = lda_separation(embeddings, labels)

    # Top-k discriminative dimensions
    mu_v = v_emb.mean(axis=0)
    mu_nv = nv_emb.mean(axis=0)
    pooled_std = np.sqrt((v_emb.std(axis=0) ** 2 + nv_emb.std(axis=0) ** 2) / 2) + 1e-8
    per_dim_d = np.abs(mu_v - mu_nv) / pooled_std
    top_k_dims = int((per_dim_d > 0.5).sum())

    return {
        "prompt": prompt_name,
        "cohen_d_mean": round(float(cd), 4),
        "lda_fisher": round(float(fisher), 4),
        "lda_auc": round(float(lda_auc), 4),
        "dims_d_gt_0.5": top_k_dims,
        "n_violent": int(v_idx.sum()),
        "n_nonviolent": int(nv_idx.sum()),
    }


def run_offline_analysis(embeddings_dir: Path, labels_dir: Path, output_path: Path):
    """
    Offline mode: analyzes pre-computed embeddings from different prompt runs.
    Expects subdirectories named after each prompt under embeddings_dir.
    Falls back to single directory if no subdirectories found.
    """
    print("\n[Offline Mode] Analyzing pre-computed embeddings")

    # Check if we have prompt-subdirectories or a flat structure
    subdirs = [d for d in embeddings_dir.iterdir() if d.is_dir() and d.name in PROMPT_LIBRARY]

    if subdirs:
        print(f"Found {len(subdirs)} prompt subdirectories")
        results = []
        for subdir in subdirs:
            prompt_name = subdir.name
            pairs = load_pairs_from_dir(subdir, labels_dir, max_videos=100)
            if not pairs:
                continue
            X = np.vstack([p[0] for p in pairs])
            y = np.concatenate([p[1] for p in pairs])
            stats = analyze_embedding_set(X, y, prompt_name)
            results.append(stats)
            print(f"  {prompt_name:30s} LDA AUC={stats['lda_auc']:.3f}  Cohen d={stats['cohen_d_mean']:.3f}")
    else:
        # Flat structure: analyze existing embeddings with current prompt only
        print("No prompt subdirectories found. Analyzing existing embeddings as 'current_prompt'.")
        pairs = load_pairs_from_dir(embeddings_dir, labels_dir, max_videos=200)
        if not pairs:
            print("No embeddings found.")
            return
        X = np.vstack([p[0] for p in pairs])
        y = np.concatenate([p[1] for p in pairs])
        stats = analyze_embedding_set(X, y, "current_prompt")
        results = [stats]
        print(f"  current_prompt  LDA AUC={stats['lda_auc']:.3f}  Cohen d={stats['cohen_d_mean']:.3f}")

        # Show which dimensions are most discriminative
        v_emb = X[y == 1]
        nv_emb = X[y == 0]
        mu_v = v_emb.mean(axis=0)
        mu_nv = nv_emb.mean(axis=0)
        pooled = np.sqrt((v_emb.std(axis=0)**2 + nv_emb.std(axis=0)**2) / 2) + 1e-8
        per_dim_d = np.abs(mu_v - mu_nv) / pooled
        top_dims = np.argsort(per_dim_d)[-10:][::-1]
        print("\n  Top 10 discriminative dimensions in current embeddings:")
        for dim in top_dims:
            print(f"    dim {dim:4d}: d={per_dim_d[dim]:.3f}  V={mu_v[dim]:.4f}  NV={mu_nv[dim]:.4f}")

    # Save results
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_path}")
    return results


def load_pairs_from_dir(emb_dir, labels_dir, max_videos=200):
    pairs = []
    for vf in sorted(emb_dir.rglob("*.npz"))[:max_videos]:
        vid = vf.stem
        for candidate in [
            labels_dir / f"{vid}.npy",
            labels_dir / "xd_violence" / f"{vid}.npy",
        ]:
            if candidate.exists():
                try:
                    emb = np.load(vf)["emb"].astype(np.float32)
                    lbl = np.load(candidate)
                    if lbl.ndim > 1:
                        lbl = lbl.max(axis=1)
                    n = min(len(emb), len(lbl))
                    pairs.append((emb[:n], (lbl[:n] > 0).astype(int)))
                except Exception:
                    pass
                break
    return pairs


# ──────────────────────────────────────────────
# Online mode: runs prompts live against VLM
# ──────────────────────────────────────────────

def run_online_analysis(
    frames_dir: Path,
    vlm_model_path: str,
    output_dir: Path,
    prompts: Optional[List[str]] = None,
    device: str = "cuda:0",
):
    """
    Runs each prompt in PROMPT_LIBRARY against a set of probe frames.
    Extracts embeddings and computes separation statistics.

    frames_dir should contain:
      frames_dir/violent/     *.jpg frames from violent events
      frames_dir/previolent/  *.jpg frames from pre-violence windows
      frames_dir/neutral/     *.jpg frames from calm scenes
    """
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("transformers / qwen_vl_utils not available. Run in offline mode instead.")
        return

    if prompts is None:
        prompts = list(PROMPT_LIBRARY.keys())

    print(f"Loading Qwen2.5-VL from {vlm_model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_path, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(vlm_model_path)
    model.eval()

    # Load probe frames
    categories = {"violent": 1, "previolent": 0.5, "neutral": 0}
    frames_by_cat = {}
    for cat, _ in categories.items():
        cat_dir = frames_dir / cat
        if cat_dir.exists():
            frames_by_cat[cat] = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png"))

    if not frames_by_cat:
        print(f"No frames found in {frames_dir}")
        return

    results = []
    for prompt_name in prompts:
        prompt_text = PROMPT_LIBRARY[prompt_name]
        print(f"\nRunning prompt: {prompt_name}")

        all_embs, all_labels = [], []

        for cat, label_val in categories.items():
            if cat not in frames_by_cat:
                continue
            label = 1 if label_val > 0 else 0
            for frame_path in frames_by_cat[cat][:20]:  # limit to 20 per category
                try:
                    from PIL import Image
                    img = Image.open(frame_path).convert("RGB")

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ]
                    }]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden = outputs.hidden_states[-1]  # (1, seq, 3584)
                        emb = hidden[:, -1, :].float().cpu().numpy().flatten()

                    all_embs.append(emb)
                    all_labels.append(label)
                except Exception as e:
                    print(f"  Error on {frame_path}: {e}")
                    continue

        if len(all_embs) == 0:
            continue

        X = np.array(all_embs)
        y = np.array(all_labels)
        stats = analyze_embedding_set(X, y, prompt_name)
        results.append(stats)
        print(f"  LDA AUC={stats['lda_auc']:.3f}  Cohen d={stats['cohen_d_mean']:.3f}  n={len(y)}")

        # Save embeddings for this prompt
        prompt_out = output_dir / prompt_name
        prompt_out.mkdir(parents=True, exist_ok=True)
        np.save(prompt_out / "embeddings.npy", X)
        np.save(prompt_out / "labels.npy", y)

    # Rank and print summary
    results.sort(key=lambda r: r["lda_auc"], reverse=True)
    print("\n" + "=" * 60)
    print("PROMPT RANKING BY LDA AUC (best separating prompts)")
    print("=" * 60)
    print(f"{'Prompt':<30} {'LDA AUC':>10} {'Cohen d':>10} {'Dims d>0.5':>12}")
    print("-" * 65)
    for r in results:
        print(f"  {r['prompt']:<28} {r['lda_auc']:>10.3f} {r['cohen_d_mean']:>10.3f} {r['dims_d_gt_0.5']:>12}")

    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prompt_ranking.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRanking saved to {csv_path}")
    return results


# ──────────────────────────────────────────────
# Print the full prompt library (no GPU needed)
# ──────────────────────────────────────────────

def print_prompt_library():
    print("\n" + "=" * 70)
    print("PROMPT LIBRARY")
    print("=" * 70)
    categories = {
        "Baseline": ["binary_violence", "minimal_description"],
        "Pre-violence behavioral": ["body_language", "proxemics", "facial_emotion", "group_dynamics"],
        "Object/environment": ["object_threat", "environment_risk", "attention_focus"],
        "Anticipatory": ["pre_event_prediction", "behavioral_escalation", "situational_awareness"],
        "Structured output": ["structured_risk", "structured_cues"],
        "Contrastive": ["normal_vs_abnormal", "rich_narrative"],
    }
    for cat, keys in categories.items():
        print(f"\n--- {cat} ---")
        for k in keys:
            print(f"\n  [{k}]")
            for line in PROMPT_LIBRARY[k].split("\n"):
                print(f"    {line}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online", "list"], default="offline",
                        help="offline: analyze existing embeddings | online: run live VLM | list: print prompts")
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--frames_dir", type=Path, default=Path("data/probe_frames"))
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("results/prompt_engineering.csv"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.mode == "list":
        print_prompt_library()
    elif args.mode == "offline":
        run_offline_analysis(args.embeddings, args.labels, args.output)
    elif args.mode == "online":
        run_online_analysis(
            frames_dir=args.frames_dir,
            vlm_model_path=args.vlm_model,
            output_dir=args.output.parent / "prompt_embeddings",
            device=args.device,
        )


if __name__ == "__main__":
    main()