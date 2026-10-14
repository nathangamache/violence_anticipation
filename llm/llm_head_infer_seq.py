#!/usr/bin/env python3
"""
Video-level LLM decision head inference.

Current setup:
- Each line in llm_data/{split}.jsonl corresponds to ONE video clip.
- Schema (example):

  {
    "id": "GoldenEye.1995__#00-24-53_00-25-06_label_B1-0-0",
    "source": "xd_violence",
    "label": 0,
    "event_frame": 0,
    "total_frames": 313,
    "fps": 24.0,
    "prompt": "...Scene graph summary... Question: Does this video contain physical violence...?",
    "response": "non-violent"   # (from a previous run, can be ignored)
  }

This script:
- Loads train.jsonl or val.jsonl.
- For each example, uses the existing `prompt` and appends strict JSON-output
  instructions.
- Calls Qwen2.5-3B-Instruct to get:

    {"label": "violent" or "non-violent", "risk": <float 0-1>}

- Evaluates video-level:
    * MSE (between numeric risk and 0/1 ground-truth label)
    * ROC-AUC (using risk as the score)
    * Accuracy (thresholding risk at 0.5, or via label string)
- Saves detailed predictions to llm_data/{split}_seq_preds_qwen.jsonl
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------
def build_prompt_for_example(example):
    """
    Take the existing 'prompt' field and append strict JSON-output instructions.

    The original prompt already explains:
      - The dataset
      - The scene graph summary over time
      - The question: violent vs non-violent

    We now force the model to output ONLY a JSON object with:
      - label: "violent" or "non-violent"
      - risk: float in [0,1]
    """
    base_prompt = example["prompt"].rstrip()

    extra = [
        "",
        "Now, using ONLY the information above, you must:",
        "1) Decide whether this clip contains clear physical violence ('violent' or 'non-violent').",
        "2) Output a calibrated probability (between 0 and 1) that the clip contains physical violence at any point.",
        "",
        "Respond in EXACTLY the following JSON format and nothing else:",
        '',
        '  {"label": "violent" or "non-violent", "risk": 0.0}',
        "",
        "Use a float between 0 and 1 for the value of 'risk'.",
        "Do not add explanations, extra keys, or any text outside the JSON object."
    ]

    prompt = base_prompt + "\n" + "\n".join(extra)
    return prompt


def parse_llm_json(text):
    """
    Extract {"label": ..., "risk": ...} from the LLM output text.

    Returns:
      (label_str, risk_float) or (None, None) if parsing fails.
    """
    import re

    # Find first {...} blob
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None, None

    blob = m.group(0)

    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None, None

    label = obj.get("label", "")
    risk = obj.get("risk", None)

    if not isinstance(label, str):
        return None, None

    label_norm = label.strip().lower()

    # Try to coerce risk to float
    try:
        risk_float = float(risk)
    except (TypeError, ValueError):
        # If risk missing / bad, we can't use this prediction
        return None, None

    # Clamp risk to [0,1] just in case
    risk_float = float(np.clip(risk_float, 0.0, 1.0))

    return label_norm, risk_float


# ---------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------
def run_video_llm_inference(
    jsonl_path,
    model_path,
    device="cuda:0",
    max_examples=None,
    bin_thresh=0.5,
    max_new_tokens=128,
):
    print(f"[Load] JSONL from {jsonl_path}")
    examples = load_jsonl(jsonl_path)
    print(f"[Load] Total video examples in file: {len(examples)}")

    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"[Subsample] Using first {len(examples)} examples")

    # Load tokenizer + model
    print(f"[LLM] Loading Qwen from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {}
    model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.to(device)
    model.eval()

    all_scores = []
    all_targets = []
    all_pred_labels = []
    used_records = []

    for ex in tqdm(examples, desc="Videos", dynamic_ncols=True):
        vid_id = ex.get("id")
        src = ex.get("source", None)
        target_label = int(ex.get("label", 0))  # 0 = non-violent, 1 = violent

        prompt = build_prompt_for_example(ex)

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        # Only decode the new tokens after the prompt
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        label_pred, risk_pred = parse_llm_json(text)
        if label_pred is None:
            print(f"[Warn] Could not parse JSON output for id={vid_id}; skipping.")
            continue

        all_scores.append(risk_pred)
        all_targets.append(target_label)
        all_pred_labels.append(label_pred)

        used_records.append(
            {
                "id": vid_id,
                "source": src,
                "gt_label": target_label,
                "pred_label": label_pred,
                "pred_risk": risk_pred,
                "raw_output": text,
            }
        )

    if not all_scores:
        raise RuntimeError("[LLM] No usable predictions parsed from outputs.")

    scores = np.array(all_scores, dtype=np.float32)   # [N_examples], in [0,1]
    targets = np.array(all_targets, dtype=np.int32)   # [N_examples], 0/1

    # Metrics
    mse = float(np.mean((scores - targets) ** 2))

    # ROC-AUC
    if np.unique(targets).size < 2:
        print("[LLM] Only one class in labels; ROC-AUC undefined.")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(targets, scores))

    # Accuracy (using risk threshold OR label string)
    # We'll just threshold risk:
    pred_binary = (scores >= bin_thresh).astype(np.int32)
    acc = float(np.mean(pred_binary == targets))

    print("\n====== LLM VIDEO-LEVEL EVAL ======")
    print(f"Num examples used:   {scores.shape[0]}")
    print(f"Video-level MSE:     {mse:.6f}")
    if np.isnan(auc):
        print("Video-level ROC-AUC: N/A (only one class present)")
    else:
        print(f"Video-level ROC-AUC: {auc:.4f}")
    print(f"Video-level Accuracy: {acc:.4f}")
    print("==================================\n")

    return {
        "mse": mse,
        "auc": auc,
        "accuracy": acc,
        "num_examples": int(scores.shape[0]),
        "records": used_records,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate (expects {split}.jsonl in data_dir).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="llm_data",
        help="Directory containing train.jsonl / val.jsonl.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-3B-Instruct",
        help="Local path or HF model id for Qwen2.5-3B-Instruct.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. 'cuda:0', 'cuda:1', or 'cpu'.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max number of examples to evaluate (for quicker runs).",
    )
    parser.add_argument(
        "--bin_thresh",
        type=float,
        default=0.5,
        help="Threshold on risk for binary prediction (violent vs non-violent).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for generation.",
    )
    args = parser.parse_args()

    jsonl_path = os.path.join(args.data_dir, f"{args.split}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    device = torch.device(args.device)

    metrics = run_video_llm_inference(
        jsonl_path=jsonl_path,
        model_path=args.model_path,
        device=device,
        max_examples=args.max_examples,
        bin_thresh=args.bin_thresh,
        max_new_tokens=args.max_new_tokens,
    )

    # Save detailed predictions JSONL alongside the dataset
    out_path = os.path.join(args.data_dir, f"{args.split}_seq_preds_qwen.jsonl")
    save_jsonl(out_path, metrics["records"])
    print(f"[Save] Wrote predictions JSONL → {out_path}")


if __name__ == "__main__":
    main()
