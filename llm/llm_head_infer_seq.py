#!/usr/bin/env python3
"""Video-level LLM head: ask Qwen2.5-3B-Instruct for strict JSON {"label", "risk"} per llm_data/{split}.jsonl row.

Reports MSE, ROC-AUC and accuracy at --bin_thresh and writes llm_data/{split}_seq_preds_qwen.jsonl.

Usage:
    python -m llm.llm_head_infer_seq --split val --data_dir llm_data --model_path <local or HF path> --device cuda:0
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def build_prompt_for_example(example):
    """Append strict JSON-output instructions to the example's existing prompt."""
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
    """Return (label, risk clipped to [0, 1]) from the first {...} in text, or (None, None)."""
    import re

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

    try:
        risk_float = float(risk)
    except (TypeError, ValueError):
        return None, None

    risk_float = float(np.clip(risk_float, 0.0, 1.0))

    return label_norm, risk_float


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
        target_label = int(ex.get("label", 0))

        prompt = build_prompt_for_example(ex)

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

    scores = np.array(all_scores, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.int32)

    mse = float(np.mean((scores - targets) ** 2))

    if np.unique(targets).size < 2:
        print("[LLM] Only one class in labels; ROC-AUC undefined.")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(targets, scores))

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

    out_path = os.path.join(args.data_dir, f"{args.split}_seq_preds_qwen.jsonl")
    save_jsonl(out_path, metrics["records"])
    print(f"[Save] Wrote predictions JSONL → {out_path}")


if __name__ == "__main__":
    main()
