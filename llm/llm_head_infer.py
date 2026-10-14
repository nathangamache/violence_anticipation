#!/usr/bin/env python3
"""Run a chat LLM (default Qwen2.5-3B-Instruct) as a risk head on llm_data/*.jsonl prompts; report MSE and ROC-AUC.

Each row needs "prompt" (or "input"); the target is read from "target_risk", then "risk", then "label".

Usage:
    python -m llm.llm_head_infer --model_path Qwen/Qwen2.5-3B-Instruct \
        --data_path llm_data/val.jsonl --out_path llm_data/val_preds_qwen.jsonl --device cuda:1
"""

import argparse
import json
import math
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[JSONL] Skip malformed line in {path}: {e}")
    print(f"[JSONL] Loaded {len(items)} samples from {path}")
    return items


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[JSONL] Wrote {len(records)} records → {path}")


def build_chat_prompt(prompt_text: str) -> List[Dict[str, str]]:
    """Wrap a prompt in a system + user chat asking for JSON {"risk", "rationale"}."""
    system_msg = (
        "You are an expert assistant for early violence risk assessment "
        "from CCTV-style video. You receive a textual summary of the current "
        "scene and must estimate the short-term risk of physical violence.\n\n"
        "Always respond with a JSON object of the form:\n"
        "{\"risk\": <float between 0 and 1>, \"rationale\": \"...\"}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    return messages


def extract_json_block(text: str) -> Optional[str]:
    """Return the outermost {...} substring of text, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_risk_from_output(raw_output: str) -> Tuple[float, str]:
    """Parse (risk in [0, 1], rationale): JSON first, then a number after "risk", then the first number; default 0.5."""
    risk_val = 0.5
    rationale = raw_output.strip()

    json_block = extract_json_block(raw_output)
    if json_block is not None:
        try:
            obj = json.loads(json_block)
            if isinstance(obj, dict) and "risk" in obj:
                rv = obj["risk"]
                if isinstance(rv, (int, float, str)):
                    risk_val = float(rv)
                if "rationale" in obj and isinstance(obj["rationale"], str):
                    rationale = obj["rationale"]
                return _clamp01(risk_val), rationale
        except Exception:
            pass

    m = re.search(r"risk[^0-9]*([01]?\.\d+|\d+\.\d+|\d+)", raw_output, flags=re.IGNORECASE)
    if m:
        try:
            risk_val = float(m.group(1))
            return _clamp01(risk_val), rationale
        except Exception:
            pass

    m2 = re.search(r"([01]?\.\d+|\d+\.\d+|\d+)", raw_output)
    if m2:
        try:
            risk_val = float(m2.group(1))
        except Exception:
            risk_val = 0.5

    return _clamp01(risk_val), rationale


def _clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.5
    return max(0.0, min(1.0, float(x)))


def load_qwen(model_path: str, device: str):
    """Load a causal LM and tokenizer from a HF repo id or local directory (CPU if CUDA is unavailable)."""
    print(f"[LLM] Loading model from: {model_path}")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device_obj)
    model.eval()

    print(f"[LLM] Loaded on device: {device_obj}")
    return tokenizer, model, device_obj


def run_llm_inference(
    data: List[Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
    max_examples: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    bin_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Generate a risk per sample; returns {"records": [...], "metrics": {"mse", "auc", "num_frames"}}."""
    if max_examples is not None and max_examples > 0:
        data = data[:max_examples]
        print(f"[LLM] Limiting to first {len(data)} examples")

    all_targets: List[float] = []
    all_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else 1.0,
        "top_p": 0.9,
    }

    for sample in tqdm(data, desc="LLM Inference", dynamic_ncols=True):
        if "prompt" in sample:
            prompt_text = sample["prompt"]
        elif "input" in sample:
            prompt_text = sample["input"]
        else:
            print(f"[LLM] Sample {sample.get('id','<no-id>')} missing 'prompt'/'input'; skipping.")
            continue

        messages = build_chat_prompt(prompt_text)

        chat_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            chat_str,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **gen_kwargs,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

        risk_pred, rationale = parse_risk_from_output(raw_output)

        target_risk = None
        if "target_risk" in sample:
            target_risk = float(sample["target_risk"])
        elif "risk" in sample:
            # "risk" is assumed to be ground truth here, not an earlier prediction.
            target_risk = float(sample["risk"])
        elif "label" in sample:
            target_risk = float(sample["label"])

        record = {
            "id": sample.get("id"),
            "video_id": sample.get("video_id"),
            "frame_idx": sample.get("frame_idx"),
            "t_sec": sample.get("t_sec"),
            "source": sample.get("source"),
            "prompt": prompt_text,
            "target_risk": target_risk,
            "pred_risk": risk_pred,
            "bin_thresh": bin_thresh,
            "raw_output": raw_output,
            "rationale": rationale,
        }
        records.append(record)

        if target_risk is not None:
            all_targets.append(target_risk)
            all_scores.append(risk_pred)

    metrics = {
        "mse": None,
        "auc": None,
        "num_frames": len(all_scores),
    }

    if all_scores:
        scores_np = np.asarray(all_scores, dtype=np.float32)
        targets_np = np.asarray(all_targets, dtype=np.float32)

        mse = float(mean_squared_error(targets_np, scores_np))
        metrics["mse"] = mse

        bin_targets = (targets_np >= bin_thresh).astype(int)
        if np.unique(bin_targets).size < 2:
            print("[LLM] Only one class in binarized labels; ROC-AUC undefined.")
            auc = float("nan")
        else:
            auc = float(roc_auc_score(bin_targets, scores_np))
        metrics["auc"] = auc

    return {
        "records": records,
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-3B-Instruct as a violence-risk decision head "
                    "on graph-to-text summaries."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help=(
            "HF repo name or local directory for the model. "
            "Examples: 'Qwen/Qwen2.5-3B-Instruct' or "
            "'~/models/Qwen2.5-3B-Instruct'"
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="llm_data/val.jsonl",
        help="Path to LLM dataset JSONL (e.g. llm_data/val.jsonl).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="llm_data/val_preds_qwen.jsonl",
        help="Where to write predictions JSONL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. 'cuda:0', 'cuda:1', or 'cpu'. "
             "You can also set CUDA_VISIBLE_DEVICES externally.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=256,
        help="Maximum number of examples to run. "
             "Set to -1 or 0 to run all.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic output.",
    )
    parser.add_argument(
        "--bin_thresh",
        type=float,
        default=0.5,
        help="Threshold for binarizing risk when computing ROC-AUC.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    max_examples = args.max_examples
    if max_examples is not None and max_examples <= 0:
        max_examples = None

    data = load_jsonl(args.data_path)

    tokenizer, model, device = load_qwen(args.model_path, args.device)

    results = run_llm_inference(
        data=data,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_examples=max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        bin_thresh=args.bin_thresh,
    )

    records = results["records"]
    metrics = results["metrics"]

    write_jsonl(args.out_path, records)

    print("\n========== LLM HEAD METRICS ==========")
    print(f"Num examples used:   {len(records)}")
    print(f"Num frames w/ label: {metrics['num_frames']}")
    if metrics["mse"] is not None:
        print(f"Frame-level MSE:     {metrics['mse']:.6f}")
    else:
        print("Frame-level MSE:     N/A (no labels found)")
    if metrics["auc"] is None or math.isnan(metrics["auc"]):
        print("Frame-level ROC-AUC: N/A (not enough label variety)")
    else:
        print(f"Frame-level ROC-AUC: {metrics['auc']:.4f}")
    print("======================================\n")


if __name__ == "__main__":
    main()
