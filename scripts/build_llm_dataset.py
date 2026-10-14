#!/usr/bin/env python3
"""Build video-level LLM prompt/response JSONL (llm_data/{train,val}.jsonl) from graphs and labels."""
import os
import json
import argparse

import numpy as np
import yaml

try:
    from core.graph import graph_to_text
except Exception:
    graph_to_text = None

from core.utils import get_device


def load_metadata(meta_dir: str) -> dict:
    """Merge metadata/{xd_violence,ucf_crime,kinetics}.json into one dict keyed by vid_id."""
    meta = {}
    if not os.path.isdir(meta_dir):
        print(f"[Meta] Metadata dir not found: {meta_dir}")
        return meta

    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        path = os.path.join(meta_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                cur = json.load(f)
            meta.update(cur)
            print(f"[Meta] Loaded {len(cur)} entries from {fname}")
        else:
            print(f"[Meta] Missing {fname}, skipping.")
    print(f"[Meta] Total metadata entries: {len(meta)}")
    return meta


def safe_load_graph(graph_path: str):
    """Load a graph .npy as a list of per-frame dicts ([] on error)."""
    try:
        arr = np.load(graph_path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                seq = arr.tolist()
            else:
                seq = arr.tolist()
        else:
            seq = list(arr)
        if not isinstance(seq, list):
            seq = list(seq)
        return seq
    except Exception as e:
        print(f"[Graph] Error loading {graph_path}: {e}")
        return []


def safe_load_labels(label_path: str):
    try:
        arr = np.load(label_path, allow_pickle=True)
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return arr
    except Exception as e:
        print(f"[Labels] Error loading {label_path}: {e}")
        return np.zeros((0,), dtype=float)


def safe_load_predictions(pred_dir: str | None, vid_id: str):
    """Return (pred, gt) curves from <pred_dir>/<vid_id>.npz, or (None, None)."""
    if pred_dir is None:
        return None, None

    path = os.path.join(pred_dir, f"{vid_id}.npz")
    if not os.path.exists(path):
        return None, None

    try:
        data = np.load(path)
        pred = np.asarray(data["pred"], dtype=float).reshape(-1)
        gt = np.asarray(data["gt"], dtype=float).reshape(-1)
        return pred, gt
    except Exception as e:
        print(f"[Pred] Error loading {path}: {e}")
        return None, None


def fallback_graph_summary(graph_seq, meta: dict | None, label: int) -> str:
    """Stats-only summary used when core.graph.graph_to_text is unavailable or fails."""
    num_frames = len(graph_seq)
    num_nodes_per_frame = []

    for f in graph_seq:
        nodes = f.get("nodes", None) if isinstance(f, dict) else None
        if nodes is None:
            num_nodes_per_frame.append(0)
        else:
            try:
                if hasattr(nodes, "shape"):
                    n = nodes.shape[0]
                else:
                    n = len(nodes)
                num_nodes_per_frame.append(int(n))
            except Exception:
                num_nodes_per_frame.append(0)

    avg_nodes = float(np.mean(num_nodes_per_frame)) if num_nodes_per_frame else 0.0
    violence_str = "violent" if label == 1 else "non-violent"
    src = meta.get("source", "unknown") if meta else "unknown"

    return (
        f"This video comes from the {src} dataset and is annotated as {violence_str}.\n"
        f"It has {num_frames} frames. On average there are about {avg_nodes:.1f} "
        f"tracked entities (people/objects) per frame."
    )


def build_single_example(
    vid_id: str,
    graph_seq,
    label_curve: np.ndarray,
    meta: dict | None,
    pred_curve: np.ndarray | None,
    bin_thresh: float = 0.5,
) -> dict:
    """Build one video-level JSONL example (metadata fields, prompt, response)."""
    if meta is None:
        meta = {}

    src = meta.get("source", "unknown")
    event_frame = meta.get("event_frame", None)
    total_frames = meta.get("total_frames", int(label_curve.shape[0]))
    fps = meta.get("fps", None)
    base_label = meta.get("label", None)

    if base_label is None:
        base_label = int((label_curve >= bin_thresh).any())
    else:
        try:
            base_label = int(base_label)
        except Exception:
            base_label = int((label_curve >= bin_thresh).any())

    violence_str = "violent" if base_label == 1 else "non-violent"

    if graph_to_text is not None:
        try:
            text_summary = graph_to_text(graph_seq, meta)
        except TypeError:
            try:
                text_summary = graph_to_text(graph_seq)
            except Exception:
                text_summary = fallback_graph_summary(graph_seq, meta, base_label)
        except Exception:
            text_summary = fallback_graph_summary(graph_seq, meta, base_label)
    else:
        text_summary = fallback_graph_summary(graph_seq, meta, base_label)

    pred_summary = ""
    if pred_curve is not None and pred_curve.size > 0:
        max_pred = float(pred_curve.max())
        mean_pred = float(pred_curve.mean())
        argmax_pred = int(pred_curve.argmax())
        pred_summary = (
            "Base transformer risk summary (scores in [0, 1]):\n"
            f"- max_risk: {max_pred:.3f} at frame {argmax_pred}\n"
            f"- mean_risk: {mean_pred:.3f}\n"
        )
        if event_frame is not None and isinstance(event_frame, int):
            if 0 <= event_frame < pred_curve.shape[0]:
                risk_at_event = float(pred_curve[event_frame])
                pred_summary += f"- risk_at_event_frame: {risk_at_event:.3f}\n"

    prompt_lines = []
    prompt_lines.append("You are a model that judges whether a short video contains physical violence.")
    prompt_lines.append("")
    prompt_lines.append(f"Video ID: {vid_id}")
    prompt_lines.append(f"Source dataset: {src}")
    prompt_lines.append(f"Total frames: {total_frames}")
    if fps is not None:
        prompt_lines.append(f"Frame rate (fps): {fps}")
    if event_frame is not None:
        prompt_lines.append(f"Annotated event frame (violence onset): {event_frame}")
    prompt_lines.append("")
    prompt_lines.append("Scene graph summary (simplified description of people, objects, and interactions over time):")
    prompt_lines.append(text_summary.strip())
    prompt_lines.append("")

    if pred_summary:
        prompt_lines.append(pred_summary.strip())
        prompt_lines.append("")

    prompt_lines.append(
        "Question: Does this video contain physical violence (fighting, assault, robbery, arson, etc.) at any point?"
    )
    prompt_lines.append("Answer with a single word: 'violent' or 'non-violent'.")

    prompt = "\n".join(prompt_lines)

    response = violence_str

    example = {
        "id": vid_id,
        "source": src,
        "label": int(base_label),
        "event_frame": event_frame,
        "total_frames": int(total_frames) if total_frames is not None else None,
        "fps": float(fps) if fps is not None else None,
        "prompt": prompt,
        "response": response,
    }

    return example


def build_examples_from_index(
    index_path: str,
    metadata: dict,
    pred_dir: str | None,
    bin_thresh: float = 0.5,
) -> list:
    """Build LLM examples for every entry of an index JSON (train_index.json or val_index.json)."""
    with open(index_path, "r") as f:
        items = json.load(f)

    print(f"[LLM] Building examples from {index_path} ({len(items)} entries)")

    examples = []
    for idx, item in enumerate(items):
        graph_path = item["graph_path"]
        label_path = item["label_path"]

        vid_id = os.path.splitext(os.path.basename(graph_path))[0]

        graph_seq = safe_load_graph(graph_path)
        if not graph_seq:
            print(f"[LLM] Skip {vid_id}: empty or invalid graph sequence.")
            continue

        label_curve = safe_load_labels(label_path)
        if label_curve.size == 0:
            print(f"[LLM] Skip {vid_id}: empty or invalid label curve.")
            continue

        meta = metadata.get(vid_id, None)
        pred_curve, _ = safe_load_predictions(pred_dir, vid_id)

        try:
            ex = build_single_example(
                vid_id=vid_id,
                graph_seq=graph_seq,
                label_curve=label_curve,
                meta=meta,
                pred_curve=pred_curve,
                bin_thresh=bin_thresh,
            )
            examples.append(ex)
        except Exception as e:
            print(f"[LLM] Error building example for {vid_id}: {e}")

    print(f"[LLM] Built {len(examples)} examples from {index_path}")
    return examples


def write_jsonl(path: str, examples: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[LLM] Wrote {len(examples)} examples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_index",
        type=str,
        default="train_index.json",
        help="Path to train index JSON (default: train_index.json).",
    )
    parser.add_argument(
        "--val_index",
        type=str,
        default="val_index.json",
        help="Path to val index JSON (default: val_index.json).",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Optional directory with per-video predictions (e.g. predictions/model_epoch19).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="llm_data",
        help="Output directory for JSONL files (default: llm_data).",
    )
    parser.add_argument(
        "--bin_thresh",
        type=float,
        default=0.5,
        help="Threshold for deciding if a video is violent from its label curve when metadata label is missing.",
    )
    args = parser.parse_args()

    if os.path.exists("configs/paths.yaml"):
        with open("configs/paths.yaml", "r") as f:
            paths_cfg = yaml.safe_load(f)
        paths = paths_cfg.get("paths", {})
        meta_dir = paths.get("metadata", os.path.join("data", "metadata"))
    else:
        meta_dir = os.path.join("data", "metadata")

    print(f"[LLM] Using metadata dir: {meta_dir}")
    metadata = load_metadata(meta_dir)

    if os.path.exists(args.train_index):
        train_examples = build_examples_from_index(
            index_path=args.train_index,
            metadata=metadata,
            pred_dir=args.pred_dir,
            bin_thresh=args.bin_thresh,
        )
    else:
        print(f"[LLM] Train index not found: {args.train_index}. No train examples.")
        train_examples = []

    if os.path.exists(args.val_index):
        val_examples = build_examples_from_index(
            index_path=args.val_index,
            metadata=metadata,
            pred_dir=args.pred_dir,
            bin_thresh=args.bin_thresh,
        )
    else:
        print(f"[LLM] Val index not found: {args.val_index}. No val examples.")
        val_examples = []

    train_out = os.path.join(args.out_dir, "train.jsonl")
    val_out = os.path.join(args.out_dir, "val.jsonl")

    if train_examples:
        write_jsonl(train_out, train_examples)
    if val_examples:
        write_jsonl(val_out, val_examples)

    if not train_examples and not val_examples:
        print("[LLM] No examples were written. Check your indexes and paths.")


if __name__ == "__main__":
    main()
