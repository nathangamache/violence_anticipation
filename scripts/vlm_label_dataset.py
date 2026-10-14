#!/usr/bin/env python3
"""
Generate VLM labels and embeddings with optional temporal context.

This script processes video frames through a VLM (Qwen2.5-VL) to generate
structured violence detection labels and embeddings. It supports three
context modes for temporal awareness:

- none: Process each frame independently (original behavior)
- text: Include text summary of previous frame(s) in the prompt
- visual: Include previous frame image(s) as multi-frame input
- both: Combine text and visual context

Violence-focused prompt modes for better embeddings:
- violence_embedding: Best for training violence detection models
- violence_anticipation: Focuses on pre-violence indicators
- violence_detailed: Most comprehensive analysis
- violence_minimal: Fast but still violence-focused

NEW: Anticipation-specific prompt modes for pre-violence detection:
- anticipation_multicue: Granular scoring of 31 individual pre-violence cues
- anticipation_simple: 7 key scores + stage classification
- anticipation_behavioral: Body language and social dynamics focus
- scene_context: Environment baseline analysis (run once per video)

Usage:
    # Original behavior (no context, default prompt)
    python vlm_label_dataset.py --graphs_dir data/graphs --context_mode none

    # Violence-focused embeddings (RECOMMENDED for violence detection)
    python vlm_label_dataset.py --graphs_dir data/graphs --prompt_mode violence_embedding

    # NEW: Anticipation multi-cue scoring (RECOMMENDED for anticipation task)
    python vlm_label_dataset.py --graphs_dir data/graphs --prompt_mode anticipation_multicue

    # NEW: Simple anticipation with stage classification
    python vlm_label_dataset.py --graphs_dir data/graphs --prompt_mode anticipation_simple

    # With text context
    python vlm_label_dataset.py --graphs_dir data/graphs --context_mode text

    # Multi-GPU sharding
    CUDA_VISIBLE_DEVICES=0 python vlm_label_dataset.py --rank 0 --world_size 2 ...
    CUDA_VISIBLE_DEVICES=1 python vlm_label_dataset.py --rank 1 --world_size 2 ...
"""

import argparse
import json
import os
import time
import atexit
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter
import re

import numpy as np
import cv2
from PIL import Image

from vlm.qwen_teacher import QwenTeacher, QwenTeacherConfig, multicue_to_vector, simple_to_vector


# ==============================================================================
# Lock File System for Multi-GPU Coordination
# ==============================================================================

class VideoLockManager:
    """
    Manages lock files to prevent multiple GPUs from processing the same video.
    
    Lock files are created in a .locks directory and contain the GPU rank and timestamp.
    Stale locks (older than timeout) are automatically cleaned up.
    """
    
    def __init__(self, lock_dir: str, rank: int = 0, stale_timeout: float = 3600):
        """
        Args:
            lock_dir: Directory to store lock files
            rank: GPU rank/identifier for this process
            stale_timeout: Seconds after which a lock is considered stale (default 1 hour)
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.stale_timeout = stale_timeout
        self.held_locks: List[Path] = []
        
        # Register cleanup on exit
        atexit.register(self.release_all_locks)
    
    def _lock_path(self, vid_id: str) -> Path:
        """Get the lock file path for a video."""
        return self.lock_dir / f"{vid_id}.lock"
    
    def _is_stale(self, lock_path: Path) -> bool:
        """Check if a lock file is stale (too old)."""
        try:
            mtime = lock_path.stat().st_mtime
            age = time.time() - mtime
            return age > self.stale_timeout
        except Exception:
            return True
    
    def try_acquire(self, vid_id: str) -> bool:
        """
        Try to acquire a lock for a video.
        
        Returns:
            True if lock acquired, False if video is already locked by another process
        """
        lock_path = self._lock_path(vid_id)
        
        # Check if lock exists
        if lock_path.exists():
            # Check if it's stale
            if self._is_stale(lock_path):
                print(f"    [Lock] Removing stale lock for {vid_id}", flush=True)
                try:
                    lock_path.unlink()
                except Exception:
                    pass
            else:
                # Lock is held by another process
                try:
                    with open(lock_path, 'r') as f:
                        lock_info = f.read().strip()
                    print(f"    [Lock] {vid_id} is locked by {lock_info}", flush=True)
                except Exception:
                    print(f"    [Lock] {vid_id} is locked (couldn't read info)", flush=True)
                return False
        
        # Try to create lock file
        try:
            # Use exclusive create to avoid race condition
            with open(lock_path, 'x') as f:
                f.write(f"rank={self.rank},time={time.time():.0f}")
            self.held_locks.append(lock_path)
            print(f"    [Lock] Acquired lock for {vid_id}", flush=True)
            return True
        except FileExistsError:
            # Another process created the lock between our check and create
            print(f"    [Lock] {vid_id} was just locked by another process", flush=True)
            return False
        except Exception as e:
            print(f"    [Lock] Failed to acquire lock for {vid_id}: {e}", flush=True)
            return False
    
    def release(self, vid_id: str):
        """Release a lock for a video."""
        lock_path = self._lock_path(vid_id)
        try:
            if lock_path.exists():
                lock_path.unlink()
                if lock_path in self.held_locks:
                    self.held_locks.remove(lock_path)
                print(f"    [Lock] Released lock for {vid_id}", flush=True)
        except Exception as e:
            print(f"    [Lock] Failed to release lock for {vid_id}: {e}", flush=True)
    
    def release_all_locks(self):
        """Release all locks held by this process (called on exit)."""
        for lock_path in list(self.held_locks):
            try:
                if lock_path.exists():
                    lock_path.unlink()
                    print(f"    [Lock] Cleanup: released {lock_path.stem}", flush=True)
            except Exception:
                pass
        self.held_locks.clear()
    
    def is_locked(self, vid_id: str) -> bool:
        """Check if a video is currently locked."""
        lock_path = self._lock_path(vid_id)
        if not lock_path.exists():
            return False
        if self._is_stale(lock_path):
            return False
        return True


def load_metadata(meta_dir: str) -> Dict[str, dict]:
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta.update(json.load(f))
    return meta


def build_video_index(root: str) -> Dict[str, str]:
    """
    Map video "stem" -> absolute path. Adds alias for *_x264.
    """
    out = {}
    rp = Path(root)
    if not rp.exists():
        return out

    for p in rp.rglob("*.mp4"):
        stem = p.stem
        out[stem] = str(p)
        out[stem.lower()] = str(p)

        if stem.endswith("_x264"):
            base = stem[:-5]
            out[base] = str(p)
            out[base.lower()] = str(p)

    return out


def resolve_video_path(vid_id: str, meta: dict, idx_kin: dict, idx_ucf: dict, idx_xd: dict) -> Optional[str]:
    source = (meta or {}).get("source", None)
    candidates = [vid_id, vid_id.lower()]

    # Sometimes kinetics metadata uses youtube_id as actual file stem
    youtube_id = (meta or {}).get("youtube_id", None)
    if youtube_id:
        candidates += [youtube_id, str(youtube_id).lower()]

    if source == "kinetics":
        for c in candidates:
            if c in idx_kin:
                return idx_kin[c]
    elif source == "ucf_crime":
        for c in candidates:
            if c in idx_ucf:
                return idx_ucf[c]
    elif source == "xd_violence":
        for c in candidates:
            if c in idx_xd:
                return idx_xd[c]

    # fallback: try all
    for c in candidates:
        if c in idx_ucf:
            return idx_ucf[c]
        if c in idx_xd:
            return idx_xd[c]
        if c in idx_kin:
            return idx_kin[c]

    return None


def read_frame_pil(cap: cv2.VideoCapture, frame_idx: int) -> Optional[Image.Image]:
    if frame_idx < 0:
        frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def safe_total_frames(cap: cv2.VideoCapture, meta: dict) -> int:
    # Prefer metadata total_frames if present
    if meta and "total_frames" in meta:
        try:
            tf = int(meta["total_frames"])
            if tf > 0:
                return tf
        except Exception:
            pass
    # fallback to cv2
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return max(tf, 0)


def existing_t_set(jsonl_path: str) -> set:
    if not os.path.exists(jsonl_path):
        return set()
    s = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                t = r.get("t", None)
                if t is not None:
                    s.add(int(t))
            except Exception:
                continue
    return s


def load_previous_context_from_jsonl(jsonl_path: str, target_t: int, window_size: int = 1) -> List[dict]:
    """
    Load previous frame context from existing JSONL for resumption.
    
    Returns list of parsed records for frames immediately before target_t.
    """
    if not os.path.exists(jsonl_path):
        return []
    
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "error" not in r and r.get("t") is not None:
                    records.append(r)
            except Exception:
                continue
    
    # Sort by t and find frames before target_t
    records.sort(key=lambda x: x.get("t", -1))
    
    previous = []
    for r in records:
        t = r.get("t", -1)
        if t < target_t:
            previous.append(r)
        else:
            break
    
    # Return last `window_size` frames
    return previous[-window_size:] if previous else []


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def print_planned_stats(planned, metadata):
    c_src = Counter()
    c_pos = Counter()
    c_xd_suffix = Counter()

    for vid_id, source in planned:
        c_src[source] += 1
        info = metadata.get(vid_id, {})
        if source == "xd_violence" and int(info.get("label", 0)) == 0:
            continue
        if int(info.get("label", 0)) == 1:
            c_pos[source] += 1
        if source == "xd_violence":
            if "_label_" in vid_id:
                suf = vid_id.split("_label_")[-1]
                c_xd_suffix[suf] += 1

    print("[Plan] counts per source:", dict(c_src))
    print("[Plan] positive labels per source:", dict(c_pos))
    if c_xd_suffix:
        print("[Plan] xd label suffix:", dict(c_xd_suffix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", type=str, required=True)
    ap.add_argument("--meta_dir", type=str, default="data/metadata")
    ap.add_argument("--out_jsonl_root", type=str, default="data/processed/vlm_labels")
    ap.add_argument("--out_npz_root", type=str, default="data/processed/vlm_emb")
    ap.add_argument("--out_features_root", type=str, default="data/processed/vlm_features",
                    help="Output directory for structured feature vectors (anticipation modes)")

    ap.add_argument("--kinetics_root", type=str, default="data/raw_videos/kinetics")
    ap.add_argument("--ucf_root", type=str, default="data/raw_videos/ucf_crime")
    ap.add_argument("--xd_root", type=str, default="data/raw_videos/xd_violence")

    ap.add_argument("--every_t", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip videos that already have .jsonl and .npz files")
    ap.add_argument("--verify_complete", action="store_true",
                    help="When used with --skip_existing, verify that existing files are complete "
                         "(have expected number of embeddings). Re-processes incomplete files.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete and regenerate all files")
    ap.add_argument("--lock_dir", type=str, default=".locks",
                    help="Directory for lock files to coordinate multi-GPU processing")
    ap.add_argument("--lock_timeout", type=float, default=3600,
                    help="Seconds after which a lock is considered stale (default: 3600 = 1 hour)")

    # sharding for 2 GPUs
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)

    # teacher config
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=512,  # Increased for multicue output
                    help="Max tokens for VLM generation (512 recommended for anticipation_multicue)")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    
    # Context/memory settings
    ap.add_argument("--context_mode", type=str, default="none",
                    choices=["none", "text", "visual", "both"],
                    help="Context mode: none=independent frames, text=text summary of previous, "
                         "visual=include previous frame image, both=text+visual")
    ap.add_argument("--context_window", type=int, default=1,
                    help="Number of previous frames to include in context (1-3 recommended)")
    ap.add_argument("--store_context_images", action="store_true",
                    help="Store PIL images in memory for visual context (uses more RAM)")
    
    # Violence-focused prompt mode for better embeddings
    ap.add_argument("--prompt_mode", type=str, default="default",
                    choices=["default", "violence_embedding", "violence_anticipation", 
                             "violence_detailed", "violence_minimal", "violence_temporal",
                             "anticipation_multicue", "anticipation_simple", 
                             "anticipation_behavioral", "scene_context"],
                    help="Prompt mode for embedding extraction. "
                         "Use 'violence_embedding' for violence detection. "
                         "Use 'anticipation_multicue' or 'anticipation_simple' for anticipation task.")
    
    # Feature extraction options
    ap.add_argument("--save_features", action="store_true",
                    help="Save structured feature vectors for anticipation modes (in addition to embeddings)")
    
    args = ap.parse_args()

    # Determine if we should save structured features
    save_features = args.save_features or args.prompt_mode in ("anticipation_multicue", "anticipation_simple")
    
    meta = load_metadata(args.meta_dir)
    if not meta:
        raise RuntimeError(f"No metadata loaded from {args.meta_dir}")

    # Build list of vids from graphs_dir (most honest source of "what exists")
    graph_files = []
    for fn in os.listdir(args.graphs_dir):
        if fn.endswith(".npy") or fn.endswith(".pt"):
            graph_files.append(fn)
    vid_ids = sorted([os.path.splitext(f)[0] for f in graph_files])

    # Shard across GPUs/processes
    vid_ids = vid_ids[args.rank::args.world_size]
    print(f"[Index] graphs={len(graph_files)} shard vids={len(vid_ids)} rank={args.rank}/{args.world_size}")
    print(f"[Config] context_mode={args.context_mode} context_window={args.context_window} prompt_mode={args.prompt_mode}")
    if save_features:
        print(f"[Config] Saving structured features to {args.out_features_root}")

    # Build video indexes (one-time pain)
    print("[Index] Scanning video roots (one-time pain)...", flush=True)
    idx_kin = build_video_index(args.kinetics_root)
    idx_ucf = build_video_index(args.ucf_root)
    idx_xd  = build_video_index(args.xd_root)
    print(f"[Index] kinetics={len(idx_kin)//2} ucf={len(idx_ucf)//2} xd={len(idx_xd)//2}", flush=True)

    # Initialize lock manager for multi-GPU coordination
    lock_manager = VideoLockManager(
        lock_dir=args.lock_dir,
        rank=args.rank,
        stale_timeout=args.lock_timeout,
    )
    print(f"[Lock] Lock directory: {args.lock_dir}, rank={args.rank}, timeout={args.lock_timeout}s", flush=True)

    # Initialize teacher with context settings AND prompt_mode
    teacher = QwenTeacher(QwenTeacherConfig(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_impl,
        use_fast_processor=False,
        # Context settings
        context_mode=args.context_mode,
        context_window_size=args.context_window,
        store_context_images=args.store_context_images or args.context_mode in ("visual", "both"),
        # Violence-focused prompt mode
        prompt_mode=args.prompt_mode,
    ))

    total_vids = len(vid_ids)
    processed_count = 0
    skipped_count = 0
    locked_count = 0
    error_count = 0

    for vid_idx, vid_id in enumerate(vid_ids):
        info = meta.get(vid_id, None)
        source = (info or {}).get("source", "unknown")

        out_jsonl_dir = os.path.join(args.out_jsonl_root, source)
        out_npz_dir   = os.path.join(args.out_npz_root, source)
        out_feat_dir  = os.path.join(args.out_features_root, source) if save_features else None
        ensure_dir(out_jsonl_dir)
        ensure_dir(out_npz_dir)
        if out_feat_dir:
            ensure_dir(out_feat_dir)

        out_jsonl = os.path.join(out_jsonl_dir, f"{vid_id}.jsonl")
        out_npz   = os.path.join(out_npz_dir, f"{vid_id}.npz")
        out_feat  = os.path.join(out_feat_dir, f"{vid_id}.npz") if out_feat_dir else None

        if args.overwrite:
            if os.path.exists(out_jsonl):
                os.remove(out_jsonl)
            if os.path.exists(out_npz):
                os.remove(out_npz)
            if out_feat and os.path.exists(out_feat):
                os.remove(out_feat)
            print(f"[{vid_idx+1}/{total_vids}] Overwriting {vid_id}...", flush=True)

        # Check if we should skip this video
        if args.skip_existing and os.path.exists(out_jsonl) and os.path.exists(out_npz):
            should_skip = True
            
            # If verify_complete is set, check that the file has expected number of embeddings
            if args.verify_complete:
                try:
                    # First we need to know T (expected timesteps)
                    graph_path_npy = os.path.join(args.graphs_dir, f"{vid_id}.npy")
                    graph_path_pt  = os.path.join(args.graphs_dir, f"{vid_id}.pt")
                    gpath_check = graph_path_npy if os.path.exists(graph_path_npy) else graph_path_pt
                    
                    if os.path.exists(gpath_check):
                        if gpath_check.endswith(".npy"):
                            seq_check = np.load(gpath_check, allow_pickle=True)
                            T_check = len(seq_check)
                        else:
                            import torch
                            seq_check = torch.load(gpath_check)
                            T_check = len(seq_check)
                        
                        expected_count = len(list(range(0, T_check, max(1, args.every_t))))
                        
                        # Check actual embedding count
                        z = np.load(out_npz)
                        actual_count = len(z["t"])
                        
                        if actual_count < expected_count:
                            should_skip = False
                            print(f"[{vid_idx+1}/{total_vids}] Incomplete: {vid_id} ({actual_count}/{expected_count} embeddings) - reprocessing", flush=True)
                            # Delete incomplete files
                            os.remove(out_jsonl)
                            os.remove(out_npz)
                            if out_feat and os.path.exists(out_feat):
                                os.remove(out_feat)
                except Exception as e:
                    # If verification fails, reprocess to be safe
                    should_skip = False
                    print(f"[{vid_idx+1}/{total_vids}] Verification failed for {vid_id}: {e} - reprocessing", flush=True)
                    if os.path.exists(out_jsonl):
                        os.remove(out_jsonl)
                    if os.path.exists(out_npz):
                        os.remove(out_npz)
                    if out_feat and os.path.exists(out_feat):
                        os.remove(out_feat)
            
            if should_skip:
                skipped_count += 1
                print(f"[{vid_idx+1}/{total_vids}] Skipping {vid_id} (already exists)", flush=True)
                continue

        # Try to acquire lock for this video
        if not lock_manager.try_acquire(vid_id):
            locked_count += 1
            print(f"[{vid_idx+1}/{total_vids}] Skipping {vid_id} (locked by another process)", flush=True)
            continue

        graph_path_npy = os.path.join(args.graphs_dir, f"{vid_id}.npy")
        graph_path_pt  = os.path.join(args.graphs_dir, f"{vid_id}.pt")
        gpath = graph_path_npy if os.path.exists(graph_path_npy) else graph_path_pt
        if not os.path.exists(gpath):
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Skipping {vid_id} (no graph file)", flush=True)
            continue

        # Determine T from graph sequence length
        try:
            if gpath.endswith(".npy"):
                seq = np.load(gpath, allow_pickle=True)
                T = len(seq)
            else:
                import torch
                seq = torch.load(gpath)
                T = len(seq)
        except Exception as e:
            error_count += 1
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Error loading graph for {vid_id}: {e}", flush=True)
            continue

        if T <= 0:
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Skipping {vid_id} (T=0)", flush=True)
            continue

        vpath = resolve_video_path(vid_id, info, idx_kin, idx_ucf, idx_xd)
        if not vpath or not os.path.exists(vpath):
            error_count += 1
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Error: Missing video for {vid_id}", flush=True)
            rec = {"vid_id": vid_id, "source": source, "error": "missing_video"}
            with open(out_jsonl, "a", buffering=1) as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            error_count += 1
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Error: Cannot open video {vid_id}", flush=True)
            rec = {"vid_id": vid_id, "source": source, "error": "video_open_failed", "video_path": vpath}
            with open(out_jsonl, "a", buffering=1) as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            continue

        total_frames = safe_total_frames(cap, info)
        if total_frames <= 0:
            cap.release()
            error_count += 1
            lock_manager.release(vid_id)
            print(f"[{vid_idx+1}/{total_vids}] Error: Bad frame count for {vid_id}", flush=True)
            rec = {"vid_id": vid_id, "source": source, "error": "bad_total_frames", "video_path": vpath}
            with open(out_jsonl, "a", buffering=1) as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            continue

        # Starting processing
        print(f"[{vid_idx+1}/{total_vids}] Processing {vid_id} ({source}, T={T}, frames={total_frames})...", flush=True)

        done_t = existing_t_set(out_jsonl) if args.skip_existing and os.path.exists(out_jsonl) else set()

        # per-video embedding buffers
        ts: List[int] = []
        raw_idxs: List[int] = []
        embs: List[np.ndarray] = []
        features: List[np.ndarray] = []  # For structured features

        # If resuming and npz exists, load existing buffers
        if args.skip_existing and os.path.exists(out_npz):
            try:
                z = np.load(out_npz)
                ts = list(map(int, z["t"].tolist()))
                raw_idxs = list(map(int, z["raw_frame_idx"].tolist()))
                embs = [e for e in z["emb"]]
            except Exception:
                ts, raw_idxs, embs = [], [], []
                done_t = set()
        
        # Load existing features if resuming
        if save_features and args.skip_existing and out_feat and os.path.exists(out_feat):
            try:
                zf = np.load(out_feat)
                features = [f for f in zf["features"]]
            except Exception:
                features = []

        existing_emb_t = set(ts)

        # IMPORTANT: Reset context at the start of each video
        teacher.reset_context()

        # sample timesteps
        t_steps = list(range(0, T, max(1, args.every_t)))
        
        # If resuming mid-video with context mode, we need to rebuild context
        # from previously processed frames
        if args.context_mode != "none" and done_t:
            first_todo_t = min(t for t in t_steps if t not in done_t) if any(t not in done_t for t in t_steps) else None
            if first_todo_t is not None:
                # Load previous context from JSONL
                prev_records = load_previous_context_from_jsonl(out_jsonl, first_todo_t, args.context_window)
                for rec in prev_records:
                    # Rebuild context (without images - they weren't stored)
                    from vlm.qwen_teacher import FrameContext
                    ctx = FrameContext(
                        t=rec.get("t", 0),
                        raw_frame_idx=rec.get("raw_frame_idx", 0),
                        parsed={
                            "violent": rec.get("violent", {}),
                            "people_count": rec.get("people_count"),
                            "actions": rec.get("actions", []),
                            "weapons": rec.get("weapons", []),
                            "scene": rec.get("scene", []),
                            "temporal_change": rec.get("temporal_change"),
                            "notes": rec.get("notes", ""),
                        },
                        image=None,
                    )
                    teacher.context.add(ctx)
                print(f"[Resume] Loaded {len(prev_records)} previous frame contexts for {vid_id}")

        for t in t_steps:
            if t in done_t and t in existing_emb_t:
                continue

            raw_frame_idx = int(round(((t + 0.5) / float(T)) * float(total_frames)))
            raw_frame_idx = max(0, min(total_frames - 1, raw_frame_idx))

            img = read_frame_pil(cap, raw_frame_idx)
            if img is None:
                rec = {"vid_id": vid_id, "source": source, "t": t, "T": T, "raw_frame_idx": raw_frame_idx,
                       "error": "frame_read_failed"}
                with open(out_jsonl, "a", buffering=1) as f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                continue

            # Use context-aware inference with temporal tracking
            # The prompt_mode is already configured in the teacher
            parsed, raw_text, emb = teacher.infer_one(
                img,
                return_embedding=True,
                t=t,
                raw_frame_idx=raw_frame_idx,
            )

            if parsed is None:
                rec = {"vid_id": vid_id, "source": source, "t": t, "T": T, "raw_frame_idx": raw_frame_idx,
                       "error": "parse_failed", "raw_model_output": raw_text}
                with open(out_jsonl, "a", buffering=1) as f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                continue

            # write JSONL record (includes new temporal_change field if using context)
            rec = {
                "vid_id": vid_id,
                "source": source,
                "t": int(t),
                "T": int(T),
                "raw_frame_idx": int(raw_frame_idx),
                "context_mode": args.context_mode,
                "prompt_mode": args.prompt_mode,  # Track which prompt was used
                **parsed,
            }
            with open(out_jsonl, "a", buffering=1) as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                f.flush()

            # append embedding
            if emb is not None and t not in existing_emb_t:
                ts.append(int(t))
                raw_idxs.append(int(raw_frame_idx))
                embs.append(emb.astype(np.float16))
                existing_emb_t.add(int(t))
                
                # Extract structured features for anticipation modes
                if save_features and parsed is not None:
                    if args.prompt_mode == "anticipation_multicue":
                        feat = multicue_to_vector(parsed)
                    elif args.prompt_mode == "anticipation_simple":
                        feat = simple_to_vector(parsed)
                    else:
                        # For other modes, create a minimal feature vector
                        feat = np.array([
                            parsed.get("overall_threat", parsed.get("violent", {}).get("score", 0.0)) or 0.0,
                        ], dtype=np.float32)
                    features.append(feat)

        cap.release()

        # save NPZ (sorted by t)
        if embs:
            order = np.argsort(np.asarray(ts, dtype=np.int32))
            t_arr = np.asarray(ts, dtype=np.int32)[order]
            r_arr = np.asarray(raw_idxs, dtype=np.int32)[order]
            e_arr = np.stack([embs[i] for i in order], axis=0)  # [N, D]

            np.savez_compressed(out_npz, t=t_arr, raw_frame_idx=r_arr, emb=e_arr)
            
            # Save structured features
            if save_features and features and out_feat:
                f_arr = np.stack([features[i] for i in order], axis=0)  # [N, F]
                np.savez_compressed(out_feat, t=t_arr, raw_frame_idx=r_arr, features=f_arr,
                                   prompt_mode=args.prompt_mode)
        
        # Release lock after successful completion
        lock_manager.release(vid_id)
        
        processed_count += 1
        feat_info = f", {len(features)} features" if save_features and features else ""
        print(f"[{vid_idx+1}/{total_vids}] Completed {vid_id} ({len(embs)} embeddings{feat_info} saved)", flush=True)

    print("\n" + "="*60, flush=True)
    print("[Done] Finished labeling+embeddings.", flush=True)
    print(f"  Processed: {processed_count}", flush=True)
    print(f"  Skipped (existing): {skipped_count}", flush=True)
    print(f"  Skipped (locked):   {locked_count}", flush=True)
    print(f"  Errors:    {error_count}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()