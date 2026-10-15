import os
import json
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


EDGE_DIM = 4  # last-dim of edge features we expect


class GraphSequenceDataset(Dataset):
    """Per-video graph sequences with frame-level risk labels.

    Items are (frames, labels[T]); each frame is a dict of float32 tensors: nodes [N, node_dim],
    edges [N, N, EDGE_DIM], scene [scene_dim], audio [audio_dim].
    """

    def __init__(self, index_json: str, min_frames: int = 16):
        if not os.path.exists(index_json):
            raise FileNotFoundError(index_json)

        with open(index_json, "r") as f:
            raw_items = json.load(f)

        self.items: List[Dict[str, Any]] = []
        self.node_dim: int = None
        self.scene_dim: int = None
        self.audio_dim: int = None

        for it in raw_items:
            gpath = it["graph_path"]
            lpath = it["label_path"]

            if not (os.path.exists(gpath) and os.path.exists(lpath)):
                continue

            try:
                graph_seq = np.load(gpath, allow_pickle=True)
                labels = np.load(lpath)

                if isinstance(graph_seq, np.ndarray):
                    frames_raw = list(graph_seq)
                else:
                    frames_raw = graph_seq

                if len(frames_raw) == 0 or len(labels) == 0:
                    continue

                frame0 = frames_raw[0]

                if self.node_dim is None:
                    self.node_dim = self._infer_node_dim(frame0)
                    self.scene_dim = self._infer_vec_dim(frame0.get("scene", None), default=512)
                    self.audio_dim = self._infer_vec_dim(frame0.get("audio", None), default=64)
                    print(f"[Dataset] Inferred dims: node={self.node_dim}, "
                          f"scene={self.scene_dim}, audio={self.audio_dim}")

                T = min(len(frames_raw), len(labels))
                if T < min_frames:
                    continue

                # validate frame0 only; failures drop the item via the except below
                _ = self._coerce_nodes(frame0.get("nodes", []))
                _edges = self._coerce_edges(frame0.get("edges", None), num_nodes=_.shape[0])
                _scene = self._coerce_vec(frame0.get("scene", None), self.scene_dim)
                _audio = self._coerce_vec(frame0.get("audio", None), self.audio_dim)

                it2 = dict(it)
                it2["length"] = int(T)
                self.items.append(it2)

            except Exception as e:
                continue

        if not self.items:
            raise RuntimeError("No valid items found in index")

        print(f"[Dataset] Final usable items: {len(self.items)}")

    def _infer_node_dim(self, frame: Dict[str, Any]) -> int:
        nodes_raw = frame.get("nodes", [])
        # nodes_raw: list of vectors, dict track_id -> vector, or empty
        if isinstance(nodes_raw, dict):
            vals = list(nodes_raw.values())
            if not vals:
                return 772  # 4 bbox coords + 768-d action vector
            sample = vals[0]
        else:
            if not nodes_raw:
                return 772
            sample = nodes_raw[0]

        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 0:
            return 772
        return int(arr.shape[-1])

    def _infer_vec_dim(self, vec_raw, default: int) -> int:
        if vec_raw is None:
            return default
        arr = np.asarray(vec_raw, dtype=np.float32)
        if arr.ndim == 0:
            return default
        return int(arr.shape[-1])

    def _coerce_nodes(self, nodes_raw) -> np.ndarray:
        """Turn various node formats into [N, node_dim] float32."""
        rows = []

        if isinstance(nodes_raw, dict):
            iterable = nodes_raw.values()
        else:
            iterable = nodes_raw

        for v in iterable:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 0:
                    continue
                rows.append(arr)
            except Exception:
                continue

        if len(rows) == 0:
            # No persons this frame → single dummy node
            return np.zeros((1, self.node_dim), dtype=np.float32)

        mat = np.stack(rows, axis=0)

        d = mat.shape[-1]
        if d == self.node_dim:
            return mat.astype(np.float32)

        if d > self.node_dim:
            return mat[..., : self.node_dim].astype(np.float32)
        else:
            pad_width = self.node_dim - d
            pad = np.zeros((mat.shape[0], pad_width), dtype=np.float32)
            return np.concatenate([mat, pad], axis=-1)

    def _coerce_edges(self, edges_raw, num_nodes: int) -> np.ndarray:
        """Return [N, N, EDGE_DIM] float32."""
        if edges_raw is None:
            return np.zeros((num_nodes, num_nodes, EDGE_DIM), dtype=np.float32)

        arr = np.asarray(edges_raw, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * EDGE_DIM, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] != EDGE_DIM:
            d = arr.shape[-1]
            if d > EDGE_DIM:
                arr = arr[..., :EDGE_DIM]
            else:
                pad = np.zeros(arr.shape[:-1] + (EDGE_DIM - d,), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=-1)

        if arr.shape[0] != num_nodes or arr.shape[1] != num_nodes:
            arr = np.zeros((num_nodes, num_nodes, EDGE_DIM), dtype=np.float32)

        return arr.astype(np.float32)

    def _coerce_vec(self, vec_raw, dim: int) -> np.ndarray:
        if vec_raw is None:
            return np.zeros((dim,), dtype=np.float32)
        arr = np.asarray(vec_raw, dtype=np.float32).flatten()
        d = arr.shape[0]
        if d == dim:
            return arr
        if d > dim:
            return arr[:dim]
        pad = np.zeros((dim - d,), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        gpath = entry["graph_path"]
        lpath = entry["label_path"]
        T = entry["length"]

        graph_seq = np.load(gpath, allow_pickle=True)
        labels = np.load(lpath).astype(np.float32)

        if isinstance(graph_seq, np.ndarray):
            frames_raw = list(graph_seq)
        else:
            frames_raw = graph_seq

        frames: List[Dict[str, torch.Tensor]] = []

        for t in range(T):
            frame = frames_raw[t]
            nodes_arr = self._coerce_nodes(frame.get("nodes", []))
            edges_arr = self._coerce_edges(frame.get("edges", None),
                                           num_nodes=nodes_arr.shape[0])
            scene_vec = self._coerce_vec(frame.get("scene", None), self.scene_dim)
            audio_vec = self._coerce_vec(frame.get("audio", None), self.audio_dim)

            frames.append(
                {
                    "nodes": torch.from_numpy(nodes_arr),
                    "edges": torch.from_numpy(edges_arr),
                    "scene": torch.from_numpy(scene_vec),
                    "audio": torch.from_numpy(audio_vec),
                }
            )

        labels_t = torch.from_numpy(labels[:T])
        return frames, labels_t


def infer_dims(dataset: GraphSequenceDataset):
    return dataset.node_dim, dataset.scene_dim, dataset.audio_dim
