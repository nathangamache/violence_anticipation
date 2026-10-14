import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, Union


GraphNodeArray = np.ndarray  # (N, D)
AdjacencyArray = np.ndarray  # (N, N)
SceneArray = np.ndarray      # (D,)
AudioArray = np.ndarray      # (A,)


def _as_scene_getter(scene: np.ndarray):
    """Return scene_at(t): a 1-D scene is shared by all frames, a 2-D scene is indexed per frame (clamped)."""
    scene = np.asarray(scene, dtype=np.float32)

    if scene.ndim == 1:
        def scene_at(_t: int) -> SceneArray:
            return scene
        return scene_at

    if scene.ndim == 2:
        T = scene.shape[0]

        def scene_at(t: int) -> SceneArray:
            idx = min(max(int(t), 0), T - 1)
            return scene[idx]
        return scene_at

    def scene_at(_t: int) -> SceneArray:
        return np.zeros((0,), dtype=np.float32)

    return scene_at


def _as_actions_for_frame(
    actions: Any,
    frame_idx: int,
) -> Dict[int, np.ndarray]:
    """Return {track_id: feature} for one frame from a dict or list/array of per-frame dicts (0-d object arrays unwrapped)."""
    if actions is None:
        return {}

    if isinstance(actions, dict):
        frame_actions = actions.get(frame_idx, {})
        return frame_actions if isinstance(frame_actions, dict) else {}

    if isinstance(actions, (list, np.ndarray)):
        if frame_idx < len(actions):
            fa = actions[frame_idx]
            if isinstance(fa, np.ndarray) and fa.dtype == object and fa.shape == ():
                fa = fa.item()
            return fa if isinstance(fa, dict) else {}

    return {}


def _bbox_to_feature(bbox: Sequence[float]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Pad/trim bbox to 4 values; return it and its first two values (top-left for xyxy) as the distance position."""
    arr = np.asarray(bbox, dtype=np.float32).flatten()

    if arr.shape[0] < 4:
        arr = np.pad(arr, (0, 4 - arr.shape[0]))
    elif arr.shape[0] > 4:
        arr = arr[:4]

    cx, cy = float(arr[0]), float(arr[1])
    return arr, (cx, cy)


def _build_adjacency(centers: List[Tuple[float, float]]) -> AdjacencyArray:
    if len(centers) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    centers_arr = np.asarray(centers, dtype=np.float32)  # (N, 2)
    diff = centers_arr[:, None, :] - centers_arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)
    adj = 1.0 / (1.0 + dist)
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def build_graph_sequence(
    tracks: List[List[Dict[str, Any]]],
    actions: Any,
    scene: np.ndarray,
    audio: np.ndarray,
    max_nodes: int = 32,
) -> List[Dict[str, Any]]:
    """Per-frame graph dicts {frame_idx, nodes (N, 4 + action dim), adj (N, N), scene, audio}; empty frames are skipped."""
    scene_at = _as_scene_getter(scene)
    audio_vec = np.asarray(audio, dtype=np.float32).flatten()

    graphs: List[Dict[str, Any]] = []

    if isinstance(tracks, list) and len(tracks) > 0 and isinstance(tracks[0], list):
        frame_iter = enumerate(tracks)
    elif isinstance(tracks, list) and len(tracks) > 0 and isinstance(tracks[0], dict):
        frame_iter = (
            (frame.get("frame_idx", idx), frame.get("tracks", []))
            for idx, frame in enumerate(tracks)
        )
    else:
        raise TypeError(
            f"Unsupported 'tracks' structure in build_graph_sequence: {type(tracks)}"
        )

    for frame_idx, frame_tracks in frame_iter:
        if not frame_tracks:
            continue

        frame_actions = _as_actions_for_frame(actions, int(frame_idx))

        node_features: List[np.ndarray] = []
        centers: List[Tuple[float, float]] = []

        for person in frame_tracks:
            if not isinstance(person, dict):
                continue
            bbox = person.get("bbox", None)
            if bbox is None:
                continue

            track_id = person.get("track_id", 0)
            bbox_feat, center = _bbox_to_feature(bbox)

            act_vec = frame_actions.get(track_id, None)
            if act_vec is None:
                # Missing action vector: the node is bbox-only (4-d); the dataset zero-pads it later.
                act_arr = np.zeros((0,), dtype=np.float32)
            else:
                act_arr = np.asarray(act_vec, dtype=np.float32).flatten()

            node_feat = np.concatenate([bbox_feat, act_arr], axis=0)
            node_features.append(node_feat)
            centers.append(center)

        if not node_features:
            continue

        nodes = np.stack(node_features, axis=0).astype(np.float32)
        centers_list = centers

        if nodes.shape[0] > max_nodes:
            nodes = nodes[:max_nodes]
            centers_list = centers_list[:max_nodes]

        adj = _build_adjacency(centers_list)

        graphs.append(
            {
                "frame_idx": int(frame_idx),
                "nodes": nodes,
                "adj": adj,
                "scene": scene_at(int(frame_idx)),
                "audio": audio_vec,
            }
        )

    return graphs


def graph_to_text(
    graph_seq: List[Dict[str, Any]],
    max_frames: int = 8,
    max_people: int = 5,
) -> str:
    """Plain-text summary of the last ``max_frames`` graphs for the LLM head."""
    if not graph_seq:
        return "No people or interactions detected in the recent frames."

    window = graph_seq[-max_frames:]

    lines: List[str] = []
    for g in window:
        frame_idx = g.get("frame_idx", 0)
        nodes = g.get("nodes", None)
        if nodes is None or nodes.shape[0] == 0:
            lines.append(f"Frame {frame_idx}: no visible people.")
            continue

        num_people = nodes.shape[0]
        lines.append(
            f"Frame {frame_idx}: about {num_people} people visible, "
            f"with several pairwise distances indicating possible interactions."
        )

    return " ".join(lines)
