import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, Union


GraphNodeArray = np.ndarray  # (N, D)
AdjacencyArray = np.ndarray  # (N, N)
SceneArray = np.ndarray      # (D,)
AudioArray = np.ndarray      # (A,)


def _as_scene_getter(scene: np.ndarray):
    """
    Normalize scene representation so we can do scene_at(t).
    - If scene is 1D → same vector for all frames.
    - If scene is 2D → per-frame row, clamp index to last frame.
    """
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

    # Fallback: zero vector
    def scene_at(_t: int) -> SceneArray:
        return np.zeros((0,), dtype=np.float32)

    return scene_at


def _as_actions_for_frame(
    actions: Any,
    frame_idx: int,
) -> Dict[int, np.ndarray]:
    """
    Normalize the actions container into a mapping:
        {track_id -> feature_vector} for the given frame.

    Handles:
    - dict[int -> dict[track_id -> feat]]
    - list/np.ndarray where actions[frame_idx] is that dict
    - scalar object arrays from np.save / np.load
    """
    if actions is None:
        return {}

    # If it is already a dict-of-dicts keyed by frame index
    if isinstance(actions, dict):
        frame_actions = actions.get(frame_idx, {})
        return frame_actions if isinstance(frame_actions, dict) else {}

    # If it's a list or numpy array of per-frame dicts
    if isinstance(actions, (list, np.ndarray)):
        if frame_idx < len(actions):
            fa = actions[frame_idx]
            # np.object array element → unwrap if scalar
            if isinstance(fa, np.ndarray) and fa.dtype == object and fa.shape == ():
                fa = fa.item()
            return fa if isinstance(fa, dict) else {}

    return {}


def _bbox_to_feature(bbox: Sequence[float]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert a bbox to a feature vector + a center coordinate for distance.
    We assume bbox is [x1, y1, x2, y2] or already normalized;
    we don't enforce semantics here, just treat first 2 dims as "center-ish".
    """
    arr = np.asarray(bbox, dtype=np.float32).flatten()

    if arr.shape[0] < 4:
        arr = np.pad(arr, (0, 4 - arr.shape[0]))
    elif arr.shape[0] > 4:
        arr = arr[:4]

    # Use first two components as "center" proxy (good enough for distances)
    cx, cy = float(arr[0]), float(arr[1])
    return arr, (cx, cy)


def _build_adjacency(centers: List[Tuple[float, float]]) -> AdjacencyArray:
    """
    Build a simple adjacency matrix based on pairwise distances between centers.
    Edge weight = 1 / (1 + distance); diagonal set to 0.
    """
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
    """
    Build a per-frame graph sequence from:
      - tracks: list over frames, each element is a list of {track_id, bbox}.
      - actions: dict or list/np.ndarray giving per-frame {track_id -> action_vec}.
      - scene: 1D (global) or 2D (per-frame) scene embedding.
      - audio: 1D audio embedding (global for the clip).

    Returns:
        List of dicts, one per frame, each containing:
            {
              "frame_idx": int,
              "nodes": (N, D) float32 array,
              "adj":   (N, N) float32 array,
              "scene": (S,)   float32 array,
              "audio": (A,)   float32 array,
            }
    """
    # Scene getter (handles global vs per-frame)
    scene_at = _as_scene_getter(scene)
    audio_vec = np.asarray(audio, dtype=np.float32).flatten()

    graphs: List[Dict[str, Any]] = []

    # --- Case 1: tracks is list-of-frames, each frame is list-of-person dicts ---
    if isinstance(tracks, list) and len(tracks) > 0 and isinstance(tracks[0], list):
        frame_iter = enumerate(tracks)
    # --- Case 2: older style (list of dicts with "tracks" key) ---
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
        # frame_tracks is expected to be a list of person dicts
        if not frame_tracks:
            # No people in this frame → we can choose to skip it
            # or create an empty graph. Skipping is usually cleaner.
            continue

        # Get actions map for this frame
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

            # Get action embedding if available for this track
            act_vec = frame_actions.get(track_id, None)
            if act_vec is None:
                # Fallback: no action feature → just zeros of some default dim
                # We don't know the true dim here, so we use 0-length and rely
                # on later code to handle pure bbox features if needed.
                act_arr = np.zeros((0,), dtype=np.float32)
            else:
                act_arr = np.asarray(act_vec, dtype=np.float32).flatten()

            node_feat = np.concatenate([bbox_feat, act_arr], axis=0)
            node_features.append(node_feat)
            centers.append(center)

        if not node_features:
            # All persons in this frame were invalid → skip
            continue

        # Stack node features
        nodes = np.stack(node_features, axis=0).astype(np.float32)
        centers_list = centers

        # Limit max_nodes if desired (for very crowded scenes)
        if nodes.shape[0] > max_nodes:
            nodes = nodes[:max_nodes]
            centers_list = centers_list[:max_nodes]

        # Build adjacency
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


# ---------------------------------------------------------------------------
# Optional: text summarizer for LLM head (graph_to_text)
# ---------------------------------------------------------------------------

def graph_to_text(
    graph_seq: List[Dict[str, Any]],
    max_frames: int = 8,
    max_people: int = 5,
) -> str:
    """
    Convert a short window of graphs into a simple text description
    for the LLM head. This does not need to be fancy; just structured.

    Assumes graph_seq is a list of graph dicts produced by build_graph_sequence.
    """
    if not graph_seq:
        return "No people or interactions detected in the recent frames."

    # Use only the last max_frames graphs
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
