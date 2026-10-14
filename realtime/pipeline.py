#!/usr/bin/env python3
import cv2
import torch

from core.extractors import (
    live_track,
    live_actions,
    live_audio,
    live_scene,
)
from core.graph import graph_to_text
from llm.llm_embedder import LLMEmbedder
from llm.llm_head import LLMFusionHead
from core.model import ViolenceTransformer


# ----------------- Base model -----------------
model = ViolenceTransformer().cuda()
model.eval()

# ----------------- LLM + fusion setup -----------------
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

llm = LLMEmbedder(
    model_name=LLM_MODEL_NAME,
    device="cuda",
)

fusion_head = LLMFusionHead(
    mode="linear",      # or "weighted"
    base_weight=0.7,
    detach_llm=True,
)
fusion_head.eval()


def build_live_graph(tracks, actions, audio, scene):
    """
    Build a single live graph_frame matching the offline structure enough
    for graph_to_text() and LLM use.

    tracks: list of dicts {track_id, bbox}
    actions: dict track_id -> embedding (np.ndarray) or similar
    audio: np.ndarray or None
    scene: np.ndarray or None
    """
    nodes = []

    for t in tracks:
        tid = int(t.get("track_id"))
        bbox = t.get("bbox", None)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        node = {
            "track_id": tid,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "position": [float(cx), float(cy)],
            "velocity": [0.0, 0.0],      # no temporal velocity yet in live mode
            "action_logits": None,
            "action_names": [],
        }

        # If live_actions produced embeddings, attach them as "logits"
        if isinstance(actions, dict) and tid in actions:
            node["action_logits"] = actions[tid]
            node["action_names"] = []   # no label space yet

        nodes.append(node)

    graph = {
        "nodes": nodes,
        "scene": scene,
        "audio": audio,
        "timestamp": None,      # you can fill this if you track elapsed time
        "scene_text": None,     # optional: add CLIP→text label later
    }

    return graph


def run_realtime(video_source=0, use_llm=True):
    """
    Real-time pipeline:
      - Reads frames from webcam/RTSP/file
      - Runs detection, tracking, actions, scene
      - Builds a sliding buffer of per-frame graphs (unused by base model here,
        because we kept your original model(buffer) API assumption very simple)
      - Base model outputs a risk score
      - Optional LLM runs on the latest frame summary
      - Fusion head combines base + LLM risk
    """
    cap = cv2.VideoCapture(video_source)
    buffer = []

    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {video_source}")
        return

    print("[Realtime] Starting stream...")
    print(f"[Realtime] LLM enabled: {use_llm}  (model={LLM_MODEL_NAME})")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Realtime] Stream ended or cannot read frame.")
            break

        # 1) Per-frame feature extraction
        tracks = live_track(frame)
        actions = live_actions(frame, tracks)
        audio   = live_audio()
        scene   = live_scene(frame)

        # 2) Build graph frame & append to buffer
        graph = build_live_graph(tracks, actions, audio, scene)
        buffer.append(graph)

        # Keep a sliding context window (~30s @ 4 FPS)
        if len(buffer) > 120:
            buffer.pop(0)

        # 3) Only start predicting once we have some context
        if len(buffer) > 30:
            with torch.no_grad():
                # --- Base model risk (assumes model(buffer) → scalar prob or logit) ---
                base_risk = model(buffer).item()

                if use_llm:
                    # --- LLM semantic risk on latest frame ---
                    latest_graph = buffer[-1]
                    summary = graph_to_text(latest_graph)
                    llm_score = llm.predict_risk(summary)  # float in [0,1]

                    base_tensor = torch.tensor([base_risk], dtype=torch.float32)
                    llm_tensor  = torch.tensor([llm_score], dtype=torch.float32)

                    fused_prob = fusion_head(base_tensor, llm_tensor)
                    fused_value = fused_prob.item()

                    print(
                        f"[Realtime] base={base_risk:.3f}  "
                        f"llm={llm_score:.3f}  fused={fused_value:.3f}"
                    )
                else:
                    print(f"[Realtime] base={base_risk:.3f}")

        cv2.imshow("Violence Anticipation", frame)
        if cv2.waitKey(1) == 27:  # ESC
            print("[Realtime] Stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
