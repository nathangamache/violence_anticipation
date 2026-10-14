import time
import cv2
import numpy as np
import torch

from core.extractors import (
    live_track,
    live_actions,
    live_scene,
    live_audio
)
from core.model import ViolenceTransformer
from realtime.pipeline import build_live_graph


def benchmark(video_path="sample.mp4", frames=50):
    print("[Benchmark] Initializing models...")
    model = ViolenceTransformer().cuda().eval()

    cap = cv2.VideoCapture(video_path)
    extractor_times = []
    model_times = []

    print(f"[Benchmark] Running on first {frames} frames...")

    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        tracks = live_track(frame)
        actions = live_actions(frame, tracks)
        audio = live_audio()
        scene = live_scene(frame)

        graph = build_live_graph(tracks, actions, audio, scene)

        t1 = time.time()

        seq_cuda = [{
            "nodes": torch.tensor(graph["nodes"]).cuda(),
            "edges": torch.tensor(graph["edges"]).cuda(),
            "audio": torch.tensor(graph["audio"]).cuda(),
            "scene": torch.tensor(graph["scene"]).cuda()
        } for _ in range(30)]

        with torch.no_grad():
            model(seq_cuda)

        t2 = time.time()

        extractor_times.append(t1 - t0)
        model_times.append(t2 - t1)

    cap.release()

    ex_mean = np.mean(extractor_times)
    mo_mean = np.mean(model_times)

    print("\n[Benchmark Results]")
    print(f"Extractor FPS: {1/ex_mean:.2f}")
    print(f"Model FPS:     {1/mo_mean:.2f}")
    print(f"Total FPS:     {1/(ex_mean + mo_mean):.2f}")

    return {
        "extractor_fps": float(1/ex_mean),
        "model_fps": float(1/mo_mean),
        "total_fps": float(1/(ex_mean + mo_mean))
    }


if __name__ == "__main__":
    benchmark()
