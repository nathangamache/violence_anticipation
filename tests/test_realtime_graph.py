import numpy as np
import torch
from realtime.pipeline import build_live_graph


def test_live_graph():
    tracks = [
        {"track_id": 1, "bbox": [0, 0, 10, 10]},
        {"track_id": 2, "bbox": [10, 10, 20, 20]}
    ]

    actions = {
        1: np.zeros(768),
        2: np.zeros(768)
    }

    audio = np.zeros(128, dtype=np.float32)
    scene = np.zeros(512, dtype=np.float32)

    graph = build_live_graph(tracks, actions, audio, scene)

    assert graph["nodes"].shape == (2, 770)
    assert graph["edges"].shape[1] == 3
    assert graph["audio"].shape == (128,)
    assert graph["scene"].shape == (512,)

    print("[PASS] test_live_graph")
