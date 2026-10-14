#!/usr/bin/env python3
import numpy as np

from core.graph import build_graph_sequence, graph_to_text


def test_build_graph_sequence_basic():
    # Fake scene/audio embeddings
    scene = np.zeros((32,), dtype=np.float32)
    audio = np.zeros((16,), dtype=np.float32)

    # Fake tracks: two frames, two people
    tracks = [
        {
            "frame_idx": 0,
            "persons": [
                {"track_id": 1, "bbox": [0, 0, 10, 10]},
                {"track_id": 2, "bbox": [100, 100, 110, 110]},
            ],
        },
        {
            "frame_idx": 1,
            "persons": [
                {"track_id": 1, "bbox": [5, 0, 15, 10]},   # moved right
                {"track_id": 2, "bbox": [100, 105, 110, 115]},  # moved down
            ],
        },
    ]

    # Fake actions: only some frames/people have logits
    actions = {
        0: {
            1: {
                "logits": np.array([0.1, 0.9], dtype=np.float32),
                "names": ["standing", "punching"],
            },
        },
        1: {
            2: {
                "logits": np.array([0.8, 0.2], dtype=np.float32),
                "names": ["walking", "kicking"],
            },
        },
    }

    graph_seq = build_graph_sequence(tracks, actions, scene, audio)

    # Basic structure
    assert isinstance(graph_seq, np.ndarray)
    assert graph_seq.dtype == object
    assert len(graph_seq) == 2

    # Each frame should be a dict with required keys
    for frame in graph_seq:
        assert isinstance(frame, dict)
        assert "nodes" in frame
        assert "scene" in frame
        assert "audio" in frame
        assert "timestamp" in frame
        assert "scene_text" in frame

        # scene/audio should be same objects we passed in (or copies)
        assert frame["scene"].shape == scene.shape
        assert frame["audio"].shape == audio.shape

    # Check node structure
    frame0 = graph_seq[0]
    nodes0 = frame0["nodes"]
    assert len(nodes0) == 2

    n0 = nodes0[0]
    assert "track_id" in n0
    assert "bbox" in n0
    assert "position" in n0
    assert "velocity" in n0
    assert "action_logits" in n0
    assert "action_names" in n0

    # Velocity on first frame should be zero (no previous position)
    vx0, vy0 = n0["velocity"]
    assert vx0 == 0.0
    assert vy0 == 0.0

    # On second frame, track 1 should have non-zero vx
    frame1 = graph_seq[1]
    nodes1 = frame1["nodes"]
    node1_track1 = [n for n in nodes1 if n["track_id"] == 1][0]
    vx1, vy1 = node1_track1["velocity"]
    assert vx1 != 0.0 or vy1 != 0.0


def test_graph_to_text_basic():
    # Build a minimal fake graph_frame
    graph_frame = {
        "nodes": [
            {
                "track_id": 1,
                "bbox": [0, 0, 10, 10],
                "position": [5.0, 5.0],
                "velocity": [1.0, 0.0],
                "action_logits": np.array([0.2, 0.8], dtype=np.float32),
                "action_names": ["standing", "punching"],
            },
            {
                "track_id": 2,
                "bbox": [100, 100, 110, 110],
                "position": [105.0, 105.0],
                "velocity": [0.0, -1.0],
                "action_logits": np.array([0.9, 0.1], dtype=np.float32),
                "action_names": ["walking", "kicking"],
            },
        ],
        "scene": np.zeros((32,), dtype=np.float32),
        "audio": np.zeros((16,), dtype=np.float32),
        "timestamp": 12.34,
        "scene_text": "parking lot at night",
    }

    summary = graph_to_text(graph_frame)
    assert isinstance(summary, str)
    assert "Time: 12.34" in summary
    assert "Person 1" in summary
    assert "Person 2" in summary
    assert "parking lot at night" in summary

    # Empty nodes → special case
    empty_frame = {
        "nodes": [],
        "scene": None,
        "audio": None,
        "timestamp": 5.0,
        "scene_text": None,
    }
    summary_empty = graph_to_text(empty_frame)
    assert "no people detected" in summary_empty.lower()
