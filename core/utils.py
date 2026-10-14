import json
import os
import numpy as np
import torch

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

def get_device():
    gpu = os.environ.get("VA_GPU", "1")  # default to GPU 0
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")