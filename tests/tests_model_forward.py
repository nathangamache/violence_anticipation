import torch
from core.model import ViolenceTransformer


def test_model_forward():
    model = ViolenceTransformer().cuda()

    seq = []
    for _ in range(30):
        seq.append({
            "nodes": torch.randn(5, 770).cuda(),
            "edges": torch.tensor([[0, 1, 0.5]], dtype=torch.float32).cuda(),
            "audio": torch.randn(128).cuda(),
            "scene": torch.randn(512).cuda()
        })

    out = model(seq)
    assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"
    assert 0 <= out.item() <= 1, "Output should be a probability in [0, 1]"

    print("[PASS] test_model_forward")
