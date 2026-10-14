import torch
from core.model import ViolenceTransformer


def export():
    print("[Export] Loading model...")
    model = ViolenceTransformer().cuda().eval()

    seq_dummy = [{
        "nodes": torch.randn(5, 770).cuda(),
        "edges": torch.tensor([[0, 1, 0.1]], dtype=torch.float32).cuda(),
        "audio": torch.randn(128).cuda(),
        "scene": torch.randn(512).cuda()
    } for _ in range(30)]

    print("[Export] Running torch.onnx.export...")
    torch.onnx.export(
        model,
        (seq_dummy,),
        "violence_transformer.onnx",
        opset_version=16,
        input_names=["graph_sequence"],
        output_names=["violence_risk"],
        dynamic_axes={"graph_sequence": {0: "batch", 1: "frames"}},
    )

    print("[Export] Model saved to violence_transformer.onnx")


if __name__ == "__main__":
    export()
