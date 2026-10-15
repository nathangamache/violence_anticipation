"""Infer the LSTM architecture of a .pt/.pth checkpoint from its weight shapes.

Usage: python inspect_checkpoint.py /path/to/best_temporal_model.pt
"""

import sys
import torch


def infer_architecture(state_dict: dict):
    print("\n── Weight shapes ──────────────────────────────────")
    for k, v in state_dict.items():
        print(f"  {k:55s} {tuple(v.shape)}")

    print("\n── Inferred architecture ──────────────────────────")
    try:
        if "frame_encoder.0.weight" in state_dict:
            fe_key = "frame_encoder.0.weight"
        elif "feature_extractor.net.0.weight" in state_dict:
            fe_key = "feature_extractor.net.0.weight"
        else:
            raise KeyError("frame_encoder.0.weight or feature_extractor.net.0.weight")

        fe_in  = state_dict[fe_key].shape[1]
        fe_out = state_dict[fe_key].shape[0]
        print(f"  input_dim   (VLM embedding):        {fe_in}")
        print(f"  encoder_dim (frame encoder out):    {fe_out}")

        # LSTM — weight_ih_l0 shape = (4*hidden, input_size)
        lstm_4h = state_dict["lstm.weight_ih_l0"].shape[0]
        lstm_h  = lstm_4h // 4
        print(f"  lstm_hidden:                        {lstm_h}")

        bidirectional = "lstm.weight_ih_l0_reverse" in state_dict
        print(f"  bidirectional:                      {bidirectional}")
        lstm_out_dim = lstm_h * 2 if bidirectional else lstm_h
        print(f"  lstm output dim:                    {lstm_out_dim}")

        num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l") and "reverse" not in k)
        print(f"  num_layers:                         {num_layers}")

        has_attention = any(k.startswith("attention") for k in state_dict)
        print(f"  attention:                          {has_attention}")

        print("\n── Suggested constructor args ─────────────────────")
        print(f"  ViolenceLSTM(")
        print(f"      input_dim={fe_in},")
        print(f"      encoder_dim={fe_out},")
        print(f"      lstm_hidden={lstm_h},")
        print(f"      num_layers={num_layers},")
        print(f"  )")

    except KeyError as e:
        print(f"  Could not auto-infer: missing key {e}")
        print("  Check the weight shapes above and set dimensions manually.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint.pt>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            print("Format: wrapped dict (model_state_dict key found)")
            print(f"Other keys: {[k for k in ckpt if k != 'model_state_dict']}")
            state_dict = ckpt["model_state_dict"]
        else:
            print("Format: raw state dict")
            state_dict = ckpt
    else:
        print(f"Unexpected checkpoint type: {type(ckpt)}")
        sys.exit(1)

    infer_architecture(state_dict)


if __name__ == "__main__":
    main()