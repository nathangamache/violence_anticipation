#!/usr/bin/env python3
import torch
import torch.nn as nn


class ViolenceTransformer(nn.Module):
    """Temporal transformer over frame dicts with "nodes" [N, node_dim], "scene" [scene_dim], "audio" [audio_dim].

    forward returns [T] logits, or sigmoid probabilities when return_logits=False.
    """

    def __init__(
        self,
        node_dim: int = 772,
        scene_dim: int = 512,
        audio_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.scene_dim = scene_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(self, frames, return_logits: bool = True):
        device = next(self.parameters()).device

        if not frames:
            dummy = torch.zeros(self.hidden_dim, device=device)
            seq = dummy.unsqueeze(0).unsqueeze(0)
        else:
            frame_vecs = []
            for f in frames:
                nodes = f["nodes"].to(device)
                if nodes.dim() == 1:
                    nodes = nodes.unsqueeze(0)

                x_nodes = self.node_proj(nodes)
                node_agg = x_nodes.mean(dim=0)

                scene = f["scene"].to(device)
                if scene.dim() > 1:
                    scene = scene.view(-1)
                scene_emb = self.scene_proj(scene)

                audio = f["audio"].to(device)
                if audio.dim() > 1:
                    audio = audio.view(-1)
                audio_emb = self.audio_proj(audio)

                frame_vecs.append(node_agg + scene_emb + audio_emb)

            seq = torch.stack(frame_vecs, dim=0).unsqueeze(0)  # [1, T, H]

        _, T, _ = seq.shape
        if T > self.max_len:
            seq = seq[:, : self.max_len, :]
            T = self.max_len

        pos_idx = torch.arange(T, device=device)
        pe = self.pos_embed(pos_idx)
        seq = seq + pe.unsqueeze(0)

        seq_out = self.temporal_encoder(seq)
        logits = self.out_head(seq_out).squeeze(-1).squeeze(0)  # [T]

        if return_logits:
            return logits
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_proba(self, frames):
        return self.forward(frames, return_logits=False)
