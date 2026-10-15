"""Finite-context causal GRU/Transformer models with censored hazard losses."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset



def context_window(video, tick, context, normalizer):
    lo = max(0, tick - context + 1)
    pad = context - (tick - lo + 1)
    present = np.asarray(video.present[lo:tick + 1]).copy()
    xs = [normalizer.transform(a[lo:tick + 1], present[:, j], j) for j, a in enumerate(video.streams)]
    if pad:
        xs = [np.pad(x, ((pad, 0), (0, 0)), mode="edge") for x in xs]
        present = np.pad(present, ((pad, 0), (0, 0)), mode="edge")
    return xs, present.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, store, selection, normalizer, context, position_control=False):
        self.store, self.normalizer, self.context = store, normalizer, context
        self.position_control = position_control
        self.samples = [(vid, int(i)) for vid, ix in selection.items() for i in ix]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vid, tick = self.samples[index]
        v = self.store[vid]
        x, present = context_window(v, tick, self.context, self.normalizer)
        return {"streams": x, "present": present, "hazard": v.labels.hazard_target[tick].astype(np.float32),
                "now": np.float32(v.labels.in_violence[tick]), "now_valid": np.float32(not v.labels.excluded[tick]),
                "weight": np.float32(np.clip(30 / max(v.labels.tte[tick], 30), 0.25, 1) if np.isfinite(v.labels.tte[tick]) else 1),
                "position": np.float32(v.t_sec[tick] / 300 if self.position_control else 0)}


class CausalTemporalModel(nn.Module):
    def __init__(self, dims, kind="gru", hidden=64, layers=2, heads=4, context=120,
                 modality_dropout=0.15, dropout=0.1, position_control=False):
        super().__init__()
        if kind not in ("gru", "transformer") or not 60 <= context <= 300:
            raise ValueError("kind must be gru/transformer and context must be 60..300 seconds")
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.kind, self.context, self.modality_dropout = kind, context, modality_dropout
        self.position_control = position_control
        # LayerNorm on a single scalar is constant, so scalar streams project before normalization.
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()) if d == 1 else
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU()) for d in dims])
        self.fuse = nn.Linear(len(dims) * (hidden + 1), hidden)
        if kind == "gru":
            self.encoder = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        else:
            block = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        pos = torch.arange(context).float()[:, None]
        frequency = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000) / hidden))
        encoding = torch.zeros(context, hidden)
        encoding[:, 0::2] = torch.sin(pos * frequency)
        encoding[:, 1::2] = torch.cos(pos * frequency[:encoding[:, 1::2].shape[1]])
        self.register_buffer("lag_encoding", encoding)
        self.hazard_head = nn.Linear(hidden + int(position_control), 6)
        self.now_head = nn.Linear(hidden + int(position_control), 1)

    def forward(self, streams, present, position=None):
        if present.shape[1] > self.context:
            raise ValueError("input exceeds the configured finite context")
        masks = present
        if self.training and self.modality_dropout:
            keep = (torch.rand(present.shape[0], 1, present.shape[2], device=present.device) >= self.modality_dropout)
            masks = present * keep
        parts = [torch.cat([proj(x) * masks[:, :, j:j + 1], masks[:, :, j:j + 1]], dim=-1)
                 for j, (proj, x) in enumerate(zip(self.projections, streams))]
        x = self.fuse(torch.cat(parts, dim=-1))
        if self.kind == "gru":
            x, _ = self.encoder(x)
        else:
            n = x.shape[1]
            x = x + self.lag_encoding[-n:]
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            x = self.encoder(x, mask=mask)
        last = x[:, -1]
        if self.position_control:
            if position is None:
                raise ValueError("position control requires elapsed time")
            last = torch.cat([last, position[:, None]], dim=-1)
        return self.hazard_head(last), self.now_head(last).squeeze(-1)

    @staticmethod
    def score30(hazard_logits):
        return -torch.expm1(F.logsigmoid(-hazard_logits[:, :4]).sum(dim=-1))


def hazard_loss(logits, target, weight):
    valid = target >= 0
    # Only positive event-bin terms far from onset are down-weighted; survival terms retain unit weight.
    weights = torch.where(target > 0, weight[:, None], 1) * valid
    loss = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0), reduction="none")
    return (loss * weights).sum() / valid.sum().clamp_min(1)


def batch_loss(model, batch, device, auxiliary_weight):
    xs = [x.to(device) for x in batch["streams"]]
    h, now = model(xs, batch["present"].to(device), batch["position"].to(device))
    loss = hazard_loss(h, batch["hazard"].to(device), batch["weight"].to(device))
    valid = batch["now_valid"].to(device)
    detection = F.binary_cross_entropy_with_logits(now, batch["now"].to(device), reduction="none")
    return loss + auxiliary_weight * (detection * valid).sum() / valid.sum().clamp_min(1)


def fit_temporal(model, train, val, device="cpu", epochs=20, patience=4, batch_size=32,
                 lr=1e-3, auxiliary_weight=0.2, seed=0, check_time=lambda: None, selection_score=None):
    if not len(train) or not len(val):
        raise ValueError("temporal training and validation need usable ticks")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, stale, history, best_state = -float("inf"), 0, [], None
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            check_time()
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device, auxiliary_weight)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            n = len(batch["now"])
            total, count = total + float(loss.detach()) * n, count + n
        model.eval()
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                check_time()
                loss = batch_loss(model, batch, device, auxiliary_weight)
                n = len(batch["now"])
                vtotal, vcount = vtotal + float(loss) * n, vcount + n
        value = vtotal / vcount
        metric = selection_score(model) if selection_score is not None else None
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val_loss": value,
                        "val_matched_far_auroc": metric})
        if metric is None or metric > best + 1e-6:
            best, stale, model.best_epoch = metric if metric is not None else best, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if metric is None or stale >= patience:
            break
    if best_state is None:
        raise ValueError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    return history


def predict_temporal(model, video, normalizer, device="cpu", batch_size=32, indices=None):
    model.eval()
    out = []
    with torch.no_grad():
        indices = np.arange(len(video.t_sec)) if indices is None else np.asarray(indices, int)
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start + batch_size]
            windows = [context_window(video, i, model.context, normalizer) for i in ix]
            xs = [torch.from_numpy(np.stack([w[0][j] for w in windows])).to(device)
                  for j in range(len(video.streams))]
            present = torch.from_numpy(np.stack([w[1] for w in windows])).to(device)
            pos = torch.tensor(video.t_sec[list(ix)] / 300, dtype=torch.float32, device=device)
            h, _ = model(xs, present, pos)
            out.append(model.score30(h).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)
