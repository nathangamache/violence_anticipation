"""Causal temporal models with extended hazards, a band head and event-level supervision."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from anticipation.models.objectives import TRAIN_HAZARD_EDGES, band_targets, event_bags, matched_weights


def context_window(video, tick, context, normalizer, history_length=None, return_history=False):
    length = context if history_length is None else min(context, history_length)
    lo = max(0, tick - length + 1)
    pad = context - (tick - lo + 1)
    present = np.asarray(video.present[lo:tick + 1]).copy()
    xs = [normalizer.transform(a[lo:tick + 1], present[:, j], j) for j, a in enumerate(video.streams)]
    history = np.r_[np.zeros(pad, bool), np.ones(tick - lo + 1, bool)]
    if pad:
        xs = [np.pad(x, ((pad, 0), (0, 0))) for x in xs]
        present = np.pad(present, ((pad, 0), (0, 0)))
    result = xs, present.astype(np.float32)
    return (*result, history) if return_history else result


class WindowDataset(Dataset):
    def __init__(self, store, selection, normalizer, context, position_control=False,
                 objective="horizon30", far_positive_downweight=False, history_dropout=0,
                 time_match_band=False):
        self.store, self.normalizer, self.context = store, normalizer, context
        self.position_control, self.objective = position_control, objective
        self.far_positive_downweight, self.history_dropout = far_positive_downweight, history_dropout
        self.samples = [(vid, int(i)) for vid, ix in selection.items() for i in ix]
        self.targets, self.band_weights = {}, {}
        times, sources, targets, keys = [], [], [], []
        for vid, ix in selection.items():
            v = store[vid]
            target, valid = band_targets(v, store.catalog.loc[vid, "role"])
            self.targets[vid] = target, valid
            for tick in ix[valid[ix]]:
                keys.append((vid, int(tick)))
                times.append(v.t_sec[tick])
                sources.append(str(store.catalog.loc[vid, "source_kind"]))
                targets.append(target[tick])
        weights = matched_weights(times, sources, targets) if time_match_band else np.ones(len(keys))
        self.band_weights = dict(zip(keys, weights))

    def __len__(self):
        return len(self.samples)

    def window(self, vid, tick):
        v = self.store[vid]
        length = None
        if self.history_dropout and np.random.random() < self.history_dropout:
            length = int(np.random.randint(1, min(self.context, tick + 1) + 1))
        x, present, history = context_window(v, tick, self.context, self.normalizer, length, True)
        return {"streams": x, "present": present, "history": history,
                "position": np.float32(v.t_sec[tick] / 300 if self.position_control else 0)}

    def __getitem__(self, index):
        vid, tick = self.samples[index]
        v = self.store[vid]
        out = self.window(vid, tick)
        weight = np.clip(30 / max(v.labels.tte[tick], 30), 0.25, 1) if self.far_positive_downweight and np.isfinite(v.labels.tte[tick]) else 1
        out.update(hazard=v.labels.hazard_target[tick].astype(np.float32), now=np.float32(v.labels.in_violence[tick]),
                   now_valid=np.float32(not v.labels.excluded[tick]), weight=np.float32(weight),
                   band=self.targets[vid][0][tick], band_weight=np.float32(self.band_weights.get((vid, tick), 0)))
        return out


class EventBagDataset(Dataset):
    def __init__(self, windows, ids, max_ticks=16, seed=0):
        self.windows, self.max_ticks = windows, max_ticks
        self.bags, self.info = event_bags(windows.store, ids, max_ticks, seed)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        vid, ticks, target = self.bags[index]
        mask = np.arange(self.max_ticks) < len(ticks)
        padded = np.pad(ticks, (0, self.max_ticks - len(ticks)), mode="edge")
        items = [self.windows.window(vid, int(tick)) for tick in padded]
        return {"streams": [np.stack([x["streams"][j] for x in items]) for j in range(len(items[0]["streams"]))],
                "present": np.stack([x["present"] for x in items]), "history": np.stack([x["history"] for x in items]),
                "position": np.array([x["position"] for x in items]), "bag_mask": mask, "target": np.float32(target)}


class CausalTemporalModel(nn.Module):
    def __init__(self, dims, kind="gru", hidden=64, layers=2, heads=4, context=180,
                 modality_dropout=0.15, dropout=0.1, position_control=False,
                 hazard_edges=TRAIN_HAZARD_EDGES, score_kind="hazard30"):
        super().__init__()
        if kind not in ("gru", "transformer") or not 60 <= context <= 300:
            raise ValueError("kind must be gru/transformer and context must be 60..300 seconds")
        if hidden < 2 or heads < 1 or hidden % heads:
            raise ValueError("hidden must be >= 2 and divisible by positive heads")
        self.kind, self.context, self.modality_dropout = kind, context, modality_dropout
        self.position_control, self.hazard_edges, self.score_kind = position_control, tuple(hazard_edges), score_kind
        # Scalar inputs project before LayerNorm to retain a nonconstant signal.
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()) if d == 1 else
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU()) for d in dims])
        self.fuse = nn.Linear(len(dims) * (hidden + 1), hidden)
        self.heads = heads
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
        self.hazard_head = nn.Linear(hidden + int(position_control), len(hazard_edges) - 1)
        self.now_head = nn.Linear(hidden + int(position_control), 1)
        self.band_head = nn.Linear(hidden + int(position_control), 1)

    def forward(self, streams, present, position=None, history=None, return_band=False):
        if present.shape[1] > self.context:
            raise ValueError("input exceeds the configured finite context")
        masks = present
        if self.training and self.modality_dropout:
            masks = present * (torch.rand(present.shape[0], 1, present.shape[2], device=present.device) >= self.modality_dropout)
        parts = [torch.cat([proj(x) * masks[:, :, j:j + 1], masks[:, :, j:j + 1]], dim=-1)
                 for j, (proj, x) in enumerate(zip(self.projections, streams))]
        x = self.fuse(torch.cat(parts, dim=-1))
        if self.kind == "gru":
            if history is None:
                x, _ = self.encoder(x)
                last = x[:, -1]
            else:
                lengths = history.sum(dim=1).cpu()
                seq = nn.utils.rnn.pad_sequence([row[valid] for row, valid in zip(x, history)], batch_first=True)
                packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
                _, state = self.encoder(packed)
                last = state[-1]
        else:
            n = x.shape[1]
            x = x + self.lag_encoding[-n:]
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            if history is not None:
                mask = mask[None] | ~history[:, None, :]
                # Padding queries attend to themselves, avoiding all-masked softmax rows.
                mask[:, torch.arange(n, device=x.device), torch.arange(n, device=x.device)] = False
                mask = mask.repeat_interleave(self.heads, dim=0)
            last = self.encoder(x, mask=mask)[:, -1]
        if self.position_control:
            if position is None:
                raise ValueError("position control requires elapsed time")
            last = torch.cat([last, position[:, None]], dim=-1)
        out = self.hazard_head(last), self.now_head(last).squeeze(-1)
        return (*out, self.band_head(last).squeeze(-1)) if return_band else out

    @staticmethod
    def score30(hazard_logits):
        return -torch.expm1(F.logsigmoid(-hazard_logits[:, :4]).sum(dim=-1))

    def hazard_band(self, logits):
        i30, i120 = self.hazard_edges.index(30.), self.hazard_edges.index(120.)
        log_survival = F.logsigmoid(-logits).cumsum(dim=-1)
        return torch.exp(log_survival[:, i30 - 1]) - torch.exp(log_survival[:, i120 - 1])


def hazard_loss(logits, target, weight):
    valid = target >= 0
    weights = torch.where(target > 0, weight[:, None], 1) * valid
    loss = F.binary_cross_entropy_with_logits(logits, target.clamp_min(0), reduction="none")
    return (loss * weights).sum() / valid.sum().clamp_min(1)


def band_loss(logits, target, weight):
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1)


def event_band_loss(logits, mask, target, temperature=0.5):
    pooled = temperature * (torch.logsumexp((logits / temperature).masked_fill(~mask, -torch.inf), dim=1) - mask.sum(dim=1).log())
    return F.binary_cross_entropy_with_logits(pooled, target)


def batch_loss(model, batch, device, auxiliary_weight, band_weight=0):
    h, now, band = model([x.to(device) for x in batch["streams"]], batch["present"].to(device),
                         batch["position"].to(device), batch["history"].to(device), True)
    loss = hazard_loss(h, batch["hazard"].to(device), batch["weight"].to(device))
    valid = batch["now_valid"].to(device)
    detection = F.binary_cross_entropy_with_logits(now, batch["now"].to(device), reduction="none")
    loss = loss + auxiliary_weight * (detection * valid).sum() / valid.sum().clamp_min(1)
    return loss + band_weight * band_loss(band, batch["band"].to(device), batch["band_weight"].to(device))


def bag_loss(model, batch, device):
    xs = [x.flatten(0, 1).to(device) for x in batch["streams"]]
    _, _, band = model(xs, batch["present"].flatten(0, 1).to(device), batch["position"].flatten().to(device),
                       batch["history"].flatten(0, 1).to(device), True)
    return event_band_loss(band.reshape(batch["bag_mask"].shape), batch["bag_mask"].to(device), batch["target"].to(device))


def fit_temporal(model, train, val, device="cpu", epochs=20, patience=4, batch_size=32,
                 lr=1e-3, auxiliary_weight=0.2, seed=0, check_time=lambda: None, selection_score=None,
                 band_weight=1., event_weight=1., event_dataset=None, event_batch_size=2):
    if not len(train) or not len(val):
        raise ValueError("temporal training and validation need usable ticks")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=0)
    bags = DataLoader(event_dataset, batch_size=event_batch_size, shuffle=True, generator=generator, num_workers=0) if event_dataset is not None and len(event_dataset) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, stale, history, best_state = -float("inf"), 0, [], None
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        bag_iter = iter(bags) if bags is not None else None
        for batch in train_loader:
            check_time()
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device, auxiliary_weight, band_weight)
            if bag_iter is not None and event_weight:
                event_batch = next(bag_iter, None)
                if event_batch is None:
                    bag_iter = iter(bags)
                    event_batch = next(bag_iter)
                loss = loss + event_weight * bag_loss(model, event_batch, device)
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
                value = batch_loss(model, batch, device, auxiliary_weight, band_weight)
                n = len(batch["now"])
                vtotal, vcount = vtotal + float(value) * n, vcount + n
        metric = selection_score(model) if selection_score is not None else None
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val_loss": vtotal / vcount,
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
            windows = [context_window(video, i, model.context, normalizer, return_history=True) for i in ix]
            xs = [torch.from_numpy(np.stack([w[0][j] for w in windows])).to(device) for j in range(len(video.streams))]
            present = torch.from_numpy(np.stack([w[1] for w in windows])).to(device)
            history = torch.from_numpy(np.stack([w[2] for w in windows])).to(device)
            pos = torch.tensor(video.t_sec[ix] / 300, dtype=torch.float32, device=device)
            h, _, band = model(xs, present, pos, history, True)
            score = torch.sigmoid(band) if model.score_kind == "band" else (model.hazard_band(h) if model.score_kind == "hazard_band" else model.score30(h))
            out.append(score.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)
