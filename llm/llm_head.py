#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Optional


class LLMFusionHead(nn.Module):
    """
    Fusion module that combines the base model's risk score
    with an LLM-derived semantic risk score.

    Typical usage:
        # base_logits: (B,) or (B, 1) from ViolenceTransformer
        # llm_scores:  (B,) or float from LLMEmbedder
        fusion = LLMFusionHead(mode="linear", detach_llm=True)

        fused_prob = fusion(base_logits, llm_scores)

    Args:
        mode: "weighted" or "linear"
            - "weighted": simple convex combination of base + llm
            - "linear":   small learnable layer on [base_prob, llm_score]
        base_weight: default weight for base model in "weighted" mode
        detach_llm: if True, LLM scores are treated as constants
                    (no gradients flow through them)
        eps: numeric epsilon for safety/clamping
    """

    def __init__(
        self,
        mode: str = "linear",
        base_weight: float = 0.7,
        detach_llm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert mode in ("weighted", "linear"), "mode must be 'weighted' or 'linear'"

        self.mode = mode
        self.base_weight = base_weight
        self.detach_llm = detach_llm
        self.eps = eps

        if mode == "linear":
            # Input: [base_prob, llm_score] → output logit
            self.linear = nn.Linear(2, 1)
            # Initialize to roughly emulate base_weight / (1-base_weight)
            with torch.no_grad():
                w = torch.tensor([base_weight, 1.0 - base_weight]).view(1, 2)
                self.linear.weight.copy_(w)
                self.linear.bias.zero_()

    def _to_column(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor shape is (B, 1)
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x

    def forward(
        self,
        base_logits: torch.Tensor,
        llm_scores: torch.Tensor | float,
    ) -> torch.Tensor:
        """
        Args:
            base_logits: raw logits from your transformer, shape (B,) or (B,1)
            llm_scores:  LLM risk scores in [0,1], shape (B,), (B,1), or scalar

        Returns:
            fused_prob: probabilities in [0,1], shape (B,1)
        """
        # Convert base logits to probability
        base_logits = self._to_column(base_logits)
        base_prob = torch.sigmoid(base_logits)

        # Convert LLM scores to tensor
        if isinstance(llm_scores, float) or isinstance(llm_scores, int):
            llm_scores = torch.tensor([llm_scores], device=base_prob.device, dtype=base_prob.dtype)
        elif isinstance(llm_scores, torch.Tensor):
            llm_scores = llm_scores.to(device=base_prob.device, dtype=base_prob.dtype)
        else:
            raise TypeError("llm_scores must be float, int, or torch.Tensor")

        llm_scores = self._to_column(llm_scores)

        if self.detach_llm and llm_scores.requires_grad:
            llm_scores = llm_scores.detach()

        # Clamp LLM score just in case
        llm_scores = torch.clamp(llm_scores, 0.0 + self.eps, 1.0 - self.eps)

        if self.mode == "weighted":
            # Simple convex combination in probability space
            w = self.base_weight
            fused = w * base_prob + (1.0 - w) * llm_scores
            fused = torch.clamp(fused, 0.0 + self.eps, 1.0 - self.eps)
            return fused

        # "linear" mode
        x = torch.cat([base_prob, llm_scores], dim=-1)  # (B, 2)
        logits = self.linear(x)  # (B, 1)
        fused_prob = torch.sigmoid(logits)
        fused_prob = torch.clamp(fused_prob, 0.0 + self.eps, 1.0 - self.eps)
        return fused_prob


def fuse_scores_weighted(
    base_prob: torch.Tensor,
    llm_score: torch.Tensor | float,
    base_weight: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Stateless helper for quick fusion in eval / debugging.

    Args:
        base_prob: probability from transformer, (B,) or (B,1)
        llm_score: semantic risk from LLM, (B,), (B,1) or scalar
        base_weight: convex weight for base model
    """
    if isinstance(llm_score, float) or isinstance(llm_score, int):
        llm_score = torch.tensor([llm_score], device=base_prob.device, dtype=base_prob.dtype)

    if base_prob.dim() == 1:
        base_prob = base_prob.unsqueeze(-1)
    if isinstance(llm_score, torch.Tensor) and llm_score.dim() == 1:
        llm_score = llm_score.unsqueeze(-1)

    llm_score = llm_score.to(base_prob.device, base_prob.dtype)

    fused = base_weight * base_prob + (1.0 - base_weight) * llm_score
    return torch.clamp(fused, 0.0 + eps, 1.0 - eps)
