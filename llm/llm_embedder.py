#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class LLMEmbedder:
    """Qwen2.5 wrapper that scores violence risk in [0, 1] from a scene summary."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device

        print(f"[LLM] Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )

        self.system_prompt = (
            "You analyze human interactions in video surveillance scenes and "
            "estimate the likelihood that physical violence will occur soon. "
            "Respond with ONLY a number between 0.0 and 1.0 with no extra text."
        )

    def build_prompt(self, summary: str) -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"Scene Description:\n{summary}\n\n"
            f"Violence likelihood (0.0–1.0):"
        )

    def _extract_number(self, text: str) -> float:
        nums = re.findall(r"\d+\.\d+|\d+", text)
        if not nums:
            return 0.0
        try:
            value = float(nums[-1])
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, value))

    def predict_risk(self, summary: str) -> float:
        prompt = self.build_prompt(summary)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        score = self._extract_number(full_text)
        return score
