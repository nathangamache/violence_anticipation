# vlm/qwen_teacher.py
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
except Exception:
    AutoProcessor = None
    AutoModelForVision2Seq = None


def _best_effort_json_parse(text: str) -> Optional[Dict[str, Any]]:
    s = text.strip()

    # prefer fenced ```json {..} ``` if present
    fence = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        blob = fence.group(1).strip()
        try:
            return json.loads(blob)
        except Exception:
            pass

    # otherwise, find first balanced {...}
    m = re.search(r"\{", s)
    if not m:
        return None

    start = m.start()
    depth = 0
    end = None
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None

    blob = s[start:end].strip()
    try:
        return json.loads(blob)
    except Exception:
        # tiny repairs
        repaired = blob.replace("'", '"')
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)
        try:
            return json.loads(repaired)
        except Exception:
            return None


@dataclass
class FrameContext:
    """Stores context from a previous frame analysis."""
    t: int
    raw_frame_idx: int
    parsed: Dict[str, Any]
    image: Optional[Image.Image] = None  # Optional: for multi-frame visual context
    
    def to_text_summary(self, max_len: int = 300) -> str:
        """Convert to a concise text summary for prompt injection."""
        parts = []
        
        # Violence assessment
        violent = self.parsed.get("violent", {})
        if isinstance(violent, dict):
            label = violent.get("label", "unknown")
            score = violent.get("score", "?")
            parts.append(f"violence={label}({score})")
        
        # People count
        pc = self.parsed.get("people_count")
        if pc is not None:
            parts.append(f"people={pc}")
        
        # Actions (truncate if needed)
        actions = self.parsed.get("actions", [])
        if actions:
            parts.append(f"actions=[{','.join(actions[:3])}]")
        
        # Weapons
        weapons = self.parsed.get("weapons", [])
        if weapons:
            parts.append(f"weapons=[{','.join(weapons[:3])}]")
        
        # Scene
        scene = self.parsed.get("scene", [])
        if scene:
            parts.append(f"scene=[{','.join(scene[:3])}]")
        
        summary = f"t={self.t}: " + ", ".join(parts)
        return summary[:max_len]


@dataclass
class ContextWindow:
    """
    Sliding window of recent frame contexts.
    Supports both text-based and multi-frame visual context.
    """
    max_frames: int = 3  # How many previous frames to remember
    frames: List[FrameContext] = field(default_factory=list)
    
    def add(self, ctx: FrameContext):
        """Add a new frame context, maintaining window size."""
        self.frames.append(ctx)
        if len(self.frames) > self.max_frames:
            # Remove oldest, also clear its image to free memory
            old = self.frames.pop(0)
            if old.image is not None:
                del old.image
    
    def clear(self):
        """Clear all context (e.g., when starting a new video)."""
        for f in self.frames:
            if f.image is not None:
                del f.image
        self.frames.clear()
    
    def get_text_context(self) -> Optional[str]:
        """Get text summary of recent frames for prompt injection."""
        if not self.frames:
            return None
        
        lines = ["Previous frame analyses:"]
        for ctx in self.frames:
            lines.append(f"  {ctx.to_text_summary()}")
        return "\n".join(lines)
    
    def get_images(self) -> List[Image.Image]:
        """Get list of previous frame images (for multi-frame input)."""
        return [f.image for f in self.frames if f.image is not None]
    
    def latest(self) -> Optional[FrameContext]:
        """Get the most recent frame context."""
        return self.frames[-1] if self.frames else None


# ==============================================================================
# Violence-Focused Prompts
# ==============================================================================

class ViolencePrompts:
    """
    Collection of violence-focused prompts for embedding extraction.
    
    These prompts guide the VLM's attention toward violence-relevant features,
    resulting in embeddings that better separate violent from non-violent content.
    """
    
    @staticmethod
    def embedding_focused() -> str:
        """
        Prompt optimized for embedding extraction.
        Focuses the model's attention on violence-relevant visual features.
        """
        return (
            "Analyze this security camera frame for signs of violence or pre-violence.\n"
            "Focus your attention on:\n"
            "1. BODY LANGUAGE: Aggressive postures, raised fists, fighting stances, defensive positions, lunging\n"
            "2. PHYSICAL CONTACT: Pushing, hitting, grabbing, choking, wrestling, kicking\n"
            "3. CROWD DYNAMICS: Panic, fleeing, circling around a fight, confrontational grouping\n"
            "4. WEAPONS: Knives, guns, bats, bottles, chairs, any object being wielded\n"
            "5. FACIAL EXPRESSIONS: Anger, fear, aggression, distress\n"
            "6. RAPID MOVEMENT: Sudden running, chasing, falling, being thrown\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "aggression_level": number 0..1,\n'
            '  "threat_indicators": [string, ...],\n'
            '  "body_language": [string, ...],\n'
            '  "actions": [string, ...],\n'
            '  "weapons": [string, ...],\n'
            '  "people_count": integer,\n'
            '  "scene": [string, ...]\n'
            "}\n"
        )
    
    @staticmethod
    def anticipation_focused() -> str:
        """
        Prompt focused on ANTICIPATING violence before it happens.
        Emphasizes pre-violence indicators and escalation patterns.
        """
        return (
            "You are a violence anticipation system. Analyze this frame for signs that "
            "violence MAY BE ABOUT TO OCCUR in the next 30 seconds.\n"
            "\n"
            "PRE-VIOLENCE INDICATORS to look for:\n"
            "- Aggressive body postures (squared shoulders, clenched fists, pointing)\n"
            "- Invasion of personal space / face-to-face confrontation\n"
            "- Verbal altercation indicators (open mouths, animated gestures)\n"
            "- Gathering crowd watching a confrontation\n"
            "- One person backing away while another advances\n"
            "- Brandishing or reaching for objects\n"
            "- Removing jackets/items (preparing for fight)\n"
            "- Sudden stillness before explosive movement\n"
            "\n"
            "ACTIVE VIOLENCE INDICATORS:\n"
            "- Physical contact (hitting, pushing, grabbing)\n"
            "- People falling or on the ground\n"
            "- Weapons being used\n"
            "- People fleeing\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "violence_imminent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "violence_active": {"label": "yes|no", "score": number 0..1},\n'
            '  "pre_violence_indicators": [string, ...],\n'
            '  "active_violence_indicators": [string, ...],\n'
            '  "threat_level": "none|low|medium|high|critical",\n'
            '  "people_count": integer,\n'
            '  "primary_concern": string\n'
            "}\n"
        )
    
    @staticmethod
    def detailed_violence() -> str:
        """
        Comprehensive violence analysis prompt with detailed categories.
        """
        return (
            "Analyze this image as a security analyst looking for violence or threats.\n"
            "\n"
            "Examine these specific aspects:\n"
            "\n"
            "PHYSICAL AGGRESSION:\n"
            "- Are any people making physical contact in an aggressive way?\n"
            "- Punching, kicking, slapping, pushing, grabbing, choking?\n"
            "- Is anyone on the ground being attacked?\n"
            "\n"
            "THREATENING POSTURES:\n"
            "- Raised fists, fighting stances, lunging positions?\n"
            "- Face-to-face confrontation with aggressive body language?\n"
            "- Someone cornering or surrounding another person?\n"
            "\n"
            "WEAPONS & OBJECTS:\n"
            "- Any visible weapons (guns, knives, bats, bottles)?\n"
            "- Objects being wielded as weapons (chairs, poles)?\n"
            "- Anyone reaching into pockets/waistbands?\n"
            "\n"
            "CROWD BEHAVIOR:\n"
            "- People fleeing or running in panic?\n"
            "- Crowd circling around a fight?\n"
            "- General chaos or disorder?\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "physical_aggression": {"present": boolean, "type": [string, ...]},\n'
            '  "threatening_postures": {"present": boolean, "details": [string, ...]},\n'
            '  "weapons": {"present": boolean, "items": [string, ...]},\n'
            '  "crowd_panic": {"present": boolean, "details": string},\n'
            '  "overall_threat": number 0..1,\n'
            '  "people_count": integer,\n'
            '  "scene_type": string,\n'
            '  "notes": string\n'
            "}\n"
        )
    
    @staticmethod
    def minimal_violence() -> str:
        """
        Short, focused prompt for faster processing.
        Still guides attention to violence but with less text.
        """
        return (
            "Security camera analysis. Detect violence, aggression, or threats.\n"
            "Look for: fighting, weapons, aggressive postures, fleeing people, confrontations.\n"
            "\n"
            "Return JSON:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": 0-1},\n'
            '  "threat_level": "none|low|medium|high",\n'
            '  "indicators": [string, ...],\n'
            '  "people_count": integer\n'
            "}\n"
        )
    
    @staticmethod
    def temporal_violence(context_text: Optional[str] = None) -> str:
        """
        Violence-focused prompt with temporal context awareness.
        """
        base = (
            "You are analyzing video frames for violence anticipation.\n"
            "Your goal is to detect violence BEFORE or AS it happens.\n"
        )
        
        if context_text:
            base += f"\n{context_text}\n\n"
            base += (
                "Consider how the situation is EVOLVING:\n"
                "- Is aggression escalating or de-escalating?\n"
                "- Are people moving closer together or apart?\n"
                "- Has anyone's posture become more threatening?\n"
                "- Are new threats emerging?\n\n"
            )
        else:
            base += "\nThis is the first frame - establish baseline threat level.\n\n"
        
        base += (
            "VIOLENCE INDICATORS TO DETECT:\n"
            "- Physical contact: hitting, pushing, grabbing, kicking\n"
            "- Aggressive postures: raised fists, fighting stance, lunging\n"
            "- Weapons: guns, knives, bats, improvised weapons\n"
            "- Pre-violence: confrontation, squaring up, removing clothing\n"
            "- Crowd response: fleeing, circling, panic\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "violence_imminent": boolean,\n'
            '  "escalation": "increasing|stable|decreasing|new_threat",\n'
            '  "threat_indicators": [string, ...],\n'
            '  "actions": [string, ...],\n'
            '  "weapons": [string, ...],\n'
            '  "people_count": integer,\n'
            '  "scene": [string, ...],\n'
            '  "confidence": number 0..1\n'
            "}\n"
        )
        return base

    # ==========================================================================
    # NEW: Multi-cue anticipation prompts for granular pre-violence detection
    # ==========================================================================
    
    @staticmethod
    def anticipation_multicue() -> str:
        """
        Multi-cue anticipation prompt that scores INDIVIDUAL pre-violence indicators.
        
        This produces structured scores for each escalation cue, enabling the model
        to learn which combinations predict violence rather than relying on a single
        binary label.
        """
        return (
            "You are an expert violence anticipation system for surveillance video.\n"
            "Your task is to detect PRE-VIOLENCE CUES that appear BEFORE violence occurs.\n"
            "\n"
            "Score each indicator from 0.0 (not present) to 1.0 (clearly present).\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "people_count": integer,\n'
            '  "scene_type": "street|indoor|parking|bar|school|transit|other",\n'
            '  \n'
            '  "confrontation": {\n'
            '    "face_to_face": 0.0-1.0,\n'
            '    "personal_space_invasion": 0.0-1.0,\n'
            '    "finger_pointing": 0.0-1.0,\n'
            '    "verbal_argument": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "aggressive_posture": {\n'
            '    "squared_shoulders": 0.0-1.0,\n'
            '    "clenched_fists": 0.0-1.0,\n'
            '    "fighting_stance": 0.0-1.0,\n'
            '    "puffed_chest": 0.0-1.0,\n'
            '    "leaning_forward_aggressive": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "movement_patterns": {\n'
            '    "advancing_on_someone": 0.0-1.0,\n'
            '    "backing_away_defensive": 0.0-1.0,\n'
            '    "circling": 0.0-1.0,\n'
            '    "sudden_movements": 0.0-1.0,\n'
            '    "cornering_someone": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "crowd_dynamics": {\n'
            '    "gathering_to_watch": 0.0-1.0,\n'
            '    "forming_circle": 0.0-1.0,\n'
            '    "phones_recording": 0.0-1.0,\n'
            '    "bystanders_backing_away": 0.0-1.0,\n'
            '    "two_groups_facing": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "preparation_cues": {\n'
            '    "removing_jacket": 0.0-1.0,\n'
            '    "rolling_up_sleeves": 0.0-1.0,\n'
            '    "handing_items_to_friend": 0.0-1.0,\n'
            '    "reaching_for_object": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "active_violence": {\n'
            '    "pushing_shoving": 0.0-1.0,\n'
            '    "punching_hitting": 0.0-1.0,\n'
            '    "kicking": 0.0-1.0,\n'
            '    "grabbing_holding": 0.0-1.0,\n'
            '    "weapon_use": 0.0-1.0,\n'
            '    "person_on_ground": 0.0-1.0,\n'
            '    "people_fleeing": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "weapons_visible": [list of visible weapons or empty],\n'
            '  "overall_threat_score": 0.0-1.0,\n'
            '  "primary_concern": "brief description of main threat or none"\n'
            "}\n"
            "\n"
            "Important:\n"
            "- Score 0.0 means definitely not present\n"
            "- Score 0.5 means uncertain or partially visible\n"
            "- Score 1.0 means clearly and obviously present\n"
            "- Be conservative: only score high if you clearly see the indicator\n"
        )
    
    @staticmethod
    def anticipation_simple() -> str:
        """
        Simplified anticipation prompt with key escalation scores.
        Faster than multicue but still captures essential pre-violence signals.
        """
        return (
            "Analyze this surveillance frame for violence risk.\n"
            "\n"
            "Score each from 0.0 (absent) to 1.0 (clearly present):\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "people_count": integer,\n'
            '  "confrontation_score": 0.0-1.0,\n'
            '  "aggressive_posture_score": 0.0-1.0,\n'
            '  "crowd_tension_score": 0.0-1.0,\n'
            '  "physical_contact_score": 0.0-1.0,\n'
            '  "weapon_present_score": 0.0-1.0,\n'
            '  "fleeing_panic_score": 0.0-1.0,\n'
            '  "overall_threat": 0.0-1.0,\n'
            '  "stage": "calm|tension|escalation|violence|aftermath",\n'
            '  "indicators": [list up to 5 specific observations]\n'
            "}\n"
            "\n"
            "Stage definitions:\n"
            "- calm: Normal activity, no tension\n"
            "- tension: Verbal argument, confrontational postures, but no contact\n"
            "- escalation: Aggressive movement, invasion of space, preparing to fight\n"
            "- violence: Active physical contact, fighting, weapon use\n"
            "- aftermath: Fight ended, people dispersing, someone injured\n"
        )
    
    @staticmethod
    def anticipation_behavioral() -> str:
        """
        Behavioral-focused prompt emphasizing body language and social dynamics.
        Best for scenarios where physical violence cues are subtle.
        """
        return (
            "You are a behavioral analyst examining surveillance footage.\n"
            "Focus on BODY LANGUAGE and SOCIAL DYNAMICS that precede violence.\n"
            "\n"
            "Analyze and score each behavioral category:\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "people_count": integer,\n'
            '  "interaction_type": "none|friendly|neutral|tense|hostile",\n'
            '  \n'
            '  "body_language": {\n'
            '    "aggressive_stance": 0.0-1.0,\n'
            '    "defensive_posture": 0.0-1.0,\n'
            '    "dominant_behavior": 0.0-1.0,\n'
            '    "submissive_behavior": 0.0-1.0,\n'
            '    "agitated_movement": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "proxemics": {\n'
            '    "intimate_distance_hostile": 0.0-1.0,\n'
            '    "blocking_exit": 0.0-1.0,\n'
            '    "surrounding_target": 0.0-1.0,\n'
            '    "approach_velocity": "slow|normal|fast|charging"\n'
            '  },\n'
            '  \n'
            '  "group_behavior": {\n'
            '    "pack_mentality": 0.0-1.0,\n'
            '    "isolating_individual": 0.0-1.0,\n'
            '    "audience_gathering": 0.0-1.0,\n'
            '    "intervention_attempt": 0.0-1.0\n'
            '  },\n'
            '  \n'
            '  "escalation_phase": "baseline|verbal|posturing|contact|violence",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "notes": "brief observation"\n'
            "}\n"
        )
    
    @staticmethod
    def anticipation_temporal(prev_scores: Optional[Dict[str, float]] = None) -> str:
        """
        Temporal-aware anticipation prompt that considers previous frame analysis.
        
        Args:
            prev_scores: Dictionary of scores from previous frame analysis
        """
        base = (
            "You are analyzing a video frame for violence anticipation.\n"
            "Consider CHANGES over time, not just the current state.\n"
        )
        
        if prev_scores:
            base += "\nPREVIOUS FRAME SCORES:\n"
            for key, val in prev_scores.items():
                base += f"  - {key}: {val:.2f}\n"
            base += "\nNote whether the situation is ESCALATING or DE-ESCALATING.\n"
        
        base += (
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "people_count": integer,\n'
            '  "confrontation_score": 0.0-1.0,\n'
            '  "aggression_score": 0.0-1.0,\n'
            '  "violence_score": 0.0-1.0,\n'
            '  "overall_threat": 0.0-1.0,\n'
            '  "trend": "escalating|stable|de-escalating|new_situation",\n'
            '  "time_to_violence": "unlikely|>60s|30-60s|10-30s|<10s|ongoing",\n'
            '  "key_changes": [list of changes from previous frame]\n'
            "}\n"
        )
        return base
    
    @staticmethod
    def scene_context() -> str:
        """
        Scene context prompt to understand environment risk factors.
        Run once at the start of a video to establish baseline.
        """
        return (
            "Analyze this surveillance scene for environmental risk factors.\n"
            "\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "location_type": "street|bar|club|school|transit|retail|parking|residential|office|stadium|other",\n'
            '  "time_of_day": "day|night|evening|unclear",\n'
            '  "crowd_density": "empty|sparse|moderate|crowded|packed",\n'
            '  "lighting_quality": "bright|adequate|dim|dark",\n'
            '  "alcohol_present": boolean,\n'
            '  "security_visible": boolean,\n'
            '  "confined_space": boolean,\n'
            '  "exit_routes_visible": integer,\n'
            '  "baseline_risk": "low|medium|high",\n'
            '  "risk_factors": [list of environmental risk factors]\n'
            "}\n"
        )


@dataclass
class QwenTeacherConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    dtype: str = "bfloat16"
    device: Optional[str] = None     # "cuda:0"
    attn_implementation: Optional[str] = "sdpa"  # "sdpa" or "eager" to avoid flash_attn
    use_fast_processor: bool = True  # set False to silence the fast-processor warning
    
    # Context/memory settings
    context_mode: str = "both"       # "none", "text", "visual", "both"
    context_window_size: int = 5     # Number of previous frames to remember
    store_context_images: bool = False  # Whether to keep PIL images in memory
    
    # Violence-focused prompt settings
    prompt_mode: str = "violence_embedding"  # "default", "violence_embedding", "violence_anticipation", 
                                             # "violence_detailed", "violence_minimal", "violence_temporal",
                                             # "anticipation_multicue", "anticipation_simple", 
                                             # "anticipation_behavioral", "scene_context"


class QwenTeacher:
    def __init__(self, cfg: QwenTeacherConfig):
        if AutoProcessor is None or AutoModelForVision2Seq is None:
            raise RuntimeError("transformers not installed or missing vision2seq classes.")

        self.cfg = cfg
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

        kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        # IMPORTANT: this prevents transformers from trying flash_attn_2
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation

        # Processor: allow opting out of "fast" behavior
        try:
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_id, trust_remote_code=True, use_fast=cfg.use_fast_processor
            )
        except TypeError:
            # older/newer transformers may not accept use_fast on AutoProcessor
            self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

        self.model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, **kwargs).to(self.device)
        self.model.eval()
        
        # Initialize context window
        self.context = ContextWindow(max_frames=cfg.context_window_size)

    def reset_context(self):
        """Reset context window (call when starting a new video)."""
        self.context.clear()

    @staticmethod
    def default_prompt() -> str:
        return (
            "You are analyzing a single video frame.\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "people_count": integer >= 0,\n'
            '  "actions": [string, ...],\n'
            '  "weapons": [string, ...],\n'
            '  "scene": [string, ...],\n'
            '  "notes": string\n'
            "}\n"
            "Rules:\n"
            "- Use short lowercase words/phrases.\n"
            "- actions: verbs like 'punching', 'running', 'arguing'. Max 5.\n"
            "- weapons: like 'knife', 'gun', 'bat'. Empty list if none.\n"
            "- scene: like 'street', 'indoor', 'night', 'crowd'. Max 5.\n"
            "- If unsure, label='uncertain' and score=0.5.\n"
            "- Do not use markdown code fences. Output raw JSON only.\n"
        )

    @staticmethod
    def context_aware_prompt(context_text: Optional[str] = None) -> str:
        """Generate prompt with temporal context awareness."""
        base = (
            "You are analyzing video frames for violence detection.\n"
        )
        
        if context_text:
            base += f"\n{context_text}\n\n"
            base += (
                "Now analyze the CURRENT frame. Consider temporal continuity:\n"
                "- Are actions escalating, de-escalating, or stable?\n"
                "- Note any significant changes from previous frames.\n"
                "- Your 'notes' field should briefly describe temporal changes if relevant.\n\n"
            )
        else:
            base += "This is the first frame in the sequence.\n\n"
        
        base += (
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "violent": {"label": "yes|no|uncertain", "score": number 0..1},\n'
            '  "people_count": integer >= 0,\n'
            '  "actions": [string, ...],\n'
            '  "weapons": [string, ...],\n'
            '  "scene": [string, ...],\n'
            '  "temporal_change": "escalating|stable|de-escalating|new_scene",\n'
            '  "notes": string\n'
            "}\n"
            "Rules:\n"
            "- Use short lowercase words/phrases.\n"
            "- actions: verbs like 'punching', 'running', 'arguing'. Max 5.\n"
            "- weapons: like 'knife', 'gun', 'bat'. Empty list if none.\n"
            "- scene: like 'street', 'indoor', 'night', 'crowd'. Max 5.\n"
            "- temporal_change: how situation evolved from previous frame(s).\n"
            "- If unsure, label='uncertain' and score=0.5.\n"
            "- Do not use markdown code fences. Output raw JSON only.\n"
        )
        return base

    def get_prompt_for_mode(self, context_text: Optional[str] = None) -> str:
        """
        Get the appropriate prompt based on the configured prompt_mode.
        
        Args:
            context_text: Optional context from previous frames
            
        Returns:
            The prompt string to use
        """
        mode = self.cfg.prompt_mode
        
        if mode == "default":
            if context_text:
                return self.context_aware_prompt(context_text)
            return self.default_prompt()
        
        elif mode == "violence_embedding":
            # Best for embedding extraction - focuses attention on violence features
            return ViolencePrompts.embedding_focused()
        
        elif mode == "violence_anticipation":
            # Best for anticipation task - emphasizes pre-violence indicators
            return ViolencePrompts.anticipation_focused()
        
        elif mode == "violence_detailed":
            # Most comprehensive analysis
            return ViolencePrompts.detailed_violence()
        
        elif mode == "violence_minimal":
            # Fastest processing, still violence-focused
            return ViolencePrompts.minimal_violence()
        
        elif mode == "violence_temporal":
            # Violence-focused with temporal context
            return ViolencePrompts.temporal_violence(context_text)
        
        # NEW anticipation modes
        elif mode == "anticipation_multicue":
            # Multi-cue scoring for granular pre-violence detection
            return ViolencePrompts.anticipation_multicue()
        
        elif mode == "anticipation_simple":
            # Simplified anticipation with key scores
            return ViolencePrompts.anticipation_simple()
        
        elif mode == "anticipation_behavioral":
            # Behavioral/body language focused
            return ViolencePrompts.anticipation_behavioral()
        
        elif mode == "scene_context":
            # Scene analysis for baseline risk
            return ViolencePrompts.scene_context()
        
        else:
            # Fallback to default
            return self.default_prompt()

    def _build_inputs(self, image: Image.Image, prompt: str, context_images: Optional[List[Image.Image]] = None):
        """
        Build model inputs, optionally including previous frame images.
        
        Args:
            image: Current frame to analyze
            prompt: Text prompt
            context_images: Optional list of previous frame images for multi-frame context
        """
        all_images = []
        content = []
        
        # Add context images first (if using visual context mode)
        if context_images:
            for i, ctx_img in enumerate(context_images):
                content.append({"type": "image", "image": ctx_img})
                content.append({"type": "text", "text": f"[Previous frame {i+1}]"})
                all_images.append(ctx_img)
        
        # Add current image
        content.append({"type": "image", "image": image})
        if context_images:
            content.append({"type": "text", "text": "[Current frame to analyze]"})
        all_images.append(image)
        
        # Add the main prompt
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": "You are a security camera analyst specializing in violence detection. Analyze images carefully for any signs of violence, aggression, or threats."},
            {"role": "user", "content": content},
        ]

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=all_images, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=[prompt], images=all_images, return_tensors="pt").to(self.device)

        return inputs

    def _extract_embedding(self, inputs) -> Optional[np.ndarray]:
        """
        Image-conditioned embedding for Qwen2.5-VL.

        Goal: DO NOT mean-pool over the entire sequence (prompt dominates).
        Instead:
        1) forward pass (no generation) w/ hidden states
        2) pool ONLY vision/image token positions (best-effort)
        3) L2-normalize for stability
        """
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

        # last layer hidden states: [B, S, H]
        hs = None
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states[-1]
        elif hasattr(out, "decoder_hidden_states") and out.decoder_hidden_states is not None:
            hs = out.decoder_hidden_states[-1]
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hs = out.last_hidden_state

        if hs is None:
            return None

        input_ids = inputs.get("input_ids", None)
        attn_mask = inputs.get("attention_mask", None)
        tok = getattr(self.processor, "tokenizer", None)

        pooled = None

        # --- Try to pool ONLY vision/image token positions ---
        if input_ids is not None and tok is not None:
            ids = input_ids[0]

            cand_ids = set()

            # common tokenizer/config attrs across VL implementations
            for a in ("image_token_id", "image_pad_token_id", "vision_token_id", "img_token_id"):
                if hasattr(tok, a):
                    try: cand_ids.add(int(getattr(tok, a)))
                    except Exception: pass
            for a in ("image_token_id", "image_pad_token_id", "vision_token_id"):
                if hasattr(self.model.config, a):
                    try: cand_ids.add(int(getattr(self.model.config, a)))
                    except Exception: pass

            # also sniff special tokens containing image/vision/img
            try:
                specials = []
                if getattr(tok, "additional_special_tokens", None):
                    specials += list(tok.additional_special_tokens)
                if getattr(tok, "special_tokens_map", None):
                    for v in tok.special_tokens_map.values():
                        if isinstance(v, str):
                            specials.append(v)

                for s in specials:
                    sl = str(s).lower()
                    if ("image" in sl) or ("vision" in sl) or ("img" in sl):
                        try:
                            cand_ids.add(int(tok.convert_tokens_to_ids(str(s))))
                        except Exception:
                            pass
            except Exception:
                pass

            # mask from candidate ids
            if cand_ids:
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for cid in cand_ids:
                    mask |= (ids == cid)

                # apply attention mask too (just in case)
                if attn_mask is not None:
                    mask = mask & attn_mask[0].bool()

                if mask.any():
                    pooled = hs[0][mask].mean(dim=0)

        # --- Fallbacks ---
        if pooled is None:
            # fallback 1: mean-pool only attended tokens
            if attn_mask is not None:
                m = attn_mask[0].bool()
                if m.any():
                    pooled = hs[0][m].mean(dim=0)
                else:
                    pooled = hs[0, -1]
            else:
                pooled = hs[0].mean(dim=0)

        pooled = pooled.float()

        # L2 normalize -> stabilizes training + makes cosine checks meaningful
        pooled = pooled / (pooled.norm(p=2) + 1e-12)

        return pooled.detach().cpu().numpy().astype(np.float32)

    def infer_one(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        return_embedding: bool = False,
        t: Optional[int] = None,
        raw_frame_idx: Optional[int] = None,
        use_context: Optional[bool] = None,
    ) -> Union[Tuple[Optional[Dict[str, Any]], str], Tuple[Optional[Dict[str, Any]], str, Optional[np.ndarray]]]:
        """
        Analyze a single frame, optionally with temporal context.
        
        Args:
            image: PIL Image of the current frame
            prompt: Custom prompt (if None, uses configured prompt_mode)
            return_embedding: Whether to return the embedding vector
            t: Current timestep index (for context tracking)
            raw_frame_idx: Raw frame index in the video (for context tracking)
            use_context: Override cfg.context_mode (True=use context, False=don't)
        
        Returns:
            (parsed_dict, raw_text) or (parsed_dict, raw_text, embedding)
        """
        # Determine whether to use context
        should_use_context = use_context if use_context is not None else (self.cfg.context_mode != "none")
        
        context_images = None
        context_text = None
        
        if should_use_context and self.context.frames:
            mode = self.cfg.context_mode
            
            if mode in ("text", "both"):
                context_text = self.context.get_text_context()
            
            if mode in ("visual", "both"):
                context_images = self.context.get_images()
                if not context_images:  # No images stored
                    context_images = None
        
        # Build prompt - use provided prompt or get from configured mode
        if prompt is not None:
            final_prompt = prompt
        else:
            final_prompt = self.get_prompt_for_mode(context_text)
        
        # Build inputs (with optional multi-frame visual context)
        inputs = self._build_inputs(image, final_prompt, context_images)

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=(self.cfg.temperature > 0),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        decoded_tail = decoded.split("assistant")[-1].strip()
        parsed = _best_effort_json_parse(decoded_tail)

        # Update context window if we have valid results and tracking info
        if parsed is not None and t is not None and raw_frame_idx is not None:
            ctx = FrameContext(
                t=t,
                raw_frame_idx=raw_frame_idx,
                parsed=parsed,
                image=image.copy() if self.cfg.store_context_images else None
            )
            self.context.add(ctx)

        if not return_embedding:
            return parsed, decoded_tail

        emb = self._extract_embedding(inputs)
        return parsed, decoded_tail, emb
    
    def infer_embedding_only(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract only the embedding without generating text response.
        
        This is faster when you only need embeddings for training/inference.
        Uses the configured prompt_mode to guide attention.
        
        Args:
            image: PIL Image of the frame
            prompt: Custom prompt (if None, uses configured prompt_mode)
            
        Returns:
            Embedding vector as numpy array, or None if extraction failed
        """
        # Use provided prompt or get from configured mode
        if prompt is None:
            prompt = self.get_prompt_for_mode()
        
        # Build inputs
        inputs = self._build_inputs(image, prompt)
        
        # Extract embedding without generation
        return self._extract_embedding(inputs)
    
    def infer_batch_sequential(
        self,
        images: List[Image.Image],
        start_t: int = 0,
        t_step: int = 1,
        return_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process a sequence of frames with context propagation.
        
        This maintains context across the batch, simulating sequential processing.
        Useful for processing clips where you want temporal awareness.
        
        Args:
            images: List of PIL Images in temporal order
            start_t: Starting timestep index
            t_step: Step between timesteps
            return_embeddings: Whether to include embeddings in results
        
        Returns:
            List of result dicts with 't', 'parsed', 'raw_text', and optionally 'embedding'
        """
        self.reset_context()
        results = []
        
        for i, img in enumerate(images):
            t = start_t + (i * t_step)
            
            if return_embeddings:
                parsed, raw_text, emb = self.infer_one(
                    img,
                    return_embedding=True,
                    t=t,
                    raw_frame_idx=t,  # Approximate
                )
                results.append({
                    "t": t,
                    "parsed": parsed,
                    "raw_text": raw_text,
                    "embedding": emb,
                })
            else:
                parsed, raw_text = self.infer_one(
                    img,
                    return_embedding=False,
                    t=t,
                    raw_frame_idx=t,
                )
                results.append({
                    "t": t,
                    "parsed": parsed,
                    "raw_text": raw_text,
                })
        
        return results
    
    def extract_embeddings_batch(
        self,
        images: List[Image.Image],
        prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for a batch of images (no text generation).
        
        This is the fastest way to get embeddings for many frames.
        Uses the configured prompt_mode to guide attention toward violence features.
        
        Args:
            images: List of PIL Images
            prompt: Custom prompt (if None, uses configured prompt_mode)
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors (or None for failed extractions)
        """
        if prompt is None:
            prompt = self.get_prompt_for_mode()
        
        embeddings = []
        iterator = images
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(images, desc="Extracting embeddings")
            except ImportError:
                pass
        
        for img in iterator:
            emb = self.infer_embedding_only(img, prompt)
            embeddings.append(emb)
        
        return embeddings


# ==============================================================================
# Utility: Parse multi-cue output into feature vector
# ==============================================================================

def multicue_to_vector(parsed: Dict[str, Any]) -> np.ndarray:
    """
    Convert anticipation_multicue output to a fixed-size feature vector.
    
    This extracts all the numeric scores into a vector that can be used
    for training a classifier or as input to a downstream model.
    
    Args:
        parsed: The parsed JSON from anticipation_multicue prompt
        
    Returns:
        numpy array of shape (31,) with all scores
    """
    features = []
    
    # Confrontation (4 features)
    conf = parsed.get("confrontation", {})
    features.extend([
        conf.get("face_to_face", 0.0),
        conf.get("personal_space_invasion", 0.0),
        conf.get("finger_pointing", 0.0),
        conf.get("verbal_argument", 0.0),
    ])
    
    # Aggressive posture (5 features)
    agg = parsed.get("aggressive_posture", {})
    features.extend([
        agg.get("squared_shoulders", 0.0),
        agg.get("clenched_fists", 0.0),
        agg.get("fighting_stance", 0.0),
        agg.get("puffed_chest", 0.0),
        agg.get("leaning_forward_aggressive", 0.0),
    ])
    
    # Movement patterns (5 features)
    mov = parsed.get("movement_patterns", {})
    features.extend([
        mov.get("advancing_on_someone", 0.0),
        mov.get("backing_away_defensive", 0.0),
        mov.get("circling", 0.0),
        mov.get("sudden_movements", 0.0),
        mov.get("cornering_someone", 0.0),
    ])
    
    # Crowd dynamics (5 features)
    crowd = parsed.get("crowd_dynamics", {})
    features.extend([
        crowd.get("gathering_to_watch", 0.0),
        crowd.get("forming_circle", 0.0),
        crowd.get("phones_recording", 0.0),
        crowd.get("bystanders_backing_away", 0.0),
        crowd.get("two_groups_facing", 0.0),
    ])
    
    # Preparation cues (4 features)
    prep = parsed.get("preparation_cues", {})
    features.extend([
        prep.get("removing_jacket", 0.0),
        prep.get("rolling_up_sleeves", 0.0),
        prep.get("handing_items_to_friend", 0.0),
        prep.get("reaching_for_object", 0.0),
    ])
    
    # Active violence (7 features)
    viol = parsed.get("active_violence", {})
    features.extend([
        viol.get("pushing_shoving", 0.0),
        viol.get("punching_hitting", 0.0),
        viol.get("kicking", 0.0),
        viol.get("grabbing_holding", 0.0),
        viol.get("weapon_use", 0.0),
        viol.get("person_on_ground", 0.0),
        viol.get("people_fleeing", 0.0),
    ])
    
    # Overall threat score (1 feature)
    features.append(parsed.get("overall_threat_score", 0.0))
    
    return np.array(features, dtype=np.float32)


def simple_to_vector(parsed: Dict[str, Any]) -> np.ndarray:
    """
    Convert anticipation_simple output to a feature vector.
    
    Args:
        parsed: The parsed JSON from anticipation_simple prompt
        
    Returns:
        numpy array of shape (7,) with all scores
    """
    return np.array([
        parsed.get("confrontation_score", 0.0),
        parsed.get("aggressive_posture_score", 0.0),
        parsed.get("crowd_tension_score", 0.0),
        parsed.get("physical_contact_score", 0.0),
        parsed.get("weapon_present_score", 0.0),
        parsed.get("fleeing_panic_score", 0.0),
        parsed.get("overall_threat", 0.0),
    ], dtype=np.float32)