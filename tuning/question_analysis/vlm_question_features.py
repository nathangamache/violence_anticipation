#!/usr/bin/env python3
"""Per-frame VLM yes/no answers to violence-indicator questions, used as features in [0, 1].

Usage:
    python vlm_question_features.py --videos_dir data/videos --embeddings_dir data/processed/vlm_emb \
        --output_dir data/processed/vlm_qa_features --device cuda:0
    python vlm_question_features.py --video_path data/videos/my_video.mp4 --device cuda:0
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2


def find_video_file(video_name: str, videos_dir: Path) -> Optional[Path]:
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MP4', '.AVI', '.MKV']
    
    for ext in extensions:
        path = videos_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    for subdir in ['', 'xd_violence', 'ucf_crime', 'XD_Violence', 'UCF_Crime', 'videos']:
        base = videos_dir / subdir if subdir else videos_dir
        if not base.exists():
            continue
        
        for ext in extensions:
            path = base / f"{video_name}{ext}"
            if path.exists():
                return path
        
        matches = list(base.glob(f"**/{video_name}.*"))
        for m in matches:
            if m.suffix.lower() in [e.lower() for e in extensions]:
                return m
    
    return None


def get_video_info(video_path: Path) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'total_frames': 0, 'fps': 1.0, 'width': 0, 'height': 0}
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS) or 1.0,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def extract_frame_from_video(video_path: Path, frame_idx: int, video_info: Dict = None) -> Optional[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    if video_info is None:
        video_info = get_video_info(video_path)
    
    total_frames = video_info.get('total_frames', 0)
    
    actual_frame = min(max(0, frame_idx), total_frames - 1) if total_frames > 0 else frame_idx
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class VideoFrameIterator:
    """Yield (frame_idx, PIL image) for every sample_rate-th frame."""
    
    def __init__(self, video_path: Path, sample_rate: int = 1):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.video_info = get_video_info(video_path)
        self.total_frames = self.video_info['total_frames']
        self.fps = self.video_info['fps']
    
    def __len__(self):
        return (self.total_frames + self.sample_rate - 1) // self.sample_rate
    
    def __iter__(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, Image.fromarray(frame_rgb)
            
            frame_idx += 1
        
        cap.release()


VIOLENCE_QUESTIONS = [
    {
        "id": "physical_contact",
        "question": "Is there physical contact between two or more people in this image?",
        "type": "binary",
        "weight": 2.0,
    },
    {
        "id": "hitting_punching", 
        "question": "Is anyone hitting, punching, kicking, or striking another person?",
        "type": "binary",
        "weight": 3.0,
    },
    {
        "id": "pushing_shoving",
        "question": "Is anyone pushing, shoving, or grabbing another person aggressively?",
        "type": "binary",
        "weight": 2.5,
    },
    {
        "id": "person_on_ground",
        "question": "Is anyone lying on the ground in a non-relaxed position (fallen, pushed down)?",
        "type": "binary",
        "weight": 2.0,
    },
    {
        "id": "restraining",
        "question": "Is anyone being physically restrained, held down, or pinned?",
        "type": "binary",
        "weight": 2.0,
    },
    
    {
        "id": "aggressive_stance",
        "question": "Are any people in aggressive stances (raised fists, confrontational positioning)?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "defensive_posture",
        "question": "Is anyone in a defensive posture (arms up protecting face, cowering, backing away)?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "running_fleeing",
        "question": "Are people running or fleeing in panic (not jogging or casual movement)?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "sudden_movement",
        "question": "Does the image show signs of sudden, rapid, or chaotic movement?",
        "type": "binary",
        "weight": 1.0,
    },
    
    {
        "id": "weapon_visible",
        "question": "Is there a weapon visible (gun, knife, bat, stick, bottle used as weapon)?",
        "type": "binary",
        "weight": 3.0,
    },
    {
        "id": "holding_object_threateningly",
        "question": "Is anyone holding an object in a threatening manner?",
        "type": "binary",
        "weight": 2.0,
    },
    
    {
        "id": "crowd_surrounding",
        "question": "Is a crowd surrounding or encircling a specific person or small group?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "crowd_dispersing",
        "question": "Is a crowd rapidly dispersing or scattering?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "group_confrontation",
        "question": "Are two groups of people facing each other in a confrontational manner?",
        "type": "binary",
        "weight": 2.0,
    },
    
    {
        "id": "distress_visible",
        "question": "Are there visible signs of distress, fear, or pain on anyone's face?",
        "type": "binary",
        "weight": 1.5,
    },
    {
        "id": "anger_aggression",
        "question": "Does anyone appear visibly angry or aggressive in their expression?",
        "type": "binary",
        "weight": 1.5,
    },
    
    {
        "id": "damage_visible",
        "question": "Is there visible damage or destruction (broken objects, overturned items)?",
        "type": "binary",
        "weight": 1.0,
    },
    {
        "id": "blood_injury",
        "question": "Is there visible blood or signs of injury on anyone?",
        "type": "binary",
        "weight": 2.5,
    },
    {
        "id": "emergency_responders",
        "question": "Are emergency responders (police, medical personnel) present or arriving?",
        "type": "binary",
        "weight": 1.0,
    },
    
    # Negative indicators: presence suggests non-violence (sports often looks like violence).
    {
        "id": "calm_interaction",
        "question": "Are people interacting calmly and peacefully?",
        "type": "binary",
        "weight": -1.0,
    },
    {
        "id": "sports_activity",
        "question": "Is this a sports or athletic activity?",
        "type": "binary", 
        "weight": -1.5,
    },
    {
        "id": "performance_acting",
        "question": "Does this appear to be a performance, movie scene, or staged event?",
        "type": "binary",
        "weight": -1.0,
    },
    {
        "id": "playful_interaction",
        "question": "Are people engaged in playful or friendly physical interaction?",
        "type": "binary",
        "weight": -1.5,
    },
    
    {
        "id": "violence_occurring",
        "question": "Overall, is violence or a physical altercation occurring in this image?",
        "type": "binary",
        "weight": 2.0,
    },
    {
        "id": "violence_imminent",
        "question": "Does the situation suggest violence is about to occur (tension, confrontation building)?",
        "type": "binary",
        "weight": 2.0,
    },
]


def get_question_list() -> List[Dict]:
    return VIOLENCE_QUESTIONS


def get_feature_names() -> List[str]:
    return [q["id"] for q in VIOLENCE_QUESTIONS]


class VLMQuestionExtractor:
    """Answer yes/no questions about an image with Qwen2/2.5-VL and map answers to scores."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        
        # Qwen2.5-VL uses different classes than Qwen2-VL
        if "Qwen2.5" in model_name or "Qwen2_5" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model_class = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model_class = Qwen2VLForConditionalGeneration
        
        from transformers import AutoProcessor
        
        self.model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        self.model.eval()
        self.questions = get_question_list()
        
        print(f"Loaded. {len(self.questions)} questions per frame.")
    
    def _parse_yes_no(self, response: str) -> float:
        """Map a free-text answer to a score in [0, 1]."""
        response = response.lower().strip()
        
        if response.startswith("yes"):
            return 1.0
        if "yes," in response or "yes." in response:
            return 1.0
        if response in ["yes", "y", "true", "correct", "definitely", "absolutely"]:
            return 1.0
        
        if response.startswith("no"):
            return 0.0
        if "no," in response or "no." in response:
            return 0.0
        if response in ["no", "n", "false", "incorrect"]:
            return 0.0
        
        if "possibly" in response or "maybe" in response or "might" in response:
            return 0.5
        if "probably yes" in response or "likely yes" in response:
            return 0.75
        if "probably no" in response or "likely no" in response or "unlikely" in response:
            return 0.25
        if "unclear" in response or "cannot tell" in response or "hard to say" in response:
            return 0.5
        
        if "yes" in response:
            return 0.8
        if "no" in response:
            return 0.2
        
        return 0.5
    
    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Ask each question separately; returns [num_questions] scores in [0, 1]."""
        features = []
        
        for q in self.questions:
            question = q["question"]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{question} Answer with only 'yes' or 'no'."},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
            
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            score = self._parse_yes_no(response)
            features.append(score)
        
        return np.array(features, dtype=np.float32)
    
    @torch.no_grad()
    def extract_features_batch_questions(self, image: Image.Image) -> np.ndarray:
        """Ask all questions in one prompt (faster, less accurate than extract_features)."""
        question_text = "Answer each question with ONLY 'yes' or 'no', one per line:\n\n"
        for i, q in enumerate(self.questions):
            question_text += f"{i+1}. {q['question']}\n"
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
        
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        lines = response.strip().split('\n')
        features = []
        
        for i, q in enumerate(self.questions):
            if i < len(lines):
                score = self._parse_yes_no(lines[i])
            else:
                score = 0.5
            features.append(score)
        
        return np.array(features, dtype=np.float32)
    
    def get_weighted_score(self, features: np.ndarray) -> float:
        """Weighted score in [0, 1]; positive and negative weights each normalized to unit sum."""
        weights = np.array([q["weight"] for q in self.questions], dtype=np.float32)
        
        pos_mask = weights > 0
        neg_mask = weights < 0
        
        if pos_mask.sum() > 0:
            weights[pos_mask] = weights[pos_mask] / weights[pos_mask].sum()
        if neg_mask.sum() > 0:
            weights[neg_mask] = weights[neg_mask] / np.abs(weights[neg_mask]).sum()
        
        score = np.dot(features, weights)
        score = (score + 1) / 2
        return float(np.clip(score, 0, 1))


def process_single_video(
    extractor: VLMQuestionExtractor,
    video_path: Path,
    output_path: Path,
    sample_rate: int = 1,
    batch_questions: bool = True,
):
    """Extract QA features for every sample_rate-th frame of one video and save them."""
    video_info = get_video_info(video_path)
    
    if video_info['total_frames'] == 0:
        print(f"Could not open video: {video_path}")
        return
    
    print(f"Processing {video_path.name}: {video_info['total_frames']} frames @ {video_info['fps']:.1f} fps")
    
    all_features = []
    all_scores = []
    frame_indices = []
    
    video_iter = VideoFrameIterator(video_path, sample_rate)
    
    for frame_idx, image in tqdm(video_iter, total=len(video_iter), desc=video_path.stem):
        try:
            if batch_questions:
                features = extractor.extract_features_batch_questions(image)
            else:
                features = extractor.extract_features(image)
            
            score = extractor.get_weighted_score(features)
            
            all_features.append(features)
            all_scores.append(score)
            frame_indices.append(frame_idx)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            all_features.append(np.full(len(extractor.questions), 0.5, dtype=np.float32))
            all_scores.append(0.5)
            frame_indices.append(frame_idx)
    
    if not all_features:
        print(f"No frames processed for {video_path}")
        return
    
    all_features = np.stack(all_features)
    all_scores = np.array(all_scores)
    frame_indices = np.array(frame_indices)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=all_features,
        scores=all_scores,
        frame_indices=frame_indices,
        feature_names=get_feature_names(),
        questions=[q["question"] for q in extractor.questions],
        weights=[q["weight"] for q in extractor.questions],
        video_info=video_info,
        sample_rate=sample_rate,
    )
    
    print(f"Saved {len(all_features)} frames to {output_path}")
    print(f"  Mean violence score: {all_scores.mean():.3f}")


def process_videos_from_embeddings(
    extractor: VLMQuestionExtractor,
    videos_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    sample_rate: int = 1,
    batch_questions: bool = True,
    source_filter: str = None,
):
    """Process videos named after the embedding files in embeddings_dir, mirroring its layout."""
    emb_files = list(embeddings_dir.rglob("*.npz"))
    
    print(f"Found {len(emb_files)} embedding files")
    
    processed = 0
    skipped = 0
    not_found = 0
    
    for emb_path in emb_files:
        video_name = emb_path.stem
        
        if source_filter:
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            
            if source_filter == 'xd_violence' and not is_xd:
                continue
            if source_filter == 'ucf_crime' and not is_ucf:
                continue
        
        rel_path = emb_path.relative_to(embeddings_dir)
        output_path = output_dir / rel_path.parent / f"{video_name}.npz"
        
        if output_path.exists():
            skipped += 1
            continue
        
        video_path = find_video_file(video_name, videos_dir)
        
        if video_path is None:
            not_found += 1
            if not_found <= 5:
                print(f"Video not found: {video_name}")
            continue
        
        process_single_video(
            extractor, video_path, output_path,
            sample_rate=sample_rate, batch_questions=batch_questions
        )
        processed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (exists): {skipped}")
    print(f"  Not found: {not_found}")


def process_video_directory(
    extractor: VLMQuestionExtractor,
    videos_dir: Path,
    output_dir: Path,
    sample_rate: int = 1,
    batch_questions: bool = True,
):
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
    
    video_files = []
    for ext in extensions:
        video_files.extend(videos_dir.rglob(f"*{ext}"))
        video_files.extend(videos_dir.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(video_files)} video files")
    
    for video_path in video_files:
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}.npz"
        
        if output_path.exists():
            print(f"Skipping (exists): {video_name}")
            continue
        
        process_single_video(
            extractor, video_path, output_path,
            sample_rate=sample_rate, batch_questions=batch_questions
        )


def analyze_features(features_dir: Path, labels_dir: Path, output_path: Path):
    """Plot the per-question mean difference between violence and non-violence frames."""
    import matplotlib.pyplot as plt
    
    all_features = []
    all_labels = []
    
    for feat_path in features_dir.rglob("*.npz"):
        video_name = feat_path.stem
        label_path = labels_dir / f"{video_name}.npy"
        
        if not label_path.exists():
            label_path = labels_dir / f"{video_name}.npz"
        
        if not label_path.exists():
            continue
        
        feat_data = np.load(feat_path)
        features = feat_data["features"]
        
        if label_path.suffix == ".npz":
            labels = np.load(label_path)["labels"]
        else:
            labels = np.load(label_path)
        
        if labels.ndim > 1:
            labels = labels.max(axis=1)
        labels = (labels > 0).astype(float)
        
        min_len = min(len(features), len(labels))
        all_features.append(features[:min_len])
        all_labels.append(labels[:min_len])
    
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"Analyzing {len(all_features)} samples")
    print(f"Violence rate: {all_labels.mean():.1%}")
    
    violence_mask = all_labels > 0.5
    
    violence_means = all_features[violence_mask].mean(axis=0)
    nonviolence_means = all_features[~violence_mask].mean(axis=0)
    
    diff = violence_means - nonviolence_means
    
    sample_path = list(features_dir.rglob("*.npz"))[0]
    feat_data = np.load(sample_path)
    feature_names = list(feat_data.get("feature_names", [f"Q{i}" for i in range(len(diff))]))
    
    sorted_idx = np.argsort(np.abs(diff))[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = range(len(diff))
    colors = ['red' if d > 0 else 'blue' for d in diff[sorted_idx]]
    
    ax.barh(y_pos, diff[sorted_idx], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('Difference (Violence - Non-Violence Mean)')
    ax.set_title('Feature Discriminative Power\n(Red = higher in violence, Blue = higher in non-violence)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved analysis to {output_path}")
    
    print("\nTop Discriminative Features:")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {diff[idx]:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="VLM Question-Based Feature Extraction")
    
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to a single video file')
    parser.add_argument('--videos_dir', type=str, default=None,
                        help='Directory containing video files')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Directory with embeddings (to find video names)')
    
    parser.add_argument('--output_dir', type=str, default='data/processed/vlm_qa_features')
    
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--batch_questions', action='store_true', default=True,
                        help='Ask all questions in one prompt (faster)')
    parser.add_argument('--individual_questions', action='store_true',
                        help='Ask each question individually (slower but more accurate)')
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'],
                        help='Filter to specific dataset')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing features for discriminative power')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Directory with extracted features (for analysis)')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Directory with labels (for analysis)')
    
    parser.add_argument('--list_questions', action='store_true',
                        help='Print the question bank and exit')
    
    args = parser.parse_args()
    
    if args.list_questions:
        print("Violence Detection Question Bank:")
        print("="*70)
        for i, q in enumerate(get_question_list()):
            print(f"{i+1:2d}. [{q['id']}] (weight={q['weight']:+.1f})")
            print(f"    {q['question']}")
        return
    
    if args.analyze:
        if not args.features_dir or not args.labels_dir:
            print("Need --features_dir and --labels_dir for analysis")
            return
        analyze_features(
            Path(args.features_dir),
            Path(args.labels_dir),
            Path(args.output_dir) / "feature_analysis.png"
        )
        return
    
    if not args.video_path and not args.videos_dir:
        print("ERROR: Specify --video_path (single video) or --videos_dir (batch processing)")
        print("\nExamples:")
        print("  # Process single video:")
        print("  python vlm_question_features.py --video_path video.mp4 --output_dir ./qa_features")
        print("")
        print("  # Process all videos matching embeddings:")
        print("  python vlm_question_features.py --videos_dir data/videos --embeddings_dir data/processed/vlm_emb --output_dir ./qa_features")
        return
    
    extractor = VLMQuestionExtractor(
        model_name=args.model,
        device=args.device,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_mode = args.batch_questions and not args.individual_questions
    
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
        
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}.npz"
        
        process_single_video(
            extractor, video_path, output_path,
            sample_rate=args.sample_rate,
            batch_questions=batch_mode,
        )
    
    elif args.videos_dir and args.embeddings_dir:
        process_videos_from_embeddings(
            extractor,
            Path(args.videos_dir),
            Path(args.embeddings_dir),
            output_dir,
            sample_rate=args.sample_rate,
            batch_questions=batch_mode,
            source_filter=args.source_filter,
        )
    
    elif args.videos_dir:
        process_video_directory(
            extractor,
            Path(args.videos_dir),
            output_dir,
            sample_rate=args.sample_rate,
            batch_questions=batch_mode,
        )
    
    print("\nDone!")
    print(f"Features saved to: {output_dir}")
    print(f"\nFeature dimension: {len(get_question_list())}")
    print("\nTo use with temporal model:")
    print(f"  python train_temporal_model.py --embeddings_dir {output_dir} --input_dim {len(get_question_list())} ...")


if __name__ == '__main__':
    main()