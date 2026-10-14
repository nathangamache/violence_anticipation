#!/usr/bin/env python3
"""
Quick Test: VLM Question Features Discriminative Power

Tests the QA-based feature extraction on a small sample to see
if the features actually discriminate violence from non-violence.

This version loads frames DIRECTLY FROM VIDEO FILES - no need for
pre-extracted frame images.

Usage:
    python test_qa_features.py \
        --videos_dir data/videos \
        --embeddings_dir data/processed/vlm_emb \
        --labels_dir data/processed/labels_multih \
        --n_samples 50 \
        --device cuda:0
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Question Bank
# ==============================================================================

VIOLENCE_QUESTIONS = [
    {"id": "physical_contact", "question": "Is there physical contact between two or more people in this image?", "weight": 2.0},
    {"id": "hitting_punching", "question": "Is anyone hitting, punching, kicking, or striking another person?", "weight": 3.0},
    {"id": "pushing_shoving", "question": "Is anyone pushing, shoving, or grabbing another person aggressively?", "weight": 2.5},
    {"id": "person_on_ground", "question": "Is anyone lying on the ground in a non-relaxed position (fallen, pushed down)?", "weight": 2.0},
    {"id": "restraining", "question": "Is anyone being physically restrained, held down, or pinned?", "weight": 2.0},
    {"id": "aggressive_stance", "question": "Are any people in aggressive stances (raised fists, confrontational positioning)?", "weight": 1.5},
    {"id": "defensive_posture", "question": "Is anyone in a defensive posture (arms up protecting face, cowering, backing away)?", "weight": 1.5},
    {"id": "running_fleeing", "question": "Are people running or fleeing in panic (not jogging or casual movement)?", "weight": 1.5},
    {"id": "sudden_movement", "question": "Does the image show signs of sudden, rapid, or chaotic movement?", "weight": 1.0},
    {"id": "weapon_visible", "question": "Is there a weapon visible (gun, knife, bat, stick, bottle used as weapon)?", "weight": 3.0},
    {"id": "holding_object_threateningly", "question": "Is anyone holding an object in a threatening manner?", "weight": 2.0},
    {"id": "crowd_surrounding", "question": "Is a crowd surrounding or encircling a specific person or small group?", "weight": 1.5},
    {"id": "crowd_dispersing", "question": "Is a crowd rapidly dispersing or scattering?", "weight": 1.5},
    {"id": "group_confrontation", "question": "Are two groups of people facing each other in a confrontational manner?", "weight": 2.0},
    {"id": "distress_visible", "question": "Are there visible signs of distress, fear, or pain on anyone's face?", "weight": 1.5},
    {"id": "anger_aggression", "question": "Does anyone appear visibly angry or aggressive in their expression?", "weight": 1.5},
    {"id": "damage_visible", "question": "Is there visible damage or destruction (broken objects, overturned items)?", "weight": 1.0},
    {"id": "blood_injury", "question": "Is there visible blood or signs of injury on anyone?", "weight": 2.5},
    {"id": "emergency_responders", "question": "Are emergency responders (police, medical personnel) present or arriving?", "weight": 1.0},
    {"id": "calm_interaction", "question": "Are people interacting calmly and peacefully?", "weight": -1.0},
    {"id": "sports_activity", "question": "Is this a sports or athletic activity?", "weight": -1.5},
    {"id": "performance_acting", "question": "Does this appear to be a performance, movie scene, or staged event?", "weight": -1.0},
    {"id": "playful_interaction", "question": "Are people engaged in playful or friendly physical interaction?", "weight": -1.5},
    {"id": "violence_occurring", "question": "Overall, is violence or a physical altercation occurring in this image?", "weight": 2.0},
    {"id": "violence_imminent", "question": "Does the situation suggest violence is about to occur (tension, confrontation building)?", "weight": 2.0},
]


# ==============================================================================
# Video Frame Loading
# ==============================================================================

def find_video_file(video_name: str, videos_dir: Path) -> Optional[Path]:
    """Find video file for a given video name."""
    
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MP4', '.AVI']
    
    # Try direct match
    for ext in extensions:
        path = videos_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    # Try in subdirectories
    for subdir in ['', 'xd_violence', 'ucf_crime', 'XD_Violence', 'UCF_Crime', 'videos']:
        base = videos_dir / subdir if subdir else videos_dir
        if not base.exists():
            continue
        
        for ext in extensions:
            path = base / f"{video_name}{ext}"
            if path.exists():
                return path
        
        # Recursive glob
        matches = list(base.glob(f"**/{video_name}.*"))
        for m in matches:
            if m.suffix.lower() in [e.lower() for e in extensions]:
                return m
    
    return None


def extract_frame_from_video(video_path: Path, frame_idx: int) -> Optional[Image.Image]:
    """Extract a single frame from a video file."""
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # If frame_idx seems like seconds (labels at 1 FPS), convert
    if frame_idx < total_frames:
        actual_frame = frame_idx
    else:
        actual_frame = min(int(frame_idx * fps), total_frames - 1) if fps > 0 else frame_idx
    
    actual_frame = min(max(0, actual_frame), total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def get_video_info(video_path: Path) -> Dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


# ==============================================================================
# VLM Extractor
# ==============================================================================

class SimpleVLMExtractor:
    """VLM-based question answering for violence detection."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda:0"):
        self.device = device
        self.questions = VIOLENCE_QUESTIONS
        
        print(f"Loading {model_name}...")
        
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
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        
        print(f"Loaded. Will ask {len(self.questions)} questions per image.")
    
    def _parse_yes_no(self, response: str) -> float:
        response = response.lower().strip()
        first_word = response.split()[0] if response.split() else ""
        
        if first_word in ['yes', 'yes,', 'yes.']:
            return 1.0
        if first_word in ['no', 'no,', 'no.']:
            return 0.0
        if 'yes' in response and 'no' not in response:
            return 0.8
        if 'no' in response and 'yes' not in response:
            return 0.2
        return 0.5
    
    @torch.no_grad()
    def extract_batch_questions(self, image: Image.Image) -> Tuple[np.ndarray, List[str]]:
        """Ask all questions in one prompt."""
        
        question_text = "Look at this image carefully. Answer each question with ONLY 'yes' or 'no', one answer per line:\n\n"
        for i, q in enumerate(self.questions):
            question_text += f"{i+1}. {q['question']}\n"
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question_text}]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        output_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        lines = response.strip().split('\n')
        features = []
        raw_answers = []
        
        for i, q in enumerate(self.questions):
            if i < len(lines):
                answer = re.sub(r'^\d+[\.\)]\s*', '', lines[i].strip())
                score = self._parse_yes_no(answer)
                raw_answers.append(answer[:50])
            else:
                score = 0.5
                raw_answers.append("[no response]")
            features.append(score)
        
        return np.array(features, dtype=np.float32), raw_answers


# ==============================================================================
# Sample Selection
# ==============================================================================

def sample_frames(
    videos_dir: Path,
    embeddings_dir: Path,
    labels_dir: Path,
    n_samples: int = 50,
    source_filter: str = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Sample frames from violence and non-violence segments."""
    
    violence_candidates = []
    nonviolence_candidates = []
    
    emb_files = list(embeddings_dir.rglob("*.npz"))
    print(f"Scanning {len(emb_files)} videos...")
    
    videos_found = 0
    
    for emb_path in tqdm(emb_files, desc="Finding videos"):
        video_name = emb_path.stem
        
        # Source filter
        if source_filter:
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            if source_filter == "xd_violence" and not is_xd:
                continue
            if source_filter == "ucf_crime" and not is_ucf:
                continue
        
        # Find video file
        video_path = find_video_file(video_name, videos_dir)
        if video_path is None:
            continue
        
        videos_found += 1
        
        # Find labels
        label_path = labels_dir / f"{video_name}.npy"
        if not label_path.exists():
            label_path = labels_dir / f"{video_name}.npz"
        if not label_path.exists():
            continue
        
        # Load labels
        if label_path.suffix == ".npz":
            label_data = np.load(label_path)
            labels = label_data.get("labels", label_data.get("arr_0", None))
        else:
            labels = np.load(label_path)
        
        if labels is None:
            continue
        
        if labels.ndim > 1:
            labels = labels.max(axis=1)
        labels = (labels > 0).astype(int)
        
        video_info = get_video_info(video_path)
        
        for idx in np.where(labels == 1)[0]:
            violence_candidates.append({
                "video_name": video_name,
                "video_path": str(video_path),
                "frame_idx": int(idx),
            })
        
        for idx in np.where(labels == 0)[0]:
            nonviolence_candidates.append({
                "video_name": video_name,
                "video_path": str(video_path),
                "frame_idx": int(idx),
            })
    
    print(f"\nVideos found: {videos_found}")
    print(f"Violence frames: {len(violence_candidates)}, Non-violence: {len(nonviolence_candidates)}")
    
    if not violence_candidates or not nonviolence_candidates:
        return [], []
    
    random.seed(42)
    violence_samples = random.sample(violence_candidates, min(n_samples, len(violence_candidates)))
    nonviolence_samples = random.sample(nonviolence_candidates, min(n_samples, len(nonviolence_candidates)))
    
    print(f"Sampled {len(violence_samples)} violence, {len(nonviolence_samples)} non-violence")
    return violence_samples, nonviolence_samples


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_results(violence_features: np.ndarray, nonviolence_features: np.ndarray, output_dir: Path):
    """Analyze discriminative power."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [q["id"] for q in VIOLENCE_QUESTIONS]
    weights = np.array([q["weight"] for q in VIOLENCE_QUESTIONS])
    
    violence_means = violence_features.mean(axis=0)
    nonviolence_means = nonviolence_features.mean(axis=0)
    diff = violence_means - nonviolence_means
    
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    print(f"\n{'Feature':<30} {'Violence':>10} {'Non-Viol':>10} {'Diff':>10}")
    print("-"*62)
    
    sorted_idx = np.argsort(np.abs(diff))[::-1]
    for idx in sorted_idx:
        d = diff[idx]
        marker = "***" if abs(d) > 0.2 else "**" if abs(d) > 0.1 else ""
        print(f"{feature_names[idx]:<30} {violence_means[idx]:>10.3f} {nonviolence_means[idx]:>10.3f} {d:>+10.3f} {marker}")
    
    # Per-feature AUC
    all_features = np.vstack([violence_features, nonviolence_features])
    all_labels = np.array([1]*len(violence_features) + [0]*len(nonviolence_features))
    
    feature_aucs = []
    for i in range(len(feature_names)):
        try:
            auc = roc_auc_score(all_labels, all_features[:, i])
        except:
            auc = 0.5
        feature_aucs.append(auc)
    
    print("\n\nPer-Feature ROC-AUC (top 10):")
    sorted_auc_idx = np.argsort(feature_aucs)[::-1]
    for idx in sorted_auc_idx[:10]:
        print(f"  {feature_names[idx]:<30} {feature_aucs[idx]:.3f}")
    
    # Logistic regression
    n = len(all_labels)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.7*n)], idx[int(0.7*n):]
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(all_features[train_idx], all_labels[train_idx])
    
    train_auc = roc_auc_score(all_labels[train_idx], clf.predict_proba(all_features[train_idx])[:, 1])
    test_auc = roc_auc_score(all_labels[test_idx], clf.predict_proba(all_features[test_idx])[:, 1])
    test_acc = accuracy_score(all_labels[test_idx], clf.predict(all_features[test_idx]))
    
    print(f"\n\nLogistic Regression on 25 QA Features:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    
    # Weighted score
    pos_weights = np.maximum(weights, 0)
    pos_weights /= pos_weights.sum() + 1e-8
    neg_weights = np.maximum(-weights, 0)
    neg_weights /= neg_weights.sum() + 1e-8
    
    violence_scores = (violence_features @ pos_weights) - (violence_features @ neg_weights)
    nonviolence_scores = (nonviolence_features @ pos_weights) - (nonviolence_features @ neg_weights)
    all_scores = np.concatenate([violence_scores, nonviolence_scores])
    weighted_auc = roc_auc_score(all_labels, all_scores)
    
    print(f"\nWeighted Score AUC: {weighted_auc:.4f}")
    print(f"  Violence mean:     {violence_scores.mean():.3f} ± {violence_scores.std():.3f}")
    print(f"  Non-violence mean: {nonviolence_scores.mean():.3f} ± {nonviolence_scores.std():.3f}")
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Score distribution
    axes[0].hist(violence_scores, bins=15, alpha=0.6, label='Violence', color='red')
    axes[0].hist(nonviolence_scores, bins=15, alpha=0.6, label='Non-violence', color='blue')
    axes[0].set_xlabel('Weighted Score')
    axes[0].set_title(f'Score Distribution (AUC={weighted_auc:.3f})')
    axes[0].legend()
    
    # Feature AUCs
    colors = ['green' if a > 0.6 else 'orange' if a > 0.55 else 'gray' for a in np.array(feature_aucs)[sorted_auc_idx]]
    axes[1].barh(range(len(feature_aucs)), np.array(feature_aucs)[sorted_auc_idx], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(feature_aucs)))
    axes[1].set_yticklabels([feature_names[i] for i in sorted_auc_idx], fontsize=8)
    axes[1].axvline(x=0.5, color='red', linestyle='--')
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('Per-Feature AUC')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis.png', dpi=150)
    plt.close()
    
    return {
        "n_violence": len(violence_features),
        "n_nonviolence": len(nonviolence_features),
        "logistic_test_auc": float(test_auc),
        "logistic_test_acc": float(test_acc),
        "weighted_score_auc": float(weighted_auc),
        "top_features": [{"name": feature_names[i], "auc": float(feature_aucs[i])} for i in sorted_auc_idx[:5]],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--source_filter', type=str, default=None)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./qa_feature_test')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample
    print("="*70)
    print("SAMPLING FRAMES")
    print("="*70)
    
    violence_samples, nonviolence_samples = sample_frames(
        Path(args.videos_dir), Path(args.embeddings_dir), Path(args.labels_dir),
        args.n_samples, args.source_filter,
    )
    
    if not violence_samples:
        print("\nERROR: No samples found! Check --videos_dir path.")
        return
    
    # Load VLM
    print("\n" + "="*70)
    print("LOADING VLM")
    print("="*70)
    extractor = SimpleVLMExtractor(args.model, args.device)
    
    # Extract
    print("\n" + "="*70)
    print("EXTRACTING QA FEATURES")
    print("="*70)
    
    violence_features, nonviolence_features = [], []
    
    print("\nViolence samples...")
    for s in tqdm(violence_samples):
        img = extract_frame_from_video(Path(s['video_path']), s['frame_idx'])
        if img:
            feat, _ = extractor.extract_batch_questions(img)
            violence_features.append(feat)
    
    print("\nNon-violence samples...")
    for s in tqdm(nonviolence_samples):
        img = extract_frame_from_video(Path(s['video_path']), s['frame_idx'])
        if img:
            feat, _ = extractor.extract_batch_questions(img)
            nonviolence_features.append(feat)
    
    violence_features = np.stack(violence_features)
    nonviolence_features = np.stack(nonviolence_features)
    
    # Save
    np.savez(output_dir / 'features.npz', violence=violence_features, nonviolence=nonviolence_features)
    
    # Analyze
    summary = analyze_results(violence_features, nonviolence_features, output_dir)
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Interpretation
    print("\n" + "="*70)
    test_auc = summary['logistic_test_auc']
    if test_auc > 0.80:
        print("✅ EXCELLENT: QA features are highly discriminative!")
    elif test_auc > 0.70:
        print("✅ GOOD: QA features show promise.")
    elif test_auc > 0.60:
        print("⚠️  MODERATE: Some signal, but may need augmentation.")
    else:
        print("❌ WEAK: QA features don't discriminate well.")
    print("="*70)
    
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()