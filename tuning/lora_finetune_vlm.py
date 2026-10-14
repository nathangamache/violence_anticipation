#!/usr/bin/env python3
"""
LoRA fine-tuning of Qwen2.5-VL as a yes/no violence VQA (V3 - Stability Fixed).

Usage:
    python tuning/lora_finetune_vlm.py --mode train \
        --videos_dir data/raw_videos/xd_violence/videos --labels_dir data/processed/labels_multih \
        --embeddings_dir data/processed/vlm_emb --output_dir ./lora_finetuned \
        --source_filter xd_violence --device cuda:0 --log_file logs/lora_train.log

    Multi-GPU: one process per GPU with --device cuda:K --worker_id K --num_workers N.
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime

# Suppress noisy warnings BEFORE imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', message='.*padding.*')
warnings.filterwarnings('ignore', message='.*token.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm

cv2.setLogLevel(0)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. Install with: pip install peft")


def setup_logging(log_file: Optional[str] = None, worker_id: int = 0):
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if worker_id > 0:
            log_path = log_path.parent / f"{log_path.stem}_worker{worker_id}{log_path.suffix}"
        
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
        force=True,
    )
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    max_samples_per_class: int = 2000
    val_split: float = 0.1
    image_size: int = 336
    
    device: str = "cuda:0"
    use_bf16: bool = True  # more stable than FP16
    
    worker_id: int = 0
    num_workers: int = 1
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def find_video_file(video_name: str, videos_dir: Path) -> Optional[Path]:
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MP4', '.AVI']
    
    for ext in extensions:
        path = videos_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    for subdir in ['', 'xd_violence', 'ucf_crime', 'XD_Violence', 'UCF_Crime']:
        base = videos_dir / subdir if subdir else videos_dir
        if not base.exists():
            continue
        for ext in extensions:
            path = base / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def extract_frame(video_path: Path, frame_idx: int, target_size: int = 336) -> Optional[Image.Image]:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # label index is treated as seconds (known misalignment, see docs/codebase/vlm_and_llm.md)
        actual_frame = min(int(frame_idx * fps), total_frames - 1)
        actual_frame = max(0, actual_frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None
        
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((target_size, target_size), Image.LANCZOS)
        
        return image
    except Exception:
        return None


class ViolenceDataset(Dataset):
    """Dataset for violence classification fine-tuning."""
    
    def __init__(
        self,
        samples: List[Dict],
        processor,
        image_size: int = 336,
        logger=None,
    ):
        self.samples = samples
        self.processor = processor
        self.image_size = image_size
        self.logger = logger
        
        self._validate_samples()
    
    def _validate_samples(self):
        valid = []
        for s in self.samples:
            if Path(s['video_path']).exists():
                valid.append(s)
        
        removed = len(self.samples) - len(valid)
        if removed > 0 and self.logger:
            self.logger.info(f"Filtered out {removed} invalid samples")
        
        self.samples = valid
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = extract_frame(
            Path(sample['video_path']),
            sample['frame_idx'],
            self.image_size
        )
        
        if image is None:
            return None
        
        target = "Yes" if sample['label'] == 1 else "No"
        
        return {
            'image': image,
            'label': sample['label'],
            'target': target,
        }


def collate_fn_simple(batch, processor, device):
    """Simplified collate that processes one sample at a time."""
    batch = [b for b in batch if b is not None]
    
    if not batch:
        return None
    
    try:
        results = []
        
        for sample in batch:
            image = sample['image']
            target = sample['target']
            label = sample['label']
            
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Is there violence in this image? Answer yes or no."},
                    ],
                },
            ]
            
            prompt_text = processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            full_text = prompt_text + target
            
            inputs = processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            
            # Process just the prompt to find where response starts
            prompt_inputs = processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            input_ids = inputs['input_ids']
            labels = input_ids.clone()
            
            labels[:, :prompt_length] = -100
            
            if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token_id is not None:
                labels[input_ids == processor.tokenizer.pad_token_id] = -100
            
            results.append({
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'pixel_values': inputs.get('pixel_values'),
                'image_grid_thw': inputs.get('image_grid_thw'),
                'labels': labels,
                'class_label': label,
            })
        
        return {
            'input_ids': torch.cat([r['input_ids'] for r in results], dim=0),
            'attention_mask': torch.cat([r['attention_mask'] for r in results], dim=0),
            'pixel_values': torch.cat([r['pixel_values'] for r in results], dim=0) if results[0]['pixel_values'] is not None else None,
            'image_grid_thw': torch.cat([r['image_grid_thw'] for r in results], dim=0) if results[0]['image_grid_thw'] is not None else None,
            'labels': torch.cat([r['labels'] for r in results], dim=0),
            'class_labels': torch.tensor([r['class_label'] for r in results]),
        }
        
    except Exception as e:
        logging.debug(f"Collate error: {e}")
        return None


def prepare_dataset(
    videos_dir: Path,
    labels_dir: Path,
    embeddings_dir: Path,
    max_samples_per_class: int = 2000,
    source_filter: str = None,
    worker_id: int = 0,
    num_workers: int = 1,
    logger=None,
) -> Tuple[List[Dict], List[Dict]]:
    """Prepare training samples."""
    
    violence_samples = []
    nonviolence_samples = []
    
    emb_files = sorted(list(embeddings_dir.rglob("*.npz")))
    
    if source_filter:
        filtered = []
        for emb_path in emb_files:
            video_name = emb_path.stem
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            if source_filter == 'xd_violence' and is_xd:
                filtered.append(emb_path)
            elif source_filter == 'ucf_crime' and is_ucf:
                filtered.append(emb_path)
        emb_files = filtered
    
    if num_workers > 1:
        emb_files = [f for i, f in enumerate(emb_files) if i % num_workers == worker_id]
    
    if logger:
        logger.info(f"Worker {worker_id}: Processing {len(emb_files)} videos")
    
    for emb_path in tqdm(emb_files, desc="Building dataset", disable=worker_id > 0):
        video_name = emb_path.stem
        
        video_path = find_video_file(video_name, videos_dir)
        if video_path is None:
            continue
        
        label_path = labels_dir / f"{video_name}.npy"
        if not label_path.exists():
            label_path = labels_dir / f"{video_name}.npz"
        if not label_path.exists():
            continue
        
        try:
            if label_path.suffix == '.npz':
                labels = np.load(label_path)
                labels = labels.get('labels', labels.get('arr_0'))
            else:
                labels = np.load(label_path)
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(int)
        except Exception:
            continue
        
        violence_idx = np.where(labels == 1)[0]
        nonviolence_idx = np.where(labels == 0)[0]
        
        max_per_video = 20
        if len(violence_idx) > max_per_video:
            violence_idx = np.random.choice(violence_idx, max_per_video, replace=False)
        if len(nonviolence_idx) > max_per_video:
            nonviolence_idx = np.random.choice(nonviolence_idx, max_per_video, replace=False)
        
        for idx in violence_idx:
            violence_samples.append({
                'video_name': video_name,
                'video_path': str(video_path),
                'frame_idx': int(idx),
                'label': 1,
            })
        
        for idx in nonviolence_idx:
            nonviolence_samples.append({
                'video_name': video_name,
                'video_path': str(video_path),
                'frame_idx': int(idx),
                'label': 0,
            })
    
    if logger:
        logger.info(f"Found {len(violence_samples)} violence, {len(nonviolence_samples)} non-violence")
    
    random.seed(42 + worker_id)
    
    if len(violence_samples) > max_samples_per_class:
        violence_samples = random.sample(violence_samples, max_samples_per_class)
    if len(nonviolence_samples) > max_samples_per_class:
        nonviolence_samples = random.sample(nonviolence_samples, max_samples_per_class)
    
    all_samples = violence_samples + nonviolence_samples
    random.shuffle(all_samples)
    
    n_val = int(len(all_samples) * 0.1)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    
    if logger:
        logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    return train_samples, val_samples


def setup_model_and_lora(config: TrainingConfig, logger):
    """Load model and apply LoRA with stability settings."""
    
    logger.info(f"Loading {config.model_name}...")
    
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {dtype}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=False,  # Use slow processor for consistency
    )
    
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT not installed. Run: pip install peft")
    
    # Prepare model for training BEFORE applying LoRA
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="gaussian",
    )
    
    model = get_peft_model(model, lora_config)
    
    model.train()
    
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Trainable: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")
    
    if trainable_params == 0:
        logger.error("No trainable parameters! Check LoRA configuration.")
        raise RuntimeError("No trainable parameters found")
    
    return model, processor
    
    return model, processor


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    logger,
    scaler=None,
):
    """Train one epoch with robust NaN handling."""
    
    model.train()
    total_loss = 0.0
    valid_steps = 0
    nan_count = 0
    skip_count = 0
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=config.worker_id > 0)
    
    for step, batch in enumerate(pbar):
        if batch is None:
            skip_count += 1
            continue
        
        try:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(config.device)
            
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(config.device)
            
            # no autocast: it causes gradient issues with PEFT
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                nan_count += 1
                if nan_count <= 10:
                    logger.warning(f"Bad loss at step {step}: {loss.item():.4f}")
                optimizer.zero_grad()
                continue
            
            loss = loss / config.gradient_accumulation_steps
            
            loss.backward()
            
            accumulated_loss += loss.item()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.max_grad_norm
                )
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    nan_count += 1
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    continue
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += accumulated_loss * config.gradient_accumulation_steps
                valid_steps += 1
                accumulated_loss = 0.0
            
            if valid_steps > 0:
                pbar.set_postfix({
                    'loss': f"{total_loss/valid_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'nan': nan_count,
                })
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at step {step}")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                skip_count += 1
            else:
                logger.error(f"Error at step {step}: {e}")
                raise
    
    avg_loss = total_loss / max(valid_steps, 1)
    logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, valid={valid_steps}, nan={nan_count}, skip={skip_count}")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, processor, config: TrainingConfig, logger):
    """Yes/no accuracy via generate (known bug: input_ids already end with the target answer)."""
    
    model.eval()
    correct = 0
    total = 0
    
    # Force left padding for generation
    processor.tokenizer.padding_side = 'left'
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=config.worker_id > 0):
        if batch is None:
            continue
        
        try:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            class_labels = batch['class_labels']
            
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(config.device)
            
            image_grid_thw = batch.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(config.device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            
            for i, output in enumerate(outputs):
                text = processor.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
                text = text.lower()
                
                pred = 1 if 'yes' in text else 0
                correct += (pred == class_labels[i].item())
                total += 1
                
        except Exception as e:
            logger.debug(f"Eval error: {e}")
            continue
    
    acc = correct / max(total, 1)
    logger.info(f"Validation: {correct}/{total} = {acc:.4f}")
    
    return acc


def train(
    config: TrainingConfig,
    train_samples: List[Dict],
    val_samples: List[Dict],
    output_dir: Path,
    logger,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, processor = setup_model_and_lora(config, logger)
    
    train_dataset = ViolenceDataset(train_samples, processor, config.image_size, logger)
    val_dataset = ViolenceDataset(val_samples, processor, config.image_size, logger)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=lambda b: collate_fn_simple(b, processor, config.device),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn_simple(b, processor, config.device),
    )
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        logger.info("Using standard AdamW")
    
    total_steps = (len(train_loader) * config.epochs) // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.learning_rate / 10,
    )
    
    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            config, epoch, logger
        )
        
        if not math.isnan(train_loss):
            history['train_loss'].append(float(train_loss))
        
        val_acc = evaluate(model, val_loader, processor, config, logger)
        history['val_acc'].append(float(val_acc))
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(output_dir / 'best_model')
            processor.save_pretrained(output_dir / 'best_model')
            logger.info(f"Saved best model (acc={best_acc:.4f})")
        
        model.save_pretrained(output_dir / f'checkpoint_epoch{epoch+1}')
    
    model.save_pretrained(output_dir / 'final_model')
    processor.save_pretrained(output_dir / 'final_model')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete. Best acc: {best_acc:.4f}")
    
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'extract'])
    
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, default=None)
    parser.add_argument('--embeddings_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./lora_finetuned')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_samples_per_class', type=int, default=2000)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--source_filter', type=str, default=None, choices=['xd_violence', 'ucf_crime'])
    
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    
    parser.add_argument('--log_file', type=str, default=None)
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_file, args.worker_id)
    logger.info(f"LoRA Fine-tuning V3 - Mode: {args.mode}")
    logger.info(f"Worker {args.worker_id}/{args.num_workers}, Device: {args.device}")
    
    if args.mode == 'train':
        if not args.labels_dir or not args.embeddings_dir:
            logger.error("Need --labels_dir and --embeddings_dir")
            return
        
        config = TrainingConfig(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_samples_per_class=args.max_samples_per_class,
            device=args.device,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )
        
        train_samples, val_samples = prepare_dataset(
            Path(args.videos_dir),
            Path(args.labels_dir),
            Path(args.embeddings_dir),
            args.max_samples_per_class,
            args.source_filter,
            args.worker_id,
            args.num_workers,
            logger,
        )
        
        output_dir = Path(args.output_dir)
        if args.num_workers > 1:
            output_dir = output_dir / f"worker_{args.worker_id}"
        
        train(config, train_samples, val_samples, output_dir, logger)


if __name__ == '__main__':
    main()