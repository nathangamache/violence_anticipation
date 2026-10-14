#!/usr/bin/env python3
"""
Pose and Motion Feature Extraction for Violence Anticipation

Extracts violence-relevant features using YOLOv8-pose:
1. Body keypoints (fighting stances, raised fists, defensive positions)
2. Interpersonal features (distance, relative positions)
3. Motion features (velocity, acceleration between frames)
4. Pose descriptors (angles, normalized distances)

These features directly capture violence-relevant signals that VLM embeddings miss.

Usage:
    python pose_feature_extraction.py \
        --videos_dir data/raw_videos/xd_violence/videos \
        --graphs_dir data/graphs \
        --output_dir data/processed/pose_features \
        --source_filter xd_violence \
        --device cuda:0
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


# COCO keypoint indices
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Key indices for violence-relevant features
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16


class PoseFeatureExtractor:
    """
    Extracts pose and motion features relevant to violence detection.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.5,
        max_persons: int = 10,
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_persons = max_persons
        
        print(f"Loading YOLOv8-pose model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        # Feature dimensions
        self.keypoint_dim = 17 * 3  # 17 keypoints x (x, y, conf)
        self.pose_descriptor_dim = 32  # Computed pose features
        self.interpersonal_dim = 20  # Features between people
        self.motion_dim = 17 * 2  # Velocity per keypoint
        
        # Total per-person features
        self.person_feature_dim = self.keypoint_dim + self.pose_descriptor_dim
        # Scene-level features
        self.scene_feature_dim = self.interpersonal_dim + 10  # + global stats
        
        # Previous frame for motion
        self.prev_keypoints = None
        self.prev_boxes = None
    
    def reset(self):
        """Reset state for new video."""
        self.prev_keypoints = None
        self.prev_boxes = None
    
    def extract_frame(self, frame: np.ndarray) -> Dict:
        """
        Extract all features from a single frame.
        
        Returns dict with:
            - keypoints: [N, 17, 3] raw keypoints
            - boxes: [N, 4] bounding boxes
            - person_features: [N, person_feature_dim]
            - scene_features: [scene_feature_dim]
            - motion_features: [N, motion_dim] (if prev frame available)
            - combined: [fixed_dim] fixed-size feature vector
        """
        # Run pose detection
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        keypoints_list = []
        boxes_list = []
        confs_list = []
        
        for r in results:
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints.data.cpu().numpy()  # [N, 17, 3]
                for i, kpt in enumerate(kpts):
                    if r.boxes is not None and i < len(r.boxes):
                        box = r.boxes.xyxy[i].cpu().numpy()
                        conf = r.boxes.conf[i].cpu().numpy()
                        keypoints_list.append(kpt)
                        boxes_list.append(box)
                        confs_list.append(conf)
        
        # Limit to max_persons (keep highest confidence)
        if len(keypoints_list) > self.max_persons:
            indices = np.argsort(confs_list)[::-1][:self.max_persons]
            keypoints_list = [keypoints_list[i] for i in indices]
            boxes_list = [boxes_list[i] for i in indices]
        
        n_persons = len(keypoints_list)
        
        if n_persons == 0:
            # No detections - return zeros
            return self._empty_features()
        
        keypoints = np.array(keypoints_list)  # [N, 17, 3]
        boxes = np.array(boxes_list)  # [N, 4]
        
        # Extract features
        person_features = self._extract_person_features(keypoints, boxes, frame.shape)
        scene_features = self._extract_scene_features(keypoints, boxes, frame.shape)
        motion_features = self._extract_motion_features(keypoints)
        
        # Combine into fixed-size vector
        combined = self._combine_features(
            person_features, scene_features, motion_features, n_persons
        )
        
        # Store for next frame
        self.prev_keypoints = keypoints.copy()
        self.prev_boxes = boxes.copy()
        
        return {
            'keypoints': keypoints,
            'boxes': boxes,
            'n_persons': n_persons,
            'person_features': person_features,
            'scene_features': scene_features,
            'motion_features': motion_features,
            'combined': combined,
        }
    
    def _empty_features(self) -> Dict:
        """Return empty features when no persons detected."""
        combined = np.zeros(self.get_feature_dim(), dtype=np.float32)
        return {
            'keypoints': np.array([]),
            'boxes': np.array([]),
            'n_persons': 0,
            'person_features': np.array([]),
            'scene_features': np.zeros(self.scene_feature_dim, dtype=np.float32),
            'motion_features': np.array([]),
            'combined': combined,
        }
    
    def get_feature_dim(self) -> int:
        """Get dimension of combined feature vector."""
        # Max persons * person features + scene features + motion summary + metadata
        return (self.max_persons * (self.pose_descriptor_dim + self.motion_dim // 2) + 
                self.scene_feature_dim + 20)
    
    def _extract_person_features(
        self, 
        keypoints: np.ndarray, 
        boxes: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract pose descriptor features for each person.
        
        Features include:
        - Normalized keypoint positions
        - Joint angles (elbow, knee, shoulder)
        - Limb lengths (normalized)
        - Pose indicators (arms raised, crouched, etc.)
        """
        h, w = frame_shape[:2]
        n_persons = len(keypoints)
        features = []
        
        for i in range(n_persons):
            kpts = keypoints[i]  # [17, 3]
            box = boxes[i]  # [4]
            
            feat = []
            
            # 1. Normalize keypoints to box
            box_w = max(box[2] - box[0], 1)
            box_h = max(box[3] - box[1], 1)
            box_cx = (box[0] + box[2]) / 2
            box_cy = (box[1] + box[3]) / 2
            
            # 2. Key pose indicators (violence-relevant)
            
            # Arms raised (wrists above shoulders)
            l_arm_raised = kpts[L_WRIST, 1] < kpts[L_SHOULDER, 1] if kpts[L_WRIST, 2] > 0.3 else 0
            r_arm_raised = kpts[R_WRIST, 1] < kpts[R_SHOULDER, 1] if kpts[R_WRIST, 2] > 0.3 else 0
            feat.extend([float(l_arm_raised), float(r_arm_raised)])
            
            # Arms extended forward (elbows relatively straight, wrists far from shoulders)
            l_arm_extended = self._arm_extension(kpts, 'left')
            r_arm_extended = self._arm_extension(kpts, 'right')
            feat.extend([l_arm_extended, r_arm_extended])
            
            # Fighting stance indicators
            shoulder_width = np.linalg.norm(kpts[L_SHOULDER, :2] - kpts[R_SHOULDER, :2])
            hip_width = np.linalg.norm(kpts[L_HIP, :2] - kpts[R_HIP, :2])
            stance_ratio = hip_width / (shoulder_width + 1e-6)  # Wide stance = higher
            feat.append(np.clip(stance_ratio, 0, 2))
            
            # Crouching (hips low relative to height)
            if kpts[L_HIP, 2] > 0.3 and kpts[R_HIP, 2] > 0.3:
                hip_y = (kpts[L_HIP, 1] + kpts[R_HIP, 1]) / 2
                crouch_ratio = (hip_y - box[1]) / box_h
                feat.append(np.clip(crouch_ratio, 0, 1))
            else:
                feat.append(0.5)
            
            # Leaning/lunging (nose far from hip center horizontally)
            if kpts[NOSE, 2] > 0.3:
                hip_cx = (kpts[L_HIP, 0] + kpts[R_HIP, 0]) / 2 if kpts[L_HIP, 2] > 0.3 else box_cx
                lean = abs(kpts[NOSE, 0] - hip_cx) / box_w
                feat.append(np.clip(lean, 0, 1))
            else:
                feat.append(0)
            
            # 3. Joint angles
            l_elbow_angle = self._joint_angle(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
            r_elbow_angle = self._joint_angle(kpts, R_SHOULDER, R_ELBOW, R_WRIST)
            l_knee_angle = self._joint_angle(kpts, L_HIP, L_KNEE, L_ANKLE)
            r_knee_angle = self._joint_angle(kpts, R_HIP, R_KNEE, R_ANKLE)
            l_shoulder_angle = self._joint_angle(kpts, L_ELBOW, L_SHOULDER, L_HIP)
            r_shoulder_angle = self._joint_angle(kpts, R_ELBOW, R_SHOULDER, R_HIP)
            
            # Normalize angles to [0, 1]
            feat.extend([
                l_elbow_angle / np.pi,
                r_elbow_angle / np.pi,
                l_knee_angle / np.pi,
                r_knee_angle / np.pi,
                l_shoulder_angle / np.pi,
                r_shoulder_angle / np.pi,
            ])
            
            # 4. Limb lengths (normalized by box size)
            l_upper_arm = np.linalg.norm(kpts[L_SHOULDER, :2] - kpts[L_ELBOW, :2]) / box_h
            l_lower_arm = np.linalg.norm(kpts[L_ELBOW, :2] - kpts[L_WRIST, :2]) / box_h
            r_upper_arm = np.linalg.norm(kpts[R_SHOULDER, :2] - kpts[R_ELBOW, :2]) / box_h
            r_lower_arm = np.linalg.norm(kpts[R_ELBOW, :2] - kpts[R_WRIST, :2]) / box_h
            
            feat.extend([
                np.clip(l_upper_arm, 0, 1),
                np.clip(l_lower_arm, 0, 1),
                np.clip(r_upper_arm, 0, 1),
                np.clip(r_lower_arm, 0, 1),
            ])
            
            # 5. Body orientation (facing camera vs side)
            shoulder_vec = kpts[R_SHOULDER, :2] - kpts[L_SHOULDER, :2]
            facing = abs(shoulder_vec[0]) / (np.linalg.norm(shoulder_vec) + 1e-6)
            feat.append(facing)
            
            # 6. Vertical distribution (head to feet ratio)
            if kpts[NOSE, 2] > 0.3 and (kpts[L_ANKLE, 2] > 0.3 or kpts[R_ANKLE, 2] > 0.3):
                head_y = kpts[NOSE, 1]
                foot_y = max(kpts[L_ANKLE, 1], kpts[R_ANKLE, 1])
                vertical_span = (foot_y - head_y) / box_h
                feat.append(np.clip(vertical_span, 0, 1.5))
            else:
                feat.append(0.5)
            
            # 7. Hand positions relative to body
            for wrist_idx in [L_WRIST, R_WRIST]:
                if kpts[wrist_idx, 2] > 0.3:
                    # Relative to hip center
                    hand_rel_x = (kpts[wrist_idx, 0] - box_cx) / box_w
                    hand_rel_y = (kpts[wrist_idx, 1] - box_cy) / box_h
                    feat.extend([np.clip(hand_rel_x, -1, 1), np.clip(hand_rel_y, -1, 1)])
                else:
                    feat.extend([0, 0])
            
            # 8. Keypoint confidence summary
            mean_conf = kpts[:, 2].mean()
            min_conf = kpts[:, 2].min()
            feat.extend([mean_conf, min_conf])
            
            # Pad to fixed size
            while len(feat) < self.pose_descriptor_dim:
                feat.append(0)
            
            features.append(np.array(feat[:self.pose_descriptor_dim], dtype=np.float32))
        
        return np.array(features)
    
    def _arm_extension(self, kpts: np.ndarray, side: str) -> float:
        """Calculate arm extension (0 = bent, 1 = fully extended)."""
        if side == 'left':
            shoulder, elbow, wrist = L_SHOULDER, L_ELBOW, L_WRIST
        else:
            shoulder, elbow, wrist = R_SHOULDER, R_ELBOW, R_WRIST
        
        if kpts[shoulder, 2] < 0.3 or kpts[elbow, 2] < 0.3 or kpts[wrist, 2] < 0.3:
            return 0.5
        
        angle = self._joint_angle(kpts, shoulder, elbow, wrist)
        # Straight arm = pi, bent = smaller
        return angle / np.pi
    
    def _joint_angle(self, kpts: np.ndarray, p1: int, p2: int, p3: int) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        if kpts[p1, 2] < 0.3 or kpts[p2, 2] < 0.3 or kpts[p3, 2] < 0.3:
            return np.pi / 2  # Default to 90 degrees
        
        v1 = kpts[p1, :2] - kpts[p2, :2]
        v2 = kpts[p3, :2] - kpts[p2, :2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.arccos(cos_angle)
    
    def _extract_scene_features(
        self,
        keypoints: np.ndarray,
        boxes: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract scene-level features (interactions between people).
        
        Features include:
        - Number of people
        - Minimum distance between people
        - Distance statistics
        - Grouping patterns
        - Face-to-face detection
        """
        h, w = frame_shape[:2]
        n_persons = len(keypoints)
        
        feat = []
        
        # 1. Person count (normalized)
        feat.append(min(n_persons / self.max_persons, 1.0))
        feat.append(float(n_persons))
        
        if n_persons < 2:
            # Pad with zeros for interpersonal features
            feat.extend([0] * (self.scene_feature_dim - 2))
            return np.array(feat[:self.scene_feature_dim], dtype=np.float32)
        
        # 2. Compute pairwise distances
        centers = []
        for i in range(n_persons):
            cx = (boxes[i, 0] + boxes[i, 2]) / 2
            cy = (boxes[i, 1] + boxes[i, 3]) / 2
            centers.append([cx, cy])
        centers = np.array(centers)
        
        distances = []
        for i in range(n_persons):
            for j in range(i + 1, n_persons):
                d = np.linalg.norm(centers[i] - centers[j])
                distances.append(d / max(w, h))  # Normalize by frame size
        
        distances = np.array(distances)
        
        # Distance statistics
        feat.append(distances.min())
        feat.append(distances.max())
        feat.append(distances.mean())
        feat.append(distances.std() if len(distances) > 1 else 0)
        
        # 3. Close proximity count (potential confrontations)
        close_threshold = 0.15  # 15% of frame size
        n_close_pairs = (distances < close_threshold).sum()
        feat.append(n_close_pairs / max(len(distances), 1))
        
        # 4. Face-to-face detection
        n_facing = 0
        for i in range(n_persons):
            for j in range(i + 1, n_persons):
                if self._are_facing(keypoints[i], keypoints[j]):
                    n_facing += 1
        feat.append(n_facing / max(len(distances), 1))
        
        # 5. Physical contact likelihood (overlapping boxes)
        n_overlapping = 0
        for i in range(n_persons):
            for j in range(i + 1, n_persons):
                if self._boxes_overlap(boxes[i], boxes[j]):
                    n_overlapping += 1
        feat.append(n_overlapping / max(len(distances), 1))
        
        # 6. Crowd clustering
        # Simple: variance of positions indicates spread vs clustering
        position_var = centers.var(axis=0).sum() / (w * h)
        feat.append(np.clip(position_var, 0, 1))
        
        # 7. Height variance (some standing, some fallen)
        heights = boxes[:, 3] - boxes[:, 1]
        height_var = heights.std() / (heights.mean() + 1e-6)
        feat.append(np.clip(height_var, 0, 2))
        
        # 8. Aspect ratio variance (some upright, some horizontal/fallen)
        aspects = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1] + 1e-6)
        aspect_var = aspects.std()
        feat.append(np.clip(aspect_var, 0, 2))
        
        # Pad to fixed size
        while len(feat) < self.scene_feature_dim:
            feat.append(0)
        
        return np.array(feat[:self.scene_feature_dim], dtype=np.float32)
    
    def _are_facing(self, kpts1: np.ndarray, kpts2: np.ndarray) -> bool:
        """Check if two people are facing each other."""
        # Use shoulder orientation
        if (kpts1[L_SHOULDER, 2] < 0.3 or kpts1[R_SHOULDER, 2] < 0.3 or
            kpts2[L_SHOULDER, 2] < 0.3 or kpts2[R_SHOULDER, 2] < 0.3):
            return False
        
        # Vector from person 1's center to person 2
        c1 = (kpts1[L_SHOULDER, :2] + kpts1[R_SHOULDER, :2]) / 2
        c2 = (kpts2[L_SHOULDER, :2] + kpts2[R_SHOULDER, :2]) / 2
        
        # Shoulder vectors (perpendicular to facing direction)
        s1 = kpts1[R_SHOULDER, :2] - kpts1[L_SHOULDER, :2]
        s2 = kpts2[R_SHOULDER, :2] - kpts2[L_SHOULDER, :2]
        
        # Connection vector
        conn = c2 - c1
        
        # Person 1 facing person 2 if shoulder vector is roughly perpendicular to connection
        # and vice versa (dot product of shoulder with connection should be small)
        dot1 = abs(np.dot(s1, conn)) / (np.linalg.norm(s1) * np.linalg.norm(conn) + 1e-6)
        dot2 = abs(np.dot(s2, conn)) / (np.linalg.norm(s2) * np.linalg.norm(conn) + 1e-6)
        
        # Both should have small dot products (shoulders perpendicular to connection line)
        return dot1 < 0.5 and dot2 < 0.5
    
    def _boxes_overlap(self, box1: np.ndarray, box2: np.ndarray) -> bool:
        """Check if two boxes overlap."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        return x1 < x2 and y1 < y2
    
    def _extract_motion_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract motion features by comparing with previous frame.
        
        Returns velocity of keypoints (normalized).
        """
        n_persons = len(keypoints)
        
        if self.prev_keypoints is None or len(self.prev_keypoints) == 0:
            # No previous frame - return zeros
            return np.zeros((n_persons, self.motion_dim), dtype=np.float32)
        
        motion_features = []
        
        for i in range(n_persons):
            if i < len(self.prev_keypoints):
                # Match by index (simple approach)
                prev_kpts = self.prev_keypoints[i]
                curr_kpts = keypoints[i]
                
                velocities = []
                for j in range(17):
                    if curr_kpts[j, 2] > 0.3 and prev_kpts[j, 2] > 0.3:
                        vel = curr_kpts[j, :2] - prev_kpts[j, :2]
                        velocities.extend(vel / 100)  # Normalize
                    else:
                        velocities.extend([0, 0])
                
                motion_features.append(np.array(velocities, dtype=np.float32))
            else:
                motion_features.append(np.zeros(self.motion_dim, dtype=np.float32))
        
        return np.array(motion_features)
    
    def _combine_features(
        self,
        person_features: np.ndarray,
        scene_features: np.ndarray,
        motion_features: np.ndarray,
        n_persons: int,
    ) -> np.ndarray:
        """
        Combine all features into a fixed-size vector.
        
        Strategy:
        - Scene features: always include
        - Person features: pad/truncate to max_persons
        - Motion features: summarize (mean, max)
        """
        combined = []
        
        # 1. Scene features
        combined.extend(scene_features.tolist())
        
        # 2. Person features (padded to max_persons)
        for i in range(self.max_persons):
            if i < len(person_features):
                combined.extend(person_features[i].tolist())
            else:
                combined.extend([0] * self.pose_descriptor_dim)
        
        # 3. Motion summary
        if len(motion_features) > 0 and motion_features.size > 0:
            motion_flat = motion_features.flatten()
            combined.append(np.abs(motion_flat).mean())  # Overall motion magnitude
            combined.append(np.abs(motion_flat).max())   # Max motion
            combined.append(np.abs(motion_flat).std())   # Motion variance
            
            # Per-keypoint motion summary (average across persons)
            if len(motion_features.shape) > 1:
                mean_motion = np.abs(motion_features).mean(axis=0)
                # Take top-k most moving keypoints
                top_k = min(5, len(mean_motion) // 2)
                top_motion = np.sort(mean_motion)[-top_k:]
                combined.extend(top_motion.tolist())
            else:
                combined.extend([0] * 5)
        else:
            combined.extend([0] * 8)
        
        # 4. Metadata
        combined.append(float(n_persons))
        combined.append(1.0 if n_persons > 0 else 0.0)  # Detection flag
        
        # Pad to fixed size
        target_dim = self.get_feature_dim()
        while len(combined) < target_dim:
            combined.append(0)
        
        return np.array(combined[:target_dim], dtype=np.float32)


def extract_video_features(
    video_path: Path,
    extractor: PoseFeatureExtractor,
    every_t: int = 1,
    total_frames_hint: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract pose features from a video.
    
    Args:
        video_path: Path to video file
        extractor: PoseFeatureExtractor instance
        every_t: Sample every N seconds
        total_frames_hint: Expected number of feature frames (from graph)
    
    Returns:
        features: [T, feature_dim] array
        frame_indices: list of frame indices that were processed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine sampling
    if total_frames_hint is not None:
        # Match the graph sampling
        t_steps = list(range(0, total_frames_hint, max(1, every_t)))
    else:
        duration = total_frames / fps
        t_steps = list(range(0, int(duration), max(1, every_t)))
    
    extractor.reset()
    features = []
    frame_indices = []
    
    for t in t_steps:
        # Convert t (seconds) to frame index
        frame_idx = int(t * fps)
        frame_idx = min(frame_idx, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            # Use empty features
            feat = extractor._empty_features()['combined']
        else:
            feat = extractor.extract_frame(frame)['combined']
        
        features.append(feat)
        frame_indices.append(frame_idx)
    
    cap.release()
    
    return np.array(features), frame_indices


def build_video_index(root: str) -> Dict[str, str]:
    """Map video stem -> absolute path."""
    out = {}
    rp = Path(root)
    if not rp.exists():
        return out
    
    for p in rp.rglob("*.mp4"):
        stem = p.stem
        out[stem] = str(p)
        out[stem.lower()] = str(p)
        if stem.endswith("_x264"):
            base = stem[:-5]
            out[base] = str(p)
            out[base.lower()] = str(p)
    
    return out


def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--graphs_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed/pose_features')
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'])
    parser.add_argument('--every_t', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='yolov8m-pose.pt')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build video index
    print("Building video index...")
    video_index = build_video_index(args.videos_dir)
    print(f"Found {len(video_index) // 2} videos")
    
    # Get video IDs from graphs
    graph_files = sorted([f for f in os.listdir(args.graphs_dir) 
                         if f.endswith('.npy') or f.endswith('.pt')])
    vid_ids = [os.path.splitext(f)[0] for f in graph_files]
    
    # Filter by source
    if args.source_filter:
        vid_ids = [v for v in vid_ids if classify_video_source(v) == args.source_filter]
    
    # Shard
    vid_ids = vid_ids[args.rank::args.world_size]
    print(f"Processing {len(vid_ids)} videos (rank {args.rank}/{args.world_size})")
    
    # Initialize extractor
    extractor = PoseFeatureExtractor(
        model_name=args.model,
        device=args.device,
    )
    
    feature_dim = extractor.get_feature_dim()
    print(f"Feature dimension: {feature_dim}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for vid_id in tqdm(vid_ids, desc="Extracting pose features"):
        source = classify_video_source(vid_id)
        out_subdir = output_dir / source
        out_subdir.mkdir(exist_ok=True)
        
        out_path = out_subdir / f"{vid_id}.npz"
        
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        
        # Find video
        video_path = video_index.get(vid_id) or video_index.get(vid_id.lower())
        if not video_path:
            errors += 1
            continue
        
        # Get T from graph
        graph_path = Path(args.graphs_dir) / f"{vid_id}.npy"
        if not graph_path.exists():
            graph_path = Path(args.graphs_dir) / f"{vid_id}.pt"
        
        try:
            if graph_path.suffix == '.npy':
                seq = np.load(graph_path, allow_pickle=True)
                T = len(seq)
            else:
                import torch
                seq = torch.load(graph_path)
                T = len(seq)
        except Exception as e:
            errors += 1
            continue
        
        try:
            features, frame_indices = extract_video_features(
                Path(video_path),
                extractor,
                every_t=args.every_t,
                total_frames_hint=T,
            )
            
            np.savez_compressed(
                out_path,
                features=features,
                frame_indices=np.array(frame_indices),
                feature_dim=feature_dim,
            )
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {vid_id}: {e}")
            errors += 1
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"Features saved to: {output_dir}")


if __name__ == '__main__':
    main()