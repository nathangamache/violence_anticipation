# Violence Anticipation System
Predicting violence **up to 30 seconds before it happens** using multimodal deep learning.

This project implements a full end-to-end research pipeline for real-time violence anticipation using:

- YOLOv8 human detection  
- DeepSORT tracking  
- VideoMAE action embeddings  
- CLIP scene embeddings  
- Audio mel-spectrogram features  
- Graph construction per video frame  
- Graph Transformer + Temporal Transformer model  
- Sliding-window anticipation training  
- Real-time inference with live cameras  

This README covers everything you need to run, train, evaluate, and develop the system.

---

# 🔥 1. Project Structure
```bash
violence_anticipation/
│
├── core/ # model + extractors + graph builder (merged)
│ ├── model.py
│ ├── graph.py
│ ├── extractors.py
│ ├── utils.py
│
├── training/
│ ├── dataset.py
│ ├── train.py
│ ├── evaluate.py
│
├── realtime/
│ ├── pipeline.py
│
├── scripts/
│ ├── preprocess_all.py
│ ├── benchmark.py
│ ├── visualize_graphs.py
│ ├── export_trace_graph.py
│
├── configs/
│ ├── model.yaml
│ ├── training.yaml
│ ├── paths.yaml
│ └── features.yaml
│
├── tests/
│ ├── test_model_forward.py
│ ├── test_graph_builder.py
│ ├── test_extractors.py
│ ├── test_realtime_graph.py
│
└── README.md
```

---

# ⚙️ 2. Installation

## Install PyTorch (CUDA recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Install the rest
```bash
pip install ultralytics deep-sort-realtime librosa sounddevice clip-anytorch
pip install matplotlib opencv-python tqdm scikit-learn
pip install transformers timm einops
```
(If you need the complete list, see the "pip install all dependencies" section below.)

---

# 🎬 3. Dataset Preparation
Place your raw videos here:

```bash
data/raw_videos/
    video001.mp4
    video002.mp4
    ...
```
Create a metadata file describing violent event positions:

```yaml
metadata.json
{
  "video001": { "event_frame": 423, "total_frames": 2000 },
  "video002": { "event_frame": 845, "total_frames": 2200 },
  ...
}
```

---

# 🔧 4. Preprocessing (tracking → features → graphs → labels)
Run the full preprocessing script:

```bash
python scripts/preprocess_all.py
```

This produces:
```bash
data/processed/
    tracks/
    actions/
    audio/
    scenes/
    graphs/
    labels/
```

Each video becomes:
- `tracks/videoID.json` – YOLO + DeepSORT results
- `actions/videoID.npy` – VideoMAE embeddings
- `audio/videoID.npy` – mel-spectrogram summary
- `scenes/videoID.npy` – CLIP embeddings
- `graphs/videoID.npy` – graph sequence list
- `labels/videoID.npy` – anticipation labels

---

# 🧠 5. Model Architecture Overview
```bash
Video → YOLO → DeepSORT
                    ↘ per-person crops → VideoMAE → action embeddings
Video → CLIP → scene embedding
Audio → mel-spectrogram → audio embedding
Tracking + actions → graph nodes
Pairwise distances → graph edges

Graph Transformer → embeds each frame as a token
Temporal Transformer → predicts violence risk timeline
```

Node vector format:
```python
node = [center_x, center_y] + action_embedding(768)
dim = 770
```

Frame embedding:
```python
frame = avg(nodes) + scene + audio
```

---

# 🎓 6. Training the Model
```bash
python training/train.py
```

This will:
- Load the processed graph + label files
- Train the ViolenceTransformer
- Evaluate per epoch
- Save models into `checkpoints/`

Train config is controlled by:
```bash
configs/training.yaml
```

---

# 📈 7. Evaluating (AUC + Time-To-Detection)
Evaluation is automatically part of training, but you can call it manually:

```python
from training.evaluate import evaluate_model
```

Metrics:
- AUC — how well predictions differentiate violence vs. non-violence
- TTD — time-before-event detection (higher = better anticipation)

---

# 🛠 8. Running Realtime Prediction (Webcam or RTSP)
```bash
python realtime/pipeline.py
```

Options:
- `0` → laptop webcam
- `"rtsp://..."` → security camera
- `"video.mp4"` → video file

Example:
```bash
python realtime/pipeline.py --source rtsp://192.168.1.91/live
```

Output:
```bash
[Realtime] Violence Risk: 0.842
[Realtime] Violence Risk: 0.914
...
```

---

# 🚀 9. Benchmarking (FPS, model speed)
```bash
python scripts/benchmark.py
```

Output looks like:
```yaml
Extractor FPS: 8.22
Model FPS:     43.11
Total FPS:     7.20
```

---

# 👀 10. Visualizing Graphs
```bash
python scripts/visualize_graphs.py
```

# 📤 11. Export ONNX Model
```bash
python scripts/export_trace_graph.py
```

This exports: violence_transformer.onnx
Suitable for:
- TensorRT
- ONNX Runtime
- Server inference

---

# 🧪 12. Running Tests
Run all tests:
```bash
pytest -q
```

Tests cover:
- Model forward pass
- Graph builder
- Extractor imports
- Realtime graph shape
- Utility functions

---

# 📦 13. Installing Everything (Full Dependency List)
```bash
pip install \
    ultralytics \
    deep-sort-realtime \
    opencv-python \
    librosa \
    sounddevice \
    matplotlib \
    clip-anytorch \
    tqdm \
    scikit-learn \
    transformers \
    timm \
    einops \
    requests \
    pillow \
    pyyaml
```

---

# 🧭 14. Roadmap
- Add multi-resolution tracking
- Improve long-range anticipation (60–120 seconds)
- Add LLM fusion for human-activity semantics
- Add dataset mixing (UCF-Crime, Moments-in-Time, RLVS)

---

# 🙌 15. Contributing
All code is modular — simply update any part of:
- core/model.py
- core/extractors.py
- core/graph.py

And ensure tests still pass with:
```bash
pytest
```