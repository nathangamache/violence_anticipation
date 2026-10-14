#!/usr/bin/env python3
import os, json, torch, cv2
import numpy as np
from tqdm import tqdm
from transformers import (
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
)
from torch.utils.data import Dataset, DataLoader

# ======================================================
# CONFIG
# ======================================================
BASE = os.path.expanduser("~/projects/violence_anticipation/data")
VIDEO_DIR = f"{BASE}/raw_videos/kinetics/videos"
META = f"{BASE}/metadata/kinetics.json"

OUT_MODEL = os.path.expanduser(
    "~/projects/violence_anticipation/models/videomae_violence_finetuned.pt"
)

BATCH_SIZE = 2
LR = 3e-5
EPOCHS = 3
MAX_FRAMES = 32
SAMPLING = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# LOAD METADATA + CLASS MAP
# ======================================================
with open(META, "r") as f:
    metadata = json.load(f)

classes = sorted(list(set([info["label"] for info in metadata.values()])))
label_to_id = {name: i for i, name in enumerate(classes)}
id_to_label = {i: name for name, i in label_to_id.items()}


# ======================================================
# DATASET
# ======================================================
class KineticsViolenceDataset(Dataset):
    def __init__(self, video_dir, metadata):
        self.video_dir = video_dir
        self.metadata = list(metadata.items())
        self.feat = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

    def __len__(self):
        return len(self.metadata)

    def sample_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % SAMPLING == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) >= MAX_FRAMES:
                break
            idx += 1

        cap.release()
        return frames

    def __getitem__(self, idx):
        clip_id, info = self.metadata[idx]
        label = label_to_id[info["label"]]
        path = f"{self.video_dir}/{clip_id}.mp4"

        frames = self.sample_video(path)
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * MAX_FRAMES

        inputs = self.feat(frames, return_tensors="pt")
        return inputs["pixel_values"][0], torch.tensor(label)


# ======================================================
# TRAINING
# ======================================================
def train():
    dataset = KineticsViolenceDataset(VIDEO_DIR, metadata)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=len(classes),
        label2id=label_to_id,
        id2label=id_to_label,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            outputs = model(pixel_values=x)
            loss = criterion(outputs.logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({"loss": loss.item()})

    torch.save(model.state_dict(), OUT_MODEL)
    print(f"[FINETUNE] Saved model to {OUT_MODEL}")


if __name__ == "__main__":
    train()
