#!/usr/bin/env python3
"""Debug which sources are detected in the dataset"""
import json
import re
from pathlib import Path
from collections import Counter

def extract_video_name(item):
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)

def classify_video_source(video_id: str) -> str:
    """Classify video source from its ID."""
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'

# Load train index
with open('train_index.json') as f:
    train_items = json.load(f)

print(f"Total items: {len(train_items)}")
print(f"\nFirst 5 video names:")
for item in train_items[:5]:
    name = extract_video_name(item)
    source = classify_video_source(name)
    print(f"  {name[:60]}... -> {source}")

# Count sources
source_counts = Counter()
for item in train_items:
    name = extract_video_name(item)
    source = classify_video_source(name)
    source_counts[source] += 1

print(f"\nSource distribution:")
for source, count in source_counts.most_common():
    print(f"  {source}: {count}")