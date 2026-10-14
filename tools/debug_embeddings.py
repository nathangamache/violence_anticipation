#!/usr/bin/env python3
"""Debug script to understand embedding structure"""
import numpy as np
from pathlib import Path
import sys

# Check a few embedding files
emb_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/processed/vlm_emb')

print(f"Scanning {emb_dir}")

for subdir in ['ucf_crime', 'xd_violence', '']:
    search_dir = emb_dir / subdir if subdir else emb_dir
    if not search_dir.exists():
        continue
    
    files = list(search_dir.glob('*.npz'))[:3] + list(search_dir.glob('*.npy'))[:3]
    
    if files:
        print(f"\n{search_dir}:")
        for f in files[:3]:
            try:
                if f.suffix == '.npz':
                    data = np.load(f)
                    print(f"  {f.name}:")
                    print(f"    Keys: {list(data.keys())}")
                    for k in list(data.keys())[:3]:
                        arr = data[k]
                        print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    arr = np.load(f)
                    print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception as e:
                print(f"  {f.name}: ERROR - {e}")