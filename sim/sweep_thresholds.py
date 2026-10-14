#!/usr/bin/env python3
"""Run sim/live_simulator.py at several thresholds and print a precision/recall/false-alarm table."""
import subprocess
import json
from pathlib import Path

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("Threshold | Precision | Recall | F1    | FalseAlarms/hr | Detection Rate")
print("-" * 80)

for thresh in thresholds:
    result = subprocess.run([
        'python', 'sim/live_simulator.py',
        '--model_path', 'runs/outputs_xd/best_model.pt',
        '--synthetic',
        '--embeddings_dir', 'data/processed/vlm_emb',
        '--labels_dir', 'data/processed/labels_multih', 
        '--test_index', 'test_index.json',
        '--source_filter', 'xd_violence',
        '--dilute_ratio', '10.0',
        '--threshold', str(thresh),
        '--output_dir', f'./simulation_sweep/thresh_{thresh}',
        '--device', 'cuda:1'
    ], capture_output=True, text=True)
    
    try:
        with open(f'./simulation_sweep/thresh_{thresh}/results.json') as f:
            r = json.load(f)
        
        fa_per_hr = r['n_false_alarms'] / (r['total_duration_sec'] / 3600)
        det_rate = r['n_detected_events'] / r['n_violence_events'] if r['n_violence_events'] > 0 else 0
        
        print(f"{thresh:.1f}       | {r['precision']:.3f}     | {r['recall']:.3f}  | {r['f1']:.3f} | {fa_per_hr:>14.1f} | {det_rate:.1%}")
    except Exception as e:
        print(f"{thresh:.1f}       | ERROR: {e}")

print("\nResults saved in ./simulation_sweep/")