"""Scan raw frames for onset curation without reading benchmark model outputs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

def decode_video(row):
    from anticipation.features.io import Video

    scale = 448 / max(float(row['width']), float(row['height']))
    return Video(row['path'], num_threads=1, width=max(1, round(float(row['width']) * scale)),
                 height=max(1, round(float(row['height']) * scale)))


MODEL = 'models/Qwen2.5-VL-7B-Instruct'
QUESTION = ('Is anyone physically attacking, hitting, pushing, grabbing, kicking, stabbing or shooting at '
            'another person right now? Include police use of force, objects thrown at a person and vehicles '
            'used as weapons. Do not count arguing, pointing or squaring up alone. Answer only Yes or No.')


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def scan_times(event, duration):
    start = max(event['clear_start'], event['onset'] - 120)
    end = min(event['onset'] + 15, np.nextafter(duration, 0))
    return np.arange(start, end + 1e-8, 1).tolist()


def flag_frames(times, scores, onset, threshold):
    times, scores = np.asarray(times), np.asarray(scores)
    pre = np.flatnonzero(times < onset)
    high = scores > threshold
    candidate = None
    for j in range(max(0, len(pre) - 4)):
        win = pre[j:j + 5]
        if high[win].sum() >= 3:
            candidate = float(times[win[high[win]][0]])
            break
    post = (times >= onset) & (times <= onset + 15)
    return {'early_violence': candidate is not None, 'candidate_onset_sec': candidate,
            'no_violence_at_onset': bool(post.any() and not high[post].any()),
            'n_pre_above': int(high[pre].sum()), 'n_post': int(post.sum()),
            'n_post_above': int(high[post].sum())}


class Scorer:
    def __init__(self, batch_size):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        torch.set_num_threads(2)
        self.torch, self.batch_size = torch, batch_size
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True,
                                                       min_pixels=448 * 448, max_pixels=448 * 448)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', local_files_only=True).eval()
        self.answer_ids = {}
        for label in ('Yes', 'No'):
            ids = self.processor.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'{label} is not a single answer token: {ids}')
            self.answer_ids[label] = ids[0]
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': QUESTION}]}]
        self.prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score(self, frames):
        from PIL import Image
        out = []
        for start in range(0, len(frames), self.batch_size):
            images = [Image.fromarray(x) for x in frames[start:start + self.batch_size]]
            inputs = self.processor(text=[self.prompt] * len(images), images=images,
                                    padding=True, return_tensors='pt').to('cuda:0')
            with self.torch.inference_mode():
                logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
                lp = logits.log_softmax(-1)
                yes, no = lp[:, self.answer_ids['Yes']], lp[:, self.answer_ids['No']]
                prob = self.torch.sigmoid(yes - no)
            out.extend([{'p_yes': float(p), 'logp_yes': float(y), 'logp_no': float(n)}
                        for p, y, n in zip(prob.cpu(), yes.cpu(), no.cpu())])
        return out


def calibration_plan(videos, catalog, seed):
    rng = np.random.default_rng(seed)
    clean = sorted(v for v, row in catalog.items() if row['role'] == 'clean_negative')
    if not clean:
        clean = sorted(v for v, row in catalog.items() if row['is_clean_negative'] == 'True')
    rng.shuffle(clean)
    plan = []
    for v in clean[:800]:
        duration = float(catalog[v]['duration_sec'])
        plan.append({'video_id': v, 't_sec': float(rng.uniform(0, max(0, duration - 1))), 'label': 'clean'})
    violent = sorted((v, s, e) for v, row in videos.items() if row['role'] == 'violent'
                     for s, e in row['segments'] if e - s >= 2)
    rng.shuffle(violent)
    for v, s, e in violent[:400]:
        plan.append({'video_id': v, 't_sec': float(rng.uniform(s, e)), 'label': 'violent'})
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='data/annotation/onset_v3')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=301015)
    ap.add_argument('--max-seconds', type=float, default=10500)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'frames').mkdir(exist_ok=True)
    raw = Path('benchmark/events.json').read_bytes()
    videos = json.loads(raw)['videos']
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    targets = [(v, e) for v, row in sorted(videos.items()) for e in row.get('events', []) if e['eligible30']]
    manifest = {'model': MODEL, 'question': QUESTION, 'seed': args.seed, 'device': 'cuda:0',
                'dtype': 'bfloat16', 'pixels': 448 * 448, 'batch_size': args.batch_size,
                'events_sha256': hashlib.sha256(raw).hexdigest(),
                'videos_sha256': hashlib.sha256(Path('benchmark/videos.csv').read_bytes()).hexdigest(),
                'n_events': len(targets), 'sampling': '1 Hz from max(clear_start,onset-120), inclusive to onset+15',
                'score': 'P(Yes | Yes or No) = sigmoid(logp_Yes - logp_No)',
                'clean_fpr_target': 0.02, 'threshold_rule': '98th percentile clean scores, strict >'}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing scan manifest differs')
    dump(manifest_path, manifest)
    started = time.monotonic()
    scorer = Scorer(args.batch_size)
    cal_path = out / 'calibration.json'
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
    else:
        plan = calibration_plan(videos, catalog, args.seed)
        scored = []
        checkpoint = out / 'calibration_frames.json'
        if checkpoint.exists():
            scored = json.loads(checkpoint.read_text())
        for start in range(len(scored), len(plan), args.batch_size):
            chunk = plan[start:start + args.batch_size]
            frames = []
            for item in chunk:
                v = decode_video(catalog[item['video_id']])
                frames.append(v.frames_at([item['t_sec']])[0])
                del v
            scores = scorer.score(frames)
            scored.extend(dict(item, **score) for item, score in zip(chunk, scores))
            dump(checkpoint, scored)
            print(f'calibration {len(scored)}/{len(plan)} elapsed={time.monotonic() - started:.1f}s', flush=True)
        dump(checkpoint, scored)
        clean = np.array([r['p_yes'] for r in scored if r['label'] == 'clean'])
        violent = np.array([r['p_yes'] for r in scored if r['label'] == 'violent'])
        threshold = float(np.quantile(clean, 0.98, method='higher'))
        cal = {'threshold': threshold, 'n_clean': len(clean), 'n_violent': len(violent),
               'clean_fpr': float(np.mean(clean > threshold)),
               'annotated_violent_tpr': float(np.mean(violent > threshold)),
               'note': 'Calibration uses annotation classes, not visual ground truth; negatives may contain violence.'}
        dump(cal_path, cal)
    flags = []
    elapsed_path = out / 'runtime.json'
    previous = json.loads(elapsed_path.read_text())['total_seconds'] if elapsed_path.exists() else 0
    for i, (vid, event) in enumerate(targets[:args.limit]):
        path = out / 'frames' / (vid.replace('/', '__') + f"__{event['index']}.json")
        if path.exists():
            result = json.loads(path.read_text())
        else:
            if previous + time.monotonic() - started > args.max_seconds:
                dump(elapsed_path, {'total_seconds': previous + time.monotonic() - started, 'complete': False})
                raise RuntimeError('GPU time budget reached; scan incomplete')
            times = scan_times(event, videos[vid]['duration_sec'])
            v = decode_video(catalog[vid])
            scores = scorer.score(v.frames_at(times))
            del v
            result = {'video_id': vid, 'event_index': event['index'], 'onset_source': videos[vid]['onset_source'],
                      'original_onset_sec': event['onset'], 'clear_start': event['clear_start'],
                      'frames': [dict(t_sec=t, **s) for t, s in zip(times, scores)]}
            dump(path, result)
        flags.append({k: v for k, v in result.items() if k != 'frames'} | flag_frames(
            [r['t_sec'] for r in result['frames']], [r['p_yes'] for r in result['frames']],
            event['onset'], cal['threshold']))
        dump(out / 'flags.json', flags)
        dump(elapsed_path, {'total_seconds': previous + time.monotonic() - started,
                           'complete': len(flags) == len(targets), 'events_scored': len(flags)})
        print(f"event {i + 1}/{len(targets)} {vid} {event['index']} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    nonflagged = [r for r in flags if not (r['early_violence'] or r['no_violence_at_onset'])]
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(nonflagged), min(40, len(nonflagged)), replace=False)
    review = [dict(r, review_reason='flagged') for r in flags if r['early_violence'] or r['no_violence_at_onset']]
    review += [dict(nonflagged[i], review_reason='random_nonflagged') for i in sorted(indices)]
    dump(out / 'review_queue.json', review)
    print(json.dumps({'calibration': cal, 'flagged': len(review) - len(indices), 'random': len(indices)}), flush=True)


if __name__ == '__main__':
    main()
