"""Check real HF cache branching and intermediate-state hooks on tiny CPU models."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from anticipation.features import vlm_escalation as v

torch.set_num_threads(4)
results = []
for model_id, cls in [(v.MODEL_25, Qwen2_5_VLForConditionalGeneration), (v.MODEL_3, Qwen3VLForConditionalGeneration)]:
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(model_id)
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 4
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.rope_scaling = dict(rope_type='default', mrope_section=[2, 3, 3])
    visual = config.vision_config
    visual.hidden_size = 64
    visual.intermediate_size = 128
    visual.out_hidden_size = 64
    visual.num_heads = 4
    visual.depth = 2
    if config.model_type == 'qwen3_vl':
        visual.deepstack_visual_indexes = [0, 1]
        visual.num_position_embeddings = 64
    else:
        visual.fullatt_block_indexes = [0, 1]
    scorer = object.__new__(v.Scorer)
    scorer.is_qwen3 = config.model_type == 'qwen3_vl'
    scorer.device, scorer.max_pixels, scorer.max_edge = 'cpu', 128*128, 128
    scorer.layers = (1, 3)
    scorer.processor = AutoProcessor.from_pretrained(model_id)
    scorer.processor.tokenizer.padding_side = 'left'
    scorer.model = cls(config).float().eval()
    scorer.attach_hooks()
    scorer.priors = {}
    for arm in (['A', 'B', 'C'] if scorer.is_qwen3 else ['A', 'B']):
        times = v.clip_times(130, arm)
        frames = np.random.default_rng(0).integers(0, 256, (len(times), 64, 96, 3), dtype=np.uint8)
        questions, columns = v.questions_and_columns(arm)
        standard, _ = scorer.logits(frames, times, arm, questions, False, False)
        cached, hidden = scorer.logits(frames, times, arm, questions)
        diff = max(float(np.abs(a-b).max()) for a, b in zip(standard, cached))
        assert diff < 1e-4, (model_id, arm, diff)
        assert hidden.shape == (256,) and np.isfinite(hidden).all()
        score, hidden2 = scorer.score(frames, times, arm)
        assert len(score) == len(columns)
        np.testing.assert_allclose(score[:6].sum(), 1, atol=1e-6)
        np.testing.assert_array_equal(hidden, hidden2)
        result = dict(model_type=config.model_type, arm=arm, max_abs_logit_diff=diff,
                      hidden_dim=len(hidden), finite=True, dtype='float32', device='cpu')
        results.append(result)
        print(result, flush=True)
Path(__file__).with_name('tiny_hf_checks.json').write_text(json.dumps(results, indent=2))

import subprocess
import tempfile
from unittest.mock import patch
from anticipation.features import io

with tempfile.TemporaryDirectory() as temp:
    video = str(Path(temp) / 'synthetic.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=2',
                    '-t', '12', '-c:v', 'libx264', '-threads', '2', '-g', '2', '-pix_fmt', 'yuv420p', video], check=True)
    cfg = dict(model_id=v.MODEL_3, hidden_layers=(1, 3), max_pixels=128*128, max_edge=128)
    with patch.object(v, '_scorer', return_value=scorer), patch.object(v, '_hidden_size', return_value=64):
        causal = {}
        for arm in 'ABC':
            for name in (v.EXTRACTOR, v.HIDDEN_EXTRACTOR):
                def extract(path):
                    return v.extract_all(path, 'cpu', arm=arm, **cfg)[name][:3]
                result = io.check_causal(extract, video, 10, atol=0)
                assert result['causal'], (arm, name, result)
                causal[arm + ':' + name] = result
        checkpoint = str(Path(temp) / 'resume.checkpoint')
        original_score, calls = scorer.score, [0]
        def pause_after_two(*args, **kwargs):
            if calls[0] == 2:
                raise TimeoutError('simulated bounded chunk')
            calls[0] += 1
            return original_score(*args, **kwargs)
        with patch.object(scorer, 'score', side_effect=pause_after_two):
            try:
                v.extract_all(video, 'cpu', arm='B', checkpoint_path=checkpoint, **cfg)
                raise AssertionError('Expected simulated deadline')
            except TimeoutError:
                pass
        assert Path(checkpoint).exists()
        resumed = v.extract_all(video, 'cpu', arm='B', checkpoint_path=checkpoint, **cfg)
        fresh = v.extract_all(video, 'cpu', arm='B', **cfg)
        for name in resumed:
            for index in range(3):
                np.testing.assert_array_equal(resumed[name][index], fresh[name][index])
        assert not Path(checkpoint).exists()
        delayed = v.extract_all(video, 'cpu', arm='B', query_start_sec=13, **cfg)
        for name in delayed:
            np.testing.assert_array_equal(delayed[name][1][:10], fresh[name][1][:10])
            assert delayed[name][2][:10].all() and not delayed[name][2][10:].any()
        x = np.zeros((8, 64, 96, 3), np.uint8)
        tx = v.clip_times(9, 'B')
        sx, hx = scorer.score(x, tx, 'B')
        scorer.score(np.full_like(x, 255), tx, 'B')
        sx2, hx2 = scorer.score(x, tx, 'B')
        np.testing.assert_array_equal(sx, sx2)
        np.testing.assert_array_equal(hx, hx2)
        with patch.object(io, 'FEATURE_ROOT', str(Path(temp) / 'features')):
            saved = v.extract_and_save(video, 'synthetic/test', 'cpu', arm='B', rationale_samples=1, **cfg)
            for name, path in saved['paths'].items():
                t, feat, valid, meta = io.load_features(name, 'synthetic/test')
                assert feat.shape == (12, meta['D']) and len(meta['columns']) == meta['D']
                assert valid.all() and meta['no_context'] and meta['prompt_sha256']
                with np.load(path, allow_pickle=False) as z:
                    assert z['feat'].dtype == np.float16 and z['t_sec'].dtype == np.float32
            import pandas as pd
            targets = pd.read_parquet(io.feature_path(v.EXTRACTOR, 'synthetic/test').replace('.npz', '.targets.parquet'))
            assert len(targets) == 6 and targets['valid'].all()
            assert targets['stage_calm'].dtype == np.float32
        empty = v.extract_all(video, 'cpu', max_duration_sec=.5, **cfg)
        assert empty[v.HIDDEN_EXTRACTOR][1].shape == (0, 256)
        pose = str(Path(temp) / 'pose.npz')
        np.savez(pose, frame_t=np.arange(13), frame_ptr=np.zeros(14, int), det_track_id=np.array([], int), frame_ok=np.ones(13, bool))
        gated = v.extract_all(video, 'cpu', pose_gating=True, pose_path=pose, **cfg)
        assert gated[v.EXTRACTOR][2].all() and not gated[v.HIDDEN_EXTRACTOR][2].any()
        assert (gated[v.EXTRACTOR][1][:, 0] == 1).all()
    report = dict(causality=causal, no_context_x_y_x=True, bounded_resume_exact=True, early_baseline_and_invalid_gap=True, npz_and_targets_contract=True,
                  empty_output_fixed_dim=True, pose_gated_hidden_invalid=True,
                  model='random tiny Qwen3 CPU; not pretrained GPU validation')
    Path(__file__).with_name('contract_checks.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)
