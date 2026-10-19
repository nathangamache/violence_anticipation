"""Draw docs/figures/system_overview.svg, the whole-system diagram (datasets to the confirmed models)."""
import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1]
W = 1830
INK, QUIET, EDGE, ACCENT, ACCENT_BG, BAND = '#1b1f24', '#5b6470', '#8a939e', '#2a78d6', '#e8f1fb', '#f4f6f8'
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
parts = []


def text(x, y, s, size=12, weight=400, fill=INK, anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def box(x, y, w, name, lines, kind='solid'):
    h = 40 + 16 * len(lines)
    stroke, fill, sw, dash = EDGE, 'none', 1.25, ''
    if kind == 'accent':
        stroke, fill, sw = ACCENT, ACCENT_BG, 2
    if kind in ('dashed', 'quiet'):
        dash = ' stroke-dasharray="5 4"'
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
    text(x + 14, y + 24, name, 13, 600, QUIET if kind == 'quiet' else INK)
    for i, s in enumerate(lines):
        text(x + 14, y + 42 + 16 * i, s, 12, 400, QUIET if (kind == 'quiet' or i == len(lines) - 1 and len(lines) > 1) else INK)
    return (x, y, w, h)


def path(d, arrow=True, dash=False):
    extra = (' stroke-dasharray="5 4"' if dash else '') + (' marker-end="url(#arr)"' if arrow else '')
    parts.append(f'<path d="{d}" fill="none" stroke="{EDGE}" stroke-width="1.25"{extra}/>')


def mid_y(b):
    return b[1] + b[3] / 2


text(30, 38, 'Violence anticipation: the whole system, from datasets to the two confirmed models', 18, 700)
text(30, 60, 'Current pipeline (October 2026) on top; the capstone-era pipeline (Nov 2025 to Apr 2026, frozen in legacy/) at the bottom. All counts from the repository.', 12, 400, QUIET)

C0, C1, C2, C3, C4, C5 = (30, 300), (380, 250), (680, 250), (1000, 260), (1310, 230), (1580, 220)
for (x, w), label in zip((C0, C1, C2, C3, C4, C5), ('1  DATASETS', '2  CATALOG AND FOLDS', '3  LABELS V3', '4  CAUSAL 1 HZ FEATURES', '5  REGISTRATION AND SCREEN', '6  MODELS AND TESTS')):
    text(x, 98, label, 11, 700, QUIET)

y = 140
ds = []
specs = [
    ('XD-Violence test', ['798 clips with audio, official onsets', 'Movie and YouTube clips'], 'solid'),
    ('XD-Violence train', ['3,950 clips from a mirror, 4 recovered', 'Onsets from HIVAU-70k annotations'], 'solid'),
    ('UCF-Crime', ['Violent 366, normal 915, other anomaly 550', 'Onsets: 42 official, 324 HIVAU; CCTV'], 'solid'),
    ('UBI-Fights', ['Fight 211, normal 749, frame-level labels', 'CCTV, no audio'], 'solid'),
    ('HIVAU-70k and UCA annotations', ['Onset times for the training splits', 'UCA captions: soft-mask sensitivity check'], 'solid'),
    ('Curated YouTube hold-out', ['92 videos, descriptive only, not in folds', 'Ids anonymised in the public repo'], 'dashed'),
    ('NTU CCTV-Fights (external)', ['985 videos, 145 eligible fights', 'Never used for training or selection'], 'dashed'),
    ('MEVA (external)', ['355 clips, 29.6 h, scripted, no audio', 'Never-violent negatives for false alarms'], 'dashed'),
    ('Tried and dropped', ['Kinetics (700 clips), RLVS (5 s clips),', 'YouTube near misses (4 of 127 usable)'], 'quiet'),
]
for name, lines, kind in specs:
    b = box(C0[0], y, C0[1], name, lines, kind)
    ds.append(b)
    y += b[3] + 12

cat = box(C1[0], 140, C1[1], 'Catalog (videos.csv, events.json)', ['7,696 videos, 429 h', 'Roles: violent 2,098, clean 4,010,', 'other anomaly 1,388, hold-out 92', 'Onsets in seconds, 10 s aftermath', 'Near-duplicate fingerprints'])
folds = box(C1[0], cat[1] + cat[3] + 30, C1[1], 'Folds v2', ['5 folds grouped by source:', 'movie, YouTube id or camera', 'Fold 1 frozen for one check'])
man = box(C1[0], folds[1] + folds[3] + 30, C1[1], 'Extract manifests', ['v1: 6,231 videos, 239 h', 'v2: 399 hard negatives (H1)', 'Clean clips capped at 300 s'])

lab = []
y = 140
for name, lines in [
    ('Qwen2.5-VL onset scan', ['551 eligible events scanned', '419 flagged + 40 random controls']),
    ('Blind visual review', ['Pass A on every case, Pass B on', 'label changes + 10% keep audit']),
    ('Reconciliation', ['212 decisions on 0.5 s', 'frame sheets, by a third reviewer']),
    ('Follow-ups and audits', ['6 cascade rounds, 64 new events', 'Lead-in span audit; all 92', 'non-flagged events reviewed']),
    ('apply_corrections: benchmark/v3', ['419 eligible events:', 'XD 243, UCF 111, UBI 65', '162 onsets unresolved, kept']),
]:
    b = box(C2[0], y, C2[1], name, lines)
    lab.append(b)
    y += b[3] + 22

feat = box(C3[0], 140, C3[1], 'Visual and pose extractors', ['SigLIP 2 embeddings (siglip2)', 'SigLIP 2 prompt scores v1, v2', 'Person-masked scene embedding', 'V-JEPA 2 large (vjepa2_l)', 'YOLO11 pose + tracking: pose_raw,', 'pose_interact, pose_state, pairs', 'Camera-compensated motion', 'Stored as npz on one 1 Hz clock'])
aud = box(C3[0], feat[1] + feat[3] + 20, C3[1], 'Audio extractors (audited only)', ['AST events, CLAP prompts,', 'voice affect, DSP levels,', 'speech toxicity (Whisper)'])
vlm = box(C3[0], aud[1] + aud[3] + 20, C3[1], 'VLM pilot (Qwen2.5-VL)', ['Arms A, B, C + hidden-state probes', 'Phase C decision: no-go'], 'quiet')

reg = box(C4[0], 140, C4[1], 'Pre-registration', ['screen_set_v1 + amendments 01-05', 'v2 rebuilt on labels v3,', 'frozen before any v3 score', 'Cohorts: 295 positive videos,', '3,220 clean, 328 H1 (102 H1p)'])
scr = box(C4[0], reg[1] + reg[3] + 22, C4[1], 'Phase A screen', ['9 blocks on folds 0, 2, 3, 4', 'P1 probe vs scene-only model N', 'Gates G1 signal, G2 specificity,', 'G3 gain over N; N1-N3 at 10 s', 'Pass: prompts v1, prompts v2', 'Shortcuts: embeddings, pose'])
ctl = box(C4[0], scr[1] + scr[3] + 22, C4[1], 'Control baselines', ['Random alarms, position null,', 'scene-only N, time-source,', 'B0 capstone MLP, Qwen teacher'])

fus = box(C5[0], 140, C5[1], 'Fusion (forward selection)', ['Early keeps prompts v1 alone', 'Imminent: prompts v2 only', 'P2 GRU registered, not run'])
mod = box(C5[0], fus[1] + fus[3] + 22, C5[1], 'Two frozen output models', ['Early: P1 on prompts v1,', 'onset 30 to 120 s ahead', 'Imminent: P1 on prompts v2,', 'onset within 10 s', 'Thresholds at 1, 2, 5 FA/h', 'Config sha256 8e4dc5eb'], 'accent')
f1 = box(C5[0], mod[1] + mod[3] + 22, C5[1], 'Fold-1 confirmation', ['1,214 videos, 75 fights', 'Run once: both confirmed', 'AUROC 0.905 and 0.893'])
ext = box(C5[0], f1[1] + f1[3] + 22, C5[1], 'External test', ['CCTV-Fights + MEVA, no tuning', 'Passes, but trivially', 'No signal within fight videos'])
rep = box(C5[0], ext[1] + ext[3] + 22, C5[1], 'Reports', ['benchmark/results/, docs/', 'RESULTS_V3.md, results page'])

bus_x = 355
for b in ds[:6]:
    path(f'M{b[0] + b[2]} {mid_y(b)}H{bus_x}', arrow=False)
path(f'M{bus_x} {mid_y(ds[0])}V{mid_y(ds[5])}', arrow=False)
path(f'M{bus_x} {cat[1] + 40}H{cat[0]}')
path(f'M{cat[0] + cat[2] / 2} {cat[1] + cat[3]}V{folds[1]}')
path(f'M{folds[0] + folds[2] / 2} {folds[1] + folds[3]}V{man[1]}')
path(f'M{cat[0] + cat[2]} {cat[1] + 40}H{lab[0][0]}')
for a, b in zip(lab, lab[1:]):
    path(f'M{a[0] + a[2] / 2} {a[1] + a[3]}V{b[1]}')
lane_y = 124
last = lab[-1]
path(f'M{last[0] + last[2]} {mid_y(last)}H950V{lane_y}H{reg[0] + reg[2] / 2}V{reg[1]}')
text(1000, lane_y - 6, 'corrected catalog and labels (benchmark/v3)', 11, 400, QUIET)
low_y = man[1] + man[3] + 0
path(f'M{man[0] + man[2] / 2} {man[1] + man[3]}V{last[1] + last[3] + 24}H975V{feat[1] + 60}H{feat[0]}')
text(700, last[1] + last[3] + 18, 'manifests drive extraction', 11, 400, QUIET)
fx = 1285
for b in (feat, aud, vlm):
    path(f'M{b[0] + b[2]} {mid_y(b)}H{fx}', arrow=False)
path(f'M{fx} {mid_y(feat)}V{mid_y(vlm)}', arrow=False)
path(f'M{fx} {mid_y(scr)}H{scr[0]}')
path(f'M{reg[0] + reg[2] / 2} {reg[1] + reg[3]}V{scr[1]}')
path(f'M{ctl[0] + ctl[2] / 2} {ctl[1]}V{scr[1] + scr[3]}')
path(f'M{scr[0] + scr[2]} {mid_y(scr)}H1560V{mid_y(fus)}H{fus[0]}')
for a, b in ((fus, mod), (mod, f1), (f1, ext), (ext, rep)):
    path(f'M{a[0] + a[2] / 2} {a[1] + a[3]}V{b[1]}')
ext_lane = max(rep[1] + rep[3], last[1] + last[3] + 24, ctl[1] + ctl[3]) + 30
xb = 345
path(f'M{ds[6][0] + ds[6][2]} {mid_y(ds[6])}H{xb}', arrow=False, dash=True)
path(f'M{ds[7][0] + ds[7][2]} {mid_y(ds[7])}H{xb}', arrow=False, dash=True)
path(f'M{xb} {mid_y(ds[6])}V{ext_lane}H1560V{mid_y(ext)}H{ext[0]}', dash=True)
text(700, ext_lane - 6, 'external data bypasses training and selection', 11, 400, QUIET)

band_y = max(ext_lane, ds[-1][1] + ds[-1][3]) + 30
band_h = 40 + 72 + 20
H = band_y + band_h + 46
parts.insert(0, f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
parts.append(f'<rect x="20" y="{band_y}" width="{W - 40}" height="{band_h}" rx="10" fill="{BAND}"/>')
text(36, band_y + 24, 'CAPSTONE ERA, NOV 2025 TO APR 2026 (frozen in legacy/; most headline numbers later shown to be leakage or label bugs)', 11, 700, QUIET)
lw, gap = 318, (W - 60 - 5 * 318) / 4
lb = []
for i, (name, lines) in enumerate([
    ('XD test + UCF-Crime', ['Qwen2.5-VL frame embeddings', 'sampled every 5 label rows']),
    ('Labels and splits', ['labels_multih: 7 horizons', 'split by clip, leaked by movie']),
    ('Models tried', ['Graph transformer, student MLPs,', 'BiLSTM, LoRA, flow/audio/pose fusion']),
    ('Scoring', ['Frame AUC, test-tuned thresholds,', 'stitched 8-hour simulations']),
    ('Honest result (B0)', ['Movie-grouped StudentMLP: AUROC', '0.58 to 0.62, about 8 s lead; kept as a control']),
]):
    lb.append(box(30 + i * (lw + gap), band_y + 40, lw, name, lines))
for a, b in zip(lb, lb[1:]):
    path(f'M{a[0] + a[2]} {mid_y(a)}H{b[0]}')

ly = H - 20
parts.append(f'<rect x="30" y="{ly - 11}" width="22" height="14" rx="3" fill="none" stroke="{EDGE}" stroke-width="1.25"/>')
text(60, ly, 'used in training and selection', 11.5, 400, QUIET)
parts.append(f'<rect x="270" y="{ly - 11}" width="22" height="14" rx="3" fill="none" stroke="{EDGE}" stroke-width="1.25" stroke-dasharray="5 4"/>')
text(300, ly, 'held out, external, or dropped: never trained on', 11.5, 400, QUIET)
parts.append(f'<rect x="620" y="{ly - 11}" width="22" height="14" rx="3" fill="{ACCENT_BG}" stroke="{ACCENT}" stroke-width="2"/>')
text(650, ly, 'the output models', 11.5, 400, QUIET)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" '
       f'aria-label="Violence anticipation system overview">'
       f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0 0L10 5L0 10z" fill="{EDGE}"/></marker></defs>' + ''.join(parts) + '</svg>\n')
open(OUT, 'w').write(svg)

