#!/usr/bin/env python3
"""Print a Video-AUC / Frame-AUC table for one or more aggregated eval JSONs.

    python3 scripts/summarize_video_auc.py outputs/pmil/eval/baseline_6set/pmil_baseline_6set.json \
        outputs/pmil/eval/pmil_main_eval/pmil_pmil_main_eval.json

The first file is treated as the reference row for the delta column.
"""

import json
import sys
from pathlib import Path

DATASETS = ['Celeb-DF-v2', 'FaceShifter', 'DeepFakeDetection', 'DFDC', 'DFDCP', 'WDF']
SHORT = {'Celeb-DF-v2': 'CDF-v2', 'FaceShifter': 'FSh', 'DeepFakeDetection': 'DFD',
         'DFDC': 'DFDC', 'DFDCP': 'DFDCP', 'WDF': 'WDF'}


def load(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    datasets = payload.get('datasets', {})
    row = {}
    for name in DATASETS:
        metrics = datasets.get(name, {}).get('metrics', {})
        row[name] = (metrics.get('video_auc'), metrics.get('auc'))
    return payload.get('run_id', Path(path).stem), row


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if len(values) == len(DATASETS) else None


def fmt(v):
    return '  -   ' if v is None else f'{v * 100:6.2f}'


def main(paths):
    rows = [load(p) for p in paths]
    for title, idx in (('Video AUC (%)', 0), ('Frame AUC (%)', 1)):
        print(f'\n{title}')
        head = f"| {'run':28s} | " + ' | '.join(f'{SHORT[d]:>6s}' for d in DATASETS) + ' |   Avg  | Δ Avg  |'
        print(head)
        print('|' + '|'.join('-' * (len(c)) for c in head.split('|')[1:-1]) + '|')
        ref = None
        for run_id, row in rows:
            vals = [row[d][idx] for d in DATASETS]
            avg = mean(vals)
            ref = avg if ref is None else ref
            delta = '  -   ' if avg is None or ref is None or row is rows[0][1] else f'{(avg - ref) * 100:+6.2f}'
            print(f'| {run_id[:28]:28s} | ' + ' | '.join(fmt(v) for v in vals) + f' | {fmt(avg)} | {delta} |')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
