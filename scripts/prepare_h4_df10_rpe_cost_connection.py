"""Prepare saved D1 cost inputs only. No real N*C evaluation or scientific runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.h4_df10_rpe_cost_connection import prepare_saved_cases

D1 = 'ffe8a71da98bec330ce1af109a9606f14e1fc598'
PATHS = ('docs/research/h4_df10_rpe_d1_20261110/result.json',
         'docs/research/h4_df10_rpe_preparation_20261110/SAVED_POINTS.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args(); output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing prepared output')
    blobs = [subprocess.check_output(['git', 'show', D1+':'+path], cwd=ROOT) for path in PATHS]
    result, points = [json.loads(b) for b in blobs]
    cases = prepare_saved_cases(result, points)
    payload = dict(status='PREPARED_WITHOUT_REAL_RESOURCE_PRODUCTS',
        saved_source=D1, original_D1_scientific_source=result['scientific_source'],
        inputs={path: hashlib.sha256(b).hexdigest() for path, b in zip(PATHS, blobs)},
        configuration=result['configuration'], cases=cases,
        N_recomputed=False, N_times_C_evaluated=False, signal_or_circuit_calls=0,
        model_fit_or_residual_analysis=False, full_RPE_qualified=False, mandatory_stop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2); f.write('\n')
    print('Prepared saved inputs only; N*C and scientific entries were not called.')


if __name__ == '__main__':
    main()
