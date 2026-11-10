"""Execute the authorized S1 saved-data plan, once into fresh output files."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_df10_price_v2 as v2

BASE = 'docs/research/h4_df10_price_v1_20261110/'
OUTPUT = ROOT/'docs/research/h4_df10_price_v2_s1_20261110'
SOURCE_FILES = ['src/trottertracks/h4_df10_price_v2.py',
                'scripts/run_h4_df10_price_v2.py']
IDS = {'PF': {8: '49709717a6c3d405c60d', 16: '8e50c5560bb8f47f026f'},
       'PR': '14bd81a375400c9b0305'}


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def load_pairs(document, stage):
    if (document['model'] != v2.V1_MODEL or document['stage'] != stage or
            document['scientific_binding']['series'] != 'H4_TOL_ONLY_DF10' or
            document['scientific_binding']['control'] != 'symmetric_directional'):
        raise ValueError('DOCUMENT_SCIENTIFIC_BINDING')
    pairs = v2.pair_rows(document['rows'])
    pfq, cohort, n = (8, 'EXPLORATION', 9) if stage == 'develop' else (16, 'CONFIRMATION', 33)
    pf = [p for p in pairs if p['method'] == 'PF']
    pr = [p for p in pairs if p['method'] == 'PR']
    if ([p['path_index'] for p in pf] != [0] or
            [p['path_index'] for p in pr] != list(range(n))):
        raise ValueError('COHORT_PATH_BINDING')
    for p in pairs:
        q, c, identity = ((pfq, 'DETERMINISTIC', IDS['PF'][pfq]) if p['method'] == 'PF'
                          else (8, cohort, IDS['PR']))
        if (p['q'], p['T'], p['cohort'], p['candidate_id']) != (q, .1*q, c, identity):
            raise ValueError('CANDIDATE_BINDING')
    return pairs


def run():
    targets = [OUTPUT/'RESULT.json', OUTPUT/'PATH_RESIDUALS.csv']
    if any(p.exists() for p in targets):
        raise FileExistsError('S1 output exists; preserve the completed result')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for path in SOURCE_FILES:
        if (ROOT/path).read_bytes() != blob(source, path):
            raise ValueError('UNCOMMITTED_S1_SOURCE: '+path)
    inputs = {name: blob(v2.CARRIER, BASE+name+'.json') for name in ('DEVELOPMENT', 'HOLDOUT')}
    docs = {name: json.loads(raw) for name, raw in inputs.items()}
    for key in ('model_source', 'model_sha256', 'scientific_binding', 'input_sha256', 'compiler_hash'):
        if docs['DEVELOPMENT'][key] != docs['HOLDOUT'][key]:
            raise ValueError('V1_INPUT_BINDING: '+key)
    training = load_pairs(docs['DEVELOPMENT'], 'develop')
    known = load_pairs(docs['HOLDOUT'], 'holdout')
    fit = v2.fixed_fit(ROOT)
    fit_calls = 0
    def counted_fit(x, y, q):
        nonlocal fit_calls
        fit_calls += 1
        return fit(x, y, q)
    result, rows = v2.campaign(training, known, counted_fit)
    assert fit_calls == 3
    result.update(schema='h4_df10_price_v2_s1_v1',
        status='S1_COMPLETE_EXPLORATORY_KNOWN_DATA', source=source,
        design_source='df1dcd7bf0469a724493e0e4193b107ad3a157f0',
        v1_carrier=v2.CARRIER, frozen_v1_model_source=docs['DEVELOPMENT']['model_source'],
        input_sha256={BASE+k+'.json': hashlib.sha256(v).hexdigest() for k, v in inputs.items()},
        reused_fit=dict(carrier=v2.CARRIER, path=v2.FIT_PATH,
                       file_sha256=hashlib.sha256(blob(v2.CARRIER, v2.FIT_PATH)).hexdigest()),
        scientific_binding=docs['DEVELOPMENT']['scientific_binding'],
        compiler_hash=docs['DEVELOPMENT']['compiler_hash'], numpy_version=v2.np.__version__,
        training_paths=len(training), known_v1_used_paths=len(known),
        path_axis_model_rows=len(rows), fit_calls=fit_calls,
        invalid_models=[c for c in result['calibrations'] if c['status'] != 'FITTED'],
        negative_predictions=sum(r['diagnostic'] == 'NEGATIVE_PRICE' for r in rows),
        nonfinite_predictions=sum(r['diagnostic'] == 'NONFINITE_PREDICTION' for r in rows),
        new_structural_predictions=0, new_trajectories=0, full_circuit_builds=0,
        transpile_calls=0, bootstrap_calls=0, new_signal_calls=0, new_N_calls=0,
        N_times_C_calls=0, long_q_prices=0, all_law_expectation=None,
        independent_V2_holdout=False, coefficient_uncertainty=None,
        long_q_qualified=False, total_prediction_CI=None, main_claim_changed=False,
        independent_scientific_review=False, mandatory_STOP=True)
    # No rounding, clipping, or replacement of invalid predictions.
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with targets[0].open('x') as stream:
        stream.write(payload)
    with targets[1].open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status=result['status'], source=source, fit_calls=fit_calls,
                          path_axis_model_rows=len(rows), mandatory_STOP=True)))


if __name__ == '__main__':
    run()
