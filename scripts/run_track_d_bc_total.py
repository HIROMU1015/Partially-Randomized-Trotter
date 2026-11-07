#!/usr/bin/env python3
"""One fixed saved-scalar A/F/S campaign, never a molecular circuit run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import track_d_bc_total as study

OUTPUT = 'artifacts/track_d_bc_total/2026-11-08/run01'
EXEC = ('src/trottertracks/track_d_bc_total.py', 'scripts/run_track_d_bc_total.py',
        'tests/check_track_d_bc_total.py', 'docs/research/track_d_bc_total_20261108/PLAN.md',
        'docs/research/track_d_bc_total_20261108/SCENARIOS.json')
REUSED = ('src/trottertracks/track_d_b6b_bc.py', 'src/trottertracks/c1_stage3_accuracy.py')


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.preflight == args.execute:
        p.error('choose exactly one of --preflight or --execute')
    inputs = study.load_inputs(ROOT)
    if args.preflight:
        print(json.dumps(dict(status='METADATA_PREFLIGHT_PASS', input_bindings=inputs['bindings'],
            direct_q=[1,2,4,8,16,32], records_per_scenario=52, scenarios=study.SCENARIOS,
            price_predictions_executed=False, H4_totals_executed=False), ensure_ascii=False, indent=2))
        return
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    hashes = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in (*EXEC, *REUSED)}
    for f in hashes:
        if (ROOT/f).read_bytes() != subprocess.check_output(['git', 'show', source+':'+f], cwd=ROOT):
            raise ValueError('UNFIXED_EXECUTION_SOURCE:'+f)
    config = json.loads((ROOT/EXEC[-1]).read_text())
    if config['scenario_models'] != study.SCENARIOS or config['base_commit'] != study.B6B:
        raise ValueError('FIXED_SCENARIO_PLAN')
    out = ROOT/OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    write(out/'SOURCE.json', dict(execution_source=source, base=study.B6B, files=hashes,
        scope='one saved-scalar conditional ideal RPE campaign; A/F/S fixed before execution',
        molecular_sampling=0, molecular_build=0, molecular_compile=0, model_fit=0, N_solver=0))
    write(out/'input_bindings.json', inputs['bindings'])
    write(out/'preflight.json', dict(status='PASS', input_binding_count=len(inputs['bindings']),
        b4_metadata=inputs['b4_metadata'], price_catalog=inputs['catalog'],
        ideal_common_N=study.bc.common_N(inputs['premises']), actual_common_N='UNQUALIFIED',
        scenarios=config, reused_aggregate='B6b aggregate_registered -> c1_stage3_accuracy.aggregate_totals'))
    def preserve(scenario, data):
        directory = out/scenario
        directory.mkdir()
        write(directory/'prices52.json', data.pop('records'))
        for name in ('contributions', 'paired_rounds'):
            rows = data.pop(name)
            with (directory/(name+'.csv')).open('x', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        write(directory/'result.json', data)
    try:
        result = study.evaluate(inputs, on_scenario=preserve)
        for f, h in hashes.items():
            if hashlib.sha256((ROOT/f).read_bytes()).hexdigest() != h:
                raise ValueError('SOURCE_CHANGED_DURING_CAMPAIGN')
        for key, r in inputs['bindings'].items():
            if hashlib.sha256(study.bc.blob(ROOT, r['commit'], r['path'])).hexdigest() != r['sha256']:
                raise ValueError('FIXED_INPUT_CHANGED:'+key)
        result.pop('scenarios')
        result['source_commit'] = source
        result['source_input_before_after_matched'] = True
        write(out/'result.json', result)
        write(out/'STOP.json', dict(status='COMPLETE_MANDATORY_STOP', mandatory_stop=True,
            next_stage_authorized=False, next_actor='user/GPT', campaign_count=1))
        print(json.dumps(dict(status=result['status'], source=source, scenarios=['A','F','S'],
            mandatory_stop=True, result=OUTPUT+'/result.json')))
    except Exception as e:
        write(out/'STOP.json', dict(status='FAILED_MANDATORY_STOP', error=type(e).__name__+':'+str(e),
            mandatory_stop=True, retry_resume_authorized=False, next_actor='user/GPT'))
        raise


if __name__ == '__main__':
    main()
