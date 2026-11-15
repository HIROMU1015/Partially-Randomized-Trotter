"""Production worker entry + actual constructor/evaluator, invented means only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
import run_pr_df_ld_split_q3_resume as R
sys.path.insert(0,str(R.Q.VIEW/'tests'))
from check_rpe_total_binding import pauli,MOCK,T


def test_original_successful_runtime_and_fixed_metadata():
    r=R.runtime_check()
    assert r['constructor']=='PASS_UNCHANGED_GUARD'and not r['guard_modified']
    assert r['expected_AST_hash']=='3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8'
    assert len(R.fixed_inputs()['cases'])==4


def test_actual_worker_pool_constructor_toy_evaluator_and_persistence(tmp_path):
    args,_=pauli();bridge=T.create(R.Q.VIEW,args['config'],original_source=MOCK)
    view=T.bind_signal(R.Q.VIEW,bridge,args['signal_packet'])
    cases=[dict(label=label,configuration=MOCK,signal_packet=MOCK,allocation=args['allocations'][0],evaluation_target=args['evaluation_target'])for label in ('ARTIFICIAL_A','ARTIFICIAL_B')]
    inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'],
        acceleration=dict(workers_per_configuration=2,outcome_batch=1,source=MOCK),
        adopted_evaluation=dict(MSE_upper=999,epsilon_E_decimal='ARTIFICIAL_ONLY'))
    for c in cases:(tmp_path/c['label']).mkdir()
    # Only metadata intake is replaced by invented objects. Actual U.worker,
    # P.worker, Accelerator constructor, T.evaluate and persistence all execute.
    with patch.object(R.U,'metadata',return_value=(bridge,view)):
        out=R.supervise(cases,inputs,MOCK,tmp_path,poll_seconds=.1)
    assert not out['errors']and not out['monitor_errors'],out
    assert len(out['receipts'])==2 and all(p['exitcode']==0 for p in out['process_exitcodes'])
    for c in cases:
        d=tmp_path/c['label'];r=json.loads((d/'RESULT.json').read_text());api=json.loads((d/'API_RETURN.json').read_text())
        assert r['scientific_evaluation_calls']==1 and r['record']==api[0]
        assert r['status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
        assert (d/'PARTIAL_COARSE.json').exists() and (d/'PARTIAL_TOTAL_DP.json').exists()
        assert (d/'RESOURCE_FINAL.json').exists() and (d/'PERSISTED.json').exists()
        assert r['record']['diagnostic']['deterministic_risk']['MSE_upper']is not None
