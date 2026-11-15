"""Actual reused supervisor/worker persistence path with fake science only."""
from pathlib import Path
import importlib.util
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('split_q3',ROOT/'scripts/run_pr_df_ld_split_q3.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)


def test_fixed_route_and_count_bounds():
    Q.U.check_source()
    shots=[dict(m=m,N_X=n,N_Y=n)for m,n in enumerate([64,60,56,52,47,43,38,34,30,25,21,17,12])]
    b=Q.bounds(dict(allocation=dict(shots=shots)))
    assert b['transition_pairs_upper']<=141559036
    shots[0].update(N_X=48,N_Y=48)
    assert 'support_source'not in Q.bounds(dict(allocation=dict(shots=shots)))


def test_worker_owned_api_and_result_save_before_supervisor(tmp_path):
    U=Q.U
    # Reused U.worker invokes P.worker; fake only its numerical body. The
    # unchanged capture/persistence code remains the actual production route.
    def fake_worker(case,inputs,source,output):
        U.T.evaluate('FAKE')
        return dict(status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',scientific_evaluation_calls=1)
    with patch.object(U.P,'worker',fake_worker),patch.object(U.T,'evaluate',return_value=[{'synthetic':True}]):
        r=U.worker(dict(label='ARTIFICIAL'),{},dict(fixture='ONLY'),str(tmp_path))
    assert r['api_return_saved']and r['result_saved']and r['scientific_evaluation_calls']==1
    assert all((tmp_path/f).exists()for f in ('API_RETURN.json','RESULT.json','RESOURCE_FINAL.json','PERSISTED.json'))
