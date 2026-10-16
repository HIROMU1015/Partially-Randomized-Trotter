"""Verify saved four-axis evidence with stdlib only; never run science."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FROZEN=HERE.parents[1]/'track_a_v1_phase_b_readiness/2026-10-16/frozen_model.json'


def read(name):
    return json.loads((HERE/name).read_bytes())


def main():
    frozen=json.loads(FROZEN.read_bytes());sha=hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    validation=read('validation_complete.json');terminal=read('execution_terminal.json')
    launcher=read('launcher_terminal.json');audit=read('completion_audit.json')
    assert terminal['status']==launcher['status']==audit['status']=='COMPLETE_MANDATORY_STOP'
    assert terminal['exit_code']==launcher['native_exit_code']==0
    assert launcher['all_reaped'] and not launcher['cleanup_errors'] and launcher['first_stop'] is None
    assert terminal['SOURCE']==audit['SOURCE']=='fcf37e6794e2e51fc95e958a75fc8f75fb049423'
    assert validation['selected_axes']==frozen['validation_axes']
    rows=validation['rows'];expected={(a['cell']['id'],a['axis']) for a in frozen['validation_axes']}
    assert len(rows)==len(expected)==4
    assert {(x['cell_id'],x['axis']) for x in rows}==expected
    assert frozen['fitted_after_validation'] is False
    table=list(csv.DictReader((HERE/'axis_validation.csv').open()))
    assert len(table)==4
    for row,saved in zip(rows,table):
        parent=next(p for p in frozen['predictions'] if (p['cell_id'],p['axis'])==(row['cell_id'],row['axis']))
        assert row['model_sha256']==sha==audit['frozen_model_sha256']
        assert row['exact_prediction']==parent['exact_prediction']
        assert row['identity_verified'] and row['status']=='MEASURED_COMPLETE'
        error=abs(Fraction(row['exact_prediction'])-row['measured'])/row['measured']
        assert error==Fraction(row['exact_relative_error'])<Fraction(1,50)
        for field in ('cell_id','axis','exact_prediction','exact_relative_error','fingerprint','compiler_hash','model_sha256'):
            assert saved[field]==str(row[field]),field
        for field in ('measured','CX'):assert int(saved[field])==row[field]
    assert terminal['call_counts']==dict(snapshot_load=1,frozen_preparation_decode=1,
        native_evolution_max=2,control_probes_max=8,measured_wrappers_max=4,native_compile_max=4)
    assert audit['call_counts']==terminal['call_counts']
    assert audit['passed_axes']==audit['registered_axes']==4
    assert audit['protected_files_unchanged']==650 and audit['runtime_profile_files_unchanged']==34273
    assert launcher['max_monitor_gap_seconds']<5
    assert audit['new_five_precision_cost_evaluation']==audit['V2_executions']==0
    assert not audit['next_stage_authorized'] and not terminal['next_stage_authorized']
    assert not audit['refit'] and not audit['automatic_retry']
    print('PASS: four exact frozen predictions, per-axis threshold, saved CSV/JSON, terminal and audit consistency')


if __name__=='__main__':main()
