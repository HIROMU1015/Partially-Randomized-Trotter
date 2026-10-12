"""Off-domain and artificial I/O controls; no registered E evaluation."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from trottertracks.algorithm_codesign.df_correction_et_phase_v7 import native_evaluation as new
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import native_evaluation as old
from trottertracks.algorithm_codesign.df_correction_et_phase_v7 import storage
from trottertracks.algorithm_codesign.df_correction_comparison import intervals


@pytest.fixture(autouse=True)
def prohibit_registered_recipes_and_synthesis(monkeypatch):
    from trottertracks.algorithm_codesign.df_correction_comparison import comparison, angles
    def reject(*args, **kwargs):
        raise AssertionError('Registered recipe/synthesis forbidden in observer tests')
    monkeypatch.setattr(comparison, 'cd_recipe', reject)
    monkeypatch.setattr(comparison, 'e_recipe', reject)
    monkeypatch.setattr(angles, 'synthesize', reject)
    intervals.configure()


@pytest.mark.parametrize('controlled', [False, True])
def test_numeric_upper_exactly_matches_old_off_domain_guard(controlled):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(2)
    qc.rz(.117, 0);qc.cx(0, 1)
    if controlled:
        qc = qc.control()
    target = Operator(qc).data.copy()
    target[0, 0] += .001j  # deliberately wrong off-domain target, strict phase retained
    a, a_ledger = old.operator_numeric_guard(qc, target, F(1, 100000000))
    b, b_ledger = new.operator_numeric_guard(qc, target, F(1, 100000000))
    assert a == b
    assert a_ledger == {k:v for k,v in b_ledger.items() if k != 'Operator_residual_upper'}
    assert b == sum(F(b_ledger[k]) for k in ('Operator_residual_upper', 'Operator_forward_guard', 'reference_radius'))


def frames():
    return [np.array([[a], [b*complex(math.cos(p), math.sin(p))]], complex)
            for a,b,p in [(3/5,4/5,.13), (5/13,12/13,-.27), (8/17,15/17,.41)]]


@pytest.mark.parametrize('failure', ['guard', 'save'])
def test_observation_saved_before_failure_and_no_next_compile(monkeypatch, failure):
    records = {};calls = []
    guard_calls = []
    def bad_guard(*args):
        guard_calls.append(1)
        return F(1, 1000000), {'Operator_residual_upper':'1/1000000',
                              'Operator_forward_guard':'0', 'reference_radius':'0',
                              'float_residual_diagnostic':0., 'model':'synthetic'}
    monkeypatch.setattr(new, 'operator_numeric_guard', bad_guard)
    def save(label, value):
        records[label] = value
        if failure == 'save' and label.startswith('NUMERIC-'):
            raise OSError('synthetic observation write failure')
    prior = [{'active_modes':2, 'controlled':c, 'circuit_fingerprint':'off-domain'} for c in (False, True)]
    with pytest.raises(ArithmeticError if failure == 'guard' else OSError):
        new.native_fixture(2, frames(), prior, lambda p:calls.append(p['compile_calls']), on_diagnostic=save)
    assert calls == [1] and guard_calls == [1]
    saved = records['NUMERIC-A-0']
    assert saved['candidate_guard_pass'] is False and saved['accepted_by_observation'] is False
    assert F(saved['combined_upper']) > F(saved['allowance'])
    assert F(saved['combined_upper']) == sum(F(v) for v in saved['components'].values())
    assert saved['native'][0]['inventory']['ordered_native_ir']


def test_large_candidate_and_accepted_result_share_same_lossless_blob(tmp_path):
    candidate = {'numeric_operator_error_upper':'0',
                 'numeric_ledger':{'Operator_residual_upper':'0', 'Operator_forward_guard':'0', 'reference_radius':'0'},
                 'inventory':{'angle_lowering_error_upper':'0', 'global_phase_error_upper':'0',
                              'ordered_native_ir':{'synthetic_literal':'x'*(2*1024*1024)}}}
    def save(label, value):
        storage.dump_json(tmp_path/(label+'.json'), value)
    new.save_numeric_observation('synthetic', candidate, F(1,5000000), save, context={'off_domain':True})
    before = sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    storage.dump_json(tmp_path/'final.json', {'native':[candidate]})
    after = sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    assert after-before < 10000
    assert storage.load_json(tmp_path/'final.json')['native'][0] == candidate
    assert all(p.stat().st_size <= 512*1024 for p in tmp_path.rglob('*') if p.is_file())


def test_old_consumed_v5_launch_is_rejected_and_marker_unchanged():
    from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import launch
    root = Path(__file__).resolve().parents[3]
    marker = root/launch.RESULT/'one_shot_marker.json'
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        launch.verify_launch(root)
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == before
