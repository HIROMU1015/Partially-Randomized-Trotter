"""Artificial parent/worker pipeline, source/input/STOP and comparison policy."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_separated_j_u_curvature_pilot as m
FIT_PYTHON = Path('/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python')


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')


def fixture(root):
    # Invented diagonal data matching only the declared scalar normalization.
    # No saved molecular NPZ, Hamiltonian or fit endpoint is loaded.
    spec = m.algorithm.pr.Objective(8, 4, 2, m.CONDITIONS['loss_scale'],
                                   m.CONDITIONS['mass_scale'], .1)
    J = np.zeros((8, 4, 4))
    def mass(t):
        a = J.copy(); a[2, 0, 0] = t
        return float(m.algorithm.pr.smooth_mass(a, spec, np))
    J[2, 0, 0] = brentq(lambda t: mass(t)-spec.mass_scale, 0., 10., xtol=1e-14)
    J[0, 0, 0] = np.sqrt(m.CONDITIONS['l2_scale']-np.sum(J*J))
    U = np.repeat(np.eye(4)[None], 8, axis=0)
    g = m.algorithm.linear.reconstruct(J, U)
    g[0, 0, 0, 0] += np.sqrt(2*spec.loss_scale)
    h = np.diag([-.3, -.1, .2, .4])
    np.savez(root/'input.npz', h=h, g=g, constant=np.array(.2))
    np.savez(root/'initial.npz', J=J, U=U, x0=np.r_[np.zeros(48), m.algorithm.linear.pack(J)])
    write(root/'initial.json', dict(sector_error=1e-8, numerical_guard=1e-9))
    write(root/'scales.json', dict(spec={k:m.CONDITIONS[k] for k in
        ('rank', 'norb', 'ld', 'loss_scale', 'mass_scale', 'epsilon')}))
    (root/'synthetic_source.py').write_text('# artificial source identity only\n')
    write(root/'SOURCE.json', dict(files={'synthetic_source.py':m.io.sha(root/'synthetic_source.py')}))
    c = dict(scope=m.SCOPE, base_commit=m.BASE, conditions=m.CONDITIONS,
        algorithm_policy=asdict(m.algorithm.DEFAULT), call_limits=m.LIMITS,
        parent_limits=m.PARENT, per_arm_limits=m.WORKER, maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH, retry=0, resume=0, mandatory_stop=True,
        input_snapshot='input.npz', initial_snapshot='initial.npz',
        initial_scales='scales.json', initial_sector='initial.json',
        source_manifest='SOURCE.json', source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        evaluation_packages={}, fit_packages={}, ffsim_sources={},
        old_endpoints={}, comparators={}, saved_histories={}, initialization_SOURCE='SYNTHETIC_ONLY',
        entry='synthetic subprocess adapter, not the scientific CLI')
    for role in m.ROLES:
        module, objective = m.common.setup(role, json.loads((root/'scales.json').read_text()))
        p = m.algorithm.Coefficients(g, U, objective)
        parts = p.parts(p.pack(J))
        c['old_endpoints'][role] = role+'_old.npz'
        np.savez(root/c['old_endpoints'][role], J=J, U=U)
        c['comparators'][role] = dict(diagnostic=role+'_old.json')
        write(root/(role+'_old.json'), dict(components=dict(combined_objective=parts['objective'])))
    for role in m.ROLES:
        old=m.io.read(root/(role+'_old.json'))
        state=dict(components=dict(objective=old['components']['combined_objective'],exact_tail_mass_Ha=1.),first_order_met=False)
        write(root/(role+'_metric.json'),state)
        write(root/(role+'_separated.json'),dict(terminal=state,status='ARTIFICIAL_OLD_UNQUALIFIED'))
        c['saved_histories'][role]=dict(old_joint_metric=role+'_metric.json',
            old_joint_optimizer=role+'_old.json',old_joint_sector='initial.json',
            old_separated_optimization=role+'_separated.json',old_separated_sector='initial.json',
            old_separated_tensor=role+'_old.json')
    del c['old_endpoints'];del c['comparators']
    names=['input.npz','initial.npz','initial.json','scales.json',*[p for x in c['saved_histories'].values() for p in x.values()]]
    c['saved_inputs']={p:m.io.sha(root/p) for p in names}
    write(root/'contract.json', c)
    return c


def test_passive_scope_source_input_and_environment_controls(tmp_path, monkeypatch):
    c = fixture(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('passive NPZ decode'))
    monkeypatch.setattr(m.algorithm, 'optimize', lambda *a, **k: pytest.fail('passive optimization'))
    assert m.preflight(tmp_path, tmp_path/'contract.json') == c
    bad = deepcopy(c); bad['algorithm_policy']['max_cycles'] += 1
    write(tmp_path/'bad.json', bad)
    with pytest.raises(ValueError, match='FIXED_SEPARATED'):
        m.preflight(tmp_path, tmp_path/'bad.json')
    original = (tmp_path/'synthetic_source.py').read_bytes()
    (tmp_path/'synthetic_source.py').write_bytes(b'artificial changed source')
    with pytest.raises(ValueError, match='SOURCE_OR_INPUT'):
        m.preflight(tmp_path, tmp_path/'contract.json')
    (tmp_path/'synthetic_source.py').write_bytes(original)
    (tmp_path/'input.npz').write_bytes(b'invented mutation')
    with pytest.raises(ValueError, match='SOURCE_OR_INPUT'):
        m.preflight(tmp_path, tmp_path/'contract.json')


@pytest.mark.parametrize('qualified,budget,expected', [
    (True, True, 'BOTH_FIRST_ORDER_AND_DF_MET'),
    (False, True, 'ONE_OR_BOTH_FIRST_ORDER_UNQUALIFIED'),
    (True, False, 'DF_BUDGET_NOT_MET'), (False, False, 'DF_BUDGET_NOT_MET')])
def test_stationarity_DF_and_tail_comparison_are_independent(qualified, budget, expected):
    rows = {k:dict(sector_error=1e-8 if budget else 1., numerical_guard=1e-9) for k in m.ROLES}
    reports = {}
    for role in m.ROLES:
        state = dict(components=dict(objective=100. if role == 'pr' else 1.,
            exact_tail_mass_Ha=1. if role == 'pr' else 2.), first_order_met=qualified)
        reports[role] = dict(optimization=dict(first_order_qualified=qualified, terminal=state),
                            old_metric=deepcopy(state))
    r = m.decide(rows, reports, {})
    assert r['status'] == expected
    assert r['qualified_tail_comparison'] == ('PR_LOWER_NUMERICAL_TAIL' if qualified and budget else None)
    assert r['cross_objective_raw_F_ranking'] is None
    assert r['PF_error'] is r['attenuation'] is r['shots'] is r['native_cost'] is r['total_cost'] is r['CA_achieved'] is None
    assert not r['rigorous_stationarity'] and r['mandatory_stop']


@pytest.mark.parametrize('fault', [None, 'representation'])
def test_artificial_full_parent_real_block_workers_sector_and_STOP(tmp_path, monkeypatch, fault):
    if not FIT_PYTHON.exists():
        pytest.skip('Dedicated synthetic fit environment unavailable')
    c = fixture(tmp_path)
    protected = dict(c['saved_inputs'])
    out = tmp_path/m.FRESH
    # The fixture root is not a checkout. Load code from this fixed worktree,
    # but pass ONLY invented input paths and a fresh pytest directory.
    code = "import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from trottertracks import c_separated_j_u_curvature_pilot as m;r=Path(sys.argv[2]);c=m.preflight(r,r/'contract.json',fit=True);m.fit_worker(r,Path(sys.argv[3]),c,sys.argv[4])"
    def launch(root, path, output, contract, python, workers, calls):
        from concurrent.futures import ThreadPoolExecutor
        def one(role):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', JAX_ENABLE_X64='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (output/(role+'_worker.txt')).open('x') as log:
                subprocess.run([str(FIT_PYTHON), '-B', '-c', code, str(ROOT/'src'),
                    str(root), str(output/role), role], env=env, stdout=log,
                    stderr=log, check=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(one, k) for k in m.ROLES]:
                f.result()
    monkeypatch.setattr(m.common, 'launch_workers', launch)
    original_check_output=m.subprocess.check_output
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw: 'SYNTHETIC_ONLY\n' if args==['git','rev-parse','HEAD'] else original_check_output(args,**kw))
    if fault:
        def mismatch(*args):
            raise ValueError('ARTIFICIAL_REPRESENTATION_FAIL')
        monkeypatch.setattr(m.molecule, 'representation_gate', mismatch)
    with threadpool_limits(limits=1):
        if fault:
            with pytest.raises(ValueError, match='ARTIFICIAL_REPRESENTATION_FAIL'):
                m.run(tmp_path, tmp_path/'contract.json', out, FIT_PYTHON)
        else:
            result = m.run(tmp_path, tmp_path/'contract.json', out, FIT_PYTHON)
    stop = m.io.read(out/'STOP.json')
    if not fault:
        assert result['status'] == 'BOTH_FIRST_ORDER_AND_DF_MET'
        assert stop['scientific_units'] == 31 and not stop['failed']
        assert result['qualified_tail_comparison'] == 'PR_LOWER_NUMERICAL_TAIL'
        assert result['cross_objective_raw_F_ranking'] is None
    else:
        assert stop['scientific_units'] == 27 and stop['failed']
        assert (out/'failure.json').exists() and not (out/'result.json').exists()
    assert stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    assert stop['call_counts']['J_solve'] == 2 and 'U_solve' not in stop['call_counts']
    assert all(m.io.sha(tmp_path/p) == h for p, h in protected.items())
    for role in m.ROLES:
        assert (out/role/'J_raw_0.npz').exists()
        assert (out/role/'optimization_trace.jsonl').exists()
        assert m.io.read(out/role/'final_diagnostic.json')['status'] == 'PASS'
    with pytest.raises(ValueError, match='FRESH_SEPARATED'):
        m.run(tmp_path, tmp_path/'contract.json', out, FIT_PYTHON)


def test_initial_and_representation_correctness_FAIL_stop_and_preserve(tmp_path, monkeypatch):
    c = fixture(tmp_path)
    write(tmp_path/'initial.json', dict(sector_error=1., numerical_guard=0.))
    c['saved_inputs']['initial.json'] = m.io.sha(tmp_path/'initial.json')
    write(tmp_path/'contract.json', c)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: 'SYNTHETIC\n')
    monkeypatch.setattr(m.common, 'launch_workers', lambda *a, **k: pytest.fail('must not launch'))
    out = tmp_path/m.FRESH
    with pytest.raises(ValueError, match='SAVED_INITIAL_DF_BUDGET'):
        m.run(tmp_path, tmp_path/'contract.json', out, FIT_PYTHON)
    assert m.io.read(out/'STOP.json')['failed']
    assert (out/'failure.json').exists() and not (out/'result.json').exists()
