"""Synthetic support only: compare the fixed decoder and certificate unchanged."""
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from trottertracks import rpe_q3_round_tables as helper

ROOT = Path(__file__).resolve().parents[1]
FIXED = 'cbc471c913072de45c25a0ed22d51dff8a9b7034'


@pytest.fixture
def risk(tmp_path):
    def read(name):
        return subprocess.check_output(['git', 'show', f'{FIXED}:src/trottertracks/{name}.py'], cwd=ROOT, text=True)

    decoder = SimpleNamespace()
    exec(read('rpe_decoder_fast'), vars(decoder))
    source = read('rpe_deterministic_risk')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_good_transitions')
    path = tmp_path/'certificate.py'
    path.write_text(ast.get_source_segment(source, node)+'\n')
    spec = importlib.util.spec_from_file_location('fixture_certificate', path)
    module = importlib.util.module_from_spec(spec)

    def integer(value, label, minimum):
        if type(value) is not int or value < minimum:
            raise ValueError(label)

    module.F = decoder
    module.S = SimpleNamespace(_integer=integer)
    module.TIE_TOLERANCE = 1e-12
    spec.loader.exec_module(module)
    return module


def supports(risk, kind):
    # No physical signal data. Two adjacent supports entirely within the same
    # nearest-branch region, with enough pairs to activate the parallel path.
    old = np.linspace(-.1, .1, 420)
    phase = np.linspace(-.3, .3, 420)
    oracle = risk.F._nearest(phase, np.zeros_like(phase), 2)[0]
    previous = dict(status='ENUMERATED', m=0, q=1, outcomes=[dict(good=True, oracle_estimate=float(s)) for s in old])
    current = dict(status='ENUMERATED', m=1, q=2, outcomes=[dict(good=True, principal=float(p), oracle_estimate=float(o)) for p,o in zip(phase,oracle)])
    if kind == 'failure':
        current['outcomes'][153]['oracle_estimate'] += .001
        current['outcomes'][156]['oracle_estimate'] += .001
    elif kind == 'tie':
        # Real unchanged decoder: put a previous estimate exactly at a branch
        # mid-point and place the tied phase at the first position.
        previous['outcomes'][0]['oracle_estimate'] = np.pi/2
        current['outcomes'].insert(0, dict(good=True, principal=0., oracle_estimate=0.))
    elif kind == 'empty':
        previous['outcomes'].clear()
    elif kind == 'incomplete':
        current['status'] = 'UNKNOWN'
    return [previous, current]


@pytest.mark.parametrize('kind', ['pass', 'failure', 'tie', 'empty', 'incomplete', 'scope'])
def test_ordered_certificate_identical(risk, kind, monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity', lambda pid: set(range(8)), raising=False)
    tables = supports(risk, kind)
    limit = 1000 if kind == 'scope' else 10**7
    original = risk.verify_good_transitions
    expected = original(tables, maximum_pairs=limit)
    scope = {'verify_good_transitions': original}
    exec('def extended(tables, limit):\n    return verify_good_transitions(tables, maximum_pairs=limit)', scope)
    with helper.parallel_good_transitions(risk, 4, scope['extended']) as active:
        assert active
        assert risk.verify_good_transitions(tables, maximum_pairs=limit) == expected
        assert scope['extended'](tables, limit) == expected
    assert risk.verify_good_transitions is original
    assert scope['verify_good_transitions'] is original
    if kind == 'pass':
        assert expected['status'] == 'VERIFIED_FINITE_BINARY64_SUPPORT'
    if kind == 'failure':
        assert expected['checked_pairs'] == 154*420
    if kind == 'tie':
        assert expected['reason'] == 'GOOD_SET_NOT_CLOSED_UNDER_ORIGINAL_DECODER'


def test_unknown_source_and_single_worker_preserve_original(risk):
    original = risk.verify_good_transitions
    with helper.parallel_good_transitions(risk, 1) as active:
        assert not active and risk.verify_good_transitions is original
    risk.verify_good_transitions = lambda *args, **kw: None
    with helper.parallel_good_transitions(risk, 4) as active:
        assert not active
        assert risk.verify_good_transitions() is None


def test_full_synthetic_hchain_q3_same_exact_bounds_and_records(tmp_path):
    # Isolate the original risk checkout from the common worker modules.
    import sys
    archive = subprocess.check_output(['git', 'archive', FIXED, 'src'], cwd=ROOT)
    subprocess.run(['tar', '-xf', '-', '-C', str(tmp_path)], input=archive, check=True)
    script = '''
import importlib.util, sys
sys.path.insert(0, sys.argv[1] + '/src')
spec = importlib.util.spec_from_file_location('hchain', sys.argv[2])
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
rows = [dict(m=m, signal=dict(raw=dict(real=.91, imag=.13))) for m in range(3)]
args = ('Full PF', rows, [2, 2, 2], dict(synthetic=True))
a = h.compute((*args, 1, .1, -.01))
b = h.compute((*args, 4, .1, -.01))
for k in ('status', 'MSE_lower', 'MSE_upper', 'result', 'partial', 'round_tables'):
    assert a[k] == b[k], k
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path), str(ROOT/'scripts/rpe_hchain_q3.py')],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
