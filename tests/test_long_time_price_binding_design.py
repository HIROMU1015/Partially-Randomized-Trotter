"""Narrow C5cc/D748 compatibility evidence on invented metadata only.

Known C gaps are reproduced, not silently patched in production. The local AST
variant tests keyword compatibility only; it is not a replacement C integration.
No fixed-menu expansion, molecular inputs, Runtime or price arithmetic.
"""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from check_pr_rpe_rmse_controller import spec
from trottertracks import candidate_price_adapter as D
from trottertracks import pr_rpe_rmse_controller as B

ROOT = Path(__file__).resolve().parents[1]
C_SOURCE = '5cc7c07e112ce97205948465c94441cba0042416'
D_OLD = '0c6e16342b3c804fa0bd89a4028d6b5bea7b0322'


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


@pytest.fixture(autouse=True)
def no_science(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('This check permits metadata only')
    monkeypatch.setattr(D.NativeRuntime, '__init__', fail)
    monkeypatch.setattr(D.Adapter, '__init__', fail)


def fixture(scalars=(None, None)):
    s = spec(M=1, delta=.125, kappa=2)
    s['h'] = .125
    s['binding'].update(system='H4', rank=3)
    s['profile'] = deepcopy(B.p.PROFILE)
    config = B.generate(s)
    jobs, rows, requests = [], [], []
    for row, scalar in zip(config['rounds'], scalars):
        c = dict(system='H4', method='PR', prefix=1, formula='2nd', K=2,
                 q=row['q_PF'], R_outer=row['R_outer'],
                 T=float(B.a.exact_scalar(row['T'])), delta=.125)
        c['id'] = B.p.candidate_id(c)
        pc = {k: deepcopy(s[k]) for k in ('binding', 'profile', 'split', 'law')}
        pc.update(physical_candidate=c, structure=deepcopy(row['structure']))
        job = dict(m=row['m'], B_configuration_id=config['id'],
                   B_candidate_id=row['candidate']['id'], B_identity=deepcopy(row['identity']),
                   B_law_binding=deepcopy(row['law_binding']), physical_context=pc)
        d = D.from_a_job(config, job, input_source={'fixture': 'INPUT'},
                        execution_source={'fixture': 'EXECUTION'}, expected_scalar=scalar)
        jobs.append(job)
        requests.append(d)
        for pending in D.pending_price_records(d):
            pending.update(d_request=deepcopy(d), physical_context={'fixture_m': row['m']})
            rows.append(pending)
    return dict(configuration=config, pending={'jobs': jobs}), rows, requests


def c_functions(*, forward_scalar=False):
    path = 'src/trottertracks/rpe_d_price_connection.py'
    tree = ast.parse(blob(C_SOURCE, path))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_requests', '_paths')]
    if forward_scalar:
        # Proposed single keyword forwarding, NOT a production C source edit.
        class Forward(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'from_a_job':
                    node.keywords.append(ast.keyword(arg='expected_scalar', value=ast.parse(
                        "pc.get('expected_scalar')", mode='eval').body))
                return node
        tree = Forward().visit(tree)
    I = SimpleNamespace(providers=lambda root: SimpleNamespace(Price=D),
                        C=SimpleNamespace(p=D.p), price_context=lambda req, m: {'fixture_m': m})
    scope = dict(I=I, PATH='PR_CANDIDATE_PAIRED_PATH_PRICE')
    exec(compile(ast.fix_missing_locations(tree), C_SOURCE + ':' + path, 'exec'), scope)
    return SimpleNamespace(**{k: scope[k] for k in ('_requests', '_paths')})


def test_default_request_retains_old_D_content():
    scope = dict(__package__='trottertracks')
    exec(compile(blob(D_OLD, 'src/trottertracks/candidate_price_adapter.py'),
                 D_OLD + ':candidate_price_adapter.py', 'exec'), scope)
    plan, _, ds = fixture()
    d = ds[0]
    old = scope['from_a_job'](plan['configuration'], plan['pending']['jobs'][0],
                             input_source=d['physical_context']['input_source'],
                             execution_source=d['physical_context']['execution_source'])
    assert old == d
    c_functions()._requests(ROOT, plan, fixture()[1])


def test_expected_scalar_preserves_B_identity_but_changes_D_keys():
    _, _, old = fixture()
    _, _, new = fixture((.125, .125))
    assert old[0]['identity'] == new[0]['identity']
    assert old[0]['law_binding'] == new[0]['law_binding']
    assert old[0]['context_key'] != new[0]['context_key']
    assert old[0]['request_id'] != new[0]['request_id']


def test_C_reconstruction_requires_new_keyword_forwarding():
    plan, rows, ds = fixture((.125, .125))
    with pytest.raises(ValueError, match='does not belong'):
        c_functions()._requests(ROOT, plan, rows)
    mapped = c_functions(forward_scalar=True)._requests(ROOT, plan, rows)
    assert set(mapped) == {d['request_id'] for d in ds}
    assert all(r['mean']['rz_count'] is None for r in rows)


def test_forwarding_alone_does_not_anchor_scalar_across_rounds():
    plan, rows, ds = fixture((.125, .25))
    # Existing C anchor fields omit expected_scalar. A trusted menu scalar
    # check and its inclusion in the anchor are still needed downstream.
    assert len(c_functions(forward_scalar=True)._requests(ROOT, plan, rows)) == 2
    assert ds[0]['physical_context']['expected_scalar'] != ds[1]['physical_context']['expected_scalar']


def test_C_path_binding_does_not_compare_requested_scalar():
    _, _, ds = fixture((.125, .125))
    d = ds[0]
    path = dict(quantity='PR_CANDIDATE_PAIRED_PATH_PRICE', request=d,
                qualified_expectation=False, cohort='INVENTED', index=0, conditional=False,
                source=dict(execution=d['physical_context']['execution_source'],
                            input=d['physical_context']['input_source']), records=[],
                resolved_context=dict(preparation_hash='MOCK_PREP', tail_hash='MOCK_TAIL',
                    lambda_R=1, scalar=.25, deterministic_fragments=[0], randomized_fragments=[1, 2]))
    grouped = c_functions()._paths({d['request_id']: d}, [path], B.a.exact_scalar)
    assert grouped[d['request_id']][0]['resolved_context']['scalar'] == .25
    assert d['physical_context']['expected_scalar'] == .125
