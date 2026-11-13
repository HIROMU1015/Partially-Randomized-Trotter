"""P1 paired development pilot with the efficient controlled ZZ lowering (metadata + runner core).

Reuses the C1 science backend (r_price_phase1_science.Backend, SOURCE 1a87e533, loaded from
the pinned blob): saved H4 inputs/preparation checks, the carrier sampler sample_requests
(PCG64/CDF law), compose -> build_native_hadamard_wrapper(include_measurement=True) ->
transpile_and_measure_cost, and the C1 Accounting structural S on the same events. The only
change is the controlled-ZZ lowering inside the RTE tail (efficient_controlled_zz). Scope:
m0 (q1/R2) and m6 (q64/R128), 2 normal paths each, X/Y per path -> at most 8 compiles.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from . import pf_pr_rpe_pipeline as p
from . import efficient_controlled_zz as E

C1_SCIENCE = '1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7'
C1_PLAN = ('2e6a0cdf5a647058cc7afeeb5741bfafa24c974b', 'docs/research/r_price_phase1_results_20261111/run01/PLAN.json')
B_CONFIG = ('030f3f548a35d98a88e62e92a35f132e0243a293', 'docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json')
B_CONFIGURATION_ID = '587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911'
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
COHORT = 'D_P1_EFFICIENT_CZZ_20261113'
ROUNDS = (0, 6)
PATHS = (0, 1)
COMPILE_BUDGET = 8
CANDIDATE_FIELDS = ('K', 'R_outer', 'T', 'delta', 'formula', 'id', 'method', 'prefix', 'q', 'system')


def _blob(root, commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)


def trial_seed_function(root):
    """The carrier trial_seed (sha256 of a JSON list); free names are checked before exec."""
    tree = ast.parse(_blob(root, CARRIER, 'scripts/h46_long_time_native.py'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trial_seed')
    import builtins
    scope = dict(hashlib=hashlib, json=json)
    loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assigned = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    free = loaded - assigned - {a.arg for a in node.args.args} - set(dir(builtins))
    if not free <= set(scope):
        raise ValueError('trial_seed needs unsupplied names: ' + ','.join(sorted(free - set(scope))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), CARRIER + ':scripts/h46_long_time_native.py', 'exec'), scope)
    return scope['trial_seed']


def make_plan(root, source_label):
    """Fixed before execution: cells, candidates, seeds, runtime paths, lowering identity."""
    store = p.SavedArtifacts(Path(root))
    config = store.read(dict(commit=B_CONFIG[0], path=B_CONFIG[1]))
    if config['id'] != B_CONFIGURATION_ID:
        raise ValueError('Canonical B configuration differs')
    c1 = store.read(dict(commit=C1_PLAN[0], path=C1_PLAN[1]))
    binding = c1['menus']['H4']['binding']
    for k, v in binding.items():
        if k not in ('T', 'profile', 'profile_is_reference_only', 'rank_is_saved_observation_not_generation_argument') and not p.same(config['spec']['binding'].get(k), v):
            raise ValueError('C1 H4 binding differs from the canonical configuration: ' + k)
    seed = trial_seed_function(root); rows = []; cells = []
    for m in ROUNDS:
        r = config['rounds'][m]; c = {k: r['physical_candidate'][k] for k in CANDIDATE_FIELDS}
        if p.candidate_id(c) != c['id'] or r['structure']['signed_weights'] != [1.0] or r['structure']['allocation'] != [c['R_outer']]:
            raise ValueError('Physical candidate/structure of m%d differs' % m)
        rows.append(dict(m=m, q_RPE=r['q_RPE'], q_PF=r['q_PF'], candidate=c, structure=deepcopy(r['structure']),
                         V2_candidate_id=r['candidate']['id'], identity_configuration_id=config['id']))
        seeds = [[seed(c, COHORT, i, outer, 0) for outer in range(c['q'])] for i in PATHS]
        cells.append(dict(id='H4_S0_m%d_q%d_R%d' % (m, c['q'], c['R_outer']), system='H4', m=m, candidate_id=c['id'],
                          role='DEVELOPMENT', cohort=COHORT, n=len(PATHS), seeds=seeds))
    return dict(schema='D_P1_EFFICIENT_CZZ_PLAN_V1', status='FIXED_BEFORE_SCIENCE', source_label=source_label,
        lowering=dict(id=E.LOWERING_ID, legacy=E.LEGACY_LOWERING_ID, provider=E.PROVIDER, p0=E.P0),
        configuration_id=config['id'], menus=dict(H4=dict(binding=binding, candidates=rows)), cells=cells,
        runtime=deepcopy(c1['runtime']), science_backend=dict(commit=C1_SCIENCE, modules=['r_price_phase1', 'r_price_phase1_runner', 'r_price_phase1_science']),
        compile_budget=COMPILE_BUDGET, trajectory_paths=len(ROUNDS) * len(PATHS), sampling_law='UNCHANGED_PCG64_NUMPY_CDF_FINITE_K2 (carrier sample_requests)',
        compiler=dict(settings_hash=p.PROFILE['compiler'], basis=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=17, qiskit='1.3.0'),
        not_in_scope=['legacy full-wrapper compile', 'P2/P3', 'fit', 'all-law E[C]', 'N/RMSE/G', 'retry/path addition'])


def load_c1_backend(root):
    """Load r_price_phase1{,_runner,_science} from the pinned C1 SOURCE into trottertracks.*."""
    root = Path(root); mods = {}
    for name in ('r_price_phase1', 'r_price_phase1_runner', 'r_price_phase1_science'):
        path = 'src/trottertracks/%s.py' % name
        module = types.ModuleType('trottertracks.' + name); module.__package__ = 'trottertracks'; module.__file__ = str(root/path)
        sys.modules['trottertracks.' + name] = module
        exec(compile(_blob(root, C1_SCIENCE, path), C1_SCIENCE + ':' + path, 'exec'), module.__dict__)
        mods[name] = module
    return mods


def event_metrics(events):
    """Counts from the sampled event dicts (same objects the structural S reads)."""
    out = dict(events=len(events), rotation_ZZ=0, rotation_Z=0, identity_applications=0, product_applications=0, order2_events=0, orders={})
    for e in events:
        out['orders'][str(e['taylor_order'])] = out['orders'].get(str(e['taylor_order']), 0) + 1
        out['order2_events'] += e['taylor_order'] == 2
        for a in e['application_sequence']:
            if a['is_identity']:
                out['identity_applications'] += 1
            elif a['role'] == 'product':
                out['product_applications'] += 1
            elif a['role'] == 'rotation':
                out['rotation_ZZ' if len(a['diagonal_pauli_support']) == 2 else 'rotation_Z'] += 1
    return out


def summarize(plan, jobs):
    """Per-round descriptive summary of the finite development cohort (n paths, X/Y paired)."""
    from fractions import Fraction
    out = []
    for cell in plan['cells']:
        rows = sorted((j for j in jobs if j['cell_id'] == cell['id']), key=lambda j: j['index'])
        entry = dict(cell_id=cell['id'], m=cell['m'], n_paths=len(rows), expectation='FINITE_DEVELOPMENT_COHORT_ONLY')
        if not rows:
            out.append(dict(entry, status='NO_PASSED_PATHS')); continue
        for a, axis in enumerate(('X', 'Y')):
            D = [r['D'][a] for r in rows]; C = [r['C'][a] for r in rows]; S = [r['S'][a] for r in rows]
            mean = Fraction(sum(D), len(D))
            var = sum((Fraction(d) - mean) ** 2 for d in D) / (len(D) - 1) if len(D) > 1 else None
            entry[axis] = dict(C=C, S=S, D=D, mean_D=str(mean), descriptive_variance_D=None if var is None else str(var),
                               D_over_C=[float(Fraction(d, c)) for d, c in zip(D, C)])
        entry['X_equals_Y'] = all(r['C'][0] == r['C'][1] and r['S'][0] == r['S'][1] for r in rows)
        entry['rotation_ZZ'] = [r['events']['rotation_ZZ'] for r in rows]
        entry['order2_events'] = [r['events']['order2_events'] for r in rows]
        entry['legacy_lowering_reference_RZ'] = dict(per_controlled_ZZ_P0=[17, 19],
            estimated_extra_RZ_if_legacy=[[17 * r['events']['rotation_ZZ'], 19 * r['events']['rotation_ZZ']] for r in rows],
            kind='P0_LOCAL_DELTA_TIMES_N_ZZ_REFERENCE_NOT_A_LEGACY_COMPILE')
        out.append(entry)
    return out


USED_BACKEND_METHODS = ('__init__', 'context', 'price_requests')


def backend_plan_keys(root, methods=USED_BACKEND_METHODS):
    """Keys the pinned C1 Backend methods used by P1 read from self.plan (AST scan of the pinned SOURCE)."""
    tree = ast.parse(_blob(Path(root), C1_SCIENCE, 'src/trottertracks/r_price_phase1_science.py'))
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Backend')
    keys = set()
    for n in (x for f in backend.body if isinstance(f, ast.FunctionDef) and f.name in methods for x in ast.walk(f)):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == 'plan'
                and isinstance(n.value.value, ast.Name) and n.value.value.id == 'self' and isinstance(n.slice, ast.Constant)):
            keys.add(n.slice.value)
    return keys


def execution_plan(plan, source, root):
    """The fixed plan plus the execution SOURCE, as the C1 Backend expects plan['source']."""
    out = dict(deepcopy(plan), source=source)
    missing = backend_plan_keys(root) - set(out)
    if missing:
        raise ValueError('Plan lacks keys read by the C1 Backend: ' + ','.join(sorted(missing)))
    return out
