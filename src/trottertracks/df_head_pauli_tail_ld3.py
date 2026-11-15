"""LD3 extension of the fixed Pauli-tail P0 builder and P1 adapter (stage 1).

The fixed modules stay byte-unchanged (P0 fd111f989, P1 05447d1d4). Only their
prefix constraints are lifted here:

* P0: the fixed ``diagnose`` body runs with its literal ``(1, 2)`` prefix loop
  as the single AST change; JW, global collection, identity and gates are the
  fixed code.
* original DF configuration: the a9abf8de PREPARE_MENU inputs path (B
  437ba4ac) with the prefix as the only change.
* P1: the fixed ``generate_configuration`` runs with a dictionary identity
  whose P0 residual locator follows the dictionary. With the saved LD1/LD2
  P0 locator it reproduces the saved configurations.

Metadata and operator dictionaries only: no mean, sampling, build/compile,
QPY, price, Q3 or shot allocation.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

from . import df_head_pauli_tail_p0 as p0
from . import df_head_pauli_tail_p1 as p1

MODULE = 'src/trottertracks/df_head_pauli_tail_ld3.py'
P0_CODE = 'fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f'
P0_MODULE = 'src/trottertracks/df_head_pauli_tail_p0.py'
P1_CODE = '05447d1d4f1e3c0e9e335a7085aa58a877c139f2'
PREPARE = dict(commit=p0.MENU, path=p0.DIRECTORY+'PREPARE_MENU.py')
# PREPARE_MENU.prepare's literal kappa reference (not a module constant there).
S0_V2 = dict(commit='3216d27a69ac8532d2a22e78f5f94491d202d238',
    path='docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json')
SAVED_P0 = dict(commit=p1.P0, directory=p1.P0DIR+'run01/')
SAVED_PAULI = dict(commit='e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7',
    path='docs/research/df_head_pauli_tail_p1_results_20261114/run01/PLAN.json',
    selector=['common_kappa_configurations'])
SAVED_PAULI_IDS = {1: 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27',
    2: 'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'}
P0_SUCCESS = 'P0_OPERATOR_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'
LD3_P0_SUCCESS = 'P0_LD3_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'


def read(root, commit, path):
    return json.loads(p0.blob(root, commit, path))


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def fixed_modules(root):
    """The reused helpers are the fixed P0/P1 module bytes."""
    out = []
    for commit, path in ((P0_CODE, P0_MODULE), (P1_CODE, p1.MODULE)):
        raw = p0.blob(root, commit, path)
        if (Path(root)/path).read_bytes() != raw:
            raise ValueError('FIXED_P0_P1_MODULE_CHANGED: '+path)
        out.append(dict(commit=commit, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    return out


def p0_metadata(root, prefixes):
    """Fixed p0.metadata bindings, with the same saved context's descriptors.

    p0.metadata still checks the LD1/LD2 MENU configurations and every saved
    input byte; only the descriptor selection is widened to ``prefixes``.
    """
    meta = p0.metadata(root)
    spec = meta['context_source']
    raw = Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('SAVED_CONTEXT_BYTES_CHANGED')
    context = json.loads(raw)
    tails = {str(p): deepcopy(context['prefixes'][str(p)]) for p in prefixes}
    if any(not math.isfinite(t[k]) for t in tails.values() for k in ('scalar', 'lambda_R')):
        raise ValueError('SAVED_TAIL_DESCRIPTOR')
    if any(tails[k] != meta['saved_tail_descriptors'][k] for k in ('1', '2') if k in tails):
        raise ValueError('SAVED_TAIL_DESCRIPTOR_DIFFERS_FROM_FIXED_P0')
    meta.update(saved_tail_descriptors=tails, tail_prefixes=list(prefixes),
        tail_descriptor_sources={str(p): dict(index=deepcopy(p0.INDEX), context_sha256=spec['sha256'],
            selector=['prefixes', str(p)]) for p in prefixes})
    return meta


def diagnose(root, ham, sector, full_state, state, meta, result, save, *, prefixes):
    """Fixed P0 diagnose; only its literal (1, 2) prefix loop becomes ``prefixes``."""
    prefixes = tuple(prefixes)
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or any(type(p) is not int or not 1 <= p < ham.n_blocks for p in prefixes)):
        raise ValueError('PREFIX_OUTSIDE_NONEMPTY_TAIL')
    node = next(n for n in ast.parse(p0.blob(root, P0_CODE, P0_MODULE)).body
        if isinstance(n, ast.FunctionDef) and n.name == 'diagnose')
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name) and n.target.id == 'prefix']
    if len(loops) != 1 or not isinstance(loops[0].iter, ast.Tuple) or ast.literal_eval(loops[0].iter) != (1, 2):
        raise ValueError('FIXED_P0_PREFIX_LOOP_NOT_FOUND')
    loops[0].iter = ast.copy_location(ast.Name(id='PREFIXES', ctx=ast.Load()), loops[0].iter)
    scope = dict(vars(p0), PREFIXES=prefixes)
    exec(compile(ast.Module(body=[node], type_ignores=[]), P0_CODE+':'+P0_MODULE, 'exec'), scope)
    return scope['diagnose'](root, ham, sector, full_state, state, meta, result, save)


def saved_control(root, result, held, prefix=2):
    """Compare this run's LD2 and common products with the saved 4220f3006 P0."""
    import numpy as np
    base = SAVED_P0['directory']
    old = read(root, SAVED_P0['commit'], base+'result.json')
    new_row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
    old_row = next(r for r in old['prefixes'] if r['prefix'] == prefix)
    def archive(name):
        with np.load(io.BytesIO(p0.blob(root, SAVED_P0['commit'], base+name)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    products = {}
    for name in ('COMMON_OPERATORS.npz', 'LD'+str(prefix)+'_OPERATORS.npz'):
        saved, now = archive(name), held[name]
        products[name] = (set(saved) == set(now)
            and all(np.array_equal(saved[k], np.asarray(now[k])) for k in saved))
    for name in ('LD'+str(prefix)+'_DICTIONARY.json', 'LD'+str(prefix)+'_JW_COEFFICIENTS.json'):
        products[name] = read(root, SAVED_P0['commit'], base+name) == json.loads(json.dumps(held[name]))
    products['dictionary_sha256'] = new_row['dictionary_sha256'] == old_row['dictionary_sha256']
    fields = sorted(set(old_row) | set(new_row))
    record_equal = {k: old_row.get(k) == new_row.get(k) for k in fields}
    common_equal = old['common_operators'] == result['common_operators']
    names = {c['name'] for c in result['checks']}
    saved_checks = {c['name']: c for c in old['checks'] if c['name'] in names}
    check_values_equal = all(saved_checks[c['name']] == c for c in result['checks'] if c['name'] in saved_checks)
    return dict(prefix=prefix, source=dict(commit=SAVED_P0['commit'], path=base+'result.json'),
        products_bitwise=products, record_fields_equal=record_equal, common_operator_metrics_equal=common_equal,
        shared_check_values_equal=check_values_equal, shared_checks=len(saved_checks),
        PASS=all(products.values()))


def original_df_configurations(root, prefixes):
    """a9abf8de PREPARE_MENU inputs path, prefix as the only change (metadata only)."""
    from . import pr_rpe_rmse_s0_v2 as fixed
    tree = ast.parse(p0.blob(root, PREPARE['commit'], PREPARE['path']))
    names = ('B', 'SIGNAL', 'S0', 'INDEX', 'IMPLEMENTATION')
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    if sorted(n.targets[0].id for n in nodes) != sorted(names):
        raise ValueError('PREPARE_MENU_CONSTANTS')
    const = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), PREPARE['commit']+':'+PREPARE['path'], 'exec'), const)
    store = fixed.p.SavedArtifacts(root)
    s0 = store.read(dict(commit=const['S0'], path=fixed.S0_DIR+'CONFIGURATION.json'))
    original = store.read(S0_V2)
    desc = store.read(const['INDEX'])['T20_reference_files']['H4']
    data = Path(desc['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != desc['sha256']:
        raise ValueError('Saved tail descriptor changed')
    context = json.loads(data)
    if context['binding'] != s0['inputs']['design']['binding']:
        raise ValueError('Saved binding differs')
    source = subprocess.check_output(['git', 'show', const['SIGNAL']+':src/trottertracks/pr_rpe_rmse_s0_v2.py'], cwd=root)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'b_metadata')
    scope = dict(vars(fixed), B_RELEASE=const['B'])
    exec(compile(ast.Module(body=[node], type_ignores=[]), PREPARE['path']+':B_metadata', 'exec'), scope)
    b = scope['b_metadata'](root)
    out = {}
    for prefix in prefixes:
        inputs = deepcopy(s0['inputs']); inputs['design']['prefix'] = prefix
        inputs['design']['source'] = dict(input_source=deepcopy(s0['inputs']['design']['source']),
            composition_recipe=deepcopy(const['IMPLEMENTATION']), status='PROPOSED_PREFIX_COMPOSITION_NOT_SCIENCE_RESULT')
        tail_source = dict(index=const['INDEX'], context_sha256=desc['sha256'], selector=['prefixes', str(prefix)])
        inputs['tail_norm'].update(prefix=prefix, lambda_R=context['prefixes'][str(prefix)]['lambda_R'], source=tail_source)
        config = b.generate(b.a_spec(inputs, 'PR')); b.validate(config)
        if config['spec']['kappa'] != original['spec']['kappa']:
            raise ValueError('Common paper kappa changed')
        out[prefix] = config
    return out


def dictionary_item(root, commit, directory, prefix, meta):
    """One committed P0 dictionary, with p1.metadata's binding/scalar checks."""
    result = read(root, commit, directory+'result.json')
    if (result['status'] not in (P0_SUCCESS, LD3_P0_SUCCESS)
            or not result['checks'] or not all(c['PASS'] for c in result['checks'])
            or result['binding'] != meta['binding']):
        raise ValueError('P0_EVIDENCE_NOT_AVAILABLE_OR_DIFFERENT')
    row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
    path = directory+'LD'+str(prefix)+'_DICTIONARY.json'
    obj = read(root, commit, path)
    if (canonical_sha(obj) != row['dictionary_sha256'] or obj['binding'] != meta['binding']
            or obj['prefix'] != prefix or 'GATES_PASS' not in row['status']
            or obj['saved_DF_scalar'] != meta['saved_tail_descriptors'][str(prefix)]['scalar']):
        raise ValueError('P0_DICTIONARY_BINDING_OR_SCALAR')
    return dict(payload=obj, dictionary_sha256=row['dictionary_sha256'], source=dict(commit=commit, path=path),
        P0_metrics=row['identity_metrics'], P0_result_source=dict(commit=commit, path=directory+'result.json'))


def dictionary_identity(item):
    """p1.dictionary_identity with the P0 residual locator of this dictionary."""
    value = p1.dictionary_identity(item)
    value['P0_operator_residual_source'] = deepcopy(item['P0_result_source'])
    return value


def generate_configuration(root, b, original, item):
    """Fixed P1 generate_configuration (common kappa only) with the identity above."""
    from .h4_pr_m1_rmse import function
    scope = dict(vars(p1), dictionary_identity=dictionary_identity)
    return function(root, P1_CODE, p1.MODULE, 'generate_configuration', scope)(b, original, item)


def exact(value):
    return Fraction(value['numerator'], value['denominator']) if isinstance(value, dict) else Fraction(value)


def schedule(config):
    """Integer per-step R_m and whole-round events q_m*R_m, checked against B fields."""
    rows = []
    for r in config['rounds']:
        c = r['candidate']; events = r['q_PF']*r['R_outer']
        if (c['R_outer'] != r['R_outer'] or c['q'] != r['q_PF'] or c['event_count'] != events
                or r['mapping']['R_outer'] != r['R_outer'] or r['mapping']['event_count'] != events):
            raise ValueError('SCHEDULE_FIELDS_DISAGREE')
        rows.append(dict(m=r['m'], T=exact(r['T']), q=r['q_PF'], R=r['R_outer'], events=events,
            paper_total_target=r['mapping']['paper_total_target'], physical_ID=c['id']))
    return rows
