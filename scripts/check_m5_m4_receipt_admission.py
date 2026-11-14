"""Receipt-only admission and fixed metadata plan; no science execution entry.

Reuse the metadata/record functions of D 62ae1bd5 and C's existing guard.
The historical trial is copied, not recomputed or assigned a new cohort.
No fit/predict/compile, input-array reader, N or G function is called here.
"""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from trottertracks import baseline_signal_pipeline as C
from trottertracks import price_implementation_guard as G
from trottertracks import paper_nominal_cost as PN
from trottertracks.r_price_phase1_science import functions

ROOT = Path(__file__).resolve().parents[1]
D_SOURCE = '62ae1bd56718287de5d958cf0ed6411f5b7395c9'
C_SOURCE = 'c628e9e8447c11c984064b922cbce6c52dc72d48'
B_SOURCE = 'eecbcdaed88b626b00858c7e94b15f598258d5ec'
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
BASE = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
CATALOG = dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',
    path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json')
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
    path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
PID = '0cd1527c80f7b9841a83'
OUTPUT = 'docs/research/m5_m4_receipt_admission_20261114'


def document(root, ref):
    return json.loads(C.blob(root, ref['commit'], ref['path']))


def d_callers(root):
    """Extract only D's existing metadata/record functions, never execution code."""
    def selected(short, constants, names, extra=None):
        path = 'scripts/' + short + '.py'
        scope = dict(ROOT=Path(root), deepcopy=deepcopy, Fraction=Fraction, json=json,
                     sys=__import__('sys'), **(extra or {}))
        nodes = [n for n in ast.parse(C.blob(root, D_SOURCE, path)).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
        if {t.id for n in nodes for t in n.targets if isinstance(t, ast.Name)} != set(constants):
            raise ValueError('Pinned D constants unavailable')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), D_SOURCE + ':' + path, 'exec'), scope)
        functions(root, D_SOURCE, path, names, scope)
        return SimpleNamespace(**scope)
    t = selected('run_truncated_p7_prices',
        ('CALLER', 'H', 'TEMPLATES', 'PREFIX', 'ROUNDS', 'T3', 'PLAN_META', 'A_MENU', 'MANIFEST',
         'NATIVE', 'WRAPPER', 'NO_RTE', 'NATIVE_POLICY', 'RECORD_SCHEMA'),
        ('_meta', 'read', 'enc', 'plans'))
    f = selected('run_full_pf_shortlist_prices',
        ('CALLER', 'RANK', 'COVERAGE', 'ROLES', 'COHORT', 'RECORD_SCHEMA', 'SCALAR_QUANTITY', 'SCALAR_SOURCE'),
        ('plans', 'price_bindings', 'c_model', 'fit_inputs'), dict(T3=t))
    return t, f


def inspect(trial, carrier, catalog, expected, provenance, profile, constant):
    """Equality and original gate checks only; no price averages or state action."""
    def require(ok, why):
        if not ok:
            raise ValueError(why)
    require(trial['candidate'] == expected == carrier['candidate'] == catalog['physical_candidate'],
            'Physical candidate/formula/q/actual width differs')
    require((expected['system'], expected['method'], expected['prefix'], expected['q']) == ('H4', 'FULL', 10, 1),
            'Wrong saved M5 scope')
    require(expected['formula'] == '4th(m5_best)' and expected['id'] == PID, 'Wrong saved M5 id')
    require(trial['scalar'] == constant, 'Hamiltonian constant scalar differs')
    require(trial['status'] == 'PASS' and trial['cohort'] == carrier['cohort'] == 'EXPLORATION'
            and trial['index'] == 0 and carrier['n'] == 1 and carrier['paired_cosine_sine'] is True,
            'Original status/cohort/index/pair differs')
    require(trial['conditional'] is False and carrier['conditional_sampling'] is False
            and trial['laws'] == [] and trial['log_B'] == 0, 'Deterministic/no-RTE condition differs')
    require(set(trial['prices']) == {'cosine', 'sine'}, 'Missing measured axis')
    require(profile['P'] == 0 and profile['control'] == 'symmetric_directional'
            and profile['wrapper'] == 'full_Hadamard_with_measurement'
            and profile['scope'] == 'single_hadamard_interrogation_without_state_preparation',
            'Wrapper/control/P0 differs')
    require(provenance['compiler'] == dict(hash=profile['compiler'], basis=['rz', 'sx', 'x', 'cx'],
            optimization_level=1, seed_transpiler=17), 'Compiler differs')
    require(provenance['runtime']['qiskit'] == profile['qiskit_version'] == '1.3.0', 'Qiskit differs')
    require(provenance['cost_policy'] == 'FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0',
            'Price quantity differs')
    required = {'low_control', 'high_control_leakage', 'independent_occupation_vector',
                'Hadamard_cosine', 'compiled_vector_cosine', 'Hadamard_sine', 'compiled_vector_sine'}
    require({x['check'] for x in trial['checks']} == required, 'Original implementation gates missing')
    require(all(x['error'] <= x['tolerance'] and x['phase_alignment'] is False for x in trial['checks']),
            'Original implementation gate failed')
    price = catalog['prices'][0]
    for axis, unit in (('X', 'cosine'), ('Y', 'sine')):
        rz = trial['prices'][unit]['rz_count']
        require(type(rz) is int and rz >= 0, 'Original integer RZ count missing')
        require(price['axes'][axis]['status'] == 'EXACT_METADATA_MATCH'
                and price['axes'][axis]['rz_count'] == carrier['mean_metrics'][unit]['rz_count'] == rz,
                'Carrier/catalog/receipt price differs')
    return dict(status='ACCEPT_SAVED_DETERMINISTIC_RECEIPT', recompile=False,
                original_preparation_hash=trial['preparation_hash'], scalar=trial['scalar'],
                classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC')


def prepare(root):
    t, f = d_callers(root)
    full, trunc = f.plans(root), t.plans(root)
    template = next(p for p in full if p['formula'] == '4th(m5_best)')
    cell = next(c for c in template['cells'] if c['m'] == 4)
    if cell['action'] != 'SAVED_RECEIPT_FOR_C_ACCEPTANCE':
        raise ValueError('D would recompile saved receipt')
    catalog = next(r for r in document(root, CATALOG)['rows'] if r['physical_candidate_id'] == PID)
    original = catalog['prices'][0]
    raw = Path(original['trial_files'][0]).read_bytes()
    trial = json.loads(raw)
    carrier = json.loads(gzip.decompress(C.blob(root, CARRIER, BASE + 'exploration_costs.json.gz')))[PID]
    provenance = json.loads(C.blob(root, CARRIER, BASE + 'scientific_provenance.json'))
    scalar_meta = document(root, MANIFEST)['shared']['full_h_preparation']
    profile = template['plan']['configuration']['spec']['profile']
    result = inspect(trial, carrier, catalog, cell['row'], provenance, profile, scalar_meta['scalar']['constant'])
    binding = template['plan']['configuration']['spec']['binding']
    for k, v in provenance['inputs']['H4']['binding'].items():
        if binding[k] != v or scalar_meta['binding'][k] != v:
            raise ValueError('Original H/state/snapshot/compiler binding differs: ' + k)
    # Historical input binding T=.8 is not the execution time T=1.6.
    checks = {}
    old = '27ebabdd2943a4de787ec8b976c38e06fda5a497'
    for path in ('scripts/h46_long_time_native.py', 'scripts/h46_long_time_signal.py',
                 'scripts/h46_m5_tol_native.py', 'scripts/h46_m5_contract.py'):
        checks[path + ':whole_file_bytes'] = C.blob(root, old, path) == C.blob(root, t.NATIVE['commit'], path)
    for old_source, path, names in ((old, t.NATIVE['path'], ('compose', 'native_context', 'compile_cached', 'cost_trial')),
                        (provenance['provider_SOURCE'], t.WRAPPER['path'], ('build_deterministic_native', 'build_native_hadamard_wrapper'))):
        for name in names:
            def node(commit):
                return ast.dump(next(n for n in ast.parse(C.blob(root, commit, path)).body
                    if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
            checks[path + ':' + name] = node(old_source) == node(t.NATIVE['commit'])
    if not all(checks.values()):
        raise ValueError('Historical native/wrapper SOURCE route differs')
    source = dict(commit=CARRIER, path=BASE + 'exploration_costs.json.gz', key=PID,
                  original_trial=dict(path=original['trial_files'][0], sha256=hashlib.sha256(raw).hexdigest()),
                  catalog=CATALOG)
    # Use D's unchanged record emitter with the ORIGINAL saved cohort.
    # This isolated namespace does not modify any D source file or new-run cohort.
    f.price_bindings.__globals__['COHORT'] = trial['cohort']
    prices, sides, proofs = f.price_bindings(root, template['plan'], cell, trial, price_source=source,
        evidence_commit='UNPUBLISHED_RECEIPT_BUNDLE', evidence_path=OUTPUT + '/IMPLEMENTATION_EVIDENCE.json')
    fit_records = f.fit_inputs(prices, route='FULL_PF', formula=template['formula'], width_over_h=template['width_over_h'])
    cells = []
    for route, templates in (('FULL_PF', full), ('TRUNCATED_PF', trunc)):
        for p in templates:
            for c in p['cells']:
                for axis in ('X', 'Y'):
                    cells.append(dict(route=route, formula=p['formula'], width_over_h=p['width_over_h'],
                        configuration_id=p['configuration_id'], m=c['m'], axis=axis, physical_candidate=deepcopy(c['row']),
                        role='HOLDOUT' if c['m'] == 6 else 'CALIBRATION',
                        action=c.get('action', 'NEW_COMPILE'),
                        features=dict(one=1, q=c['row']['q']),
                        model_group=[route, p['formula'], p['width_over_h'], axis]))
    result.update(source=source, native_source_checks=checks, configuration_id=template['configuration_id'],
        physical_candidate=cell['row'], profile=profile, original_checks=trial['checks'],
        original_scalar_metadata=dict(source=MANIFEST, selector=['shared','full_h_preparation','scalar','constant']),
        quantities_reestimated=False, model_fit=None, G=None, comparison_policy_adopted=False)
    return dict(admission=result, prices=prices, sides=sides, proofs=proofs, fit_records=fit_records,
                cells=cells, raw=raw, plan=template['plan'])


def probe(root, packet, bundle_commit):
    """Guard compatibility only; its policy is NOT a selected comparison policy."""
    prices, sides = deepcopy(packet['prices']), deepcopy(packet['sides'])
    for side in sides:
        side['implementation_evidence']['commit'] = bundle_commit
    config = packet['plan']['configuration']
    policy = dict(id='TECHNICAL_COMPATIBILITY_PROBE_ONLY_NOT_ADOPTED',
        source=dict(commit=C_SOURCE, path='src/trottertracks/price_implementation_guard.py'),
        profile=deepcopy(config['spec']['profile']),
        by_method=dict(FULL_PF=dict(native_policy='H46_LONG_TIME_NATIVE_COMPOSE',
            rte_rotation_lowering=G.NO_RTE, scalar_quantity='HAMILTONIAN_CONSTANT')))
    cohorts = {(config['rounds'][4]['candidate']['id'], a): 'EXPLORATION' for a in ('X', 'Y')}
    checked = G.check(root, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
    B = C.published_api(str(root)).B
    tiers = {p['axis']: PN._tier(B, p, 'FULL_PF') for p in prices}
    return dict(guard=checked, tiers=tiers, sides=sides, comparison_policy_adopted=False, G=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output only')
    args = parser.parse_args()
    packet = prepare(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, key in (('ADMISSION.json','admission'), ('PRICES.json','prices'), ('SIDECARS_UNBOUND.json','sides'),
                      ('IMPLEMENTATION_EVIDENCE.json','proofs'), ('SAVED_CALIBRATION_INPUTS.json','fit_records'),
                      ('CELL_PLAN.json','cells')):
        (args.output / name).write_text(json.dumps(packet[key], indent=2) + '\n')
    (args.output / 'ORIGINAL_TRIAL.json').write_bytes(packet['raw'])
    print('ACCEPT_SAVED_DETERMINISTIC_RECEIPT; no compile, fit, prediction, N or G')


if __name__ == '__main__':
    main()
