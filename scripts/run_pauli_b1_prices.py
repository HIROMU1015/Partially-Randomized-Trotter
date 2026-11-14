"""PR-Pauli B1-W/V/CAL finite cell caller. Default metadata only; no proposal is adopted.

Reuse B0 canonical planner, P0 dictionaries, native DF head, normal sampler,
D2 emitter, full-space oracle, measured X/Y wrappers and compiler. No fit/G.
Non-configuration cells have their own dictionary/context-bound physical ID;
they are C calibration inputs, never relabelled as canonical B round prices.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_pauli_ld1_m0_pilot as B0

CALLER = 'scripts/run_pauli_b1_prices.py'
A_SOURCE = '7a0c9468b2499fac9fa1cd279ae2980c9dab048e'
A_PATH = 'src/trottertracks/finite_price_design.py'
A_DESIGNS = 'docs/research/a_full_h_p2_pauli_b1_price_design_20261114/DESIGNS.json'
B0_SOURCE = dict(commit='8502dd33ad16dc012fb3558668ef3e4cd8a4405c',
                 path='docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02/RESULT.json')


@lru_cache(maxsize=1)
def design_api():
    path = ROOT / A_PATH
    if path.read_bytes() != B0.blob(A_SOURCE, A_PATH):
        raise ValueError('A design checker SOURCE differs')
    spec = importlib.util.spec_from_file_location('_b1_A_design', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def proposal(prefix=1, kind='B1-CAL', *, reuse_b0=False):
    if prefix not in (1, 2) or kind not in ('B1-W', 'B1-V', 'B1-CAL'):
        raise ValueError('Supported prefixes LD1/LD2 and B1-W/V/CAL required')
    if kind == 'B1-W' and prefix != 2:
        raise ValueError('LD1 wiring B0 is already complete; do not repeat B0')
    if reuse_b0 and (prefix, kind) != (1, 'B1-V'):
        raise ValueError('B0 can only be predeclared as LD1 m0 path0 of a wiring/variability cohort')
    table = json.loads(B0.blob(A_SOURCE, A_DESIGNS))['pauli']['LD%d' % prefix]
    d = next(d for d in table['designs'] if d['design'] == 'B1_CAL_SINGLE_STEP_PLUS_Q8_AND_PAIRED')
    domain = d['plan']['domain']
    cells = d['cells'] if kind == 'B1-CAL' else [dict(cell_id='M0', role='CALIBRATION',
              q=table['rounds']['0']['q'], R=table['rounds']['0']['R'], n_paths=1 if kind == 'B1-W' else 2)]
    cells = [dict(deepcopy(domain), **deepcopy(c), T=.1 * c['q'], s=c['q'] * c['R'],
                  expectation='FINITE_COHORT_ONLY', cohort='D_PAULI_LD%d_%s_%s_PROPOSED_V1' % (prefix, kind, c['cell_id'])) for c in cells]
    if reuse_b0:
        cells[0].update(cohort=B0.connection().PILOT_COHORT, declared_reuse=dict(path_indices=[0], source=deepcopy(B0_SOURCE)))
    return dict(prefix=prefix, kind=kind, purpose='MODEL_DESIGN' if kind == 'B1-CAL' else 'WIRING_OR_VARIABILITY_ONLY',
                feature_map='PR_STEP_V1', cells=cells, axes=['X', 'Y'], adopted=False, reuse_b0=reuse_b0,
                design_source=dict(commit=A_SOURCE, path=A_DESIGNS, status='PROPOSAL_NOT_ADOPTED'))


def validate(plan, *, execution=False):
    D = design_api()
    try:
        summary = D.check_plan(plan)
    except ValueError as exc:
        if (plan.get('purpose') != 'WIRING_OR_VARIABILITY_ONLY' or not str(exc).startswith('Unidentified calibration design:')):
            raise
        # Wiring is deliberately rank 1. A's domain/pair/role checks already ran;
        # retain its rank rejection as a prohibition on price-model calibration.
        summary = dict(status='WIRING_ONLY_NOT_A_PRICE_MODEL', model_rejection=str(exc),
                       identifiability=D.identifiability(plan['cells'], plan['feature_map']))
    fixed = proposal(plan['prefix'], plan['kind'], reuse_b0=plan.get('reuse_b0', False))
    if plan['design_source'] != fixed['design_source']:
        raise ValueError('A design SOURCE differs')
    if (plan.get('axes') != ['X', 'Y'] or plan['feature_map'] != fixed['feature_map']
            or plan['purpose'] != fixed['purpose'] or len(plan['cells']) != len(fixed['cells'])):
        raise ValueError('Incomplete or foreign Pauli design')
    for c, expected in zip(plan['cells'], fixed['cells'], strict=True):
        keys = ('cell_id', 'role', 'q', 'R', 's', 'T', 'n_paths', 'composed_of', 'declared_reuse') + D.DOMAIN_KEYS
        if any(c.get(k) != expected.get(k) for k in keys) or c['s'] != c['q'] * c['R'] or not c.get('cohort'):
            raise ValueError('Pauli q/R/s/T/domain/reuse differs from A: ' + c['cell_id'])
        if c.get('declared_reuse') and c['cohort'] != B0.connection().PILOT_COHORT:
            raise ValueError('B0 original cohort must be preserved')
    if execution and plan.get('adopted') is not True:
        raise ValueError('Proposed B1 plan is not adopted for scientific execution')
    return summary


@lru_cache(maxsize=2)
def canonical(prefix):
    req, packet = B0.canonical_request(prefix=prefix)
    return req, packet, B0.connection().load_dictionary(ROOT, packet['dictionary_identity'])


def request_for(plan, cell):
    """Metadata only. Keep canonical IDs for rounds; new physical cell IDs for arbitrary q/R."""
    I, DC, G = B0._meta(); P = B0.connection()
    req, packet, dictionary = canonical(plan['prefix'])
    m = 0 if cell['cell_id'] == 'M0' else int(cell['cell_id'][1]) if cell['cell_id'] in ('M5_ROUND', 'M6_ROUND') else None
    if m is not None:
        physical = P.round_request(packet, m, dictionary)
        context = I.price_context(req, m)
        if physical['candidate_id'] != context['physical_candidate']['id']:
            raise ValueError('A dictionary request differs from C canonical physical ID')
    else:
        anchor = I.price_context(req, 0)
        identity = dict(kind='NON_CONFIGURATION_PAULI_PRICE_CELL', binding=anchor['binding'], split=anchor['split'],
                        profile=anchor['profile'], route=P.ROUTE, lowering=P.LOWERING_ID,
                        q=cell['q'], R=cell['R'], T=cell['T'], K=2)
        physical = dict(m=None, prefix=plan['prefix'], q=cell['q'], R=cell['R'], K=2, T=Fraction(cell['T']),
                        step_time=cell['T'] / cell['q'], lambda_P=dictionary['lambda_P'],
                        candidate_id='PAULI_PRICE_CELL:' + P.canonical_sha256(identity), physical_cell=identity,
                        normalization='EXISTING_FINITE_K_RULE_NOT_EVALUATED_IN_METADATA')
    if (physical['prefix'], physical['q'], physical['R'], float(physical['T'])) != (plan['prefix'], cell['q'], cell['R'], cell['T']):
        raise ValueError('Pauli physical schedule/prefix differs from selected cell')
    physical['T'] = float(physical['T'])  # preserve the original binary64 time in child JSON
    return physical, req, packet, dictionary


def seed_table(plan, cell_id):
    validate(plan); P = B0.connection(); cells = {c['cell_id']: c for c in plan['cells']}; cell = cells[cell_id]
    if cell.get('composed_of'):
        return [[seed_table(plan, token.split('#')[0])[int(token.split('#')[1][1:])][int(token.split('#')[2][1:])]
                 for token in cell['composed_of']]]
    physical = request_for(plan, cell)[0]
    return [[P.seed(physical['candidate_id'], cell['cohort'], i, outer, 0) for outer in range(cell['q'])]
            for i in range(cell['n_paths'])]


def check_composition(result, parents):
    expected = [step for p in parents for step in p['result']['event_metadata']]
    actual = result['event_metadata']
    # outer is a position in the composed path; everything else, including the
    # D2 product tokens and aggregate step phase, must be the same parent data.
    strip = lambda steps: [{k: v for k, v in s.items() if k != 'outer'} for s in steps]
    if json.dumps(strip(actual), sort_keys=True) != json.dumps(strip(expected), sort_keys=True):
        raise ValueError('Paired event/label/product/step phase/seed order differs')
    if any(p['result']['scalar'] != result['scalar'] for p in parents):
        raise ValueError('Paired parent scalar quantity differs')
    return dict(status='PAIRED_STEP_METADATA_IDENTICAL', scalar='ONCE_AT_COMPOSED_TOTAL_T')


def frozen_model_gate(plan, reference):
    """Read C's published model metadata only; never fit, predict or judge a holdout."""
    if not isinstance(reference, dict) or len(reference.get('commit', '')) != 40 or not reference.get('path'):
        raise ValueError('Pinned frozen model SOURCE required')
    models = []
    def walk(value):
        if isinstance(value, dict):
            if value.get('model') == 'PRICE_LINEAR_FEATURE_MODEL_V1': models.append(value)
            else:
                for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
    walk(json.loads(B0.blob(reference['commit'], reference['path'])))
    D = design_api(); cal = {c['cell_id']: c for c in plan['cells'] if c['role'] == 'CALIBRATION'}
    for axis in ('X', 'Y'):
        hits = [m for m in models if m.get('domain', {}).get('axis') == axis]
        if len(hits) != 1: raise ValueError('One frozen model per X/Y axis required')
        m = hits[0]; expected = dict(D._domain(next(iter(cal.values()))), axis=axis)
        if (m.get('status') != 'CALIBRATED_UNVALIDATED' or m.get('qualified_for_candidate') is not False
                or m.get('features') != list(D.FEATURE_MAPS[plan['feature_map']][0])
                or set(m.get('coefficients', {})) != set(D.FEATURE_MAPS[plan['feature_map']][0])
                or m.get('domain') != expected or m.get('design_source') != plan['design_source']
                or sorted(r['cell_id'] for r in m.get('calibration', [])) != sorted(cal)):
            raise ValueError('Frozen model domain/features/calibration/design differs')
        for r in m['calibration']:
            c = cal[r['cell_id']]; s = r['source']
            if s.get('cohort') != c['cohort'] or s.get('step_seeds') != seed_table(plan, c['cell_id']):
                raise ValueError('Frozen calibration cohort/seed SOURCE differs')
    return dict(status='FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION', source=deepcopy(reference))


def preflight(plan):
    status = validate(plan)
    return dict(status='METADATA_ONLY_NOT_EXECUTION', design=status, purpose=plan['purpose'], prefix=plan['prefix'],
        A_order_proposal=['B1-W LD2', 'B1-CAL LD1', 'resource observation', 'B1-CAL LD2'], B1_V='OPTIONAL_NOT_ADOPTED',
        acquisition_budget=dict(new_compiles=2 * sum(c['n_paths'] - len((c.get('declared_reuse') or {}).get('path_indices', [])) for c in plan['cells']),
                                reused_compiles=2 * sum(len((c.get('declared_reuse') or {}).get('path_indices', [])) for c in plan['cells'])),
        cells=[dict(cell_id=c['cell_id'], role=c['role'], request=request_for(plan, c)[0], cohort=c['cohort'],
                    step_seeds=seed_table(plan, c['cell_id']), composed_of=c.get('composed_of'), reuse=c.get('declared_reuse'))
               for c in plan['cells']],
        existing_evidence=dict(B0=deepcopy(B0_SOURCE), use='wiring; predeclared same-cell cohort path0 only',
            P3_DF_tail_Pauli_JW='DIFFERENT_LAW_HEAD_K_OR_DICTIONARY_NOT_REUSABLE',
            LD2='NO_SAVED_B1_WIRING_RECEIPT_AT_FIXED_SOURCES',
            recheck_before_science=['table/catalog', 'original trial/receipt', 'relevant running jobs', 'SOURCE/cohort/seed']),
        adopted=plan.get('adopted', False), mean=None, fit=None, N=None, G=None, mandatory_stop=True)


def acquire_path(native, ctx, dictionary, request, *, cohort, index, seeds, operators=None, blocks=None, compile_cached=None):
    """Scientific API; B0 core and full 2^n oracle unchanged. Tests supply a toy saved-operator fixture."""
    if request['prefix'] != dictionary['prefix'] or len(seeds) != request['q']:
        raise ValueError('Prefix/dictionary or seed table mismatch')
    if operators is None:
        operators, blocks, _ = B0.p0_arrays(request['prefix'])
    return B0.pilot_core(native, ctx, dictionary, request, operators, blocks, cohort=cohort,
                         compile_cached=compile_cached or native.compile_cached, index=index, step_seeds=seeds, event_metadata=True)


def ld2_wiring_gate(reference):
    """Published LD2 m0 wiring result metadata only; no replay of old circuits or numerical gates."""
    if not isinstance(reference, dict) or len(reference.get('commit', '')) != 40 or not reference.get('path'):
        raise ValueError('Published LD2 B1-W SOURCE required before LD2 calibration')
    data = json.loads(B0.blob(reference['commit'], reference['path']))
    wire = proposal(2, 'B1-W'); expected = request_for(wire, wire['cells'][0])[0]
    paths = (data.get('child') or {}).get('paths', [])
    required = {'low_control', 'high_control_leakage', 'independent_full_space_vector', 'Hadamard_cosine',
                'Hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'}
    if (data.get('status') != 'FINITE_CELL_COMPLETE' or len(paths) != 1 or paths[0].get('request') != expected
            or paths[0]['result'].get('status') != 'PASS'
            or not required <= {g['check'] for g in paths[0]['result'].get('checks', [])}):
        raise ValueError('LD2 B1-W physical/gate metadata incomplete')
    return dict(status='LD2_WIRING_METADATA_CHECKED', source=deepcopy(reference))


def records(plan, cell, paths, *, source, evidence_path):
    """Finite path receipts -> C records/round sidecars; no fit or promotion of the law expectation."""
    validate(plan); D = design_api(); P = B0.connection(); physical, req, packet, dictionary = request_for(plan, cell)
    if len(paths) != cell['n_paths'] or [p['index'] for p in paths] != list(range(cell['n_paths'])):
        raise ValueError('Incomplete/duplicate path set')
    if len({p['result']['scalar'] for p in paths}) != 1:
        raise ValueError('Mixed scalar quantities across paths')
    for p, seeds in zip(paths, seed_table(plan, cell['cell_id']), strict=True):
        if p['cohort'] != cell['cohort'] or p['request'] != physical or p['result']['seeds'] != seeds:
            raise ValueError('Axis/path/cohort/seed/physical binding differs')
    means = {a: {k: sum(Fraction(p['result']['prices'][a][k]) for p in paths) / len(paths)
                  for k in paths[0]['result']['prices'][a]} for a in ('cosine', 'sine')}
    result = dict(law_expectation=False, model_qualified=False, C_inputs=[])
    if plan['purpose'] == 'MODEL_DESIGN':
        M = D.c_model(ROOT)
        result['C_inputs'] = [M.record(cell_id=cell['cell_id'], role=cell['role'], route=P.ROUTE, axis=axis,
            features=D.features(cell, plan['feature_map']), rz_count=means[a]['rz_count'], kind=cell['kind'], source=source,
            implementation=cell['implementation'], profile=cell['profile'], n_paths=len(paths), law=cell['law'])
            for axis, a in (('X', 'cosine'), ('Y', 'sine'))]
    if physical['m'] is not None:
        I, DC, G = B0._meta(); m = physical['m']; config = req['configuration']; B = I.providers(str(ROOT)).B
        numeric = {a: {k: float(v) for k, v in metrics.items()} for a, metrics in means.items()}
        prices, sides, proofs = P.price_records([r for r in DC.pauli_requests(ROOT, req) if r['m'] == m], config,
            trial_prices=numeric, scalar=paths[0]['result']['scalar'], native_source=dict(source, path=P.MODULE),
            cohort=cell['cohort'], price_source=source, evidence_commit='FILL_WITH_PUBLISHED_COMMIT',
            evidence_path=evidence_path, measurement=B.p.measurement)
        for price in prices:
            price.update(n_paths=len(paths)); price['diagnostic'].update(pilot_role=plan['purpose'], law_expectation=False)
        result.update(prices=prices, sidecars=sides, implementation_evidence=proofs)
    return result


def check_round_records(plan, cell, prices, sides, *, evidence_commit):
    I, DC, G = B0._meta(); P = B0.connection()
    if (ROOT / 'src/trottertracks/price_implementation_guard.py').read_bytes() != B0.blob(
            'c628e9e8447c11c984064b922cbce6c52dc72d48', 'src/trottertracks/price_implementation_guard.py'):
        raise ValueError('Fixed C guard differs')
    physical, req, packet, dictionary = request_for(plan, cell); m = physical['m']
    if m is None or len(prices) != 2 or {p['axis'] for p in prices} != {'X', 'Y'}:
        raise ValueError('Non-configuration calibration cells are C inputs, not B round identities')
    config = req['configuration']; sides = deepcopy(sides)
    for s in sides:
        s['implementation_evidence']['commit'] = evidence_commit
    cohorts = {(config['rounds'][m]['candidate']['id'], axis): cell['cohort'] for axis in ('X', 'Y')}
    policy = dict(id='B1_TECHNICAL_COMPATIBILITY_NOT_ADOPTED', source=dict(commit=A_SOURCE, path=A_DESIGNS),
        profile=config['spec']['profile'], by_method={P.ROUTE: dict(native_policy=P.NATIVE_POLICY,
        rte_rotation_lowering=P.LOWERING_ID, scalar_quantity=P.SCALAR_QUANTITY)})
    checked = G.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
    return dict(checked, bound_sidecars=sides)


def execute(job):
    plan = job['plan']; validate(plan, execution=True)
    cell = next(c for c in plan['cells'] if c['cell_id'] == job['cell_id'])
    if cell['role'] != job['stage'] or job.get('step_seeds') != seed_table(plan, cell['cell_id']):
        raise ValueError('Wrong stage or incomplete explicit seed table')
    if not job.get('duplicate_preflight_complete'):
        raise ValueError('Current table/trial/receipt/running-job duplicate check required')
    if cell['role'] == 'HOLDOUT' and not job.get('frozen_model_source'):
        raise ValueError('HOLDOUT requires frozen C model SOURCE')
    if cell['role'] == 'HOLDOUT':
        frozen_model_gate(plan, job['frozen_model_source'])
    if plan['prefix'] == 2 and plan['purpose'] == 'MODEL_DESIGN':
        ld2_wiring_gate(job.get('ld2_wiring_source'))
    parents = job.get('parents', [])
    if cell.get('composed_of') and (len(parents) != cell['q'] or
            [p['result']['seeds'][0] for p in parents] != job['step_seeds'][0]):
        raise ValueError('Paired parent step seed receipts required')
    if cell.get('composed_of'):
        for token, parent in zip(cell['composed_of'], parents, strict=True):
            cid, index, _ = token.split('#'); old = next(c for c in plan['cells'] if c['cell_id'] == cid)
            if (parent['request'] != request_for(plan, old)[0] or parent['cohort'] != old['cohort']
                    or parent['index'] != int(index[1:])):
                raise ValueError('Paired parent dictionary/physical/cohort/index binding differs')
    physical, req, packet, dictionary = request_for(plan, cell)
    out, private = (ROOT / job['output']).resolve(), (ROOT / job['private']).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Public output must be in this isolated worktree')
    if out.exists() or private.exists():
        raise FileExistsError('Fresh cell namespace required')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for path in (CALLER, B0.CALLER, B0.CONNECTION, A_PATH):
        if (ROOT / path).read_bytes() != B0.blob(commit, path):
            raise ValueError('Execution SOURCE differs from HEAD')
    P = B0.connection()
    for reference in (P.NATIVE, P.WRAPPER, P.D2_SOURCE):
        if (ROOT / reference['path']).read_bytes() != B0.blob(reference['commit'], reference['path']):
            raise ValueError('Pinned native/wrapper/D2 SOURCE differs')
    if plan.get('reuse_b0'):
        saved = json.loads(B0.blob(B0_SOURCE['commit'], B0_SOURCE['path']))['result']
        if (physical['candidate_id'] != B0.preflight()['canonical_physical_candidate']['id']
                or saved['seeds'] != job['step_seeds'][0]):
            raise ValueError('B0 physical/cohort/seed provenance differs')
    imports = json.loads(subprocess.check_output([sys.executable, '-B', str(ROOT / B0.CALLER), '--import-preflight'], cwd=ROOT))
    if imports.get('status') != 'RESTORE_IMPORT_CHAIN_READY_NO_SCIENCE':
        raise ValueError('B0 import-only preflight failed')
    out.mkdir(parents=True); private.mkdir(parents=True)
    task = dict(job, output=str(out), private=str(private), request=physical,
                dictionary_identity=packet['dictionary_identity'], result=str(private / 'child.json'))
    (private / 'job.json').write_text(json.dumps(task, default=str))
    code = subprocess.call([sys.executable, '-B', str(ROOT / CALLER), '--child', str(private / 'job.json')], cwd=ROOT)
    child = json.loads((private / 'child.json').read_text()) if (private / 'child.json').exists() else None
    ok = code == 0 and child is not None and child['status'] == 'PASS' and len(child['paths']) == cell['n_paths']
    error = None
    source = dict(commit=commit, result_path=str(out.relative_to(ROOT)) + '/RESULT.json', caller=CALLER,
                  prefix=plan['prefix'], kind=plan['kind'], cell=cell['cell_id'], cohort=cell['cohort'], step_seeds=job['step_seeds'])
    if ok:
        try:
            r = records(plan, cell, child['paths'], source=source,
                        evidence_path=str(out.relative_to(ROOT)) + '/IMPLEMENTATION_EVIDENCE.json')
            for name, value in (('C_INPUTS.json', r['C_inputs']), ('PRICES.json', r.get('prices', [])),
                               ('SIDECARS_UNBOUND.json', r.get('sidecars', [])), ('IMPLEMENTATION_EVIDENCE.json', r.get('implementation_evidence', {}))):
                write(out / name, value)
        except Exception as exc:
            ok, error = False, type(exc).__name__ + ': ' + str(exc)
    write(out / 'RESULT.json', dict(status='FINITE_CELL_COMPLETE' if ok else 'STOP_CELL', child=child, error=error,
          source=source, mandatory_stop=True, mean=None, fit=None, N=None, G=None))
    return 0 if ok else 2


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, default=str); f.write('\n')


def child(path):
    job = json.loads(Path(path).read_text()); plan = job['plan']; validate(plan, execution=True)
    import h46_long_time_signal as signal
    signal.install(); B0.provider_checks(); B0.register_pauli_modules()
    import qiskit
    if qiskit.__version__ != '1.3.0':
        raise ValueError('Qiskit 1.3.0 required')
    P = B0.connection(); dictionary = P.load_dictionary(ROOT, job['dictionary_identity'])
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(job['output']), Path(job['private'])
    ctx = native.native_context('H4', (plan['prefix'],)); results = []
    if ctx.ham.constant != B0.HAMILTONIAN_CONSTANT:
        raise ValueError('Hamiltonian constant differs from the B0 bound snapshot')
    cell = next(c for c in plan['cells'] if c['cell_id'] == job['cell_id'])
    try:
        for index, seeds in enumerate(job['step_seeds']):
            if plan.get('reuse_b0') and index == 0:
                # Saved B0 remains the original path, with its original SOURCE. No new sample/compile.
                r = json.loads(B0.blob(B0_SOURCE['commit'], B0_SOURCE['path']))['result']
                results.append(dict(index=0, cohort=cell['cohort'], request=job['request'], result=r,
                                    reused=True, original_source=deepcopy(B0_SOURCE), independent_new=False))
                continue
            r = acquire_path(native, ctx, dictionary, job['request'], cohort=cell['cohort'], index=index, seeds=seeds)
            if cell.get('composed_of'):
                r['paired_check'] = check_composition(r, job['parents'])
            results.append(dict(index=index, cohort=cell['cohort'], request=job['request'], result=r, reused=False))
        write(Path(job['result']), dict(status='PASS', paths=results)); return 0
    except Exception as exc:
        write(Path(job['result']), dict(status='STOP', error=type(exc).__name__ + ': ' + str(exc), paths=results)); return 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=int, choices=(1, 2), default=1)
    p.add_argument('--kind', choices=('B1-W', 'B1-V', 'B1-CAL'), default='B1-CAL')
    p.add_argument('--declare-b0-path0', action='store_true', help='metadata declaration only, LD1 B1-V same cohort')
    p.add_argument('--execute-approved-job', help='adopted finite-cell JSON; separate science permission')
    p.add_argument('--child', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        return child(a.child)
    if a.execute_approved_job:
        return execute(json.loads(Path(a.execute_approved_job).read_text()))
    print(json.dumps(preflight(proposal(a.prefix, a.kind, reuse_b0=a.declare_b0_path0)), indent=2, default=str)); return 0


if __name__ == '__main__':
    sys.exit(main())
