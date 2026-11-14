"""Full-H P-2 finite cell adapter. Default is metadata; all A designs remain proposals.

Reuse P-1's source checks, sampler, trial, ZZ swap, parity and corrected phase
gates. No price fit/prediction/holdout judgement here. A complete adopted plan,
explicit stage/cell/cohort/seed table and fresh output are needed for execution.
Chat permission remains external; no approval certificate or grant is required.
"""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import ast
import hashlib
import importlib
import json
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_full_h_kappa2_p1 as P1

CALLER = 'scripts/run_full_h_kappa2_p2.py'
A_SOURCE = '7a0c9468b2499fac9fa1cd279ae2980c9dab048e'
A_MODULE = dict(commit=A_SOURCE, path='src/trottertracks/finite_price_design.py')
A_DESIGNS = 'docs/research/a_full_h_p2_pauli_b1_price_design_20261114/DESIGNS.json'
DESIGNS = dict(D6='D6_INDEPENDENT_Q8', D7='D7_INDEPENDENT_Q8_PLUS_PAIRED_Q2',
               D7_H2='D7_INDEPENDENT_Q8_PLUS_PAIRED_Q2', D0='D0_EXISTING_Q1_ONLY')
OLD_PRICE = dict(commit='a04470c0ad489eea1798856e648f863f9092cfb1',
                 path='docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/PRICES.json')


@lru_cache(maxsize=1)
def design_api():
    if (ROOT / A_MODULE['path']).read_bytes() != P1.blob(A_SOURCE, A_MODULE['path']):
        raise ValueError('A design checker SOURCE differs')
    return P1.load_blob_module(A_MODULE, '_p2_A_design')


def proposal(name='D7'):
    table = json.loads(P1.blob(A_SOURCE, A_DESIGNS))['full_h']
    d = next(d for d in table['designs'] if d['design'] == DESIGNS[name])
    # D0's domain is shared with the identified design; its deficient cells are kept for rejection tests.
    domain = next(x['plan']['domain'] for x in table['designs'] if x['design'] == DESIGNS['D7'])
    cells = [dict(deepcopy(domain), **deepcopy(c), expectation='FINITE_COHORT_ONLY',
                  T=.1 * c['q'], s=c['q'] * c['R'],
                  cohort='D_FULL_H_P2_%s_%s_PROPOSED_V1' % (name, c['cell_id'])) for c in d['cells']]
    if name == 'D7_H2':
        next(c for c in cells if c['cell_id'] == 'M6_ROUND')['n_paths'] = 2
    return dict(name=name, feature_map='FULL_H_GROUP_V1', cells=cells, axes=['X', 'Y'], adopted=False,
                design_source=dict(commit=A_SOURCE, path=A_DESIGNS, selector=DESIGNS[name], status='PROPOSAL_NOT_ADOPTED'))


def validate(plan, *, execution=False):
    D = design_api()
    summary = D.check_plan(plan)  # FIRST: rejects the old rank-2 design before inputs/context/output.
    expected = proposal(plan['name'])
    if plan['design_source'] != expected['design_source']:
        raise ValueError('A design SOURCE differs')
    if plan['feature_map'] != expected['feature_map'] or plan.get('axes') != ['X', 'Y']:
        raise ValueError('Full-H feature map and both axes required')
    if len(plan['cells']) != len(expected['cells']):
        raise ValueError('Incomplete finite design')
    for c, fixed in zip(plan['cells'], expected['cells'], strict=True):
        fields = ('cell_id', 'role', 'q', 'R', 'n_paths', 'T', 's', 'composed_of') + D.DOMAIN_KEYS
        if any(c.get(k) != fixed.get(k) for k in fields) or c['s'] != c['q'] * c['R']:
            raise ValueError('q/R/s/T, domain, role or finite cell differs from A: ' + c['cell_id'])
        if not c.get('cohort') or c['cohort'] == P1.COHORT:
            raise ValueError('Fresh declared P-2 cohort required; P-1 wiring excluded')
    if execution and plan.get('adopted') is not True:
        raise ValueError('Proposed plan is not adopted for scientific execution')
    return summary


def physical_row(cell):
    """The original full_baseline_v2_requests physical cid rule; no fake configuration round."""
    row = dict(system='H4', method='PR', formula='2nd', prefix=0, T=cell['T'], q=cell['q'], R_outer=cell['R'],
               K=2, delta=cell['T'] / cell['q'], s=cell['s'], family='FULL_H_PURE_RTE', L_D=0)
    tree = ast.parse((ROOT / 'scripts/h46_long_time_signal.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cid')
    scope = dict(json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'h46_long_time_signal:cid', 'exec'), scope)
    row['id'] = 'FULL_H_' + scope['cid'](row)
    return row


def seed_table(plan, cell_id):
    validate(plan)
    cells = {c['cell_id']: c for c in plan['cells']}; cell = cells[cell_id]
    if cell.get('composed_of'):
        result = []
        for token in cell['composed_of']:
            parent, path, group = token.split('#')
            result.append(seed_table(plan, parent)[int(path[1:])][int(group[1:])])
        return [result]
    f = P1.seed_function(); row = physical_row(cell)
    return [[f(row, cell['cohort'], index, outer, 0) for outer in range(cell['q'])]
            for index in range(cell['n_paths'])]


def event_projection(batches):
    return [[e.to_dict() if hasattr(e, 'to_dict') else deepcopy(e) for e in batch] for batch in batches]


def check_composition(actual, parents):
    """Ordered request boundaries are retained; equality includes event phase/application order."""
    expected = [batch for p in parents for batch in p['events']]
    # RTEEvent.to_dict retains tuples; published JSON receipts restore lists.
    # Compare the same ordered JSON representation, without rounding any value.
    if json.dumps(actual['events'], sort_keys=True) != json.dumps(expected, sort_keys=True) or actual['seeds'] != [s for p in parents for s in p['seeds']]:
        raise ValueError('Paired seed/event order or phase differs from parent group paths')
    if all('trial' in p for p in parents) and any(p['trial']['scalar'] != actual['trial']['scalar'] for p in parents):
        raise ValueError('Paired parent scalar quantity differs')
    return dict(status='PAIRED_EVENTS_IDENTICAL', requests=len(expected), scalar='ONCE_AT_COMPOSED_TOTAL_T')


def parent_receipts(plan, cell, references, execution_commit):
    """Read the saved successful G566 result; never regenerate a parent trajectory."""
    parents = []
    if len(references) != cell['q']:
        raise ValueError('Paired parent receipts required')
    for token, ref, seed in zip(cell['composed_of'], references, seed_table(plan, cell['cell_id'])[0], strict=True):
        cid, index, group = token.split('#'); parent_cell = selected(plan, 'CALIBRATION', cid)
        path = (ROOT / ref['path']).resolve()
        if not path.is_relative_to(ROOT) or ref['selector'] != ['child', 'paths', int(index[1:])]:
            raise ValueError('Paired parent receipt locator differs')
        raw = path.read_bytes(); saved = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != ref['sha256'] or group != 'g0'
                or saved['status'] != 'FINITE_CELL_COMPLETE' or saved['child']['status'] != 'PASS'
                or saved['source']['result_path'] != ref['path']
                or saved['source']['commit'] != execution_commit or saved['source']['caller'] != CALLER
                or saved['source']['cell'] != cid or saved['source']['cohort'] != parent_cell['cohort']
                or saved['source']['step_seeds'] != seed_table(plan, cid)):
            raise ValueError('Paired parent SOURCE/result binding differs')
        parent = saved['child']['paths'][int(index[1:])]
        if (parent['row'] != physical_row(parent_cell) or parent['cohort'] != parent_cell['cohort']
                or parent['index'] != int(index[1:]) or parent['seeds'] != [seed]
                or parent['trial']['status'] != 'PASS' or parent['xy_phase_gate']['status'] != 'PASS'):
            raise ValueError('Paired parent physical/cohort/index/seed/check binding differs')
        parents.append(dict(parent, receipt_source=dict(ref, execution_commit=execution_commit)))
    return parents


def restore_parent_requests(row, prep, seeds, parents, scalar):
    """Reuse the existing saved-event decoder and request dataclasses, with no sampler."""
    from h46_same_event import decode_event
    from h46_m5_contract import core
    from h46_pr_pilot_repair import canonical_tau
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    steps = core.schedule(row['formula'], row['q'], 'ABS_TIME_PER_OUTER', row['R_outer'], T=row['T'])
    if len(parents) != len(steps) or seeds != [p['seeds'][0] for p in parents]:
        raise ValueError('Paired parent order/seed allocation differs')
    requests, laws = [], []
    for step, seed, parent in zip(steps, seeds, parents, strict=True):
        if (parent['trial']['scalar'] != scalar or parent['trial']['preparation_hash'] != prep.preparation_hash
                or len(parent['events']) != 1 or len(parent['events'][0]) != step['rte_steps']
                or parent['row']['T'] != step['signed_time']):
            raise ValueError('Paired parent scalar/preparation/time/event allocation differs')
        events = tuple(decode_event(rte, e) for e in parent['events'][0])
        tau = canonical_tau(prep, step['signed_time'], step['rte_steps'])
        config, distribution = rte.make_rte_config(prep.rte_preparation.symbolic_tail,
            evolution_time=step['signed_time'], rte_steps=step['rte_steps'], finite_taylor_order=row['K'],
            truncation_tolerance=_explicit_cutoff_tolerance(tau, row['K']), seed=seed)
        law = dict(steps=step['rte_steps'], orders=list(distribution.orders), probabilities=list(distribution.order_probabilities))
        if parent['trial']['laws'] != [law]:
            raise ValueError('Paired parent finite law differs')
        for event in events:
            prep.rte_preparation.resolve_event_basis_definitions(event)
        occurrence = DFRTEEventSequenceCircuitRequest(events=events, component_specs=prep.rte_preparation.component_specs,
            controlled=True, ancilla_qubit=prep.num_system_qubits, tail_id=config.tail_id,
            tail_hash=config.tail_hash, occurrence_rte_steps=step['rte_steps'])
        requests.append(DFPartialS2StepRequest(preparation=prep, step_time=step['signed_time'], rte_config=config,
            rte_distribution=distribution, rte_occurrence=occurrence, controlled=True,
            ancilla_qubit=prep.num_system_qubits, seed=seed))
        laws.append(law)
    check_composition(dict(events=event_projection([r.rte_occurrence.events for r in requests]),
                           seeds=seeds, trial=dict(scalar=scalar)), parents)
    return requests, list(seeds), laws


def acquire_path(full, row, cohort, index, seeds, *, compile_cached=None, parents=None):
    """Scientific API: existing normal sampler and P-1 trial; explicit seeds only replace trial_seed.

    Called only after validate(execution=True). Tests pass a toy context/compiler.
    Paired requests are restored and checked before compose; normal sampling is unchanged.
    """
    native = full.setup(); ctx = full.context(row['system'])
    facade = SimpleNamespace(setup=lambda: native, context=lambda system: ctx)
    captured, wrappers, fps, compile_receipts = {}, {}, {}, {}
    original = native.compose; compiler = compile_cached or native.compile_cached
    def fixed_seed(r, co, i, outer, piece):
        if r != row or co != cohort or i != index or piece != 0 or not 0 <= outer < len(seeds):
            raise ValueError('Unexpected group seed request')
        return seeds[outer]
    def compose(r, prep, requests, scalar, fingerprint):
        result = original(r, prep, requests, scalar, fingerprint)
        captured.setdefault('events', event_projection([r.rte_occurrence.events for r in requests]))
        captured.update(scalar=scalar, n=prep.num_system_qubits)
        return result
    def compile_one(wrapper, fingerprint, representative):
        axis = 'sine' if any(i.operation.name == 'sdg' and wrapper.find_bit(i.qubits[0]).index == ctx.ham.n_qubits
                            for i in wrapper.data[-4:]) else 'cosine'
        wrappers[axis] = wrapper; fps[axis] = fingerprint
        started = time.monotonic()
        result = compiler(wrapper, fingerprint, representative)
        record, _, reused = result
        path = native.PRIVATE / 'compiled_cache' / (fingerprint + '.json')
        raw = path.read_bytes() if path.exists() else None
        if raw is not None and json.loads(raw) != record:
            raise ValueError('Saved individual compile record differs from returned record')
        compile_receipts[axis] = dict(record=deepcopy(record), fingerprint=fingerprint, reused=bool(reused),
            locator=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None),
            wall_seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(native, 'trial_seed', fixed_seed))
        stack.enter_context(patch.object(native, 'compose', compose))
        stack.enter_context(patch.object(native, 'compile_cached', native.compile_cached))
        if parents is not None:
            restored = restore_parent_requests(row, ctx.preparations[row['prefix']], seeds, parents,
                ctx.ham.constant + ctx.preparations[row['prefix']].extracted_identity_coefficient)
            def replay(r, prep, co, i):
                if r != row or prep is not ctx.preparations[row['prefix']] or co != cohort or i != index:
                    raise ValueError('Unexpected paired request replay')
                return restored
            stack.enter_context(patch.object(native, 'sample_requests', replay))
        trial, facts, ok = P1.pilot(facade, row, compile_cached=compile_one, cohort=cohort, index=index)
    provider = importlib.import_module('trotterlib.df_rte_qiskit')
    xy = P1.xy_phase_gate(wrappers, captured['n'], captured['events'], captured['scalar'], captured['scalar'], row['T'], provider=provider)
    if not ok or xy['status'] != 'PASS' or len(captured['events']) != row['q'] or any(len(b) != row['R_outer'] for b in captured['events']):
        raise ValueError('P-1 correctness/phase/XY/allocation gate failed')
    return dict(trial=trial, checks=facts, xy_phase_gate=xy, events=captured['events'], seeds=list(seeds),
                fingerprints=fps, compile_receipts=compile_receipts, index=index, cohort=cohort, row=row,
                sampling='SAVED_PARENT_EVENTS_NO_DRAW' if parents is not None else 'ORIGINAL_NORMAL_SAMPLER',
                parents=[p['receipt_source'] for p in parents] if parents is not None else [])


def selected(plan, stage, cell_id):
    validate(plan)
    cell = next(c for c in plan['cells'] if c['cell_id'] == cell_id)
    if cell['role'] != stage:
        raise ValueError('CALIBRATION/HOLDOUT stage mismatch')
    return cell


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
    walk(json.loads(P1.blob(reference['commit'], reference['path'])))
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
    summary = validate(plan)
    return dict(status='TECHNICAL_PLAN_ONLY_NOT_EXECUTION', plan=summary, adopted=plan.get('adopted', False),
                A_resource_order_proposal=[['G566', 'P2R566 (D7 only)', 'Q8R566', 'M5_ROUND'], ['G4528', 'G36223', 'M6_ROUND']],
                ordering_condition='M5 remains HOLDOUT and cannot run before full CALIBRATION/C freeze; resolve the proposed resource order before adoption',
                cells=[dict(cell_id=c['cell_id'], role=c['role'], row=physical_row(c), cohort=c['cohort'],
                            step_seeds=seed_table(plan, c['cell_id']), composed_of=c.get('composed_of')) for c in plan['cells']],
                duplicate_preflight=dict(source=OLD_PRICE, P1='WIRING_ONLY_DIFFERENT_R',
                    legacy_Full_H='OLD_LOWERING_NOT_REUSABLE', A_scope=dict(commit=A_SOURCE, path=A_DESIGNS),
                    recheck_before_science=['table/catalog', 'original trial/receipt', 'relevant running jobs', 'SOURCE/cohort/seed']),
                mean=None, fit=None, N=None, G=None, mandatory_stop=True)


def records(plan, cell, results, *, source, prepared=None, evidence_path='IMPLEMENTATION_EVIDENCE.json'):
    """Supplied path prices -> C records; no fit. Round sidecars use P-1's unchanged C format."""
    validate(plan); D = design_api(); M = D.c_model(ROOT)
    if len(results) != cell['n_paths'] or [r['index'] for r in results] != list(range(cell['n_paths'])):
        raise ValueError('Incomplete/duplicate path set')
    if len({r['trial']['scalar'] for r in results}) != 1:
        raise ValueError('Mixed scalar quantities across paths')
    for r, seeds in zip(results, seed_table(plan, cell['cell_id']), strict=True):
        if r['cohort'] != cell['cohort'] or r['seeds'] != seeds or r['row'] != physical_row(cell):
            raise ValueError('Path physical/cohort/seed binding differs')
    means = {a: {k: sum(Fraction(r['trial']['prices'][a][k]) for r in results) / len(results)
                 for k in results[0]['trial']['prices'][a]} for a in ('cosine', 'sine')}
    inputs = [M.record(cell_id=cell['cell_id'], role=cell['role'], route=cell['route'], axis=axis,
        features=D.features(cell, plan['feature_map']), rz_count=means[a]['rz_count'], kind=cell['kind'], source=source,
        implementation=cell['implementation'], profile=cell['profile'], n_paths=len(results), law=cell['law'])
        for axis, a in (('X', 'cosine'), ('Y', 'sine'))]
    out = dict(C_inputs=inputs, path_receipts=results, law_expectation=False, model_qualified=False)
    if cell['role'] == 'CALIBRATION':
        prices, sides, proofs = [], [], {}
        directory = str(Path(source['result_path']).parent)
        for record in inputs:
            axis = record['axis']; a = 'cosine' if axis == 'X' else 'sine'
            paths = []
            for r in results:
                receipt = r['compile_receipts'][a]; compiled = receipt['record']
                if (r['trial']['status'] != 'PASS' or r['xy_phase_gate']['status'] != 'PASS'
                        or r['checks']['scalar_event_phases']['status'] != 'PASS'
                        or r['checks']['builder_lowering_id'] != P1.LOWERING_ID
                        or r['checks']['parity']['status'] != 'PASS'
                        or compiled['compiler'] != cell['profile']['compiler']
                        or compiled['fingerprint'] != r['fingerprints'][a] or receipt['fingerprint'] != compiled['fingerprint']
                        or compiled['metrics'] != r['trial']['prices'][a] or not receipt['locator']['sha256']):
                    raise ValueError('Calibration path checks/compile record/price binding differs')
                price = dict(record='DIRECT_COMPILED_PATH_PRICE_V1', cell_id=cell['cell_id'], axis=axis,
                    index=r['index'], cohort=r['cohort'], seeds=r['seeds'], physical=r['row'],
                    classification='DIRECT_COMPILED_FINITE_SAMPLE', expectation='FINITE_COHORT_ONLY',
                    metrics=compiled['metrics'], fingerprint=compiled['fingerprint'], source=source,
                    compile_receipt=receipt, path_receipt=dict(path=source['result_path'], selector=['child','paths',r['index']]))
                prices.append(price)
                paths.append(dict(price=price, checks=r['checks'], xy_phase_gate=r['xy_phase_gate'],
                    scalar=r['trial']['scalar'], preparation_hash=r['trial']['preparation_hash'],
                    sampling=r['sampling'], parents=r.get('parents', []),
                    trial_checks=r['trial']['checks'], resources={k:r['trial'][k] for k in ('wall_seconds','peak_RSS_bytes')}))
            proof = dict(calibration_record=record, paths=paths, implementation=cell['implementation'], profile=cell['profile'],
                native_source=P1.NATIVE, wrapper_source=P1.WRAPPER, lowering_source=P1.NEW_ZZ, preparation_source=P1.FULL_H,
                measurement=dict(axis=axis, include_measurement=True, state_preparation_cost_included=False),
                correctness_scope='ACTION_AND_COMPILED_VECTOR_ON_INDEX0; PHASE_PARITY_XY_ALL_PATHS',
                paired_dependence=cell.get('composed_of'), law_expectation=False)
            proofs[axis] = proof
            sides.append(dict(record='PRICE_CALIBRATION_IMPLEMENTATION_SIDECAR_V1', cell_id=cell['cell_id'], axis=axis,
                price_source=source, implementation_evidence=dict(commit=None,path=evidence_path,selector=[axis]),
                price_records=dict(commit=None,path=directory+'/PRICES.json',selector=[]),
                C_input=dict(commit=None,path=directory+'/C_INPUTS.json',selector=[len(sides)]),
                path_receipts=dict(commit=None,path=source['result_path'],selector=['child','paths']),
                C_admission='NOT_PERFORMED_NON_CONFIGURATION_RECORD'))
        out.update(prices=prices, sidecars=sides, implementation_evidence=proofs)
    if cell['cell_id'] in ('M5_ROUND', 'M6_ROUND'):
        prepared = prepared or P1.plan_row()[0]; m = int(cell['cell_id'][1])
        req = prepared['plan']['pending']['requests'][m]
        if req['physical_candidate'] != physical_row(cell):
            raise ValueError('Holdout round differs from C canonical physical row')
        numeric = {a: {k: float(v) for k, v in metrics.items()} for a, metrics in means.items()}
        prices, sides, proofs = P1.price_bindings(ROOT, prepared, dict(scalar=results[0]['trial']['scalar'], prices=numeric),
            price_source=source, evidence_commit='FILL_WITH_PUBLISHED_COMMIT', evidence_path=evidence_path,
            m=m, cohort=cell['cohort'])
        for p in prices:
            p.update(n_paths=len(results)); p['diagnostic'].update(pilot_role='P2_FINITE_CELL', model_calibration_point=False)
        out.update(prices=prices, sidecars=sides, implementation_evidence=proofs)
    return out


def bind_calibration_publication(plan, cell_id, commit, directory):
    """D1's SOURCE/record check using C's pinned metadata reader; not C scientific admission."""
    import revalidate_full_h_kappa2_p1_run02 as R
    cell = selected(plan, 'CALIBRATION', cell_id)
    read = lambda filename: json.loads(P1.blob(commit, directory+'/'+filename))
    result = read('RESULT.json')
    if result['status'] != 'FINITE_CELL_COMPLETE' or result['child']['status'] != 'PASS':
        raise ValueError('Complete published calibration receipt required')
    produced = records(plan, cell, result['child']['paths'], source=result['source'],
                       evidence_path=directory+'/IMPLEMENTATION_EVIDENCE.json')
    for name,key in (('C_INPUTS.json','C_inputs'),('PRICES.json','prices'),
                     ('SIDECARS_UNBOUND.json','sidecars'),('IMPLEMENTATION_EVIDENCE.json','implementation_evidence')):
        if read(name) != produced[key]:
            raise ValueError('Published calibration record/receipt/evidence differs: '+name)
    sides = deepcopy(produced['sidecars']); reader = R.c_guard()
    for side in sides:
        for key in ('implementation_evidence','price_records','C_input','path_receipts'):
            side[key]['commit'] = commit
        if reader.evidence(ROOT, side['implementation_evidence']) != produced['implementation_evidence'][side['axis']]:
            raise ValueError('C metadata reader returned different implementation evidence')
    return dict(status='D1_PUBLICATION_LINKS_CHECKED_USING_C_EVIDENCE_READER_NOT_C_ADMISSION',
                publication=dict(commit=commit,directory=directory), bound_sidecars=sides,
                checked_path_axes=len(produced['prices']), C_scientific_admission='NOT_PERFORMED',
                canonical_round_guard_invoked=False, law_expectation=False)


def check_round_records(prepared, cell, prices, sides, *, evidence_commit):
    """C's fixed guard on round records; non-configuration C inputs have no fabricated B identity."""
    if cell['cell_id'] not in ('M5_ROUND', 'M6_ROUND') or len(prices) != 2 or {p['axis'] for p in prices} != {'X', 'Y'}:
        raise ValueError('Canonical round and both X/Y prices required; non-configuration cells use C inputs')
    import revalidate_full_h_kappa2_p1_run02 as R
    config = prepared['plan']['configuration']; m = int(cell['cell_id'][1])
    sides = deepcopy(sides)
    for s in sides:
        s['implementation_evidence']['commit'] = evidence_commit
    cohorts = {(config['rounds'][m]['candidate']['id'], a): cell['cohort'] for a in ('X', 'Y')}
    policy = R.policy(config); policy['id'] = 'P2_TECHNICAL_COMPATIBILITY_NOT_ADOPTED'
    checked = R.c_guard().check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
    return dict(checked, bound_sidecars=sides)


def execute(job):
    """One explicit finite cell, fresh output only. No automatic stage progression or retry."""
    plan = job['plan']; validate(plan, execution=True)
    cell = selected(plan, job['stage'], job['cell_id'])
    if job.get('step_seeds') != seed_table(plan, cell['cell_id']):
        raise ValueError('Explicit seed table missing/different')
    if not job.get('duplicate_preflight_complete'):
        raise ValueError('Current table/trial/receipt/running-job duplicate preflight required')
    if cell['role'] == 'HOLDOUT' and not job.get('frozen_model_source'):
        raise ValueError('HOLDOUT requires C frozen model SOURCE and separate stage decision')
    if cell['role'] == 'HOLDOUT':
        frozen_model_gate(plan, job['frozen_model_source'])
    out, private = (ROOT / job['output']).resolve(), (ROOT / job['private']).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Public output must be in this isolated worktree')
    if out.exists() or private.exists():
        raise FileExistsError('Fresh cell namespace required')
    if cell.get('composed_of') and len(job.get('parents', [])) != cell['q']:
        raise ValueError('Paired parent seed/event receipts required before context')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parents = parent_receipts(plan,cell,job['parents'],commit) if cell.get('composed_of') else None
    for path in (CALLER, P1.CALLER, A_MODULE['path']):
        if (ROOT / path).read_bytes() != P1.blob(commit, path):
            raise ValueError('Execution SOURCE differs from HEAD')
    P1.source_checks()
    P1.child_import_preflight(ROOT)
    # Only the child takes the scientific path, install() before project imports.
    out.mkdir(parents=True); private.mkdir(parents=True)
    P1.write(private / 'job.json', dict(job, parents=parents, output=str(out), private=str(private), result=str(private / 'child.json')))
    code = subprocess.call([sys.executable, '-B', str(ROOT / CALLER), '--child', str(private / 'job.json')], cwd=ROOT)
    child = json.loads((private / 'child.json').read_text()) if (private / 'child.json').exists() else None
    ok = code == 0 and child is not None and child['status'] == 'PASS' and len(child['paths']) == cell['n_paths']
    error = None
    source = dict(commit=commit, result_path=str(out.relative_to(ROOT)) + '/RESULT.json', caller=CALLER,
                  plan=plan['name'], cell=cell['cell_id'], cohort=cell['cohort'], step_seeds=job['step_seeds'])
    if ok:
        try:
            produced = records(plan, cell, child['paths'], source=source,
                               evidence_path=str(out.relative_to(ROOT)) + '/IMPLEMENTATION_EVIDENCE.json')
            for name, value in (('C_INPUTS.json', produced['C_inputs']), ('PRICES.json', produced.get('prices', [])),
                                ('SIDECARS_UNBOUND.json', produced.get('sidecars', [])),
                                ('IMPLEMENTATION_EVIDENCE.json', produced.get('implementation_evidence', {}))):
                P1.write(out / name, value)
        except Exception as exc:
            ok, error = False, type(exc).__name__ + ': ' + str(exc)
    P1.write(out / 'RESULT.json', dict(status='FINITE_CELL_COMPLETE' if ok else 'STOP_CELL', child=child, error=error,
              source=source,
              mandatory_stop=True, mean=None, fit=None, N=None, G=None))
    return 0 if ok else 2


def child(path):
    job = json.loads(Path(path).read_text()); plan = job['plan']; validate(plan, execution=True)
    results = []
    try:
        import h46_long_time_signal as signal
        signal.install(); P1.provider_checks()
        import qiskit
        if qiskit.__version__ != '1.3.0':
            raise ValueError('Qiskit 1.3.0 required')
        full = P1.full_h_module(job['output'], job['private']); cell = selected(plan, job['stage'], job['cell_id'])
        for index, seeds in enumerate(job['step_seeds']):
            r = acquire_path(full, physical_row(cell), cell['cohort'], index, seeds, parents=job.get('parents'))
            if cell.get('composed_of'):
                r['paired_check'] = check_composition(r, job['parents'])
            results.append(r)
        P1.write(Path(job['result']), dict(status='PASS', paths=results))
        return 0
    except Exception as exc:
        P1.write(Path(job['result']), dict(status='STOP', error=type(exc).__name__ + ': ' + str(exc), paths=results))
        return 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', choices=sorted(DESIGNS), default='D7')
    p.add_argument('--execute-approved-job', help='JSON of an adopted finite cell; separate science permission')
    p.add_argument('--child', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.execute_approved_job:
        return execute(json.loads(Path(args.execute_approved_job).read_text()))
    print(json.dumps(preflight(proposal(args.design)), indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
