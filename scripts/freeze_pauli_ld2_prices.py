"""C2: exact five-cell/axis LD2 OLS and four pre-observation predictions.

Reuse pinned A admission/feature code and C exact OLS unchanged. Read only
published calibration receipts and fixed plan metadata; never load a compiler,
sampler, holdout observation, price validation or risk/cost evaluator.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/freeze_pauli_ld2_prices.py'
BASE = 'docs/research/d_pauli_ld2_b1_cal_results_20261115'
OUT = BASE + '/c2_freeze'
D2 = '5dac74959ed9ea49465d393fc792ef5b4727edba'
PLAN = '043c7567451e8f502a1cf0f425d32efe0a50f5dc'
DESIGN = '7a0c9468b2499fac9fa1cd279ae2980c9dab048e'
READER = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
META = 'docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json'
TEMPLATES = 'docs/research/d_pauli_b1_caller_preparation_20261114/JOB_TEMPLATES.json'
CELLS = ('S66', 'S526', 'S4205', 'Q8R66', 'P2R66')
FEATURES = ['one', 'pf_step_count', 'event_count']
READS = set()


@lru_cache(None)
def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def ref(commit, path, *selector):
    return dict(commit=commit, path=path, selector=list(selector))


def read(reference):
    READS.add((reference['commit'], reference['path']))
    value = json.loads(blob(reference['commit'], reference['path']))
    for key in reference.get('selector', []):
        value = value[key]
    return value


@lru_cache(None)
def design_api():
    module = ModuleType('_c2_fixed_finite_price_design')
    module.__file__ = DESIGN + ':src/trottertracks/finite_price_design.py'
    exec(compile(blob(DESIGN, 'src/trottertracks/finite_price_design.py'),
                 module.__file__, 'exec'), module.__dict__)
    return module


def same(a, b, label):
    if a != b:
        raise ValueError(label)


def prepare():
    """Reuse D2's saved reader checks and fixed locators; no fit/prediction."""
    delivery = read(ref(D2, BASE + '/C_HANDOFF.json'))
    publications = read(ref(D2, BASE + '/RAW_PUBLICATIONS.json'))
    plan = read(ref(D2, BASE + '/JOBS/S66.json', 'plan'))
    template = read(ref(PLAN, TEMPLATES, 'jobs', 'LD2_B1-CAL', 'plan'))
    meta = read(ref(PLAN, META, 'dictionaries', 'LD2'))
    same(plan['cells'], template['cells'], 'Fixed plan/template cells differ')
    same(plan['design_source'], template['design_source'], 'Design SOURCE differs')
    same((plan['prefix'], plan['feature_map'], plan['axes']),
         (2, 'PR_STEP_V1', ['X', 'Y']), 'LD2 feature/axis scope differs')
    same((len(delivery['records']), len(delivery['individual_compile_receipts'])),
         (10, 18), 'Expected ten cell records / eighteen receipts')
    same([c['cell_id'] for c in delivery['cells']], list(CELLS), 'Calibration cell scope differs')
    same(delivery['implementation_source'], READER, 'D2 reader SOURCE differs')
    same(delivery['plan_source'], PLAN, 'D2 plan SOURCE differs')
    D = design_api()
    summary = D.check_plan(plan)
    M = D.c_model(ROOT)  # Existing pinned C implementation; no fit call here.
    groups = {axis: [] for axis in ('X', 'Y')}
    inputs = []
    for cell in delivery['cells']:
        name = cell['cell_id']
        fixed = next(c for c in meta['cells'] if c['cell_id'] == name)
        declared = next(c for c in plan['cells'] if c['cell_id'] == name)
        for key in ('q', 'R', 'T', 'm', 'physical_id', 'n_paths', 'cohort', 'step_seeds', 'role'):
            same(cell[key], fixed[key], name + ' metadata differs: ' + key)
        same(cell['result'], publications[name], name + ' raw publication differs')
        records_ref = ref(D2, cell['records_path'])
        records = read(records_ref)
        sides = read(ref(D2, cell['sidecars_path']))
        check_ref = ref(D2, cell['reader_check_path'])
        checked = read(check_ref)
        same(checked['status'], 'CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION', 'Saved reader failed')
        same(checked['records'], records, 'Saved reader inputs differ')
        same([r['axis'] for r in records], ['X', 'Y'], 'Ordered separate X/Y records required')
        for position, (record, side, trace) in enumerate(zip(records, sides, checked['checked'], strict=True)):
            axis = record['axis']
            index = next(i for i, r in enumerate(delivery['records'])
                         if (r['cell_id'], r['axis']) == (name, axis))
            same(record, delivery['records'][index], 'Authoritative C_HANDOFF record differs')
            same(side, next(s for s in delivery['sidecars']
                            if (s['cell_id'], s['axis']) == (name, axis)), 'Sidecar differs')
            same(record['source']['publication'], side['C_input'], 'Raw record locator differs')
            same(record['source']['implementation_evidence'], trace['implementation_evidence'], 'Proof locator differs')
            same(record['features'], {k: M._enc(v) for k, v in D.features(declared, 'PR_STEP_V1').items()},
                 'Exact feature values differ')
            same(record['n_paths'], fixed['n_paths'], 'Finite cohort size differs')
            same(record['source']['cohort'], fixed['cohort'], 'Cohort differs')
            same(record['source']['step_seeds'], fixed['step_seeds'], 'Seeds differ')
            proof = read(side['implementation_evidence'])
            same(proof['dictionary_identity'], meta['dictionary_identity'], 'LD2 dictionary differs')
            request = proof['request']
            same(request['candidate_id'], fixed['physical_id'], 'Canonical physical ID differs')
            for key in ('q', 'R', 'T', 'm'):
                same(request[key], fixed[key], 'Physical request differs: ' + key)
            same(proof['measurement']['axis'], axis, 'Measurement axis differs')
            same(proof['paired_dependence'], declared.get('composed_of'), 'Paired ancestry differs')
            receipt_refs = []
            for item in trace['paths']:
                authoritative = next(t for t in delivery['individual_compile_receipts']
                    if (t['cell_id'], t['axis'], t['index']) == (name, axis, item['index']))
                same(authoritative, dict(cell_id=name, **item), 'Individual receipt mapping differs')
                receipt = read(item['receipt'])
                same((receipt['cell_id'], receipt['axis'], receipt['index'], receipt['cohort']),
                     (name, axis, item['index'], fixed['cohort']), 'Receipt identity differs')
                same(receipt['record']['fingerprint'], item['fingerprint'], 'Receipt fingerprint differs')
                same(receipt['seeds'], fixed['step_seeds'][item['index']], 'Receipt seed differs')
                same(receipt['dictionary_identity'], meta['dictionary_identity'], 'Receipt dictionary differs')
                same(receipt['execution_source'], side['execution_source'], 'Receipt execution SOURCE differs')
                receipt_refs.append(deepcopy(authoritative))
            same(len(receipt_refs), record['n_paths'], 'Receipt count differs')
            groups[axis].append(record)
            inputs.append(dict(input_id=name + ':' + axis, record=record, cell_metadata=fixed,
                authoritative_locator=ref(D2, BASE + '/C_HANDOFF.json', 'records', index),
                published_record_locator=dict(records_ref, selector=[position]),
                sidecar=side, saved_reader_check=check_ref, individual_receipts=receipt_refs,
                execution_source=deepcopy(side['execution_source']), raw_publication=publications[name],
                physical_request_locator=ref(side['implementation_evidence']['commit'],
                    side['implementation_evidence']['path'], axis, 'request'),
                dictionary_identity=meta['dictionary_identity']))
    same(sum(len(i['individual_receipts']) for i in inputs), 18, 'Not eighteen receipts')
    same(groups['X'][4]['source']['step_seeds'],
         [[groups['X'][0]['source']['step_seeds'][0][0], groups['X'][0]['source']['step_seeds'][1][0]]],
         'P2 must retain S66 path0 then path1')
    # Canonical identity, feature and fresh-cohort metadata only; no observation files.
    parents = read(ref(D2, BASE + '/JOBS/P2R66.json', 'parents'))
    same([p['selector'] for p in parents], [['child', 'paths', 0], ['child', 'paths', 1]],
         'Ordered P2 parent locators differ')
    for parent in parents:
        same(parent['commit'], publications['S66']['commit'], 'Parent raw SOURCE differs')
        same(parent['execution_commit'], publications['S66']['execution_commit'], 'Parent execution SOURCE differs')
    holds = []
    for m, q, R, T, n in ((5, 32, 33, 3.2, 2), (6, 64, 66, 6.4, 1)):
        c = next(c for c in meta['cells'] if c['m'] == m)
        p = next(c for c in plan['cells'] if c['cell_id'] == c_id(m))
        for key in ('q', 'R', 'T', 's', 'n_paths', 'cohort', 'role'):
            same(c[key], p[key], 'Holdout plan/metadata differs: ' + key)
        same((c['q'], c['R'], c['T'], c['n_paths'], c['role'], c['axes']),
             (q, R, T, n, 'HOLDOUT', ['X', 'Y']), 'Fixed holdout scope differs')
        same(len(c['step_seeds']), n, 'Holdout fresh path count differs')
        if c['composed_of'] or not c['physical_id'].startswith('RMSE:'):
            raise ValueError('Fresh canonical round holdout required')
        cal_seeds = {s for i in inputs for row in i['record']['source']['step_seeds'] for s in row}
        hold_seeds = [s for row in c['step_seeds'] for s in row]
        if set(hold_seeds) & cal_seeds or len(hold_seeds) != q*n or len(set(hold_seeds)) != q*n:
            raise ValueError('Holdout seeds overlap calibration or repeat')
        holds.append(deepcopy(c))
    return dict(plan=plan, metadata=meta, groups=groups, inputs=inputs, holds=holds, parents=parents, design_check=summary)


def c_id(m):
    return 'M%d_ROUND' % m


def calibrate(prepared, source):
    D = design_api(); M = D.c_model(ROOT)
    models = {}; predictions = []
    for axis in ('X', 'Y'):
        admitted = D.fit_identified(ROOT, prepared['groups'][axis], plan=prepared['plan'], axis=axis)
        model = admitted['model']
        model.update(axis=axis, fit_source=deepcopy(source), input_ids=[c + ':' + axis for c in CELLS],
            application_scope='LD2_FINITE_COHORT_CALIBRATION_AND_DECLARED_M5_M6_PREDICTIONS_ONLY',
            coefficient_uncertainty=None, law_expectation=False)
        models[axis] = model
        for c in prepared['holds']:
            values = D.features(c, 'PR_STEP_V1')
            y = M.predict(model, values)
            predictions.append(dict(prediction_id='LD2:m%d:%s' % (c['m'], axis), axis=axis,
                cell_id=c['cell_id'], m=c['m'], canonical_physical_id=c['physical_id'],
                q=c['q'], R=c['R'], T=c['T'], T_exact=M._enc(Fraction(str(c['T']))), features={k: M._enc(v) for k, v in values.items()},
                predicted_RZ=M._enc(y), display_RZ=str(y), n_paths=c['n_paths'],
                cohort=c['cohort'], step_seeds=c['step_seeds'], expectation='FINITE_COHORT_ONLY',
                metadata_source=ref(PLAN, META, 'dictionaries', 'LD2', 'cells',
                    next(i for i, x in enumerate(prepared['metadata']['cells']) if x['m'] == c['m'])),
                observed_RZ=None, holdout_pass=None, independent_holdout_status='NOT_EVALUATED'))
    return dict(schema='C2_PAULI_LD2_FROZEN_PRICE_MODELS_V1',
        status='LD2_FINITE_COHORT_CALIBRATED_COEFFICIENTS_AND_PREDICTIONS_FROZEN_HOLDOUT_UNEVALUATED',
        status_ja='LD2有限cohortで較正済み・係数と予測を凍結・独立holdout未評価',
        fit_source=source, authoritative_input=ref(D2, BASE + '/C_HANDOFF.json'),
        plan_source=ref(PLAN, META, 'dictionaries', 'LD2'),
        feature_map='PR_STEP_V1', features=FEATURES,
        formula='C_a(q,R)=beta0_a+beta1_a*q+beta2_a*q*R; a=X,Y',
        method='EXACT_NORMAL_EQUATION_LEAST_SQUARES', weighting='ONE_EQUAL_WEIGHT_PER_CELL_PER_AXIS',
        models=models, predictions=predictions, inputs=prepared['inputs'],
        model_count=2, prediction_count=4, input_record_count=10, individual_receipt_count=18,
        fit_calls=2, prediction_calls=4, independent_calibration_paths=8, paired_derived_paths=1,
        paired_dependence=dict(child='P2R66', ordered_parents=['S66/path0', 'S66/path1'],
            independent_of_parents=False, ordered_parent_locators=prepared['parents'], source=ref(D2, BASE + '/JOBS/P2R66.json', 'parents')),
        scope=dict(input_binding=prepared['metadata']['input_binding'],
            dictionary_identity=prepared['metadata']['dictionary_identity'],
            saved_lambda_P=prepared['metadata']['saved_lambda_P'], saved_s_R=prepared['metadata']['saved_s_R'],
            head_order=prepared['metadata']['head_order'], domain=prepared['metadata']['domain']),
        design_check=prepared['design_check'], holdout_tolerance=None,
        holdout_tolerance_status='PAULI_TOLERANCE_NOT_ADOPTED_PF_2_PERCENT_RULE_NOT_APPLIED',
        planned_later_holdout_compiles=6, holdout_observation_reads=0, holdout_evaluations=0,
        new_compiles=0, new_sampling=0, independent_holdout_status='NOT_EVALUATED',
        LD2_order2_product_coverage='UNKNOWN_UNOBSERVED',
        LD2_nontrivial_event_phase_coverage='UNKNOWN_UNOBSERVED',
        law_expectation=False, qualified_for_candidate=False, E_C=None, CI=None, N=None, G=None,
        mandatory_stop=True)


def check_d2_consumer(prepared, reference, *, frozen=None):
    """Unchanged pinned D2 gate body; seed lookup uses its saved fixed metadata.

    This tests the metadata consumer only, not D2's full request/compile runner.
    """
    code = blob(READER, 'scripts/run_pauli_b1_prices.py').decode()
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'frozen_model_gate')
    def reader(commit, path):
        if frozen is not None:
            same((commit, path), (reference['commit'], reference['path']), 'Unexpected consumer read')
            return json.dumps(frozen).encode()
        return blob(commit, path)
    seeds = {c['cell_id']: c['step_seeds'] for c in prepared['metadata']['cells']}
    scope = dict(json=json, deepcopy=deepcopy, B0=SimpleNamespace(blob=reader),
                 design_api=design_api, seed_table=lambda plan, cell: seeds[cell])
    exec(compile(ast.Module(body=[node], type_ignores=[]), READER + ':frozen_model_gate', 'exec'), scope)
    return scope['frozen_model_gate'](prepared['plan'], reference)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    prepared = prepare()
    if not args.execute:
        print(json.dumps(dict(metadata_records=10, receipts=18, rank=prepared['design_check']['identifiability']['rank'],
                              fit_calls=0, prediction_calls=0)))
        return
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    same(blob(commit, SCRIPT), (ROOT / SCRIPT).read_bytes(), 'Commit execution SOURCE before fitting')
    source = dict(commit=commit, path=SCRIPT,
        admission=ref(DESIGN, 'src/trottertracks/finite_price_design.py'),
        exact_OLS=deepcopy(design_api().C_MODEL), C2_base='12372fa6b53c878dc5dd124e036572ba0cd82061')
    target = ROOT / OUT / 'run01'
    target.mkdir(parents=True, exist_ok=False)
    result = calibrate(prepared, source)
    # In-memory metadata interface check; committed-blob check follows publication.
    interface = check_d2_consumer(prepared, ref(commit, OUT + '/run01/FROZEN_MODELS.json'), frozen=result)
    (target / 'FROZEN_MODELS.json').write_text(json.dumps(result, indent=2) + '\n')
    checks = dict(execution_source=source, finished_UTC=datetime.now(timezone.utc).isoformat(),
        fit_calls=2, prediction_calls=4, input_records=10, receipts=18,
        new_compiles=0, new_sampling=0, holdout_observation_reads=0, holdout_evaluations=0,
        interface_check=interface, interface_check_mode='EXACT_D2_GATE_BODY_WITH_SAVED_PLAN_SEED_LOOKUP',
        input_reads=[ref(c, p) for c, p in sorted(READS)], mandatory_stop=True)
    (target / 'CHECKS.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({k: checks[k] for k in ('fit_calls', 'prediction_calls', 'input_records', 'receipts', 'holdout_observation_reads')}))


if __name__ == '__main__':
    main()
