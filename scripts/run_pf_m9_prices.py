"""Only the twelve C-frozen m9 axes; reuse the deterministic PF callers unchanged.

Metadata selection/publication and exact supplied-prediction judgement only here.
Native context/compose/compiler/axis gates remain the Stage 2 implementation.
No m6-reader changes, prediction evaluation, fit, extrapolation, N or G.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pf_price_stages as S
import pf_m6_holdout_evidence as M6
import run_full_pf_shortlist_prices as F
import run_truncated_p7_prices as T

MODULE = 'scripts/run_pf_m9_prices.py'
STUDY = 'docs/research/d_pf_m9_holdout_20261115'
PRED = dict(commit='b1a7cd358daeab51ce538c747d0cdb8ad86c97d4', path='docs/research/pf_m9_prediction_freeze_20261115/run01/PREDICTIONS.json')
PRED_SHA = 'e5e00558669fd9568d40fcc2deba5e157a2c21c5505c5bb3268ea1a185701fc5'
POLICY = dict(commit='375f026a30e3c1ed3179a73cbbe333578abff00f', path='docs/research/pf_m9_prediction_freeze_20261115/M9_POLICY.json')
HANDOFF = '04a68f5932f53377595b2d69bbf8f46e02e82a01'
ROUTES = dict(FULL_PF=F, TRUNCATED_PF=T)
TARGETS = {'FULL_PF': {'2nd': ('a4ec53ff4e60d60a2930', 224), '4th': ('b2c479e7805c4d1f0b30', 96), '4th(m5_best)': ('a53547a5fb707b9550d5', 32)},
           'TRUNCATED_PF': {'2nd': ('1bd0a5dd4bbf876da1d6', 224), '4th': ('9924c226ee246283ec54', 96), '4th(m5_best)': ('60111109e0024d1a1064', 32)}}
RAW_FILES = ('RESULT.json', 'PRICES.json', 'SIDECARS_UNBOUND.json', 'IMPLEMENTATION_EVIDENCE.json')


def check(ok, text):
    if not ok:
        raise ValueError(text)


def read(ref):
    return json.loads(S.read_pinned(ROOT, ref))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, default=str); f.write('\n')


def paths(route):
    module = ROUTES[route]
    return module.OUTPUT_ROOT + '/m9_holdout/run01', module.PRIVATE_ROOT / 'm9_holdout/run01'


def validate(doc, templates, alignment, policy, models):
    """No predicted RZ arithmetic. Compare exact descriptors, binary64 widths and original coefficients."""
    check(doc['policy'] == policy and policy['rounds'] == [9] and policy['value'] == dict(numerator=1, denominator=50), 'M9 policy differs')
    check(len(doc['rows']) == 12 and doc['accepted_predictions'] == 12 and doc['held_predictions'] == 0 and doc['observation_reads'] == 0, 'Exactly twelve unopened predictions required')
    seen, selected = set(), {r: [] for r in ROUTES}
    for index, row in enumerate(doc['rows']):
        route, formula, width, axis = row['model_group']
        check(route in ROUTES and formula in TARGETS[route] and axis in ('X', 'Y') and row['axis'] == axis, 'Foreign model group/axis')
        p = next(p for p in templates[route] if p['formula'] == formula)
        cfg = p['plan']['configuration']; req = next(r for r in p['plan']['pending']['requests'] if r['m'] == 9)
        c = req['physical_candidate']; fixed = models[row['model_group_id']]
        check(row['model_group_id'] == S.key(row['model_group']) and row['coefficients'] == fixed['coefficients'] and row['domain'] == fixed['domain'], 'Original frozen model metadata differs')
        check((c['id'], c['q']) == TARGETS[route][formula] and c['T'] == 51.2 and c['prefix'] == (10 if route == 'FULL_PF' else 7), 'Fixed physical scope differs')
        check(row['m'] == 9 and row['physical_candidate'] == c and row['physical_id'] == c['id'] and row['q'] == c['q'] and row['features'] == dict(one=1, q=c['q']), 'Physical point/q/round differs')
        check(row['T'].hex() == c['T'].hex() and row['actual_delta'].hex() == c['delta'].hex() == row['actual_delta_binary64_hex'], 'Binary64 T/width differs')
        check(row['configuration_id'] == p['configuration_id'] and width == p['width_over_h'] and row['canonical_id'] == cfg['rounds'][9]['candidate']['id'], 'Configuration/canonical ID differs')
        source = row['alignment_source']; a = alignment
        for key in source['selector']: a = a[key]
        check(a['configuration_id'] == row['configuration_id'] and a['physical_candidate'] == c and a['axis'] == axis and a['m'] == 9, 'Alignment locator differs')
        check(row['signed_structure'] == a['structure'] and row['signed_structure']['canonical'] == cfg['rounds'][9]['structure'] and row['signed_structure']['physical'] == req['structure'], 'Signed structure/order differs')
        check(row['input_binding'] == fixed['input_locators'][0]['input_binding'] and row['input_binding'] == {k: cfg['spec']['binding'][k] for k in row['input_binding']} and row['split'] == cfg['spec']['split'], 'Input binding/split differs')
        check(row['scalar'] == fixed['input_locators'][0]['scalar'] and row['fragment_order'] == 'saved_snapshot_order' and row['scalar']['quantity'] == ROUTES[route].SCALAR_QUANTITY and row['scalar']['application'] == 'ONCE_AT_TOTAL_T', 'Scalar/fragment contract differs')
        check(row['observed_RZ'] is None and row['decision'] is None and row['qualified_for_candidate'] is False and M6.exact(row['predicted_RZ']) > 0, 'Prediction already opened/invalid')
        key = (row['model_group_id'], row['configuration_id'], row['physical_id'], axis, row['m'])
        check(key not in seen, 'Duplicate prediction axis'); seen.add(key)
        selected[route].append(dict(prediction_index=index, model_group_id=row['model_group_id'], configuration_id=p['configuration_id'], axis=axis, physical_id=c['id']))
    check(all(len(v) == 6 for v in selected.values()), 'Six axes per route required')
    return selected


def predictions():
    raw = S.read_pinned(ROOT, PRED)
    check(hashlib.sha256(raw).hexdigest() == PRED_SHA, 'Prediction bytes differ')
    return json.loads(raw)


def templates_and_selection():
    doc = predictions(); templates = {route: m.plans(ROOT) for route, m in ROUTES.items()}
    alignment = read(doc['rows'][0]['alignment_source'])
    frozen = read(M6.FROZEN)
    selected = validate(doc, templates, alignment, read(POLICY), frozen['models'])
    chosen = {}
    for route, plans in templates.items():
        chosen[route] = []
        for p in plans:
            req = next(r for r in p['plan']['pending']['requests'] if r['m'] == 9)
            cells = [dict(m=9, formula=p['formula'], row=ROUTES[route].native_row(req['physical_candidate']),
                          role='M9_HOLDOUT', model_groups=[r['model_group'] for r in doc['rows'] if r['configuration_id'] == p['configuration_id']])]
            chosen[route].append(dict(p, cells=cells))
    return doc, chosen, selected


def preopen():
    doc, templates, selected = templates_and_selection()
    inherited = {n: read(dict(commit=HANDOFF, path='docs/research/pf_m9_prediction_freeze_20261115/' + n + '.json')) for n in ('PREOPEN_CHECK', 'PUBLICATION_PRECHECK')}
    check(all(d['status'] == 'CLEAR_IN_CHECKED_SOURCE_AND_SERVER_METADATA' and d['observed_price_files_opened'] == 0 for d in inherited.values()), 'C preopening state differs')
    prior = read(dict(commit=M6.INDEX['commit'], path=M6.INDEX['path']))
    snapshots = {}
    for route, module in ROUTES.items():
        snapshots[route] = {}
        for stage in S.STAGES:
            rel, _ = S.paths(module.stage_spec(), stage)
            snapshots[route][stage] = S.committed(ROOT, rel, S.STAGE_FILES + S.BOUND_FILES, commit=M6.INDEX['commit'] if stage == 'CALIBRATION' else 'a55813373b84b9128cbd10c6812c0d26143a4280')
    return dict(status='M9_FROZEN_PREDICTIONS_BOUND_BEFORE_OBSERVATION', predictions=PRED, sha256=PRED_SHA,
                policy=doc['policy'], selected=selected, protected_snapshots=snapshots, inherited_C_checks=inherited,
                observed_m9_prices_opened=0, prediction_recalculations=0, fit_calls=0)


def duplicate_check(route, templates):
    """Continue the scoped D/C checks. Findings are held, not an excuse for recompilation."""
    module = ROUTES[route]; cells = [c for p in templates for c in p['cells']]
    base = S.duplicate_preflight(ROOT, module.stage_spec(), cells)
    findings = []
    for other in ROUTES:
        rel, private = paths(other)
        for candidate in (ROOT / rel / 'RESULT.json', private / 'child.json'):
            if candidate.exists():
                d = json.loads(candidate.read_text())
                for r in d['results']:
                    if any(r['cell']['row']['id'] == c['row']['id'] for c in cells):
                        findings.append(dict(kind='M9_NAMESPACE_ALREADY_ACQUIRED', path=str(candidate)))
    for pid in subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid='], text=True).split():
        if int(pid) == os.getpid(): continue
        try: args = Path('/proc', pid, 'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, PermissionError): continue
        if args and Path(args[0]).name.startswith('python') and any(Path(a).name == Path(MODULE).name for a in args[1:]) and ('--execute-approved-route' in args or '--child' in args):
            findings.append(dict(kind='M9_RELEVANT_JOB_REQUIRES_OVERLAP_REVIEW', pid=int(pid)))
    check(not findings, 'M9 reuse/job review required: ' + json.dumps(findings))
    return dict(base, continued_from=dict(commit='bcb67b2b2bb78776db1dd8d0625a3facb37cd298', path='docs/research/d_pf_allround_price_coverage_plan_20261115/COVERAGE_DELTA.json'), m9_namespace_findings=findings)


def child(job_path):
    job = json.loads(Path(job_path).read_text()); route = job['route']; module = ROUTES[route]
    # Stdlib-only here: install provider BEFORE importing scientific packages.
    raw = subprocess.check_output(['git', 'show', PRED['commit']+':'+PRED['path']], cwd=ROOT)
    check(hashlib.sha256(raw).hexdigest() == PRED_SHA, 'Child prediction bytes changed')
    rows = [r for r in json.loads(raw)['rows'] if r['model_group'][0] == route]
    wanted = []
    for formula, _ in T.TEMPLATES:
        mine = [r for r in rows if r['physical_candidate']['formula'] == formula]
        wanted.append(dict(m=9, formula=formula, row=mine[0]['physical_candidate'], role='M9_HOLDOUT', model_groups=[r['model_group'] for r in mine]))
    check(job['cells'] == wanted and all(r['scalar']['value'] == job['expected_scalar'] for r in rows), 'Child cells/scalar changed')
    for path in (MODULE, F.CALLER, T.CALLER, S.MODULE, 'scripts/pf_m6_holdout_evidence.py'):
        check((ROOT/path).read_bytes() == subprocess.check_output(['git', 'show', job['execution_SOURCE']+':'+path], cwd=ROOT), 'Child SOURCE changed: '+path)
    with (Path(job['private']) / 'CHILD_STARTED').open('x') as f:
        f.write('ONE_SHOT_NO_RETRY\n')
    result = dict(results=[], resources=None)
    try:
        import h46_long_time_signal as signal
        signal.install(); T.provider_checks()
        import h46_long_time_native as native
        native.PUBLIC, native.PRIVATE = Path(job['public']), Path(job['private'])
        prefix = 10 if route == 'FULL_PF' else 7
        ctx = native.native_context('H4', (prefix,)); prep = ctx.preparations[prefix]
        scalar = ctx.ham.constant + (prep.extracted_identity_coefficient if route == 'TRUNCATED_PF' else 0.)
        check(scalar.hex() == job['expected_scalar'].hex(), 'Fixed m9 total scalar differs before build')
        check(tuple(prep.deterministic_fragment_indices) == tuple(range(prefix)), 'Fragment order differs before build')
        native.CONTEXT['H4'] = ctx
        kwargs = dict(expected_tail_scalar=job['saved_tail_scalar']) if route == 'TRUNCATED_PF' else {}
        def checked(native, ctx, row, cohort, **kw):
            trial, facts, ok = (F.full_trial if route == 'FULL_PF' else T.checked_trial)(native, ctx, row, cohort, **kw)
            required = {'low_control', 'high_control_leakage', 'independent_occupation_vector', 'Hadamard_cosine', 'Hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'}
            return trial, facts, ok and required <= {c['check'] for c in trial['checks']}
        result['results'], result['resources'] = S.run_cells(native, ctx, checked,
            module.COHORT, job['cells'], kw=kwargs)
        result['context_gate'] = dict(scalar=scalar, scalar_hex=scalar.hex(), preparation_hash=prep.preparation_hash,
                                     fragment_order=list(prep.deterministic_fragment_indices))
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    write(job['result'], result)
    return 0 if len(result['results']) == 3 and all(r['ok'] for r in result['results']) else 2


def execute(route):
    gate = preopen(); doc, templates, _ = templates_and_selection(); module = ROUTES[route]
    if route == 'TRUNCATED_PF':
        rel, _ = paths('FULL_PF')
        check(json.loads((ROOT / rel / 'RESULT.json').read_text())['status'] == 'M9_PRICES_COMPLETE', 'Full route abnormal; no new p7 work')
    commit = S.source_gate(ROOT, (MODULE, F.CALLER, T.CALLER, S.MODULE, 'scripts/pf_m6_holdout_evidence.py'))
    duplicates = duplicate_check(route, templates[route])
    extra = module.stage_spec()['job_extra'](ROOT)
    out_rel, private = paths(route); out = ROOT / out_rel
    out.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
    cells = [c for p in templates[route] for c in p['cells']]
    scalars = {r['scalar']['value'] for r in doc['rows'] if r['model_group'][0] == route}
    check(len(scalars) == 1, 'One route scalar expected')
    job = dict(route=route, cells=cells, public=str(out / 'child_public'), private=str(private),
               result=str(private / 'child.json'), execution_SOURCE=commit, expected_scalar=next(iter(scalars)), **extra)
    write(private / 'job.json', job)
    started = time.monotonic()
    code = subprocess.call([sys.executable, '-B', str(ROOT / MODULE), '--child', str(private / 'job.json')], cwd=ROOT)
    data = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else dict(results=[], error='Child output missing', resources=None)
    good = [r for r in data['results'] if r['ok']]
    source = dict(module.stage_spec()['source'], commit=commit, caller=MODULE, underlying_caller=module.CALLER,
                  stage='M9_HOLDOUT', predictions=PRED, policy_source=POLICY, route=route)
    prices, sides, proofs = [], [], {}
    record_error = None
    try:
        for result in good:
            p = next(p for p in templates[route] if p['formula'] == result['cell']['formula'])
            a, b, c = module.price_bindings(ROOT, p['plan'], result['cell'], result['trial'], price_source=dict(source, record='RESULT.json'), evidence_commit=S.PLACEHOLDER, evidence_path=out_rel + '/IMPLEMENTATION_EVIDENCE.json')
            for price in a:
                price['diagnostic'].update(stage='M9_HOLDOUT', compiled_wrapper=result['wrappers']['cosine' if price['axis'] == 'X' else 'sine'])
            prices += a; sides += b; proofs.update(c)
        # C's descriptors are checked before opening any model comparison.
        for price in prices:
            pred = next(r for r in doc['rows'] if r['configuration_id'] == price['configuration_id'] and r['axis'] == price['axis'])
            check(price['m'] == 9 and price['physical_context']['physical_candidate'] == pred['physical_candidate'] and price['price_profile'] == pred['domain']['profile'], 'Acquired record differs from fixed prediction domain')
    except Exception as exc:
        record_error = type(exc).__name__ + ': ' + str(exc)
        prices, sides, proofs = [], [], {}  # raw trials remain preserved; no unguarded prices promoted
    write(out / 'PRICES.json', prices); write(out / 'SIDECARS_UNBOUND.json', sides); write(out / 'IMPLEMENTATION_EVIDENCE.json', proofs)
    after = preopen()['protected_snapshots']
    ok = code == 0 and len(good) == 3 and record_error is None and after == gate['protected_snapshots']
    write(out / 'RESULT.json', dict(status='M9_PRICES_COMPLETE' if ok else 'STOP_M9_ACQUISITION', route=route,
        source=source, predictions_sha256=PRED_SHA, duplicate_preflight=duplicates, context_gate=data.get('context_gate'),
        results=data['results'], error=data.get('error'), record_error=record_error, child_exit=code, resources=data['resources'],
        elapsed_seconds=time.monotonic()-started, protected_before=gate['protected_snapshots'], protected_after=after,
        valid_axis_records=len(prices), finished_UTC=datetime.now(timezone.utc).isoformat(), retry=False,
        fit_calls=0, prediction_recalculations=0, mandatory_stop=True))
    print(json.dumps(dict(route=route, status='M9_PRICES_COMPLETE' if ok else 'STOP_M9_ACQUISITION', output=out_rel)))
    return 0 if ok else 2


def bind(route, publication):
    """Published direct observations -> existing C implementation guard; no model judgement."""
    _, templates, _ = templates_and_selection(); module = ROUTES[route]
    rel, _ = paths(route); out = ROOT / rel
    fixed = S.committed(ROOT, rel, RAW_FILES, commit=publication)
    T._meta()
    from trottertracks import price_implementation_guard as G
    prices = json.loads((out / 'PRICES.json').read_text()); sides = json.loads((out / 'SIDECARS_UNBOUND.json').read_text())
    for side in sides:
        check(side['implementation_evidence']['commit'] == S.PLACEHOLDER, 'Unexpected bound locator')
        side['implementation_evidence']['commit'] = publication
    checks = []
    for p in templates[route]:
        cfg = p['plan']['configuration']; mine = [v for v in prices if v['configuration_id'] == cfg['id']]
        if not mine: continue
        policy = dict(id=read(POLICY)['id'], source=S.GUARD_SOURCE, profile=cfg['spec']['profile'], by_method={route: dict(native_policy=T.NATIVE_POLICY, rte_rotation_lowering=T.NO_RTE, scalar_quantity=module.SCALAR_QUANTITY)})
        cohorts = {(cfg['rounds'][v['m']]['candidate']['id'], v['axis']): module.COHORT for v in mine}
        checks.append(dict(configuration_id=cfg['id'], guard=G.check(ROOT, cfg, mine, cohorts=cohorts, bindings=sides, policy=policy)))
    check(sum(len(c['guard']['checked']) for c in checks) == len(prices), 'Not all acquired axes guard-checked')
    write(out / 'SIDECARS.json', sides)
    write(out / 'GUARD_CHECK.json', dict(evidence=fixed, policy=POLICY, route=route, checks=checks, model_qualified=False))
    return dict(route=route, checked=len(prices), output=rel)


def evaluate(publications):
    doc = predictions(); rows = []; observations = {}; problems = {}
    for route, module in ROUTES.items():
        if route not in publications:
            problems[route] = 'No published acquisition/implementation evidence'; continue
        try:
            rel, _ = paths(route); out = ROOT / rel
            S.committed(ROOT, rel, RAW_FILES + S.BOUND_FILES, commit=publications[route])
            stage = json.loads((out / 'RESULT.json').read_text()); guard = json.loads((out / 'GUARD_CHECK.json').read_text())
            check(guard['policy'] == POLICY and stage['predictions_sha256'] == PRED_SHA, 'Wrong policy/prediction')
            prices = json.loads((out / 'PRICES.json').read_text())
            check(sum(len(c['guard']['checked']) for c in guard['checks']) == len(prices), 'Guard missing')
            for i, price in enumerate(prices):
                pred = next(r for r in doc['rows'] if r['configuration_id'] == price['configuration_id'] and r['axis'] == price['axis'])
                check(price['m'] == 9 and price['physical_context']['physical_candidate'] == pred['physical_candidate'] and price['price_profile'] == pred['domain']['profile'] and price['classification'] == 'DIRECT_COMPILED_DETERMINISTIC' and price['expectation'] == 'DETERMINISTIC', 'Foreign observed receipt')
                observations[pred['model_group_id']] = dict(observed_RZ=price['mean']['rz_count'], receipt=dict(commit=publications[route], path=rel+'/PRICES.json', selector=[i], axis=price['axis'], physical_id=pred['physical_id'], execution_source=stage['source']))
        except Exception as exc: problems[route] = type(exc).__name__ + ': ' + str(exc)
    for i, pred in enumerate(doc['rows']):
        row = {k: deepcopy(pred[k]) for k in ('model_group_id', 'model_group', 'configuration_id', 'physical_id', 'axis', 'm', 'q', 'predicted_RZ')}
        row['prediction_source'] = dict(PRED, selector=['rows', i, 'predicted_RZ'])
        route = pred['model_group'][0]; obs = observations.get(pred['model_group_id'])
        judged = M6.judge_row(row, obs, problem=problems.get(route) or (None if obs else 'No valid independent m9 axis receipt'))
        judged['acquisition_implementation'] = 'VALID_DIRECT' if obs and route not in problems else 'UNKNOWN'
        rows.append(judged)
    counts = {s: sum(r['decision'] == s for r in rows) for s in ('PASS', 'FAIL', 'UNKNOWN')}
    campaign = 'FAIL' if counts['FAIL'] else 'UNKNOWN' if counts['UNKNOWN'] else 'PASS'
    return dict(status='M9_ONLY_INDIVIDUAL_JUDGEMENT', predictions=PRED, predictions_sha256=PRED_SHA, policy=read(POLICY), rows=rows, counts=counts,
        campaign=campaign, campaign_pass=counts['PASS'] == 12, valid_direct_prices=sum(r['acquisition_implementation'] == 'VALID_DIRECT' for r in rows),
        publications=publications, analysis_source=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        fit_calls=0, prediction_recalculations=0, other_rounds_qualified=False, N=None, G=None, mandatory_stop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preopen-output', type=Path); p.add_argument('--execute-approved-route', choices=ROUTES)
    p.add_argument('--child', type=Path); p.add_argument('--bind-route', choices=ROUTES); p.add_argument('--publication')
    p.add_argument('--evaluate-publications', type=Path); p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.child: return child(a.child)
    if a.preopen_output: write(a.preopen_output, preopen()); return 0
    if a.execute_approved_route: return execute(a.execute_approved_route)
    if a.bind_route: print(json.dumps(bind(a.bind_route, a.publication))); return 0
    if a.evaluate_publications:
        result = evaluate(json.loads(a.evaluate_publications.read_text())); a.output.mkdir(exist_ok=False, parents=True)
        write(a.output / 'RESULT.json', result)
        with (a.output/'HOLDOUT.csv').open('x', newline='') as f:
            w=csv.writer(f, lineterminator='\n'); w.writerow(['model_group_id','physical_id','axis','m','q','predicted_numerator','predicted_denominator','observed_RZ','relative_error_numerator','relative_error_denominator','acquisition_implementation','decision','receipt'])
            for r in result['rows']:
                e=r['relative_error'] or {}; w.writerow([r['model_group_id'],r['physical_id'],r['axis'],r['m'],r['q'],r['predicted_RZ']['numerator'],r['predicted_RZ']['denominator'],r['observed_RZ'],e.get('numerator'),e.get('denominator'),r['acquisition_implementation'],r['decision'],json.dumps(r['receipt'])])
        print(json.dumps(dict(campaign=result['campaign'],counts=result['counts']))); return 0
    p.error('Select one fixed m9 operation')


if __name__ == '__main__': sys.exit(main())
