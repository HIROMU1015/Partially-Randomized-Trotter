"""Full PF (all 11 blocks) prices paired with the Truncated p7 T3 cells: S2 16h/7, Y4 16h/3, M5 16h.

Same deterministic long-time native path, wrapper, compiler and price quantity as
`run_truncated_p7_prices.py` (whose functions are imported, not modified):
method FULL, prefix 10 = one-body + fragments 0..9, scalar = Hamiltonian constant
only (no tail identity), signed pieces/actual widths from the regenerated C
configurations (constructor of C 3a8b37f1 COVERAGE). Calibration m4/m5 and
holdout m6 follow C's plan; a cell whose X and Y both have an EXACT saved receipt
(M5 m4) is NOT recompiled and is handed to C for acceptance.

Default: metadata preflight. `--execute-approved-full-pf` (separate approval) runs
the provider-root child for the new cells only (at most 16 wrappers). No retry.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import run_truncated_p7_prices as T3  # noqa: E402  (frozen T3 caller; functions reused only)

CALLER = 'scripts/run_full_pf_shortlist_prices.py'
RANK = 10
COVERAGE = dict(commit='3a8b37f1a282c0c06ac1a363b070f4376e9ff41a',
                path='docs/research/full_pf_latest_price_guard_20261114/COVERAGE.json')
C_PLAN = dict(commit='cfc0fae6f1399bbc46689b4d88998b415fa47781',
              path='docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md')
C_MODEL = dict(commit='cfc0fae6f1399bbc46689b4d88998b415fa47781', path='src/trottertracks/price_model_validation.py')
ROLES = {4: 'CALIBRATION', 5: 'CALIBRATION', 6: 'HOLDOUT'}
COHORT = 'D_FULL_PF_SHORTLIST_DETERMINISTIC_V1'
SCALAR_QUANTITY = 'HAMILTONIAN_CONSTANT'
SCALAR_SOURCE = dict(T3.NATIVE, selector='cost_trial scalar = ctx.ham.constant for method FULL (saved snapshot constant)')
OUTPUT = 'docs/research/d_full_pf_shortlist_prices_20261114/run01'
PRIVATE_OUTPUT = Path('/home/AbeHiromu/projects/d-full-pf-shortlist-20261114-private/run01')
SCOPE = dict(templates=3, rounds=[4, 5, 6], cells=9, new_cells=8, saved_receipt_cells=1, compile_maximum=16,
             sampling=0, retry=False, fit=None, N=None, G=None)


def plans(root=ROOT):
    """Regenerate the C Full PF configurations; pin IDs/physical IDs/receipts to C COVERAGE."""
    C, I = T3._meta(); api = I.providers(str(root))
    cov = T3.read(C, COVERAGE); t3 = T3.read(C, T3.T3)['per_formula']
    target = T3.read(C, T3.MANIFEST)['shared']['target']
    out = []
    for (formula, width), cand in zip(T3.TEMPLATES, cov['candidates'], strict=True):
        con = cand['constructor']
        base = T3.read(C, con['base'])
        for key in con['base_selector']:
            base = base[key]
        if (con['formula'] != formula or Fraction(con['delta_pf']['numerator'], con['delta_pf']['denominator']) != width
                or (con['M'], con['rank']) != (12, RANK) or t3[formula]['full_pf_shortlist']['configuration_id'] != cand['configuration_id']):
            raise ValueError('C COVERAGE/T3 Full PF shortlist differs')
        spec = api.D.full_pf_spec(base, formula=formula, delta_pf=T3.enc(width), M=12, rank=RANK)
        config = C.configure(root, spec, api=api)
        if config['id'] != cand['configuration_id']:
            raise ValueError('Regenerated Full PF configuration differs from C COVERAGE')
        plan = I.plan(root, config, target=target, signal_source=dict(price_only=CALLER), references=())
        cells = []
        for req in plan['pending']['requests']:
            if req['m'] not in ROLES:
                continue
            c = req['physical_candidate']
            rows = [r for r in cov['coverage'] if r['configuration_id'] == config['id'] and r['m'] == req['m']]
            if len(rows) != 2 or any(r['physical_candidate']['id'] != c['id'] or r['q_PF'] != c['q'] for r in rows):
                raise ValueError('Physical round differs from C COVERAGE at m=' + str(req['m']))
            if c['method'] != 'FULL' or c['prefix'] != RANK or c['formula'] != formula:
                raise ValueError('Full PF row must keep all blocks and the template formula')
            saved = all(r['receipt_catalog_status'] == 'EXACT_METADATA_MATCH' and r['direct_evidence'] for r in rows)
            cells.append(dict(m=req['m'], role=ROLES[req['m']], row=deepcopy(c),
                actual_delta=deepcopy(rows[0]['actual_delta']),
                signed_pieces=len(plan['configuration']['rounds'][req['m']]['structure']['signed_weights']) * c['q'],
                action='SAVED_RECEIPT_FOR_C_ACCEPTANCE' if saved else 'NEW_COMPILE',
                saved_receipts=[dict(axis=r['axis'], evidence=deepcopy(r['direct_evidence'])) for r in rows] if saved else None))
        out.append(dict(formula=formula, width_over_h=str(width / T3.H), configuration_id=config['id'], plan=plan, cells=cells))
    return out


def native_row(c):
    return dict(system='H4', method='FULL', prefix=c['prefix'], formula=c['formula'], T=c['T'], q=c['q'],
                delta=c['delta'], R_outer=None, K=None, id=c['id'])


def full_trial(native, ctx, row, cohort):
    """T3's checked trial plus the Full PF facts: all blocks, no tail identity correction."""
    prep = ctx.preparations[row['prefix']]
    if row['method'] != 'FULL' or row['prefix'] != ctx.ham.n_blocks or prep.extracted_identity_coefficient != 0.:
        raise ValueError('Full PF keeps every block and has no tail identity correction')
    trial, facts, ok = T3.checked_trial(native, ctx, row, cohort)
    ok = ok and trial['scalar'] == ctx.ham.constant and facts['retained_blocks'] == ctx.ham.n_blocks + 1
    return trial, facts, ok


def child(job_path):
    job = json.loads(Path(job_path).read_text())
    import h46_long_time_signal as signal
    signal.install()
    T3.provider_checks()
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(job['public']), Path(job['private'])
    ctx = native.native_context('H4', (RANK,))
    native.CONTEXT['H4'] = ctx
    results = []
    for cell in job['cells']:
        trial, facts, ok = full_trial(native, ctx, cell['row'], COHORT)
        results.append(dict(cell=cell, trial=trial, facts=facts, ok=ok))
        if not ok:
            break
    Path(job['result']).write_text(json.dumps(results, indent=2, default=str))
    return 0 if len(results) == len(job['cells']) and all(r['ok'] for r in results) else 2


def price_bindings(root, plan, cell, trial, *, price_source, evidence_commit, evidence_path):
    """FULL_PF records/sidecars in the same shape as the Truncated T3 ones."""
    C, I = T3._meta()
    profile = plan['configuration']['spec']['profile']; m = cell['m']; row = plan['configuration']['rounds'][m]
    prices, sides, proofs = [], [], {}
    for req in [r for r in I.price_requests(root, plan) if r['m'] == m]:
        axis = req['axis']; metrics = trial['prices']['cosine' if axis == 'X' else 'sine']
        price = deepcopy(req)
        price.update(cohort=COHORT, n_paths=1, mean=dict(rz_count=metrics['rz_count']), metrics=deepcopy(metrics),
            classification='DIRECT_DETERMINISTIC', expectation='EXACT_DETERMINISTIC_CIRCUIT', price_status='UNQUALIFIED_FOR_C_MODEL',
            source=deepcopy(price_source), price_profile=dict(profile, lowering_id=T3.NATIVE_POLICY),
            profile_extension=dict(lowering_id=T3.NATIVE_POLICY, source=T3.NATIVE), normalization_is_price_multiplier=False)
        facts = dict(method='FULL_PF', profile=deepcopy(profile), native_policy=T3.NATIVE_POLICY,
            rte_rotation_lowering=T3.NO_RTE, rte_lowering_source=None, native_source=deepcopy(T3.NATIVE),
            wrapper_source=deepcopy(T3.WRAPPER), scalar_source=deepcopy(SCALAR_SOURCE), scalar_quantity=SCALAR_QUANTITY,
            scalar_application='ONCE_AT_TOTAL_T', measurement=I.providers(str(root)).B.p.measurement(row, axis),
            resolved_scalar=trial['scalar'], expected_scalar=trial['scalar'])
        key = 'FULL_%s_m%d%s' % (plan['configuration']['spec']['formula'], m, axis)
        side = dict(identity=deepcopy(req['identity']), law_binding=deepcopy(req['law_binding']), axis=axis, cohort=COHORT,
            price_source=deepcopy(price_source), physical_context=deepcopy(req['physical_context']),
            profile_extension=dict(lowering_id=T3.NATIVE_POLICY, source=T3.NATIVE), implementation=facts,
            implementation_evidence=dict(commit=evidence_commit, path=evidence_path, selector=[key]))
        proofs[key] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis', 'cohort',
                       'physical_context', 'profile_extension', 'implementation')}
        prices.append(price); sides.append(side)
    return prices, sides, proofs


def c_model(root=ROOT):
    """C's price_model_validation from its pinned blob (record/fit/validate; not reimplemented)."""
    name = 'trottertracks._price_model_validation_cfc0fae6'
    if name not in sys.modules:
        src = str(root / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        import trottertracks  # noqa: F401
        mod = types.ModuleType(name); mod.__package__ = 'trottertracks'; sys.modules[name] = mod
        code = subprocess.check_output(['git', 'show', C_MODEL['commit'] + ':' + C_MODEL['path']], cwd=root)
        exec(compile(code, C_MODEL['commit'] + ':' + C_MODEL['path'], 'exec'), mod.__dict__)
    return sys.modules[name]


def fit_inputs(prices, *, route, formula, width_over_h, roles=ROLES, root=ROOT):
    """Full PF / Truncated price records -> C calibration/holdout records (features one, q_PF)."""
    V = c_model(root); out = []
    for p in prices:
        c = p['physical_context']['physical_candidate']; m = p['m']
        out.append(V.record(cell_id='%s:%s:%s:m%d' % (route, formula, width_over_h, m), role=roles[m], route=route,
            axis=p['axis'], features=dict(one=1, q=c['q']), rz_count=p['mean']['rz_count'],
            kind='DIRECT_COMPILED_DETERMINISTIC', source=p['source'],
            implementation=dict(lowering_id=p['profile_extension']['lowering_id'], native=p['profile_extension']['source']),
            profile=p['price_profile']))
    return out


def preflight(root=ROOT):
    out = []
    for p in plans(root):
        out.append(dict(formula=p['formula'], width_over_h=p['width_over_h'], configuration_id=p['configuration_id'],
                        cells=[dict(m=c['m'], role=c['role'], id=c['row']['id'], q=c['row']['q'], signed_pieces=c['signed_pieces'],
                                    actual_delta=c['actual_delta'], action=c['action']) for c in p['cells']]))
    new = sum(c['action'] == 'NEW_COMPILE' for t in out for c in t['cells'])
    return dict(status='PREFLIGHT_METADATA_ONLY', scope=SCOPE, new_cells=new, compile_maximum=2 * new,
                templates=out, output=OUTPUT, cohort=COHORT, plan=C_PLAN)


def execute(root=ROOT):
    C, I = T3._meta()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (CALLER, T3.CALLER):
        if C.blob(root, commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ from HEAD: ' + path)
    for d in (T3.NATIVE, T3.WRAPPER):
        if C.blob(root, d['commit'], d['path']) != (Path(root) / d['path']).read_bytes():
            raise ValueError('Pinned native SOURCE differs: ' + d['path'])
    templates = plans(root)
    new = [dict(c, formula=p['formula'], row=native_row(c['row'])) for p in templates for c in p['cells'] if c['action'] == 'NEW_COMPILE']
    if len(new) != 8:
        raise ValueError('Fixed Full PF scope is 8 new cells / 16 wrappers')
    out = Path(root) / OUTPUT; out.mkdir(parents=True, exist_ok=False); PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=False)
    job = dict(cells=new, public=str(out / 'child_public'), private=str(PRIVATE_OUTPUT), result=str(PRIVATE_OUTPUT / 'child.json'))
    (PRIVATE_OUTPUT / 'job.json').write_text(json.dumps(job))
    start = time.perf_counter()
    code = subprocess.call([sys.executable, '-B', str(Path(root) / CALLER), '--child', str(PRIVATE_OUTPUT / 'job.json')], cwd=root)
    results = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else []
    status = 'FULL_PF_PRICES_COMPLETE' if code == 0 else 'STOP_FULL_PF_PRICE_CHECK'
    source = dict(commit=commit, caller=CALLER, native=T3.NATIVE, wrapper=T3.WRAPPER, cohort=COHORT)
    if code == 0:
        prices, sides, proofs, fits = [], [], {}, []
        for p in templates:
            got = []
            for c in p['cells']:
                if c['action'] != 'NEW_COMPILE':
                    continue
                r = next(x for x in results if x['cell']['row']['id'] == c['row']['id'])
                a, b, d = price_bindings(root, p['plan'], c, r['trial'], price_source=dict(source, record='RESULT.json'),
                    evidence_commit='FILL_WITH_PUBLISHED_COMMIT', evidence_path=OUTPUT + '/IMPLEMENTATION_EVIDENCE.json')
                prices += a; sides += b; proofs.update(d); got += a
            fits += fit_inputs(got, route='FULL_PF', formula=p['formula'], width_over_h=p['width_over_h'])
        saved = [dict(configuration_id=p['configuration_id'], m=c['m'], receipts=c['saved_receipts'])
                 for p in templates for c in p['cells'] if c['action'] != 'NEW_COMPILE']
        for name, value in (('PRICES.json', prices), ('SIDECARS.json', sides), ('IMPLEMENTATION_EVIDENCE.json', proofs),
                            ('C_FIT_INPUTS.json', fits), ('SAVED_RECEIPTS_FOR_C.json', saved)):
            (out / name).write_text(json.dumps(value, indent=2, default=str))
    (out / 'RESULT.json').write_text(json.dumps(dict(status=status, source=source, scope=SCOPE, results=results,
        elapsed_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat(),
        mandatory_stop=True, retry_performed=False, fit=None, N=None, G=None), indent=2, default=str))
    print(json.dumps(dict(status=status)))
    return 0 if code == 0 else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-approved-full-pf', action='store_true', help='8 new deterministic cells, 16 wrappers (separate approval)')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.execute_approved_full_pf:
        return execute(ROOT)
    print(json.dumps(preflight(ROOT), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
