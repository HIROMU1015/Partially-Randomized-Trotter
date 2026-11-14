"""T3 price builder: Truncated PF (B0I, p7) S2 16h/7, Y4 16h/3, M5 16h, deterministic.

No new builder: the Full PF long-time native path is reused unchanged
(`h46_long_time_native.native_context/compose/cost_trial` -> `build_deterministic_native`
-> `build_native_hadamard_wrapper` -> fixed compiler), with the differences
carried by the physical row only:
- prefix 7: `prepare(ham, 7)` keeps one-body + fragments 0..6 in saved order
  (native_context's DF_GENERATION_ORDER check);
- B0I scalar: cost_trial uses c + prep.extracted_identity_coefficient (the
  discarded-tail identity) once at total T; the run also requires it to equal
  the saved prefix-7 scalar used by the B0I signals;
- signed pieces / actual widths: formula and q come from the regenerated C
  configuration (early rounds keep their own actual width);
- identities: canonical rows from C.configure/prepare; physical IDs are checked
  against the fixed T2 PLAN_METADATA.
Same quantity as Full PF: FULL_WRAPPER_MEAN_COMPILED_PRICE, measured X/Y wrapper.

Record schema (this version, on C bridge c628e9e8): classification
DIRECT_COMPILED_DETERMINISTIC / expectation DETERMINISTIC (B PRICE_KINDS); the
04bead11 labels are kept only as a diagnostic. Circuits/cells/gates unchanged.

Stages (scripts/pf_price_stages.py): C's CELL_PLAN assigns m4/m5 to CALIBRATION
and m6 to HOLDOUT; each stage is executed, published and bound separately and
HOLDOUT needs C's frozen models. The 62ae1bd5 all-cells path is not kept.
M5/m4 X/Y are reserved for saved C-admitted receipts, not new jobs. While C's
p7 admission/revised plan is pending, metadata selection works but execution stops.

Default: metadata preflight. `--execute-approved-stage CALIBRATION|HOLDOUT`
(separate approval) runs the child (original install() first, provider c31ef424
bytes checked) for that stage's cells only. No retry.
"""
from copy import deepcopy
from fractions import Fraction
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import pf_price_stages as S  # noqa: E402  (stage selection/gates only; stdlib imports at module level)

CALLER = 'scripts/run_truncated_p7_prices.py'
H = Fraction(.1)
TEMPLATES = (('2nd', 16 * H / 7), ('4th', 16 * H / 3), ('4th(m5_best)', 16 * H))
PREFIX = 7
ROUNDS = (4, 5, 6)  # T3 plan: m4/m5 calibration, m6 holdout (C scope; roles from C CELL_PLAN)
T3 = dict(commit='5fb2c32e39e63267ea1748861269c362a93a1b86',
          path='docs/research/truncated_pf_t2_results_20261114/T3_CANDIDATES.json')
PLAN_META = dict(commit='c3ae31f353d27a9f720f51fa0b089acd3d57e211',
                 path='docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json')
A_MENU = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
              path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json')
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
                path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
NATIVE = dict(commit='af284d853890155bacd19e61850c196bd1d5c660', path='scripts/h46_long_time_native.py')
WRAPPER = dict(commit=NATIVE['commit'], path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
SCALAR_SOURCE = dict(commit='5cc7c07e112ce97205948465c94441cba0042416', path='src/trottertracks/baseline_signal_numerics.py',
                     selector='Context.tail_scalars[7] (saved INDEX context prefixes.7.scalar)')
COHORT = 'D_TRUNCATED_P7_DETERMINISTIC_T3_V1'
RECORD_SCHEMA = 'D_PRICE_RECORD_B_KINDS_V2'  # B kinds: DIRECT_COMPILED_DETERMINISTIC / DETERMINISTIC
NO_RTE = 'NOT_APPLICABLE_NO_RTE'
NATIVE_POLICY = 'H46_LONG_TIME_NATIVE_COMPOSE'
SCALAR_QUANTITY = 'HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY'
OUTPUT_ROOT = 'docs/research/d_truncated_p7_t3_prices_20261114'  # + /calibration/run01, /holdout/run01
PRIVATE_ROOT = Path('/home/AbeHiromu/projects/d-truncated-p7-t3-20261114-private')
SCOPE = dict(templates=3, rounds=list(ROUNDS), cells=9, new_cells=8, saved_receipt_cells=1,
             axes=['X', 'Y'], compile_maximum=16,
             stage_wrappers=dict(CALIBRATION=10, HOLDOUT=6), sampling=0, retry=False, fit=None, N=None, G=None)


def _meta():
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    from trottertracks import baseline_signal_pipeline as C
    from trottertracks import rpe_optimization_pipeline as I
    return C, I


def read(C, d):
    return json.loads(C.blob(ROOT, d['commit'], d['path']))


def enc(v):
    return dict(numerator=v.numerator, denominator=v.denominator)


def plans(root=ROOT):
    """Regenerate the three fixed p7 B0I configurations; IDs from T3/PLAN_METADATA."""
    C, I = _meta(); api = I.providers(str(root))
    t3 = read(C, T3)['per_formula']; meta = read(C, PLAN_META)
    table = {(r['prefix'], r['formula'], r['requested_delta_over_h']): r for r in meta['configurations_table']}
    base = read(C, A_MENU)['spec']; target = read(C, MANIFEST)['shared']['target']
    out = []
    for formula, width in TEMPLATES:
        top = t3[formula]['ranked'][0]
        if top['prefix'] != PREFIX or Fraction(top['width']) != width / H:
            raise ValueError('T3 top candidate moved')
        spec = api.D.full_pf_spec(base, formula=formula, delta_pf=enc(width), M=12, rank=10)
        spec['L_D'] = PREFIX
        spec['split'].update(L_D=PREFIX, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
        spec['split']['retained_hamiltonian'].update(construction='B0I', prefix=PREFIX)
        config = C.configure(root, spec, api=api)
        fixed = table[(PREFIX, formula, str(width / H))]
        if config['id'] != top['configuration_id'] or config['id'] != fixed['configuration_id']:
            raise ValueError('Regenerated configuration differs from T3/PLAN_METADATA')
        plan = I.plan(root, config, target=target, signal_source=dict(price_only=CALLER), references=())
        cells = []
        for req, r in zip(plan['pending']['requests'], fixed['rounds'], strict=True):
            c = req['physical_candidate']
            if c['id'] != r['physical_candidate_id'] or req['q_PF'] != r['q_PF'] or c['method'] != 'B0I' or c['prefix'] != PREFIX:
                raise ValueError('Regenerated physical round differs at m=' + str(req['m']))
            if req['m'] in ROUNDS:
                cells.append(dict(m=req['m'], row=deepcopy(c), signed_pieces=r['signed_pieces'],
                                  actual_delta_over_h=r['actual_delta_over_h'],
                                  action=S.SAVED if c['id'] == S.SAVED_IDS['TRUNCATED_PF'] else S.NEW))
        out.append(dict(formula=formula, width_over_h=str(width / H), configuration_id=config['id'], plan=plan, cells=cells))
    return out


def native_row(c):
    return dict(system='H4', method='B0I', prefix=c['prefix'], formula=c['formula'], T=c['T'], q=c['q'],
                delta=c['delta'], R_outer=None, K=None, id=c['id'])


def provider_checks():
    """Child: deterministic native/provider modules equal c31ef424 blobs."""
    for name in ('trotterlib.df_trotter.ops', 'trottertracks.resource_applicability.ax2b_native_df_v5',
                 'trottertracks.resource_applicability.h6_hash_reuse_v1', 'trotterlib.rte_compiled_cost'):
        module = importlib.import_module(name)
        rel = 'src/' + name.replace('.', '/') + '.py'
        if Path(module.__file__).read_bytes() != subprocess.check_output(
                ['git', 'show', 'c31ef424175c02379eafebe8fc6c7548c1d9aabe:' + rel], cwd=ROOT):
            raise ValueError('Provider SOURCE mismatch: ' + name)
    return True


def checked_trial(native, ctx, row, cohort, *, expected_tail_scalar=None):
    """cost_trial index0 plus explicit block/scalar/piece facts."""
    from h46_m5_contract import core
    prep = ctx.preparations[row['prefix']]
    blocks = len(prep.deterministic_blocks)
    if blocks != row['prefix'] + 1 or tuple(prep.deterministic_fragment_indices) != tuple(range(row['prefix'])):
        raise ValueError('Retained one-body + prefix fragments/order differ')
    scalar = ctx.ham.constant + prep.extracted_identity_coefficient
    if expected_tail_scalar is not None and prep.extracted_identity_coefficient != expected_tail_scalar:
        raise ValueError('B0I tail identity differs from the saved signal scalar')
    steps = core.schedule(row['formula'], row['q'], 'PER_PIECE', 1, T=row['T'])
    trial = native.cost_trial((row, cohort, 0))
    facts = dict(retained_blocks=blocks, fragment_order=list(prep.deterministic_fragment_indices),
                 scalar=scalar, tail_identity=prep.extracted_identity_coefficient, signed_pieces=len(steps),
                 negative_pieces=sum(s['signed_time'] < 0 for s in steps),
                 gates=[c['check'] for c in trial['checks']])
    ok = (trial['status'] == 'PASS' and trial['scalar'] == scalar and set(trial['prices']) == {'cosine', 'sine'}
          and set(facts['gates']) >= {'low_control', 'high_control_leakage', 'independent_occupation_vector',
                                     'Hadamard_cosine', 'Hadamard_sine'})
    return trial, facts, ok


def child(job_path):
    job = json.loads(Path(job_path).read_text())
    import h46_long_time_signal as signal
    signal.install()
    provider_checks()
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = Path(job['public']), Path(job['private'])
    ctx = native.native_context('H4', (PREFIX,))
    native.CONTEXT['H4'] = ctx
    results, resources = S.run_cells(native, ctx, checked_trial, COHORT, job['cells'],
                                     kw=dict(expected_tail_scalar=job['saved_tail_scalar']), workers=job.get('workers'))
    Path(job['result']).write_text(json.dumps(dict(results=results, resources=resources), indent=2, default=str))
    return 0 if len(results) == len(job['cells']) and all(r['ok'] for r in results) else 2


def price_bindings(root, plan, cell, trial, *, price_source, evidence_commit, evidence_path):
    C, I = _meta()
    profile = plan['configuration']['spec']['profile']; m = cell['m']
    row = plan['configuration']['rounds'][m]
    prices, sides, proofs = [], [], {}
    for req in [r for r in I.price_requests(root, plan) if r['m'] == m]:
        axis = req['axis']; metrics = trial['prices']['cosine' if axis == 'X' else 'sine']
        price = deepcopy(req)
        price.update(cohort=COHORT, n_paths=1, mean=dict(rz_count=metrics['rz_count']), metrics=deepcopy(metrics),
            classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC', price_status='DIRECT_DETERMINISTIC_AVAILABLE',
            record_schema=RECORD_SCHEMA, diagnostic=dict(record_role='EXACT_DETERMINISTIC_CIRCUIT_PRICE', model_qualified=False),
            source=deepcopy(price_source), price_profile=dict(profile, lowering_id=NATIVE_POLICY),
            profile_extension=dict(lowering_id=NATIVE_POLICY, source=NATIVE), normalization_is_price_multiplier=False)
        facts = dict(method='TRUNCATED_PF', profile=deepcopy(profile), native_policy=NATIVE_POLICY,
            rte_rotation_lowering=NO_RTE, rte_lowering_source=None, native_source=deepcopy(NATIVE),
            wrapper_source=deepcopy(WRAPPER), scalar_source=deepcopy(SCALAR_SOURCE), scalar_quantity=SCALAR_QUANTITY,
            scalar_application='ONCE_AT_TOTAL_T', measurement=I.providers(str(root)).B.p.measurement(row, axis),
            resolved_scalar=trial['scalar'], expected_scalar=trial['scalar'])
        key = '%s_m%d%s' % (plan['configuration']['spec']['formula'], m, axis)
        side = dict(identity=deepcopy(req['identity']), law_binding=deepcopy(req['law_binding']), axis=axis,
            cohort=COHORT, price_source=deepcopy(price_source), physical_context=deepcopy(req['physical_context']),
            profile_extension=dict(lowering_id=NATIVE_POLICY, source=NATIVE), implementation=facts,
            implementation_evidence=dict(commit=evidence_commit, path=evidence_path, selector=[key]))
        proofs[key] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis', 'cohort',
                       'physical_context', 'profile_extension', 'implementation')}
        prices.append(price); sides.append(side)
    return prices, sides, proofs


def c_model(root=ROOT):
    """C's price_model_validation (in this tree at c628e9e8; record/fit/validate, not reimplemented)."""
    src = str(Path(root) / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    from trottertracks import price_model_validation
    return price_model_validation


def fit_inputs(prices, *, route, formula, width_over_h, roles, root=ROOT):
    """Full PF / Truncated price records -> C calibration/holdout records (features one, q_PF).

    `roles` maps m to its stage role (from C CELL_PLAN); one call = one formula/actual width.
    """
    V = c_model(root); out = []
    for p in prices:
        c = p['physical_context']['physical_candidate']; m = p['m']
        out.append(V.record(cell_id='%s:%s:%s:m%d' % (route, formula, width_over_h, m), role=roles[m], route=route,
            axis=p['axis'], features=dict(one=1, q=c['q']), rz_count=p['mean']['rz_count'],
            kind='DIRECT_COMPILED_DETERMINISTIC', source=p['source'],
            implementation=dict(lowering_id=p['profile_extension']['lowering_id'], native=p['profile_extension']['source']),
            profile=p['price_profile']))
    return out


def saved_tail_scalar(root):
    """Execution only: the saved prefix-7 scalar used by the B0I signals (gate inside checked_trial)."""
    C, _ = _meta()
    from trottertracks import baseline_signal_numerics as N
    restored = N.load_shared_h4(root, read(C, MANIFEST)['shared']['binding'])
    return dict(saved_tail_scalar=float(restored[2]['prefixes'][str(PREFIX)]['scalar']))


def stage_spec():
    return dict(route='TRUNCATED_PF', caller=CALLER, sources=(CALLER, S.MODULE), cohort=COHORT,
                scalar_quantity=SCALAR_QUANTITY, output=OUTPUT_ROOT, private=PRIVATE_ROOT, plans=plans,
                native_row=native_row, bindings=price_bindings, job_extra=saved_tail_scalar,
                source=dict(caller=CALLER, native=NATIVE, wrapper=WRAPPER, scalar=SCALAR_SOURCE, cohort=COHORT))


def stage_summary(root, spec):
    """Metadata only: the CELL_PLAN stage split of the regenerated cells."""
    templates = spec['plans'](root); out = {}
    for stage in S.STAGES:
        chosen = S.select(root, templates, spec['route'], stage)
        out[stage] = dict(output=S.paths(spec, stage)[0], new_wrappers=2 * len(S.new_cells(chosen)),
            cells=[dict(formula=p['formula'], width_over_h=p['width_over_h'], m=c['m'], id=c['row']['id'], q=c['row']['q'],
                        action=c.get('action', S.NEW), model_groups=[S.key(g) for g in c['model_groups']])
                   for p in chosen for c in p['cells']])
    return templates, out


def preflight(root=ROOT):
    templates, stages = stage_summary(root, stage_spec())
    out = []
    for p in templates:
        out.append(dict(formula=p['formula'], width_over_h=p['width_over_h'], configuration_id=p['configuration_id'],
                        cells=[dict(m=c['m'], id=c['row']['id'], q=c['row']['q'], signed_pieces=c['signed_pieces'], action=c['action'],
                                    actual_delta_over_h=c['actual_delta_over_h']) for c in p['cells']]))
    saved = None if S.P7_ADMISSION is None else S.saved_calibration(root, S.select(root, templates, 'TRUNCATED_PF', 'CALIBRATION'))
    return dict(status='PREFLIGHT_METADATA_ONLY', scope=SCOPE, templates=out, stages=stages, cell_plan=S.P7_CELL_PLAN,
                revised_plan=S.plan_status(), cohort=COHORT,
                saved_calibration=saved, saved_binding_status='PENDING_C_ACCEPTANCE' if saved is None else 'C_ADMITTED')


def execute(root=ROOT, stage=None, frozen=None):
    return S.execute_stage(root, stage_spec(), stage, frozen=frozen)


def _pinned(text):
    commit, _, path = (text or '').partition(':')
    return dict(commit=commit, path=path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-approved-stage', choices=sorted(S.STAGES),
                        help='one stage of the approved deterministic compile (C CELL_PLAN roles; separate approval)')
    parser.add_argument('--frozen-models', help='HOLDOUT only: COMMIT:PATH of C frozen calibration models')
    parser.add_argument('--bind-stage-evidence', choices=sorted(S.STAGES), help='after publishing a stage: fill the commit, C guard')
    parser.add_argument('--evidence-commit', help='published commit of the stage outputs (with --bind-stage-evidence)')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.execute_approved_stage:
        frozen = _pinned(args.frozen_models) if args.frozen_models else None
        return execute(ROOT, args.execute_approved_stage, frozen)
    if args.bind_stage_evidence:
        print(json.dumps(S.bind(ROOT, stage_spec(), args.bind_stage_evidence, args.evidence_commit)))
        return 0
    print(json.dumps(preflight(ROOT), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
