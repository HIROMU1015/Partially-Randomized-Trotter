"""ARTIFICIAL endpoints, shots, wrappers and receipts only; never saved H8 values.

The registry file supplies candidate IDs only. Every scalar, cost and receipt
below is manufactured by this test module.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
sys.path.insert(0, str(ROOT/'scripts'))
import h8_noncancellation as n
import h8_noncancellation_contract as c
import run_h8_noncancellation_diagnostic as runner
import audit_h8_noncancellation_evidence as auditor

REGISTRY = json.loads((ROOT/'references/registration_v1.json').read_bytes())
CELLS = {x['id']: x for x in REGISTRY['old'] + REGISTRY['additional']}
OLD = [x['id'] for x in REGISTRY['old']]
OLD_B0I = [x for x in OLD if CELLS[x]['method'] == 'B0I']
UNION = [f'B2_p7_q{q}_2nd_r4_K{K}' for q in (2, 4, 8) for K in (2, 4)]
REF = dict(real=.5, imag=-.25)
U = 1e-9
WAXES = ('cosine', 'sine')
BINDING = dict(compiler=n.COMPILER, source=c.PROVIDER, seed_source=c.SEED_SOURCE, result=c.D0['result'],
               analysis_source=c.D0['analysis_source'], input='e'*64, allowed_wrapper_sources=[c.PROVIDER])
PERMIT = c.Permit(synthetic=True, scope=n.SCOPE)


class ArtificialSavedPermit:
    """Mock the saved-data permit for manufactured decoder bytes only."""
    synthetic = False

    def require(self, kind):
        if kind != 'SAVED_H8':
            raise AssertionError('ARTIFICIAL_DECODER_SCOPE')


def h(*parts):
    raw = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(raw[:8], 'big')/2**64 - .5


def add(a, b):
    return {k: a[k] + b[k] for k in ('real', 'imag')}


def sub(a, b):
    return {k: a[k] - b[k] for k in ('real', 'imag')}


def allowance(u=U, margin=1e-12):
    return dict(kind='EMPIRICAL', certified=False, real=u, imag=u, log_B_margin=margin)


def det_record(cell, discard=None, PF=None):
    cid, scale = cell['id'], .004
    d = discard or {a: scale*h(cid, 'd', a) for a in n.AXES}
    p = PF or {a: scale*h(cid, 'p', a) for a in n.AXES}
    if PF is None and h(cid, 'cancel') > 0:
        p = {a: -.9*d[a] + .0002*h(cid, 'x', a) for a in n.AXES}
    ref = dict(REF)
    if cell['method'] == 'B1':
        corrected = add(ref, p)
        signals, parts = dict(corrected=corrected), dict(PF=sub(corrected, ref))
    else:
        truncated = add(ref, d)
        corrected = add(truncated, p)
        signals = dict(corrected=corrected, exact_truncated=truncated)
        parts = dict(discard=sub(truncated, ref), PF=sub(corrected, truncated))
    decomposition = dict(total_signed=sub(corrected, ref))
    if cell['id'] not in OLD_B0I:
        decomposition['signed_parts'] = parts
    return dict(cell=deepcopy(cell), signals=signals, reference=ref, error_decomposition=decomposition,
                allowance=allowance(), log_B=0.)


def PR_cell(cid):
    q, K = int(cid.split('_q')[1].split('_')[0]), int(cid.split('_K')[1])
    return dict(id=cid, method='B2', prefix=7, q=q, formula='2nd', r=4, R=4*q, K=K)


def PR_record(cid):
    cell = PR_cell(cid)
    ref = dict(REF)
    tail = add(ref, {a: .002*h(cid, 'o', a) for a in n.AXES})
    corrected = add(tail, {a: .002*h(cid, 't', a) for a in n.AXES})
    signals = dict(raw=add(corrected, dict(real=1e-3, imag=0.)), corrected=corrected, exact_tail=tail)
    b = 1 + .0004*(1 + h(cid, 'b'))
    return dict(cell=cell, signals=signals, independent_forward_signals=deepcopy(signals), reference=ref,
                error_decomposition=dict(total_signed=sub(corrected, ref), signed_parts=dict(
                    outer_PF=sub(tail, ref), finite_RTE=sub(corrected, tail))),
                allowance=allowance(), b=b, log_B=cell['R']*math.log(b), numerical_allowance_certified=False)


def signed_rows(records):
    rows = []
    for cid, cell in CELLS.items():
        s, ref = records[cid]['signals'], records[cid]['reference']
        for axis in n.AXES:
            parts = (dict(PF=s['corrected'][axis]-ref[axis]) if cell['method'] == 'B1' else
                     dict(discard=s['exact_truncated'][axis]-ref[axis],
                          PF=s['corrected'][axis]-s['exact_truncated'][axis]))
            rows.append(dict(cell={k: cell[k] for k in ('id', 'method', 'prefix', 'q', 'formula')}, axis=axis,
                             status='PASS_EMPIRICAL_NUMERICAL_CONSISTENCY', first_failure=None, failures=[],
                             checks=[], signed_parts=parts, reconstructed=cid in OLD_B0I, provenance={},
                             rigorous_bias_guarantee=None))
    return rows


def wrapper(phase, cid, replica, seed, axis, rz):
    event = hashlib.sha256(f'{phase}|{cid}|{replica}'.encode()).hexdigest()
    return dict(phase=phase, cell_id=cid, replica=replica, seed=seed, axis=axis, control='symmetric_directional',
                source=c.PROVIDER, compiler_hash=n.COMPILER, event_digest=event, fingerprint='f'*64,
                RZ=rz, CX=rz//2, RZ_depth=rz//3, CX_depth=rz//4, depth=rz//5, size=rz*3)


def B2_row(cid, axis, prediction, measured, model_sha):
    return dict(case_id=cid, axis=axis, measured_RZ=measured, model_sha256=model_sha, profile_sha256=n.PROFILE,
                compiler_hash=n.COMPILER, native_exit_code=0, reaped=True, fingerprint_before='a'*64,
                fingerprint_after='a'*64, max_control_error=1e-14,
                model_check=dict(exact_prediction=prediction, measured_RZ=measured, PASS=True,
                                 exact_relative_error=str(abs(F(prediction)-measured)/measured)))


MODEL = dict(schema='H8_FROZEN_COST_MODEL_V1', S2='a+b*q+d*prefix*q', S4='a+3*b*q+3*d*prefix*q',
             profile_sha256=n.PROFILE, compiler_hash=n.COMPILER, old_B0_used=False, S4_measured_validation=False,
             models=[dict(method=m, axis=a, coefficients=['1500', '1300', '3600'], rank=3, TRAIN_ids=[], checks=[],
                          status='S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS',
                          S4_status='UNVALIDATED_SCHEDULE_COMPOSITION_HYPOTHESIS', full_grid_guaranteed=False)
                     for m in ('B0', 'B0I') for a in WAXES])
MODEL_SHA = n.d0.digest(MODEL)


def B2_proof():
    saved = [B2_row('B0_p15_q1_4th', a, '1175933/7', 168000, MODEL_SHA) for a in WAXES]
    new = [B2_row('B0I_p21_q8_4th', a, '1852640', 1852000, MODEL_SHA) for a in WAXES]
    registry = dict(original_execution_status='STOP', model_sha256=MODEL_SHA, axes=deepcopy(saved))
    proof = dict(
        driver_terminal=dict(status='COMPLETE_MANDATORY_STOP', scope=n.B2['scope'], SOURCE=n.B2['SOURCE'],
                             fit_performed=False, retry=False, next_stage_authorized=False,
                             validation=dict(status=n.B2['validation']), axes=new, saved_B0_axes=deepcopy(saved)),
        saved_B0_acceptance=dict(acceptance_scope=n.B2['scope'], original_B2_status='STOP', axes=deepcopy(saved),
                                 accepted_by_new_grant=True),
        launcher_terminal=dict(status='COMPLETE_MANDATORY_STOP', native_exit_code=0, first_stop=None,
                               cleanup_errors=[], all_reaped=True, process_thread_violation=None,
                               max_monitor_gap_seconds=1.5),
        final_gate=dict(status='PASS_MANDATORY_STOP', SOURCE=n.B2['SOURCE'], profile_sha256=n.PROFILE),
        launcher_log_terminal=dict(status='COMPLETE_MANDATORY_STOP', SOURCE=n.B2['SOURCE'], reported_exit_code=0),
        witness_native_exit=dict(status='NATIVE_LAUNCHER_WAIT_RECORDED', native_exit_code=0, wait_completed=True,
                                 witness_stop_requested=False))
    return proof, registry


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def receipt(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def objects(mutate=lambda role, value: None):
    """Ordered artificial input roles; mutate(role, obj) runs before encoding."""
    records = {cid: det_record(cell) for cid, cell in CELLS.items()}
    records.update({cid: PR_record(cid) for cid in UNION})
    ordered = list(CELLS) + UNION
    usual = {(cid, e, f): n.d0.accounting(records[cid], e, f) for cid in ordered
             for e in n.EPSILONS for f in n.FACTORS}
    raw, out = {}, {}

    def put(role, value, data=None):
        mutate(role, value)
        out[role] = value
        raw[role] = encoded(value) if data is None else data
        return raw[role]

    put('registry', REGISTRY, (ROOT/'references/registration_v1.json').read_bytes())
    dummy = [dict(cell=dict(id='ARTIFICIAL_OTHER_B2_' + str(i), method='B2')) for i in range(3)]
    old_signals = dict(signals=[records[x] for x in OLD if CELLS[x]['method'] != 'B0I'] +
                       [records[x] for x in UNION] + dummy)
    put('old_signals', old_signals)
    put('old_B0I', [dict(prefix=CELLS[x]['prefix'], q=CELLS[x]['q'], scalar={}, signal=records[x]) for x in OLD_B0I])
    extra = []
    for x in REGISTRY['additional']:
        row = deepcopy(records[x['id']])
        row['precision_rows'] = {str(e): {str(f): usual[x['id'], e, f] for f in n.FACTORS} for e in n.EPSILONS}
        extra.append(row)
    decoded = encoded(dict(status='COMPLETE_MANDATORY_STOP', signals=extra))
    zipped = gzip.compress(decoded, mtime=0)
    put('A_signals', None, zipped)
    put('A_export', dict(gzip_bytes=len(zipped), gzip_sha256=hashlib.sha256(zipped).hexdigest(),
                         original_json_bytes=len(decoded), original_json_sha256=hashlib.sha256(decoded).hexdigest()))
    put('full_reference', dict(reference=dict(REF), numerical_allowance_certified=False,
                               reference_agreement=dict(certified=False)))
    rows = []
    for cid in UNION:
        q = PR_cell(cid)['q']
        for i in range(32):
            seed = c.task_seed(BINDING, cid, 'confirmation', i)
            rz = 1000 + 37*(i % 7) + 13*q
            rows += [wrapper('confirmation', cid, i, seed, 'cosine', rz),
                     wrapper('confirmation', cid, i, seed, 'sine', rz + 5*(i % 3))]
        for i in range(8):
            rows += [wrapper('exploration', cid, i, 7 + i, a, 10**9) for a in WAXES]
    for cid, cell in CELLS.items():
        if cell['method'] == 'B1':
            rows += [wrapper('exploration', cid, 0, 11, a, 50000 + cell['prefix']*cell['q'] + (a == 'sine'))
                     for a in WAXES]
    header = ['phase', 'cell_id', 'replica', 'seed', 'axis', 'control', 'source', 'compiler_hash', 'event_digest',
              'fingerprint', 'CX', 'CX_depth', 'RZ', 'RZ_depth', 'depth', 'size']
    mutate('wrapper_rows', rows)
    csv = '\n'.join([','.join(header)] + [','.join(str(r[k]) for k in header) for r in rows]) + '\n'
    put('wrapper_metrics', None, csv.encode())
    out['wrapper_rows'] = rows
    registrations = [dict(index=k, task=dict(cell_id=cid, replica=i, seed=c.task_seed(BINDING, cid, 'confirmation', i),
                                              phase='confirmation', kind='COST', ordinary=False, validate=False,
                                              cell=PR_cell(cid), expected_signal=records[cid]['signals']['corrected']))
                     for k, (cid, i) in enumerate((cid, i) for cid in UNION for i in range(32))]
    selection = dict(selected_ids=list(UNION), registrations=registrations, fixed_before_confirmation=True,
                     exploration_confirmation_pooled=False, signal_digest=n.d0.digest(old_signals['signals']))
    put('confirmation_selection', selection)
    put('old_completion', dict(SOURCE=c.PROVIDER, seed_source=c.SEED_SOURCE, input_SHA=BINDING['input'],
                               compiler_hash=n.COMPILER, selected_ids=list(UNION),
                               Taylor_orders=dict(confirmation={'0': 50, '2': 3}, exploration={'0': 40, '2': 2, '4': 1})))
    contract_value = dict(binding=dict(BINDING, selection_digest=n.d0.digest(out['confirmation_selection'])),
                          input_receipts=[dict(logical_name=name, **receipt(raw[role])) for name, role in (
                              ('signal_summary.json', 'old_signals'), ('wrapper_metrics.csv', 'wrapper_metrics'),
                              ('selection_before_confirmation.json', 'confirmation_selection'),
                              ('completion_audit.json', 'old_completion'))])
    put('old_D0_contract', contract_value)
    pairs = {cid: n.d0.paired_rows([r for r in out['wrapper_rows'] if r['phase'] == 'confirmation'
                                    and r['cell_id'] == cid], cell_id=cid, phase='confirmation',
                                   compiler_hash=n.COMPILER) for cid in UNION}
    exploration, confirmation = [], []
    for cid in OLD:
        if CELLS[cid]['method'] == 'B0I':
            continue
        for e in n.EPSILONS:
            for f in n.FACTORS:
                ok, _ = n.eligibility(usual[cid, e, f])
                metrics = {}
                if ok and CELLS[cid]['method'] == 'B1':
                    cell = CELLS[cid]
                    metrics = dict(RZ=dict(axis_means=dict(cosine=50000 + cell['prefix']*cell['q'],
                                                           sine=50001 + cell['prefix']*cell['q'])))
                exploration.append(dict(cell_id=cid, method=CELLS[cid]['method'], epsilon=e, factor=str(f),
                                        accounting=usual[cid, e, f], status='DERIVED_SAVED_SCALAR' if ok else
                                        'INELIGIBLE', metrics=metrics))
    for cid in UNION:
        for e in n.EPSILONS:
            for f in n.FACTORS:
                a = usual[cid, e, f]
                ok, shots = n.eligibility(a)
                metric = (dict(RZ=n.d0.weighted_metric(pairs[cid], {x: shots[n.CHANNEL[x]] for x in WAXES}, 'RZ',
                                                       point_rule='sum_axis_products')) if ok else {})
                confirmation.append(dict(cell_id=cid, method='B2', epsilon=e, factor=str(f), accounting=a,
                                         status='DERIVED_SAVED_SCALAR' if ok else 'INELIGIBLE', metrics=metric,
                                         sample_phase='confirmation', sample_count=32))
    D0_raw = put('D0_accounting', dict(H8_EXPLORATION92=exploration, H8_CONFIRMATION_UNION6=confirmation))
    put('D0_terminal', dict(status='COMPLETE_MANDATORY_STOP', first_stop=None,
                            output_file_receipts=[dict(name='candidate_accounting_v1.json', **receipt(D0_raw))]))
    put('D0_provenance', dict(SOURCE=c.D0['SOURCE'], fixed_D0_SOURCE=c.D0['analysis_source'],
                              contract_sha256=hashlib.sha256(raw['old_D0_contract']).hexdigest()))
    B0I_raw = put('B0I_accounting', [dict(cell_id=x, epsilon=e, factor=str(f), accounting=usual[x, e, f])
                                     for x in OLD_B0I for e in n.EPSILONS for f in n.FACTORS])
    put('B0I_file_receipts', dict(files=[dict(name='case_accounting_v1.json', **receipt(B0I_raw)),
                                         dict(name='case_scalar_signal_receipts_v1.json', **receipt(raw['old_B0I']))]))
    put('B1_models', deepcopy(MODEL))
    put('B1_anchors', dict(rows=[dict(method=m, prefix=p, q=q, axis=a, formula='2nd', saved_RZ=90000 + 10*p + q,
                                      runtime_profile_sha256=n.PROFILE, compiler_hash=n.COMPILER,
                                      control='symmetric_directional', origin_SOURCE=n.ANCHOR_SOURCE[m],
                                      evidence=dict(input_bound=True, compiler_bound=True))
                                 for m in ('B0', 'B0I') for p in (14, 21) for q in ((1, 8) if m == 'B0' else (1, 2, 4, 8))
                                 for a in WAXES]))
    put('B1_terminal', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=c.B1['SOURCE'], model_sha256=MODEL_SHA))
    put('B1_launcher_native_exit', dict(native_exit_code=0, wait_completed=True))
    proof, B0_registry = B2_proof()
    put('B2_saved_B0_registry', B0_registry)
    signed = dict(schema='H8_SIGNED_COMPONENT_VALIDATION_RESULT_V1', status='PASS_EMPIRICAL_NUMERICAL_CONSISTENCY',
                  first_stop=None, tolerances=dict(n.TOLERANCES), rows=signed_rows(records),
                  rigorous_bias_guarantee=None, post_execution_binding_verified=True,
                  input_receipts=[dict(role=role, path='ARTIFICIAL', **receipt(raw[role])) for role in c.SIGNED_ROLES],
                  **n.SIGNED)
    signed_raw = put('signed_axis_results', signed)
    put('signed_binding', dict(raw_axis_result_sha256=hashlib.sha256(signed_raw).hexdigest(),
                               raw_axis_result_bytes=len(signed_raw), SOURCE=n.SIGNED['SOURCE'],
                               plan_digest=n.SIGNED['plan_sha256'], grant_digest=n.SIGNED['grant_sha256']))
    out['B2'] = proof
    return out, raw


def write_snapshot(root, raw):
    files = []
    for role, data in raw.items():
        suffix = '.gz' if role == 'A_signals' else '.csv' if role == 'wrapper_metrics' else '.json'
        rel = 'inputs/' + role + suffix
        (root/'inputs').mkdir(parents=True, exist_ok=True)
        (root/rel).write_bytes(data)
        files.append(dict(role=role, snapshot_path=rel, **receipt(data)))
    return dict(schema='H8_NONCANCELLATION_INPUTS_V1', files=files, real_numerical_analysis_authorized=False)


def write_B2(root, proof):
    rows = []
    for role in c.B2_ROLES:
        rel = auditor.B2_RECEIPT_PATHS[role]
        path = root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = encoded(proof[role])
        path.write_bytes(data)
        rows.append(dict(role=role, relative_path=rel, **receipt(data)))
    return dict(required_SOURCE=n.B2['SOURCE'], scope=n.B2['scope'], receipts=rows, binding_complete=True)


_BASE = {}


def base():
    """Load the artificial bytes through the real adapter once, then deep-copy."""
    if not _BASE:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values, raw = objects()
            manifest = write_snapshot(root, raw)
            data, registry, _ = c.load_saved(manifest, root, ArtificialSavedPermit())
        data['kind'] = 'ARTIFICIAL'
        data['B2'] = values['B2']
        _BASE['data'] = data
    return deepcopy(_BASE['data'])


def run(data, stages=n.STAGES, **kwargs):
    with patch.object(n, 'MODEL_SHA', MODEL_SHA):
        return n.evaluate(data, REGISTRY, PERMIT, stages=stages, **kwargs)


def record(data, cid):
    return next(r for r in data['records'] if r['cell']['id'] == cid)


def resave(data, cid):
    """Recompute this artificial candidate's saved usual rows after a deliberate edit."""
    data['saved_usual'] = [r for r in data['saved_usual'] if r['cell_id'] != cid]
    data['saved_usual'] += [dict(cell_id=cid, epsilon=e, factor=str(f), accounting=n.d0.accounting(record(data, cid), e, f))
                            for e in n.EPSILONS for f in n.FACTORS]


def rebuild_det(data, cid, discard, PF):
    new = det_record(CELLS[cid], discard, PF)
    index = next(i for i, r in enumerate(data['records']) if r['cell']['id'] == cid)
    data['records'][index] = new
    rows = {(r['cell']['id'], r['axis']): r for r in signed_rows({cid: new, **{k: det_record(v) for k, v in CELLS.items()
                                                                              if k != cid}})}
    for r in data['signed']['rows']:
        if r['cell']['id'] == cid:
            r['signed_parts'] = rows[cid, r['axis']]['signed_parts']
    resave(data, cid)


def condition(result, e, f):
    return next(x for x in result['cost_stage']['comparisons'] if x['epsilon'] == e and x['factor'] == f)


class EnvelopeAndShotTests(unittest.TestCase):
    def test_cancellation_is_removed_per_axis(self):
        e = n.envelope('B0', dict(discard=.3, PF=-.29), .3 - .29)
        self.assertAlmostEqual(e['usual_signed'], .01)
        self.assertAlmostEqual(e['empirical_noncancellation'], .59)
        self.assertIsNone(e['rigorous_bias_bound'])

    def test_full_PF_uses_PF_only_and_equals_usual(self):
        e = n.envelope('B1', dict(PF=-.02), -.02)
        self.assertEqual(e['usual_signed'], e['empirical_noncancellation'])
        with self.assertRaisesRegex(ValueError, 'SIGNED_COMPONENT_SET'):
            n.envelope('B1', dict(discard=.01, PF=-.03), -.02)

    def test_PR_components_named_outer_PF_and_finite_RTE(self):
        e = n.envelope('B2', dict(outer_PF=.004, finite_RTE=-.001), .003)
        self.assertAlmostEqual(e['empirical_noncancellation'], .005)
        with self.assertRaisesRegex(ValueError, 'SIGNED_COMPONENT_SET'):
            n.envelope('B2', dict(discard=.004, PF=-.001), .003)

    def test_components_must_close_to_saved_total(self):
        with self.assertRaisesRegex(ValueError, 'DO_NOT_CLOSE'):
            n.envelope('B0', dict(discard=.1, PF=.1), .3)

    def test_nonfinite_and_overflow_fail_closed(self):
        for parts, total, reason in [(dict(discard=float('nan'), PF=0.), 0., 'NONFINITE_NUMBER'),
                                     (dict(discard=float('inf'), PF=0.), 0., 'NONFINITE_NUMBER'),
                                     (dict(discard=1e308, PF=-1e308), 0., 'NONFINITE_NONCANCELLATION_ENVELOPE'),
                                     (dict(discard=True, PF=0.), 0., 'FINITE_REAL_NUMBER_REQUIRED')]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                n.envelope('B0', parts, total)

    def test_usual_accounting_is_the_frozen_D0_text(self):
        r = det_record(CELLS['B0_p7_q1_2nd'])
        for e in n.EPSILONS:
            for f in n.FACTORS:
                self.assertEqual(n.frozen_accounting(r, e, f), n.d0.accounting(r, e, f))

    def test_only_the_bias_magnitude_changes(self):
        r = PR_record(UNION[0])
        magnitude = dict(real=.003, imag=.0021)
        for e in n.EPSILONS:
            for f in n.FACTORS:
                got = n.frozen_accounting(r, e, f, magnitude)
                want = n.d0._rule.u_aware_shots(magnitude, {a: f*r['allowance'][a] for a in n.AXES}, epsilon=e,
                    log_B_upper=r['log_B'] + f*r['allowance']['log_B_margin'], evidence_kind='EMPIRICAL',
                    evidence_ref=r['cell']['id'] + '_signal.json', alpha_axis=.025)
                for key in ('axes', 'N_total', 'eligibility_under_declared_allowance'):
                    self.assertEqual(got[key], want[key])

    def test_shot_integer_formula_epsilon_axis_and_alpha(self):
        r = det_record(CELLS['B0_p7_q1_2nd'])
        r['allowance'] = allowance(U, 0.)
        magnitude = dict(real=.002, imag=.001)
        a = n.frozen_accounting(r, .05, 1., magnitude)
        for axis in n.AXES:
            headroom = .05/math.sqrt(2) - magnitude[axis] - U
            self.assertEqual(a['axes'][axis]['shots'], max(1, math.ceil(2*1.**2/headroom**2*math.log(2/.025))))
        self.assertEqual(a['N_total'], sum(a['axes'][x]['shots'] for x in n.AXES))

    def test_factor_one_and_ten_are_multipliers_not_tenth(self):
        r = det_record(CELLS['B0_p7_q1_2nd'])
        r['allowance'] = allowance(1e-5, 1e-6)
        magnitude = dict(real=.002, imag=.001)
        one, ten = (n.frozen_accounting(r, .01, f, magnitude) for f in (1., 10.))
        self.assertEqual(one['axes']['real']['allowance'], 1e-5)
        self.assertEqual(ten['axes']['real']['allowance'], 10.*1e-5)
        self.assertLess(one['N_total'], ten['N_total'])
        with self.assertRaisesRegex(ValueError, 'REGISTERED_PRECISION_AND_FACTOR'):
            n.frozen_accounting(r, .01, .1, magnitude)

    def test_axis_wise_eligibility_and_unresolved_states(self):
        r = det_record(CELLS['B0_p7_q1_2nd'])
        self.assertEqual(n.eligibility(n.frozen_accounting(r, .01, 1., dict(real=1e-4, imag=.008)))[0], False)
        self.assertEqual(n.eligibility(n.frozen_accounting(r, .01, 1., dict(real=1e-4, imag=1e-4)))[0], True)
        r['allowance'] = allowance(1e-3)
        boundary = .01/math.sqrt(2)
        undetermined = n.frozen_accounting(r, .01, 1., dict(real=boundary, imag=1e-4))
        self.assertEqual(undetermined['eligibility_under_declared_allowance'], 'UNDETERMINED')
        self.assertIsNone(n.eligibility(undetermined)[0])

    def test_u_over_headroom_guard_is_unresolved_not_eligible(self):
        r = det_record(CELLS['B0_p7_q1_2nd'])
        r['allowance'] = allowance(1e-4)
        a = n.frozen_accounting(r, .01, 1., dict(real=.0065, imag=.0065))
        self.assertEqual(a['eligibility_under_declared_allowance'], 'ELIGIBLE')
        self.assertFalse(a['cost_acquisition_eligible'])
        self.assertIsNone(n.eligibility(a)[0])


class EvaluationTests(unittest.TestCase):
    def test_shots_only_full_domain_and_labels(self):
        result = run(base(), stages=n.STAGES[:1])
        self.assertEqual(result['status'], 'COMPLETE_NONCANCELLATION_SHOTS_ONLY_MANDATORY_STOP')
        self.assertEqual((len(result['accounting_rows']), len(result['axis_envelopes'])), (1900, 380))
        self.assertEqual(result['gates']['saved_usual_rows_reproduced'], 1900)
        self.assertEqual(result['cost_stage']['status'], 'NOT_EXECUTED')
        self.assertIsNone(result['cost_stage']['PR_reduction_percent'])
        self.assertIsNone(result['rigorous_bias_guarantee'])
        self.assertEqual(sum(r['reconstructed_from_saved_endpoints'] for r in result['axis_envelopes']), 24)
        for row in result['axis_envelopes']:
            self.assertGreaterEqual(row['empirical_noncancellation_bias'], row['usual_signed_bias'])
            self.assertIsNone(row['rigorous_bias_bound'])
        for row in result['accounting_rows']:
            self.assertIsNone(row['rigorous_bias_guarantee'])
            self.assertEqual(row['bias_kind'], 'EMPIRICAL_NONCANCELLATION_DIAGNOSTIC')
            if row['eligible'] and row['usual_saved']['eligibility'] == 'ELIGIBLE':
                self.assertGreaterEqual(F(row['shot_ratio_to_usual']), 1)

    def test_cancellation_changes_eligibility_but_usual_row_is_unchanged(self):
        data = base()
        rebuild_det(data, 'B0_p7_q1_2nd', dict(real=.02, imag=.001), dict(real=-.0199, imag=-.0009))
        rows = {(r['cell_id'], r['epsilon'], r['factor']): r for r in run(data, n.STAGES[:1])['accounting_rows']}
        row = rows['B0_p7_q1_2nd', .01, 1.]
        self.assertEqual(row['usual_saved']['eligibility'], 'ELIGIBLE')
        self.assertEqual(row['eligibility'], 'INELIGIBLE')

    def test_saved_usual_mismatch_or_missing_row_stops(self):
        data = base()
        data['saved_usual'][0]['accounting']['N_total'] = 12345
        with self.assertRaisesRegex(ValueError, 'SAVED_USUAL_ACCOUNTING_NOT_REPRODUCED'):
            run(data, n.STAGES[:1])
        data = base()
        data['saved_usual'].pop()
        with self.assertRaisesRegex(ValueError, 'EXACT_SAVED_USUAL_ROWS'):
            run(data, n.STAGES[:1])

    def test_missing_or_failed_signed_axis_is_kept_unresolved_and_nulls_domain(self):
        for cid, old_null in (('B0_p7_q1_2nd', True), ('B0_p8_q1_2nd', False)):
            for mode in ('missing', 'failed'):
                data = base()
                if mode == 'missing':
                    data['signed']['rows'] = [r for r in data['signed']['rows']
                                              if not (r['cell']['id'] == cid and r['axis'] == 'imag')]
                else:
                    row = next(r for r in data['signed']['rows'] if r['cell']['id'] == cid and r['axis'] == 'imag')
                    row.update(status='FAIL_OR_MISSING', first_failure='ARTIFICIAL')
                result = run(data)
                rows = [r for r in result['accounting_rows'] if r['cell_id'] == cid]
                self.assertEqual(len(rows), 10)
                self.assertTrue(all(r['eligibility'] == 'UNRESOLVED_COMPONENTS' and r['eligible'] is None for r in rows))
                for comparison in result['cost_stage']['comparisons']:
                    scopes = comparison['scopes']
                    self.assertEqual(scopes['strengthened184']['deterministic']['status'], 'UNRESOLVED_FULL_DOMAIN')
                    self.assertIsNone(scopes['strengthened184']['PR_reduction_percent_exact'])
                    if old_null:
                        self.assertIsNone(scopes['old32']['PR_reduction_percent_exact'])
                        self.assertIn(cid, scopes['old32']['deterministic']['unresolved'])
                    self.assertIn(cid, scopes['strengthened184']['deterministic']['unresolved'])

    def test_duplicate_missing_or_extra_records_stop(self):
        data = base()
        data['records'].append(deepcopy(data['records'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_RECORD'):
            run(data, n.STAGES[:1])
        data = base()
        data['records'].pop(0)
        with self.assertRaisesRegex(ValueError, 'EXACT_184_PLUS_UNION6'):
            run(data, n.STAGES[:1])
        data = base()
        data['confirmation']['selected_ids'] = UNION[:5]
        with self.assertRaisesRegex(ValueError, 'FIXED_CONFIRMATION_UNION6'):
            run(data, n.STAGES[:1])

    def test_PR_component_failure_is_unresolved_and_nulls_PR_ratio(self):
        data = base()
        record(data, UNION[0])['error_decomposition']['signed_parts']['outer_PF']['real'] += 1e-6
        result = run(data)
        self.assertEqual(result['gates']['PR_component_checks'][UNION[0]]['status'], 'FAIL_OR_MISSING')
        for comparison in result['cost_stage']['comparisons']:
            for scope in comparison['scopes'].values():
                self.assertEqual(scope['confirmed_PR']['status'], 'UNRESOLVED_FULL_DOMAIN')
                self.assertIsNone(scope['PR_over_deterministic_exact'])

    def test_PR_forward_and_reference_checks(self):
        for field in ('independent_forward_signals', 'reference'):
            data = base()
            target = record(data, UNION[1])
            if field == 'reference':
                target['reference']['imag'] += 1e-6
                target['error_decomposition']['total_signed']['imag'] -= 1e-6
                target['error_decomposition']['signed_parts']['outer_PF']['imag'] -= 1e-6
                resave(data, UNION[1])
            else:
                target[field]['exact_tail']['real'] += 1e-6
            result = run(data, n.STAGES[:1])
            first = result['gates']['PR_component_checks'][UNION[1]]['first_failure']
            self.assertIn('reference' if field == 'reference' else 'forward', first)

    def test_PR_normalization_binding_stops(self):
        data = base()
        record(data, UNION[2])['log_B'] += 1e-12
        resave(data, UNION[2])
        with self.assertRaisesRegex(ValueError, 'PR_SAVED_NORMALIZATION_LOG_BINDING'):
            run(data, n.STAGES[:1])

    def test_nonfinite_validated_component_stops(self):
        data = base()
        data['signed']['rows'][0]['signed_parts']['PF'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'NONFINITE_NUMBER'):
            run(data, n.STAGES[:1])

    def test_registered_stage_set_and_permit_kind(self):
        with self.assertRaisesRegex(ValueError, 'REGISTERED_STAGE_SET'):
            run(base(), ('COST_COMPARISON',))
        data = base()
        data['kind'] = 'SAVED_H8'
        with self.assertRaisesRegex(ValueError, 'DATA_KIND_AUTHORIZATION'):
            run(data, n.STAGES[:1])
        with self.assertRaisesRegex(ValueError, 'APPROVED_GRANT_PERMIT_REQUIRED'):
            c.Permit(synthetic=False, scope=n.SCOPE)

    def test_rare_orders_are_unobserved_not_zero_cost(self):
        layers = run(base(), n.STAGES[:1])['rare_layers']
        fourth = next(x for x in layers['confirmation'] if x['order'] == 4)
        self.assertEqual((fourth['status'], fourth['population_layer_cost']), ('UNOBSERVED', None))
        self.assertFalse(fourth['mean_cost_upper_bound_proved'])

    def test_deterministic_and_input_is_not_mutated(self):
        data = base()
        before = deepcopy(data)
        first, second = run(data), run(data)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(data['records'], before['records'])

    def test_guard_interrupt_propagates(self):
        def guard():
            raise InterruptedError('OWN_STOP')
        with self.assertRaisesRegex(InterruptedError, 'OWN_STOP'):
            run(base(), n.STAGES[:1], guard=guard)


class ReviewFindingTests(unittest.TestCase):
    """Two review findings (2026-10-17): stage scope of PR paired gates and S4 label."""

    def test_PR_paired_cost_gates_are_cost_stage_only_and_say_so(self):
        data = base()
        item = next(x for x in data['PR_saved_metrics'] if x['status'] == 'DERIVED_SAVED_SCALAR')
        item['metrics']['RZ']['G_standard_error'] += 1.
        shots = run(data, n.STAGES[:1])
        for gate in ('PR_saved_paired_metric_reproduction', 'PR_confirmation_n32_pairing'):
            self.assertEqual(shots['gates'][gate], dict(stage='COST_COMPARISON', executed=False,
                             status='NOT_EXECUTED_IN_NONCANCELLATION_SHOTS_STAGE', checked=0))
        with self.assertRaisesRegex(ValueError, 'PR_SAVED_PAIRED_COST_NOT_REPRODUCED'):
            run(data)
        gates = run(base())['gates']
        derived = sum(1 for x in base()['PR_saved_metrics'] if x['status'] == 'DERIVED_SAVED_SCALAR')
        self.assertEqual(gates['PR_saved_paired_metric_reproduction'],
                         dict(stage='COST_COMPARISON', executed=True, status='REPRODUCED', checked=derived))
        self.assertEqual(gates['PR_confirmation_n32_pairing'],
                         dict(stage='COST_COMPARISON', executed=True, status='PASS', checked=6))

    def test_S4_model_is_not_labelled_validated_without_bound_B2_PASS(self):
        shots = run(base(), n.STAGES[:1])
        self.assertNotIn('S4_model_validated_on', shots['lineage_limitations'])
        self.assertEqual(shots['lineage_limitations']['S4_model_validation_scope'],
                         'PLANNED_FOUR_REGISTERED_B2_AXES_VALIDATED_ONLY_WHEN_B2_FULL_PASS_BOUND')
        self.assertEqual(shots['S4_model_status'], 'UNVALIDATED_B2_FULL_PASS_NOT_BOUND_S4_MODEL_NOT_USED')
        self.assertNotIn('FIXED_B1_S4_MODEL', json.dumps(shots))
        cost = run(base())
        self.assertEqual(cost['S4_model_status'], 'VALIDATED_ON_FOUR_REGISTERED_B2_AXES_ONLY_B2_FULL_PASS_BOUND')


class B2GateTests(unittest.TestCase):
    def broken(self, edit):
        data = base()
        edit(data['B2'], data)
        return data

    def test_B2_gate_runs_before_any_accounting(self):
        data = self.broken(lambda p, d: p['driver_terminal'].update(status='STOP'))
        with patch.object(n, 'frozen_accounting', side_effect=AssertionError('MUST_NOT_ACCOUNT')):
            with self.assertRaisesRegex(ValueError, 'B2_DRIVER_TERMINAL_COMPLETE_REQUIRED'):
                run(data)

    def test_incomplete_or_failed_B2_is_rejected(self):
        cases = {
            'partial_three_axes': (lambda p, d: p['driver_terminal']['axes'].pop(), 'EXACT_B2_FOUR_AXES'),
            'axis_not_pass': (lambda p, d: p['driver_terminal']['axes'][1]['model_check'].update(PASS=False), 'B2_AXIS_NOT_PASS'),
            'three_percent': (lambda p, d: p['driver_terminal']['axes'][0].update(measured_RZ=1800000), 'B2_AXIS_NOT_PASS'),
            'driver_native_exit': (lambda p, d: p['launcher_terminal'].update(native_exit_code=-9), 'NATIVE_EXIT_CLEANUP'),
            'monitor_gap': (lambda p, d: p['launcher_terminal'].update(max_monitor_gap_seconds=6.), 'NATIVE_EXIT_CLEANUP'),
            'final_gate': (lambda p, d: p['final_gate'].update(status='STOP'), 'B2_FINAL_GATE_PASS_REQUIRED'),
            'launcher_wait': (lambda p, d: p['witness_native_exit'].update(native_exit_code=1), 'B2_LAUNCHER_NATIVE_WAIT'),
            'other_SOURCE': (lambda p, d: p['driver_terminal'].update(SOURCE='0'*40), 'B2_DRIVER_TERMINAL'),
            'saved_B0': (lambda p, d: p['saved_B0_acceptance'].update(accepted_by_new_grant=False), 'SAVED_B0_ACCEPTANCE'),
            'refit': (lambda p, d: p['driver_terminal'].update(fit_performed=True), 'B2_DRIVER_TERMINAL'),
        }
        for name, (edit, reason) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, reason):
                run(self.broken(edit))


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(base())

    def test_cost_stage_complete_old32_and_184_separate(self):
        r = self.result
        self.assertEqual(r['status'], 'COMPLETE_NONCANCELLATION_SHOTS_AND_COST_COMPARISON_MANDATORY_STOP')
        self.assertEqual(len(r['cost_stage']['comparisons']), 10)
        rows = {(x['cell_id'], x['epsilon'], x['factor']): x for x in r['accounting_rows']}
        checked = 0
        for comparison in r['cost_stage']['comparisons']:
            e, f = comparison['epsilon'], comparison['factor']
            for name, ids in (('old32', OLD), ('strengthened184', list(CELLS))):
                scope = comparison['scopes'][name]
                eligible = [rows[x, e, f] for x in ids if rows[x, e, f]['eligible']]
                det = scope['deterministic']
                if det['best'] is None:
                    continue
                self.assertIn(det['best']['cell_id'], ids)
                self.assertEqual(F(det['best']['G_exact']), min(F(x['G_exact']) for x in eligible))
                pr = scope['confirmed_PR']['best']
                if pr is not None:
                    ratio = F(pr['G_exact'])/F(det['best']['G_exact'])
                    self.assertEqual(scope['PR_over_deterministic_exact'], str(ratio))
                    self.assertEqual(scope['PR_reduction_percent_exact'], str(100*(1-ratio)))
                    checked += 1
            before, after = (comparison['scopes'][s]['PR_reduction_percent_exact'] for s in ('old32', 'strengthened184'))
            if before is not None and after is not None:
                self.assertEqual(comparison['reduction_change_percentage_points_exact'], str(F(after)-F(before)))
        self.assertGreater(checked, 0)

    def test_measured_first_labels_and_model_values(self):
        units = {(u['cell_id'], u['axis']): u for u in self.result['cost_stage']['unit_costs']}
        self.assertEqual(units['B0_p15_q1_4th', 'cosine']['status'], 'B2_MEASURED_REGISTERED_S4_AXIS')
        self.assertEqual(units['B0I_p21_q8_4th', 'sine']['RZ'], '1852000')
        self.assertEqual(units['B0_p14_q1_2nd', 'cosine']['status'], 'B1_REGISTERED_ANCHOR_MEASURED')
        self.assertEqual(units['B0I_p14_q2_2nd', 'sine']['RZ'], str(90000 + 140 + 2))
        legacy = units['B1_p28_q1_2nd', 'sine']
        self.assertEqual((legacy['status'], legacy['RZ']), ('LEGACY_FULL_PF_SAVED_N1_MEASURED', str(50000 + 28 + 1)))
        s2 = units['B0_p8_q1_2nd', 'cosine']
        self.assertEqual((s2['status'], s2['RZ'], s2['prefix_outside_B1_fit_range']),
                         ('FIXED_B1_S2_MODEL_ESTIMATE', str(1500 + 1300 + 3600*8), True))
        s4 = units['B0_p16_q2_4th', 'cosine']
        self.assertEqual((s4['status'], s4['RZ'], s4['prefix_outside_B1_fit_range']),
                         ('FIXED_B1_S4_MODEL_ESTIMATE_B2_FOUR_AXES_ONLY', str(1500 + 6*(1300 + 3600*16)), False))
        self.assertFalse(s4['full_grid_error_guaranteed'])
        self.assertNotIn(('B0_p7_q2_2nd', 'cosine'), [k for k, v in units.items() if v['measured']])

    def test_PR_paired_reweighting_and_sample_SE(self):
        rows = [x for x in self.result['accounting_rows'] if x['method'] == 'B2' and x['eligible']]
        self.assertTrue(rows)
        data = base()
        for row in rows[:3]:
            pairs = n.d0.paired_rows([w for w in data['wrappers'] if w['phase'] == 'confirmation'
                                      and w['cell_id'] == row['cell_id']], cell_id=row['cell_id'],
                                     phase='confirmation', compiler_hash=n.COMPILER)
            totals = [row['shots']['real']*p['cosine']['RZ'] + row['shots']['imag']*p['sine']['RZ'] for p in pairs]
            self.assertEqual(len(totals), 32)
            self.assertEqual(row['G_exact'], str(F(sum(totals), 32)))
            self.assertEqual(row['G_standard_error'], statistics.stdev(totals)/math.sqrt(32))
            self.assertEqual(row['cost_status'], 'PR_FINITE_SAMPLE_MEAN_CONFIRMATION_N32')
            self.assertFalse(row['population_mean_guaranteed'])

    def test_ineligible_rows_never_get_zero_cost(self):
        for row in self.result['accounting_rows']:
            if row['eligible'] is not True:
                self.assertIsNone(row['G_exact'])
                self.assertEqual(row['cost_status'], 'NOT_ELIGIBLE_OR_UNRESOLVED_NO_COST')

    def test_minimum_ties_unresolved_and_missing(self):
        rows = [dict(cell_id='b', eligible=True, G_exact='10', cost_status='X'),
                dict(cell_id='a', eligible=True, G_exact='20/2', cost_status='X'),
                dict(cell_id='c', eligible=True, G_exact='11', cost_status='X'),
                dict(cell_id='d', eligible=False, G_exact=None)]
        best = n.minimum(rows)
        self.assertEqual((best['best']['cell_id'], best['exact_ties']), ('a', ['a', 'b']))
        self.assertEqual(n.minimum(rows + [dict(cell_id='e', eligible=None, G_exact=None)])['status'],
                         'UNRESOLVED_FULL_DOMAIN')
        self.assertEqual(n.minimum(rows[3:])['status'], 'NO_ELIGIBLE_CANDIDATE')
        self.assertEqual(n.minimum([dict(cell_id='f', eligible=True, G_exact=None)])['status'],
                         'UNRESOLVED_ELIGIBLE_COST')

    def test_PR_saved_metric_mismatch_stops(self):
        data = base()
        item = next(x for x in data['PR_saved_metrics'] if x['status'] == 'DERIVED_SAVED_SCALAR')
        item['metrics']['RZ']['G_standard_error'] += 1.
        with self.assertRaisesRegex(ValueError, 'PR_SAVED_PAIRED_COST_NOT_REPRODUCED'):
            run(data)

    def test_confirmation_n32_seed_and_no_exploration_pooling(self):
        data = base()
        data['wrappers'] = [w for w in data['wrappers'] if not (w['cell_id'] == UNION[0] and w['replica'] == 31
                                                                and w['phase'] == 'confirmation')]
        with self.assertRaisesRegex(ValueError, 'CONFIRMATION_N32_PAIRS_REQUIRED'):
            run(data)
        data = base()
        relabeled = [dict(w, phase='confirmation') for w in data['wrappers']
                     if w['cell_id'] == UNION[0] and w['phase'] == 'exploration']
        data['wrappers'] += relabeled
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_PAIRED_AXIS'):
            run(data)
        data = base()
        target = next(w for w in data['wrappers'] if w['phase'] == 'confirmation' and w['cell_id'] == UNION[1])
        for w in data['wrappers']:
            if w['phase'] == 'confirmation' and w['cell_id'] == UNION[1] and w['replica'] == target['replica']:
                w['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'CONFIRMATION_SEED_BINDING'):
            run(data)

    def test_missing_PR_pairs_are_missing_cost_not_zero_and_null_ratio(self):
        data = base()
        data['wrappers'] = [w for w in data['wrappers'] if not (w['cell_id'] == UNION[0] and w['phase'] == 'confirmation')]
        for item in data['PR_saved_metrics']:
            if item['cell_id'] == UNION[0] and item['status'] == 'DERIVED_SAVED_SCALAR':
                item.update(status='MISSING', metrics={})
        result = run(data)
        rows = [x for x in result['accounting_rows'] if x['cell_id'] == UNION[0] and x['eligible']]
        self.assertTrue(rows)
        self.assertTrue(all(x['G_exact'] is None and x['G_standard_error'] is None and
                            x['cost_status'] == 'PR_PAIRED_COST_MISSING' for x in rows))
        nulls = [cmp for cmp in result['cost_stage']['comparisons']
                 if cmp['scopes']['old32']['confirmed_PR']['status'] == 'UNRESOLVED_ELIGIBLE_COST']
        self.assertTrue(nulls)
        self.assertTrue(all(cmp['scopes'][s]['PR_reduction_percent_exact'] is None for cmp in nulls for s in cmp['scopes']))

    def test_saved_eligible_PR_cost_missing_with_pairs_stops(self):
        data = base()
        item = next(x for x in data['PR_saved_metrics'] if x['status'] == 'DERIVED_SAVED_SCALAR')
        item.update(status='MISSING', metrics={})
        with self.assertRaisesRegex(ValueError, 'PR_SAVED_ELIGIBLE_COST_MISSING'):
            run(data)

    def test_full_PF_saved_mean_and_anchor_lineage(self):
        data = base()
        data['full_PF_saved_axis_means']['B1_p28_q1_2nd'] = [dict(cosine=1, sine=1)]
        with self.assertRaisesRegex(ValueError, 'LEGACY_FULL_PF_SAVED_MEAN_MISMATCH'):
            run(data)
        data = base()
        data['B1_anchors']['rows'][0]['origin_SOURCE'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'B1_ANCHOR_LINEAGE'):
            run(data)
        data = base()
        data['B1_model']['models'][0]['coefficients'][0] = '1'
        with self.assertRaisesRegex(ValueError, 'FIXED_B1_MODEL'):
            run(data)


class ContractTests(unittest.TestCase):
    @contextmanager
    def prepared(self, stages=n.STAGES[:1]):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, input_root = root/'source', root/'inputs'
            values, raw = objects()
            manifest = write_snapshot(input_root, raw)
            b2 = write_B2(root/'B2', values['B2'])
            files = []
            for path in c.SOURCE_PATHS:
                f = source/path
                f.parent.mkdir(parents=True, exist_ok=True)
                data = (json.dumps(manifest).encode() if path == c.MANIFEST else (ROOT/path).read_bytes())
                f.write_bytes(data)
                files.append(dict(path=path, git_blob=c.git_blob(data), **receipt(data)))
            plan = dict(schema=c.PLAN_SCHEMA, scope=n.SCOPE, SOURCE='a'*40, base_commits=dict(c.BASE),
                        science=deepcopy(c.SCIENCE), accounting=deepcopy(c.ACCOUNTING), bias_kinds=dict(c.BIAS_KINDS),
                        prohibited=dict(c.PROHIBITED), environment=c.environment(), source_files=files,
                        input_manifest=manifest, input_manifest_sha256=c.digest(manifest), B2_dependency=b2,
                        resource_estimates=dict(new_output_bytes=16*2**20, new_output_inodes=4))
            grant = dict(schema=c.GRANT_SCHEMA, scope=n.SCOPE, SOURCE='a'*40, approved=True, runtime_authorization=True,
                         authorized_stages=list(stages), one_shot=True, automatic_retry=False,
                         next_stage_authorized=False, authorization_evidence='ARTIFICIAL_UNITTEST_APPROVAL',
                         input_root=str(input_root), output_directory=str(root/'output'), B2_receipt_root=str(root/'B2'),
                         fresh_gate=dict(source_checked=True, environment_checked=True,
                                         resources_noninterference_checked=True, B2_noninterference_checked=True,
                                         checked_UTC=datetime.now(timezone.utc).isoformat()))
            grant['plan_sha256'] = c.digest(plan)
            yield root, source, plan, grant

    def test_artificial_contract_authorizes_only_with_explicit_grant(self):
        with self.prepared() as (_, source, plan, grant):
            permit, *_ = c.authorize(plan, grant, source_root=source)
            self.assertFalse(permit.synthetic)
            for flag in ('approved', 'runtime_authorization'):
                draft = dict(grant, **{flag: False})
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED'):
                    c.authorize(plan, draft, source_root=source)

    def test_invalid_grant_rejected_before_any_input_or_output(self):
        with patch.object(c, 'load_saved', side_effect=AssertionError('MUST_NOT_READ')):
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_NUMERICAL_EXECUTION'):
                runner.execute({}, dict(approved=False, runtime_authorization=False))

    def test_cost_stage_requires_bound_B2_full_PASS_receipts(self):
        with self.prepared(n.STAGES) as (_, source, plan, grant):
            c.authorize(plan, grant, source_root=source)
            plan['B2_dependency'] = dict(plan['B2_dependency'], binding_complete=False)
            grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'B2_FULL_PASS_RECEIPT_BINDING_REQUIRED'):
                c.authorize(plan, grant, source_root=source)
            plan['B2_dependency'] = dict(plan['B2_dependency'], binding_complete=True, required_SOURCE='0'*40)
            grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'B2_FULL_PASS_RECEIPT_BINDING_REQUIRED'):
                c.authorize(plan, grant, source_root=source)

    def test_frozen_scope_relaxations_rejected_even_with_rehashed_grant(self):
        edits = {
            'factor_tenth': lambda p, g: p['accounting'].update(allowance_factors=[1., .1]),
            'alpha': lambda p, g: p['accounting'].update(alpha_axis=.05),
            'guard': lambda p, g: p['accounting'].update(maximum_u_over_headroom=.1),
            'tolerance': lambda p, g: p['accounting']['closure'].update(abs_tol=1e-9),
            'science': lambda p, g: p['science'].update(deterministic_candidates=183),
            'rigorous': lambda p, g: p['bias_kinds'].update(rigorous_bias_guarantee='abs(total)'),
            'prohibited': lambda p, g: p['prohibited'].update(PR_sampling=1),
            'base': lambda p, g: p['base_commits'].update(signed_result='0'*40),
            'stages': lambda p, g: g.update(authorized_stages=['COST_COMPARISON']),
            'retry': lambda p, g: g.update(automatic_retry=True),
            'evidence': lambda p, g: g.update(authorization_evidence=' '),
            'B2_noninterference': lambda p, g: g['fresh_gate'].update(B2_noninterference_checked=False),
            'stale': lambda p, g: g['fresh_gate'].update(checked_UTC='2020-01-01T00:00:00+00:00'),
            'environment': lambda p, g: p['environment'].update(statistics_sha256='0'*64),
        }
        for name, edit in edits.items():
            with self.prepared() as (_, source, plan, grant):
                edit(plan, grant)
                grant['plan_sha256'] = c.digest(plan)
                with self.subTest(case=name), self.assertRaises(ValueError):
                    c.authorize(plan, grant, source_root=source)

    def test_source_manifest_paths_and_output_exclusivity(self):
        with self.prepared() as (root, source, plan, grant):
            (source/c.SOURCE_PATHS[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'FIXED_INPUT'):
                c.authorize(plan, grant, source_root=source)
        with self.prepared() as (root, source, plan, grant):
            plan['source_files'][1]['git_blob'] = '0'*40
            grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'SOURCE_GIT_BLOB'):
                c.authorize(plan, grant, source_root=source)
        with self.prepared() as (root, source, plan, grant):
            plan['input_manifest']['files'].pop()
            plan['input_manifest_sha256'] = c.digest(plan['input_manifest'])
            grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'FROZEN_INPUT_MANIFEST'):
                c.authorize(plan, grant, source_root=source)
        with self.prepared() as (root, source, plan, grant):
            Path(grant['output_directory'] + '.one_shot_lock').write_text('used')
            with self.assertRaisesRegex(ValueError, 'EXCLUSIVE'):
                c.authorize(plan, grant, source_root=source)
            grant['output_directory'] = str(Path(grant['input_root'])/'out')
            with self.assertRaisesRegex(ValueError, 'EXCLUSIVE'):
                c.authorize(plan, grant, source_root=source)
            for path in ('/tmp/forbidden', str(root/'..'/'escape')):
                with self.assertRaises(ValueError):
                    c.home_path(path)

    def test_strict_JSON_and_bytes(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                c.strict(raw)

    def test_adapter_path_matches_direct_artificial_evaluation(self):
        data = base()
        self.assertEqual(len(data['records']), 190)
        self.assertEqual(len(data['saved_usual']), 1900)
        self.assertEqual(len(data['PR_saved_metrics']), 60)
        result = run(data)
        self.assertEqual(result['status'], 'COMPLETE_NONCANCELLATION_SHOTS_AND_COST_COMPARISON_MANDATORY_STOP')

    def test_adapter_lineage_tampering_rejected(self):
        cases = {
            'D0_wrapper_receipt': (lambda role, v: role == 'old_D0_contract' and v['input_receipts'][1].update(sha256='0'*64), 'D0_INPUT_RECEIPT'),
            'selection_digest': (lambda role, v: role == 'old_D0_contract' and v['binding'].update(selection_digest='0'*64), 'D0_SELECTION_SIGNAL_BINDING'),
            'pooled_selection': (lambda role, v: role == 'confirmation_selection' and v.update(exploration_confirmation_pooled=True), 'D0_SELECTION_SIGNAL_BINDING'),
            'signed_raw_sha': (lambda role, v: role == 'signed_binding' and v.update(raw_axis_result_sha256='0'*64), 'SIGNED_RUN01_BINDING'),
            'signed_input_role': (lambda role, v: role == 'signed_axis_results' and v['input_receipts'][1].update(sha256='0'*64), 'SIGNED_INPUT_ROLE'),
            'signed_not_pass': (lambda role, v: role == 'signed_axis_results' and v.update(status='STOP'), 'SIGNED_RUN01_BINDING'),
            'seed': (lambda role, v: role == 'confirmation_selection' and v['registrations'][0]['task'].update(seed=1), 'CONFIRMATION_SEED_DERIVATION'),
            'D0_terminal': (lambda role, v: role == 'D0_terminal' and v.update(status='FAIL_CLOSED_STOP'), 'D0_RESULT_LINEAGE'),
            'B0I_receipt': (lambda role, v: role == 'B0I_file_receipts' and v['files'][0].update(sha256='0'*64), 'OLD_B0I_ACCOUNTING_LINEAGE'),
            'B1_exit': (lambda role, v: role == 'B1_launcher_native_exit' and v.update(native_exit_code=1), 'B1_RESULT_LINEAGE'),
            'completion': (lambda role, v: role == 'old_completion' and v.update(compiler_hash='0'*64), 'RESOURCE_MAP_COMPLETION_BINDING'),
            'A_export': (lambda role, v: role == 'A_export' and v.update(original_json_sha256='0'*64), 'A_EXPORT_BYTES'),
            'reference_certified': (lambda role, v: role == 'full_reference' and v.update(numerical_allowance_certified=True), 'REFERENCE_CERTIFICATION'),
        }
        for name, (mutate, reason) in cases.items():
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, raw = objects(mutate)
                manifest = write_snapshot(root, raw)
                with self.subTest(case=name), self.assertRaisesRegex(ValueError, reason):
                    c.load_saved(manifest, root, ArtificialSavedPermit())

    def test_input_byte_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, raw = objects()
            manifest = write_snapshot(root, raw)
            (root/'inputs/B1_terminal.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'FIXED_INPUT_BYTES_SHA'):
                c.load_saved(manifest, root, ArtificialSavedPermit())

    def test_runner_end_to_end_shots_only_then_one_shot_reuse_rejected(self):
        with self.prepared() as (root, source, plan, grant), patch.object(c, 'ROOT', source):
            terminal = runner.execute(plan, grant)
            self.assertEqual(terminal['status'], 'COMPLETE_NONCANCELLATION_SHOTS_ONLY_MANDATORY_STOP')
            self.assertEqual((terminal['child_processes_started'], terminal['cost_stage']), (0, 'NOT_EXECUTED'))
            result = json.loads((root/'output/diagnostic_result.json').read_bytes())
            self.assertTrue(result['post_execution_binding_verified'])
            self.assertEqual(len(result['input_receipts']), len(plan['input_manifest']['files']))
            self.assertTrue((root/'output.one_shot_lock').is_file())
            with self.assertRaisesRegex(ValueError, 'EXCLUSIVE'):
                runner.execute(plan, grant)

    def test_runner_end_to_end_with_bound_B2_cost_stage(self):
        with self.prepared(n.STAGES) as (root, source, plan, grant), patch.object(c, 'ROOT', source), \
                patch.object(n, 'MODEL_SHA', MODEL_SHA):
            terminal = runner.execute(plan, grant)
            self.assertEqual(terminal['status'], 'COMPLETE_NONCANCELLATION_SHOTS_AND_COST_COMPARISON_MANDATORY_STOP')
            result = json.loads((root/'output/diagnostic_result.json').read_bytes())
            self.assertEqual(len(result['input_receipts']), len(plan['input_manifest']['files']) + 6)

    def test_runner_STOP_keeps_lock_reason_and_writes_no_result(self):
        with self.prepared() as (root, source, plan, grant), patch.object(c, 'ROOT', source):
            with patch.object(c, 'load_saved', side_effect=ValueError('FIRST_INPUT_FAILURE')):
                terminal = runner.execute(plan, grant)
            self.assertEqual(terminal['status'], 'STOP')
            self.assertIn('FIRST_INPUT_FAILURE', terminal['first_stop'])
            self.assertFalse(terminal['automatic_retry'] or terminal['partial_result_claimed'])
            self.assertTrue((root/'output.one_shot_lock').is_file())
            self.assertFalse((root/'output/diagnostic_result.json').exists())

    def test_runner_post_read_change_and_own_signal_stop(self):
        with self.prepared() as (root, source, plan, grant), patch.object(c, 'ROOT', source):
            with patch.object(c, 'read_verified', side_effect=[*([b'']*0), ValueError('LATER_INPUT_CHANGE')]), \
                    patch.object(c, 'load_saved', return_value=(base(), REGISTRY, [dict(path='x', bytes=0, sha256='')])), \
                    patch.object(c, 'authorize', return_value=(PERMIT, root/'inputs', root/'output', None)):
                terminal = runner.execute(plan, grant)
            self.assertIn('LATER_INPUT_CHANGE', terminal['first_stop'])
            self.assertFalse((root/'output/diagnostic_result.json').exists())
        with self.prepared() as (root, source, plan, grant), patch.object(c, 'ROOT', source):
            def interrupted(*args, **kwargs):
                os.kill(os.getpid(), signal.SIGTERM)
                kwargs['guard']()
            with patch.object(n, 'evaluate', side_effect=interrupted):
                terminal = runner.execute(plan, grant)
            self.assertEqual(terminal['first_stop'], 'OWN_PROCESS_SIGNAL:' + str(int(signal.SIGTERM)))
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_default_CLIs_read_no_saved_values(self):
        for script in ('run_h8_noncancellation_diagnostic.py', 'audit_h8_noncancellation_evidence.py'):
            raw = subprocess.check_output([sys.executable, '-I', '-B', str(ROOT/'scripts'/script)])
            self.assertEqual(json.loads(raw)['saved_H8_numerical_analysis'], 0)

    def test_draft_files_rejected_by_CLI_without_output(self):
        with self.prepared() as (root, source, plan, grant):
            draft = dict(grant, approved=False, runtime_authorization=False)
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'grant.json').write_text(json.dumps(draft))
            done = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'scripts/run_h8_noncancellation_diagnostic.py'),
                                   '--execute-approved', '--plan', str(root/'plan.json'), '--grant',
                                   str(root/'grant.json')], capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn('EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED', done.stdout)
            self.assertFalse((root/'output').exists() or (root/'output.one_shot_lock').exists())


def floats(value):
    if isinstance(value, float):
        return 1
    if isinstance(value, dict):
        return sum(floats(v) for v in value.values())
    if isinstance(value, list):
        return sum(floats(v) for v in value)
    return 0


class AuditTests(unittest.TestCase):
    def test_metadata_audit_writes_no_scientific_float(self):
        with tempfile.TemporaryDirectory() as td, patch.object(n, 'MODEL_SHA', MODEL_SHA):
            root = Path(td)
            values, raw = objects()
            manifest = write_snapshot(root/'snap', raw)
            result = auditor.audit(manifest, root/'snap')
            self.assertEqual(result['status'], 'AUDIT_COMPLETE_MANDATORY_STOP')
            self.assertTrue(result['input_bytes_all_PASS'])
            self.assertTrue(all(result['lineage'].values()), result['lineage'])
            PR = result['PR_confirmation']
            self.assertTrue(PR['membership_consistent'])
            self.assertTrue(all(x['data_sufficient_for_future_approved_computation'] for x in PR['candidates'].values()))
            self.assertEqual(PR['confirmation_unobserved_orders'], [4])
            det = result['deterministic']
            self.assertEqual((det['signed_axes_PASS'], det['signed_axes_reconstructed'], det['u_logB_margin_present'],
                              det['saved_usual_rows'], det['saved_usual_rows_extra_or_missing']), (368, 24, 184, 1900, 0))
            self.assertEqual(floats(result), 0)
            self.assertEqual(result['arithmetic_on_saved_scientific_values'], 0)

    def test_audit_detects_missing_component_and_byte_change(self):
        def mutate(role, value):
            if role == 'old_signals':
                rec = next(x for x in value['signals'] if x['cell']['id'] == UNION[3])
                rec['error_decomposition']['signed_parts'].pop('finite_RTE')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, raw = objects(mutate)
            manifest = write_snapshot(root, raw)
            result = auditor.audit(manifest, root)
            self.assertFalse(result['PR_confirmation']['candidates'][UNION[3]]['data_sufficient_for_future_approved_computation'])
            (root/'inputs/D0_terminal.json').write_bytes(b'{}')
            self.assertEqual(auditor.audit(manifest, root)['status'], 'STOP_INPUT_BYTES')

    def test_B2_probe_requires_complete_four_axis_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values, _ = objects()
            proof = values['B2']
            proof['driver_terminal']['status'] = 'STOP'
            write_B2(root, proof)
            (root/'config.json').write_text(json.dumps(dict(SOURCE=n.B2['SOURCE'])))
            (root/'output01').mkdir(exist_ok=True)
            probe = auditor.B2_probe(root/'config.json')
            self.assertTrue(probe['all_final_receipts_present'])
            self.assertFalse(probe['B2_full_four_axis_PASS_receipts'])
            self.assertEqual(probe['access'], 'READ_ONLY')


if __name__ == '__main__':
    unittest.main()
