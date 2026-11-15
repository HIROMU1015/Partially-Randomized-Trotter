"""Synthetic checks for scripts/pf_qpy_closure_audit.py (no transpile, no saved/private input).

Artificial circuits are written directly in the rz/sx/x/cx basis; QPY round trips use a
temporary directory. Run: python -B tests/check_pf_qpy_closure_audit.py
"""
import ast
import gzip
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.synthesis import OneQubitEulerDecomposer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pf_qpy_closure_audit as A  # noqa: E402

RESULTS = []


def check(name, ok, **detail):
    RESULTS.append(dict(check=name, ok=bool(ok), **detail))
    print(('PASS ' if ok else 'FAIL ') + name, detail if detail else '')


def original_wire_function():
    text = subprocess.run(['git', 'show', ':'.join(A.WIRE_COMPARISON)], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'wires_without_final_ancilla_run')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'wire_comparison', 'exec'), scope)
    return scope['wires_without_final_ancilla_run']


def random_basis_circuit(rng, n, gates):
    qc = QuantumCircuit(n + 1)
    a = n
    for _ in range(gates):
        k = rng.randrange(5)
        q, r = rng.randrange(n + 1), rng.randrange(n + 1)
        if k == 0:
            qc.rz(rng.uniform(-4, 4), q)
        elif k == 1:
            qc.sx(q)
        elif k == 2 and q != r:
            qc.cx(q, r)
        elif k == 3:
            qc.x(q)
        else:
            qc.cx(a, rng.randrange(n)) if n else None
    qc.rz(rng.uniform(-3, 3), a); qc.sx(a); qc.rz(rng.uniform(-3, 3), a)
    qc.add_register(ClassicalRegister(1)); qc.measure(a, 0)
    return qc


STEP_WIDTH = 0.25
SCALAR = 2.293146549384713
THETA0 = 0.7123


def periodic_circuit(q, *, perturb=False, theta_shift=0.0):
    """Head / q body copies / tail per wire, final ancilla run rz(theta1(q)) sx rz(theta2)."""
    n = 3
    a = n
    qc = QuantumCircuit(n + 1)
    # head: differs per wire
    qc.rz(0.11, 0); qc.sx(0); qc.cx(0, 1); qc.rz(0.37, 2)
    for step in range(q):
        # body: one-body-like directional barrier on every system wire plus couplers
        for k in range(n):
            qc.cx(a, k); qc.rz(0.05 * (k + 1), k); qc.cx(a, k)
        qc.cx(0, 1); qc.rz(0.21 if not (perturb and step == q // 2) else math.nextafter(0.21, 1), 1); qc.cx(0, 1)
        qc.sx(2); qc.rz(-0.4, 2); qc.cx(1, 2)
        qc.rz(0.9, a)
    # tail
    qc.cx(a, 2); qc.rz(0.6, 0); qc.sx(1)
    T = STEP_WIDTH * q
    qc.rz(A.wrap(THETA0 - SCALAR * T + theta_shift), a); qc.sx(a); qc.rz(1.234, a)
    qc.add_register(ClassicalRegister(1)); qc.measure(a, 0)
    return qc, T


def write_qpy(path, circuit):
    stream = io.BytesIO(); qpy.dump(circuit, stream)
    path.write_bytes(gzip.compress(stream.getvalue(), compresslevel=1, mtime=0))


def main():
    rng = random.Random(1115)
    original = original_wire_function()
    agree = 0
    for _ in range(200):
        qc = random_basis_circuit(rng, rng.randrange(1, 5), rng.randrange(1, 80))
        w0, r0 = original(qc)
        w1, tail, _ = A.wires_without_final_ancilla_run(qc, encode=lambda p: tuple(round(float(x), 12) for x in p))
        agree += w0 == w1 and r0 == sum(t[0] == 'rz' for t in tail)
    check('reused_wire_comparison_matches_51d324ea', agree == 200, agree=agree)

    check('longest_run_first_tie', A.longest_true_run(np.array([1, 1, 0, 1, 1, 0], bool)) == (0, 2))
    check('longest_run_empty', A.longest_true_run(np.array([], bool)) == (0, 0))
    check('wrap_range', all(-math.pi < A.wrap(x) <= math.pi for x in (-10, -math.pi, 0, math.pi, 3 * math.pi, 1e3)))

    # sign of the T-dependent first RZ in the theta=pi/2 ZSX branch (synthesis only)
    dec = OneQubitEulerDecomposer('ZSX')
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    def first_rz(phi, sine):
        diag = np.diag([1, np.exp(1j * phi)])
        if sine:
            diag = np.diag([1, -1j]) @ diag
        circ = dec(H @ diag @ np.diag([np.exp(-0.3j), np.exp(0.3j)]), simplify=True)
        return [i.operation.name for i in circ.data], [float(i.operation.params[0]) for i in circ.data if i.operation.name == 'rz']
    forms, shifts, xy = set(), [], []
    for phi1, phi2 in ((0.2, 1.7), (-2.0, 0.4), (3.0, -1.1)):
        f1, a1 = first_rz(phi1, False); f2, a2 = first_rz(phi2, False); fy, ay = first_rz(phi1, True)
        forms |= {tuple(f1), tuple(f2), tuple(fy)}
        shifts.append(abs(A.wrap(a2[0] - a1[0] - (phi2 - phi1))))
        xy.append(abs(A.wrap(ay[0] - a1[0] + math.pi / 2)))
    check('zsx_form_rz_sx_rz', forms == {('rz', 'sx', 'rz')}, forms=sorted(forms))
    check('first_rz_shifts_with_input_phase', max(shifts) < 1e-12, max_residual=max(shifts))
    check('sine_axis_shift_minus_pi_over_2', max(xy) < 1e-12, max_residual=max(xy))
    names, angles = A.synthesize_input_run([('rz', [0.3]), ('p', [1.1]), ('h', [])])
    check('synthesize_input_run_form', names == ['rz', 'sx', 'rz'] and len(angles) == 2, names=names)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qs = (7, 14, 28, 224)
        def targets(perturb_q=None, rz_off=0, tag='base'):
            out, total = [], {}
            for q in qs:
                qc, T = periodic_circuit(q, perturb=(q == perturb_q))
                fp = f'{tag}{q}'
                write_qpy(tmp / f'{fp}_compiled.qpy.gz', qc)
                ops = qc.count_ops()
                metrics = dict(rz_count=ops.get('rz', 0) + (rz_off if q == 224 else 0), rz_depth=0, cx_count=ops.get('cx', 0),
                               cx_depth=0, total_depth=0, circuit_size=qc.size())
                (tmp / f'{fp}.json').write_text(json.dumps(dict(fingerprint=fp, metrics=metrics,
                    compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b')))
                total[q] = ops.get('rz', 0)
                out.append(dict(m=q, q=q, fingerprint=fp, compiled=str(tmp / f'{fp}_compiled.qpy.gz'),
                                cache_json=str(tmp / f'{fp}.json'), compiled_bytes=1, receipt_metrics=metrics,
                                trial_metrics=metrics, T=T.hex(), T_schedule=T.hex(), scalar=SCALAR.hex(), cache_reused=False))
            return out, total
        group = 'FULL_PF|2nd|16/7|X'
        tg, total = targets()
        b = total[14] - total[7]
        b //= 7
        frozen = {group: dict(a=total[7] - 7 * b, b=b)}
        unobserved = {f'FULL_PF|2nd|{m}': dict(T_hex=(STEP_WIDTH * 7 * 2 ** (m - 4)).hex(), q=7 * 2 ** (m - 4), physical_id='x')
                      for m in A.UNOBSERVED}
        prog = tmp / 'p.jsonl'
        res = A.audit_group(group, tg, None, frozen, unobserved, prog)
        check('artificial_all_pass', all(res[k] == 'PASS' for k in ('A1', 'A2', 'A3', 'A4', 'A5')),
              status={k: res[k] for k in ('A1', 'A2', 'A3', 'A4', 'A5')}, c=res.get('c'))
        check('artificial_barriers_detected', all(w['directional_barrier'] for w in res['wires'][:-1]))
        check('artificial_margins_listed', len(res['unobserved']) == 5 and all(r['margin'] is not None for r in res['unobserved']))
        check('no_price_values_in_unobserved', all(not any('rz' in k.lower() and 'count' in k.lower() for k in r) for r in res['unobserved']))

        tg2, _ = targets(perturb_q=224, tag='perturbed')
        res2 = A.audit_group(group, tg2, None, frozen, unobserved, prog)
        check('one_ulp_body_perturbation_fails_A2', res2['A2'] == 'FAIL', A2=res2['A2'])

        frozen_bad = {group: dict(a=frozen[group]['a'], b=frozen[group]['b'] + 1)}
        res3 = A.audit_group(group, tg, None, frozen_bad, unobserved, prog)
        check('wrong_frozen_slope_fails_A3', res3['A3'] == 'FAIL', A3=res3['A3'])

        tg4, _ = targets(rz_off=1, tag='rzoff')
        try:
            A.audit_group(group, tg4, None, frozen, unobserved, prog)
            stopped = False
        except A.AuditStop as exc:
            stopped = 'RZ_TOKEN_TOTAL_MISMATCH' in str(exc)
        check('rz_total_mismatch_stops', stopped)

        tg5 = [dict(t) for t in tg]
        tg5[1]['T'] = (float.fromhex(tg5[1]['T']) + 0.5).hex(); tg5[1]['T_schedule'] = tg5[1]['T']
        res5 = A.audit_group(group, tg5, None, frozen, unobserved, prog)
        check('broken_angle_relation_gives_A4_unknown', res5['A4'] == 'UNKNOWN' and
              all(r['final_run'] == 'UNKNOWN_A4' for r in res5['unobserved']))

        tg6 = [dict(t) for t in tg]
        tg6[0]['compiled'] = str(tmp / 'missing.qpy.gz')
        res6 = A.audit_group(group, tg6, None, frozen, unobserved, prog)
        check('missing_qpy_unknown_not_recompiled', res6['A1'] == 'UNKNOWN' and res6['A2'] == 'UNKNOWN')

        tiny = dict(unobserved)
        # a round whose extrapolated first RZ sits within 1e-6 of the omission point
        th9 = A.wrap(THETA0 - SCALAR * STEP_WIDTH * 224)
        T_bad = STEP_WIDTH * 224 + (th9 - 1e-8) / SCALAR
        tiny['FULL_PF|2nd|7'] = dict(T_hex=T_bad.hex(), q=56, physical_id='x')
        res7 = A.audit_group(group, tg, None, frozen, tiny, prog)
        check('small_margin_marks_unknown', res7['unobserved'][0]['final_run'] == 'UNKNOWN_SMALL_MARGIN',
              margin=res7['unobserved'][0]['margin'])
    ok = all(r['ok'] for r in RESULTS)
    print(json.dumps(dict(checks=len(RESULTS), failed=[r['check'] for r in RESULTS if not r['ok']])))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
