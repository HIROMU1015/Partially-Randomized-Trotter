#!/usr/bin/env python3
"""Synthetic tests for scripts/pauli_ld2_step_audit.py (S0 rules).

No transpile, no saved input, no H4 build: toy uncompiled wrappers follow the native
step layout (U blocks, D2-controlled tail with crz + tail p, D(f1) D(f0) D(1b) directional
blocks with ancilla rz markers, wrapper h/p/[sdg]/h/measure). A toy "compiled" circuit is
produced by a fixed local rewrite (crz -> rz cx rz cx, h -> rz sx rz, ...) and the final
ancilla run is replaced by the ZSX synthesis of the wrapper input run.
"""
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A  # noqa: E402

from qiskit import QuantumCircuit, qpy  # noqa: E402

N_SYS, ANC = 3, 3
SCALAR = 2.515477156730702
HALF_PI = math.pi / 2


def rotation(qc, label, angle):
    active = [(j, axis) for j, axis in enumerate(reversed(label)) if axis != 'I']
    for q, axis in active:
        if axis == 'X':
            qc.h(q)
        elif axis == 'Y':
            qc.sdg(q)
            qc.h(q)
    target = active[-1][0]
    for q, _ in active[:-1]:
        qc.cx(q, target)
    qc.crz(angle, ANC, target)
    for q, _ in reversed(active[:-1]):
        qc.cx(q, target)
    for q, axis in reversed(active):
        if axis == 'X':
            qc.h(q)
        elif axis == 'Y':
            qc.h(q)
            qc.s(q)


def d_block(qc, angles, gp, rzz=None, basis=0.5):
    qc.ry(basis, 2)
    if rzz:
        left, right, a = rzz
        qc.cx(left, right)
        qc.cx(ANC, right)
        qc.rz(-a, right)
        qc.cx(ANC, right)
        qc.cx(left, right)
    for q, a in enumerate(angles):
        qc.cx(ANC, q)
        qc.rz(-a, q)
        qc.cx(ANC, q)
    qc.rz(2 * gp, ANC)
    qc.ry(-basis, 2)


def step(qc, events, product=None, near_zero=False):
    for ang in (0.11, 0.22, 0.33):
        qc.ry(ang, 0)
        qc.cx(0, 1)
        qc.rz(ang, 1)
        qc.cx(0, 1)
        qc.ry(-ang, 0)
    for label, angle in events:
        rotation(qc, label, angle)
    if product is not None:
        qc.cz(ANC, product)
    qc.p(0.0, ANC)
    d_block(qc, (0.31, 8.673617379884035e-19 if near_zero else 0.32, 0.33), 0.07, rzz=(0, 1, 0.41), basis=0.6)
    d_block(qc, (0.21, 0.22, 0.23), 0.05, rzz=(1, 2, 0.42), basis=0.7)
    d_block(qc, (0.11, 0.12, 0.13), 0.03, basis=0.5)


def wrapper(steps, T, axis, product_step=None, near_zero=False):
    qc = QuantumCircuit(N_SYS + 1, 1)
    qc.h(ANC)
    for s, events in enumerate(steps):
        step(qc, events, product=1 if product_step == s else None, near_zero=near_zero)
    qc.p(-SCALAR * T, ANC)
    if axis == 'Y':
        qc.sdg(ANC)
    qc.h(ANC)
    qc.measure(ANC, 0)
    return qc


def toy_compile(wr):
    """Fixed local rewrite to the rz/sx/cx basis (no transpile)."""
    out = QuantumCircuit(N_SYS + 1, 1)
    data = [i for i in wr.data]
    idx = lambda i: tuple(wr.find_bit(q).index for q in i.qubits)
    last_anc_2q = max(k for k, i in enumerate(data) if ANC in idx(i) and len(i.qubits) > 1)
    tail = []
    for k, inst in enumerate(data):
        name, qb, pa = inst.operation.name, idx(inst), inst.operation.params
        if qb == (ANC,) and k > last_anc_2q and name != 'measure':
            tail.append((name, [float(p) for p in pa]))
            continue
        if name == 'measure':
            names, angles = A.synthesize_input_run(tail)
            it = iter(angles)
            for nm in names:
                (out.rz(next(it), ANC) if nm == 'rz' else out.sx(ANC))
            out.measure(ANC, 0)
        elif name == 'crz':
            out.rz(pa[0] / 2, qb[1]); out.cx(*qb); out.rz(-pa[0] / 2, qb[1]); out.cx(*qb)
        elif name == 'cz':
            out.rz(HALF_PI, qb[1]); out.sx(qb[1]); out.rz(HALF_PI, qb[1]); out.cx(*qb)
            out.rz(HALF_PI, qb[1]); out.sx(qb[1]); out.rz(HALF_PI, qb[1])
        elif name == 'h':
            out.rz(HALF_PI, qb[0]); out.sx(qb[0]); out.rz(HALF_PI, qb[0])
        elif name == 'sdg':
            out.rz(-HALF_PI, qb[0])
        elif name == 's':
            out.rz(HALF_PI, qb[0])
        elif name == 'ry':
            out.sx(qb[0]); out.rz(float(pa[0]), qb[0]); out.sx(qb[0])
        elif name == 'p':
            if float(pa[0]) != 0.0:
                out.rz(float(pa[0]), qb[0])
        elif name in ('cx', 'rz'):
            getattr(out, name)(*([float(p) for p in pa] + list(qb)))
        else:
            raise ValueError(name)
    return qiskit_like_identity_removal(out)


def qiskit_like_identity_removal(qc):
    """Toy stand-in for Qiskit 1.3.0: drop rz with |angle| < 1e-12, then cancel adjacent identical cx."""
    keep = [i for i in qc.data if not (i.operation.name == 'rz' and abs(float(i.operation.params[0])) < 1e-12)]
    changed = True
    while changed:
        changed = False
        for k in range(len(keep) - 1):
            a, b = keep[k], keep[k + 1]
            if a.operation.name == 'cx' and b.operation.name == 'cx' and a.qubits == b.qubits:
                del keep[k:k + 2]
                changed = True
                break
    out = QuantumCircuit(N_SYS + 1, 1)
    for i in keep:
        out.append(i.operation, i.qubits, i.clbits)
    return out


EV = {
    'a': [('XZI', 0.1), ('IYZ', -0.1), ('ZII', 0.1)],
    'b': [('IIX', 0.1), ('ZZI', 0.1), ('YIZ', -0.1)],
    'c': [('XIZ', -0.1), ('IZZ', 0.1), ('IXI', 0.1)],
    'd': [('ZIY', 0.1), ('XXI', 0.1), ('IIZ', -0.1)],
    'e': [('YZI', 0.1), ('IIY', 0.1), ('ZIZ', 0.1)],
    'f': [('IXZ', 0.1), ('ZYI', -0.1), ('XII', 0.1)],
    'g': [('IZX', -0.1), ('XIY', 0.1), ('ZZZ', 0.1)],
    'w': [('XIZ', 0.1)],
}
PATHS = {  # path_id: (kind, q, R, T, steps, composed_of)
    'W_M0/p0': ('WIRING', 1, 1, 0.1, ['w'], None),
    'S66/p0': ('CALIBRATION', 1, 3, 0.1, ['a'], None),
    'S66/p1': ('CALIBRATION', 1, 3, 0.1, ['b'], None),
    'P2R66/p0': ('CALIBRATION', 2, 3, 0.2, ['a', 'b'], ['S66/p0', 'S66/p1']),
    'Q8R66/p0': ('CALIBRATION', 3, 3, 0.30000000000000004, ['c', 'd', 'e'], None),
    'M5_ROUND/p0': ('HOLDOUT', 2, 3, 0.2, ['f', 'g'], None),
}


def dump_qpy(circuit, path):
    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    path.write_bytes(gzip.compress(buf.getvalue()))


def metrics_of(qc):
    tok = A.tokenize(qc)
    return dict(rz_count=tok['counts']['rz'], cx_count=tok['counts']['cx'], circuit_size=tok['counts']['size'],
                rz_depth=0, cx_depth=0, total_depth=0)


class Fixture:
    """Writes toy QPY + cache JSON into a temp tree and an INPUTS.json bound to them."""

    def __init__(self, tmp, mutate=None, product=False, near_zero=False):
        self.tmp = Path(tmp)
        self.paths = []
        for pid, (kind, q, R, T, steps, composed) in PATHS.items():
            rec = dict(path_id=pid, kind=kind, cell=pid.split('/')[0], index=int(pid[-1]), m=None, q=q, R=R,
                       T_hex=float(T).hex(), physical_id='TOY:' + pid, cohort='TOY', composed_of=composed, axes={})
            for axis in ('X', 'Y'):
                wr = wrapper([EV[s] for s in steps], T, axis, product_step=0 if (product and pid == 'S66/p0') else None,
                             near_zero=near_zero)
                cp = toy_compile(wr)
                if mutate:
                    cp = mutate(pid, axis, cp) or cp
                fp = hashlib.sha256((pid + axis).encode()).hexdigest()
                base = self.tmp / 'private'
                base.mkdir(parents=True, exist_ok=True)
                files = {}
                for kind_f, circ in (('wrapper', wr), ('compiled', cp)):
                    p = base / f'{fp}_{kind_f}.qpy.gz'
                    dump_qpy(circ, p)
                    files[kind_f] = dict(path=str(p), sha256=A.sha256_file(p), bytes=p.stat().st_size, exists=True)
                met = metrics_of(toy_compile(wr))      # receipt metrics come from the unmutated circuit
                cj = base / f'{fp}.json'
                cj.write_text(json.dumps(dict(compiler=A.COMPILER, fingerprint=fp, metrics=met)))
                files['cache_json'] = dict(path=str(cj), sha256=A.sha256_file(cj), bytes=cj.stat().st_size, exists=True)
                rec['axes'][axis] = dict(fingerprint=fp, cache_reused=False, metrics=met, receipt={}, files=files)
            self.paths.append(rec)
        self.out = self.tmp / 'out'
        (self.out / 'run01').mkdir(parents=True)
        inputs = dict(scalar=dict(value_hex=float(SCALAR).hex()), paths=self.paths,
                      round_T={str(m): dict(numerator=(Tq := (float(0.1) * 2 ** m)).as_integer_ratio()[0],
                                            denominator=Tq.as_integer_ratio()[1]) for m in range(13)})
        (self.out / 'INPUTS.json').write_text(json.dumps(inputs))

    def run(self):
        A.OUT, A.RUN, A.WORK = self.out, self.out / 'run01', self.out / 'run01' / 'work'
        outcomes = [A.run_path((p, SCALAR)) for p in self.paths]
        A.summarize()
        result = json.loads((A.RUN / 'AUDIT_RESULT.json').read_text())
        return outcomes, result


def run_fixture(**kw):
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp, **kw)
        outcomes, result = fx.run()
        stats = (A.RUN / 'STEP_STATS.csv').read_text()
        segs = (A.RUN / 'SEGMENTS.csv').read_text()
        paths = {p.name: json.loads(p.read_text()) for p in A.RUN.glob('PATH_*.json')}
        return outcomes, result, stats, segs, paths


class StepAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = run_fixture()

    def test_01_clean_toy_all_pass(self):
        outcomes, result, _, _, _ = self.base
        self.assertTrue(all(o['status'] == 'DONE' for o in outcomes), outcomes)
        self.assertEqual(result['P1'], 'PASS')
        self.assertEqual(result['P2'], 'PASS')
        self.assertEqual(result['P3']['status'], 'PASS', result['P3'])
        self.assertEqual(result['P4']['status'], 'PASS_EXACT', result['P4'])
        self.assertTrue(result['P4']['rz_identity_exact'])
        self.assertEqual(result['P5']['status'], 'PASS', result['P5'])
        self.assertEqual(result['P6']['status'], 'PASS')

    def test_02_stats_exclude_paired_and_y(self):
        _, result, stats, segs, _ = self.base
        rows = [r.split(',') for r in segs.strip().splitlines()]
        head = rows[0]
        ind = [dict(zip(head, r)) for r in rows[1:]]
        self.assertTrue(all(r['independent_sample'] == 'False' for r in ind if r['path_id'] == 'P2R66/p0'))
        srows = [dict(zip(stats.splitlines()[0].split(','), r.split(','))) for r in stats.strip().splitlines()[1:]]
        n_first_R3 = next(int(r['n']) for r in srows if r['R'] == '3' and r['context'] == 'FIRST')
        self.assertEqual(n_first_R3, 4)     # S66 p0, p1, Q8R66, M5 first steps; P2R66 excluded; X only
        r1 = next(r for r in srows if r['R'] == '1')
        self.assertEqual(r1['note'], 'VARIANCE_NOT_IDENTIFIABLE_N_LT_2')

    def test_03_missing_barrier_pair_fails_alignment(self):
        def mutate(pid, axis, cp):
            if pid == 'Q8R66/p0':
                data = list(cp.data)
                # drop the two CX of a D-region pair on wire 2 (keep its rz: RZ count unchanged)
                idx = lambda i: tuple(cp.find_bit(q).index for q in i.qubits)
                k = next(k for k, i in enumerate(data) if i.operation.name == 'cx' and idx(i) == (ANC, 2)
                         and data[k + 1].operation.name == 'rz'
                         and math.isclose(float(data[k + 1].operation.params[0]), -0.13))
                new = QuantumCircuit(N_SYS + 1, 1)
                for kk, inst in enumerate(data):
                    if kk in (k, k + 2):
                        continue
                    new.append(inst.operation, inst.qubits, inst.clbits)
                return new
        _, result, _, _, paths = run_fixture(mutate=mutate)
        self.assertEqual(result['P2'], 'FAIL')
        self.assertTrue(paths['PATH_Q8R66_p0.json']['axes']['X']['P2'].startswith('FAIL:ANCILLA_CX_ALIGNMENT'))

    def test_04_barrier_angle_change_fails(self):
        def mutate(pid, axis, cp):
            if pid == 'S66/p1':
                new = QuantumCircuit(N_SYS + 1, 1)
                done = False
                for inst in cp.data:
                    op = inst.operation
                    if not done and op.name == 'rz' and math.isclose(float(op.params[0]), -0.13):
                        new.rz(-0.13 + 1e-9, inst.qubits[0]); done = True
                        continue
                    new.append(op, inst.qubits, inst.clbits)
                return new
        _, result, _, _, paths = run_fixture(mutate=mutate)
        self.assertIn('BARRIER_RZ_ANGLE', paths['PATH_S66_p1.json']['axes']['X']['P2'])

    def test_05_cross_cut_two_qubit_token_fails(self):
        def mutate(pid, axis, cp):
            if pid == 'M5_ROUND/p0':
                data = list(cp.data)
                idx = lambda i: tuple(cp.find_bit(q).index for q in i.qubits)
                # first D(1b) pair on wire 0 of step 1 is the 10th cx(anc,0) (pairs: f1 2, f0 2, 1b 2 ... per step)
                ks = [k for k, i in enumerate(data) if i.operation.name == 'cx' and idx(i) == (ANC, 0)]
                k = ks[5]            # second cx of the last pair on wire 0 in step 1
                new = QuantumCircuit(N_SYS + 1, 1)
                for kk, inst in enumerate(data):
                    new.append(inst.operation, inst.qubits, inst.clbits)
                    if kk == k:
                        new.cx(0, 1); new.cx(0, 1)   # 2q tokens after wire-0 cut, before wire-1 cut
                return new
        _, result, _, _, paths = run_fixture(mutate=mutate)
        self.assertIn('MULTI_QUBIT_TOKEN_ACROSS_CUT', paths['PATH_M5_ROUND_p0.json']['axes']['X']['P2'])

    def test_06_rz_mismatch_is_stop(self):
        def mutate(pid, axis, cp):
            if pid == 'S66/p0' and axis == 'Y':
                cp.rz(0.5, 0)
                return cp
        outcomes, result, _, _, _ = run_fixture(mutate=mutate)
        stop = [o for o in outcomes if o['path_id'] == 'S66/p0']
        self.assertEqual(stop[0]['status'], 'STOP')
        self.assertIn('RZ_TOKEN_MISMATCH', stop[0]['reason'])
        self.assertIn('S66/p0', result['stops'])

    def test_07_product_token_flagged(self):
        _, result, _, _, paths = run_fixture(product=True)
        self.assertGreater(paths['PATH_S66_p0.json']['axes']['X']['products_total'], 0)
        self.assertGreater(result['products_total'], 0)

    def test_08_p4_levels(self):
        def lead(pid, axis, cp):
            if pid == 'P2R66/p0':
                data = list(cp.data)
                idx = lambda i: tuple(cp.find_bit(q).index for q in i.qubits)
                # first event of step 2 is 'IIX' on wire 0: change the leading rz of its h
                cut0 = [k for k, i in enumerate(data) if i.operation.name == 'cx' and idx(i) == (ANC, 0)][5]
                k = next(k for k in range(cut0 + 1, len(data))
                         if data[k].operation.name == 'rz' and idx(data[k]) == (0,)
                         and math.isclose(float(data[k].operation.params[0]), HALF_PI)
                         and data[k + 1].operation.name == 'sx' and k > cut0 + 20)
                new = QuantumCircuit(N_SYS + 1, 1)
                for kk, inst in enumerate(data):
                    if kk == k:
                        new.rz(HALF_PI + 0.25, 0)
                        continue
                    new.append(inst.operation, inst.qubits, inst.clbits)
                return new
        _, result, _, _, _ = run_fixture(mutate=lead)
        self.assertEqual(result['P4']['status'], 'PASS_LEADING_1Q_RUN_ONLY')
        self.assertTrue(any(x != 'EXACT' for x in result['P4']['second_body_vs_S66p1_per_wire']))

        def two_q(pid, axis, cp):
            if pid == 'P2R66/p0':
                data = list(cp.data)
                idx = lambda i: tuple(cp.find_bit(q).index for q in i.qubits)
                ks = [k for k, i in enumerate(data) if i.operation.name == 'cx' and idx(i) == (1, 2)]
                k = ks[-6]
                new = QuantumCircuit(N_SYS + 1, 1)
                for kk, inst in enumerate(data):
                    new.append(inst.operation, inst.qubits, inst.clbits)
                    if kk == k:
                        new.cx(1, 2); new.cx(1, 2)
                return new
        _, result, _, _, _ = run_fixture(mutate=two_q)
        self.assertEqual(result['P4']['status'], 'FAIL')

    def test_09_xy_difference_detected(self):
        def mutate(pid, axis, cp):
            if pid == 'Q8R66/p0' and axis == 'Y':
                data = list(cp.data)
                new = QuantumCircuit(N_SYS + 1, 1)
                swapped = False
                for inst in data:
                    if not swapped and inst.operation.name == 'sx' and cp.find_bit(inst.qubits[0]).index == 2:
                        new.x(2); swapped = True
                        continue
                    new.append(inst.operation, inst.qubits, inst.clbits)
                return new
        _, result, _, _, _ = run_fixture(mutate=mutate)
        self.assertFalse(result['P5']['compiled_X_equals_Y_except_final_run'])

    def test_10_tokens_equal_pf_original(self):
        src = subprocess.run(['git', 'show', 'd3338c7ef:scripts/pf_qpy_closure_audit.py'], cwd=ROOT,
                             check=True, capture_output=True, text=True).stdout
        ns = {'__name__': 'pf_original', '__file__': str(ROOT / 'scripts' / 'pf_qpy_closure_audit.py')}
        exec(compile(src, 'pf_qpy_closure_audit@d3338c7ef', 'exec'), ns)
        for steps, axis in ((['a'], 'X'), (['c', 'd', 'e'], 'Y'), (['w'], 'X')):
            cp = toy_compile(wrapper([EV[s] for s in steps], 0.1, axis))
            pf_wires, pf_tail, pf_counts = ns['wires_without_final_ancilla_run'](cp)
            tok = A.tokenize(cp)
            run_pos, _ = A.final_ancilla_run(tok)
            mine = [[tok['instrs'][j] for j in w] for w in tok['wires']]
            mine[ANC] = [tok['instrs'][j] for j in tok['wires'][ANC] if j not in set(run_pos)]
            self.assertEqual(mine, pf_wires)
            self.assertEqual([tok['instrs'][j] for j in run_pos], pf_tail)
            self.assertEqual((tok['counts']['rz'], tok['counts']['cx'], tok['counts']['size']),
                             (pf_counts['rz'], pf_counts['cx'], pf_counts['size']))

    def test_11_lcp_lcs_and_levels(self):
        import numpy as np
        a = np.array([1, 2, 3, 4, 9], dtype=np.uint64)
        b = np.array([1, 2, 3, 5, 9], dtype=np.uint64)
        self.assertEqual(A.lcp_len([a, b]), 3)
        self.assertEqual(A.lcs_len([a, b]), 1)
        mq = np.array([False, False, True, False, False])
        rz = np.zeros(5, dtype=bool)
        c = np.array([7, 2, 3, 4, 9], dtype=np.uint64)
        d = np.array([8, 8, 1, 2, 3, 4, 9], dtype=np.uint64)          # different deterministic prefix
        mqd = np.array([True, False, False, False, True, False, False])
        self.assertEqual(A.leveled_compare((a, rz, mq), (a, rz, mq), 0, 0), 'EXACT')
        self.assertEqual(A.leveled_compare((a, rz, mq), (d, np.zeros(7, bool), mqd), 0, 2), 'EXACT')
        self.assertEqual(A.leveled_compare((a, rz, mq), (c, rz, mq), 0, 0), 'LEADING_1Q_RUN_ONLY')
        self.assertEqual(A.leveled_compare((a, rz, mq), (b, rz, mq), 0, 0), 'DIFFER')

    def test_12_structure_counts_events(self):
        wr = wrapper([EV['c'], EV['d'], EV['e']], 0.3, 'X')
        st = A.uncompiled_structure(wr, 3)
        self.assertEqual([s['n_event'] for s in st['steps']], [3, 3, 3])
        self.assertEqual(len(st['steps'][0]['blocks']), 3)
        self.assertEqual(len({s['signature'] for s in st['steps']}), 1)
        with self.assertRaises(A.StructureFail):
            A.uncompiled_structure(wr, 2)


class RevisionS0R1Tests(unittest.TestCase):
    """Regression for the run01 failure (near-zero D pair removed by the compiler) and the S0-r1 fix."""

    def tearDown(self):
        A.S0R1_IDENTITY_PAIRS = True

    def test_13_frozen_rule_reproduces_run01_failure(self):
        A.S0R1_IDENTITY_PAIRS = False
        _, result, _, _, paths = run_fixture(near_zero=True)
        self.assertEqual(result['P2'], 'FAIL')
        msg = paths['PATH_Q8R66_p0.json']['axes']['X']['P2']
        self.assertTrue(msg.startswith('FAIL:ANCILLA_CX_ALIGNMENT'), msg)
        # deficit is exactly 2 CX per step (one identity pair per step in the toy)
        got, exp = msg.split(':')[-1].split('vs')
        self.assertEqual(int(exp) - int(got), 2 * 3)
        # statuses are not vacuous when no path is segmented
        self.assertEqual(result['P3']['status'], 'UNKNOWN_NO_SEGMENTED_PATH')
        self.assertEqual(result['P6']['status'], 'UNKNOWN_NO_SEGMENTED_PATH')
        self.assertTrue(result['P5']['compiled_X_equals_Y_except_final_run'])   # computed without P2

    def test_14_revised_rule_passes_and_counts_identity_pairs(self):
        A.S0R1_IDENTITY_PAIRS = True
        _, result, _, _, paths = run_fixture(near_zero=True)
        self.assertEqual(result['P2'], 'PASS')
        self.assertEqual(result['P3']['status'], 'PASS', result['P3'])
        self.assertEqual(result['P4']['status'], 'PASS_EXACT')
        self.assertEqual(result['P6']['status'], 'PASS')
        steps = paths['PATH_Q8R66_p0.json']['axes']['X']['steps']
        self.assertEqual([st['identity_pairs'] for st in steps], [1, 1, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
