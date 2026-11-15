"""LD3 calibration driver and audit/price checks: saved metadata, provider toy, synthetic tokens.

No H4 build, sampling of H4 events, transpile, QPY read or price of the real cell.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ld3_cal', ROOT / 'scripts/run_pauli_ld3_cal.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
sys.path[:0] = [str(D.PIN_ROOT / 'scripts'), str(D.PIN_ROOT / 'tests'), str(ROOT / 'scripts')]
import h46_long_time_signal as signal  # noqa: E402  provider root first, as the real child
signal.install()
import check_pauli_ld1_m0_pilot_circuits as T  # noqa: E402  original toy helpers (jw_saved, one_body_saved)
import pauli_ld3_cal_audit as AU  # noqa: E402
import numpy as np  # noqa: E402


def toy_ld3():
    """Toy 4-mode DF Hamiltonian with five fragments; native prepare(ham, 3) head; toy Pauli tail."""
    from trotterlib.df_hamiltonian import DFHamiltonian
    from h46_new2_native import OccupationOracle, prepare
    P = T.P
    rng = np.random.default_rng(23)
    x = rng.normal(size=(4, 4)); g = []
    for _ in range(5):
        y = rng.normal(size=(4, 4)); g.append(.3 * (y + y.T))
    ham = DFHamiltonian(.41, .3 * (x + x.T), np.array([.4, .3, .25, .2, .1]), tuple(g), {})
    full = np.zeros(16, complex)
    for i in range(16):
        if bin(i).count('1') == 2:
            full[i] = rng.normal() + 1j * rng.normal()
    full /= np.linalg.norm(full)
    oracle = OccupationOracle(ham, full, prefix=ham.n_blocks)
    qstate = np.zeros(16, complex); qstate[oracle.qindices] = oracle.initial
    ctx = SimpleNamespace(ham=ham, oracle=oracle, qstate=qstate, preparations={3: prepare(ham, 3)})
    a = T.jw_saved(4)
    H1 = T.one_body_saved(ham.one_body, a)
    F = [ham.lambdas[k] * T.one_body_saved(g[k], a) @ T.one_body_saved(g[k], a) for k in range(3)]
    sR = ctx.preparations[3].extracted_identity_coefficient
    terms = []
    for label, c in (('YXIZ', -.31), ('ZZII', .22), ('IXYI', .17), ('XIIX', -.12), ('IIZI', .09)):
        terms.append(dict(label_qiskit=label, label_in_saved_tensor_order=label[::-1], coefficient=c,
                          coefficient_hex=float(c).hex(), sign=1 if c > 0 else -1))
    lam = math.fsum(abs(t['coefficient']) for t in terms)
    for t in terms:
        t['probability'] = abs(t['coefficient']) / lam
    d = dict(prefix=3, lambda_P=lam, saved_DF_scalar=sR, pauli_identity=sR, terms=terms)
    A_s = P.pauli_sum(d, 'saved'); tail = A_s + sR * np.eye(16)
    operators = dict(head=H1 + sum(F), tail=tail, Pauli_nonidentity=A_s)
    blocks = np.array([H1, *F, tail])
    request = dict(m=None, prefix=3, q=2, R=3, K=2, T=.2, step_time=.1, lambda_P=lam,
                   candidate_id='PAULI_PRICE_CELL:TOY_LD3', physical_cell=dict(profile=dict(compiler='TOY')))
    return ctx, d, operators, blocks, request


def synthetic_arrays(entry_first, entry_later, suffix, bodies_by_path, end):
    """Per-wire (hash, rz, mq) segments built from known deterministic parts and bodies."""
    out = {}
    for pid, bodies in bodies_by_path.items():
        wires = []
        for w in range(len(suffix)):
            segs = []
            for s, body in enumerate(bodies):
                entry = entry_first[w] if s == 0 else entry_later[w]
                h = np.array(entry + body[w] + suffix[w], dtype=np.uint64)
                rz = np.array([int(t % 2 == 0) for t in h], dtype=np.int64)
                segs.append((h, rz, np.zeros(len(h), bool)))
            e = np.array(end[w], dtype=np.uint64)
            wires.append(dict(segs=segs, end=(e, np.array([int(t % 2 == 0) for t in e], dtype=np.int64), np.zeros(len(e), bool))))
        out[pid] = wires
    return out


class LD3CalTests(unittest.TestCase):
    def test_metadata_reproduces_LD2_cell_and_binds_LD3(self):
        m = json.loads(subprocess.check_output([sys.executable, '-B', str(ROOT / D.SCRIPT), '--metadata'], cwd=ROOT))
        self.assertTrue(m['rule_check']['equal'])
        r = m['request']
        self.assertEqual((r['prefix'], r['q'], r['R'], r['T'], r['step_time']), (3, 8, 660, 0.8, 0.1))
        self.assertTrue(r['candidate_id'].startswith('PAULI_PRICE_CELL:'))
        self.assertNotEqual(r['candidate_id'], D.M12_ID)
        self.assertEqual([len(row) for row in m['step_seeds']], [8, 8])
        self.assertTrue(m['seed_checks']['unique_new']); self.assertFalse(m['seed_checks']['old_cohort_overlap'])
        self.assertEqual(r['physical_cell']['split']['tail_dictionary'], m['dictionary_identity'])
        for field, value in (('cohort', 'WRONG'), ('compile_maximum', 5)):
            bad = deepcopy(m); bad[field] = value
            with self.assertRaises(ValueError):
                D.check_binding(bad, m)
        bad = deepcopy(m); bad['step_seeds'][1][7] += 1
        with self.assertRaises(ValueError):
            D.check_binding(bad, m)

    def test_toy_prefix3_path_accelerated_equals_base_and_receipts(self):
        import h46_long_time_native as native
        B1 = D.legacy(); ctx, d, ops, blocks, req = toy_ld3()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); native.PRIVATE = folder
            cache = folder / 'compiled_cache'; cache.mkdir(); calls = []
            def fake(wrapper, fp, representative):
                record = dict(fingerprint=fp, compiler='TOY', metrics={k: 1 for k in native.METRICS})
                calls.append(fp); (cache / (fp + '.json')).write_text(json.dumps(record))
                for suffix in ('_wrapper.qpy.gz', '_compiled.qpy.gz'):
                    (cache / (fp + suffix)).write_bytes(b'ARTIFICIAL')
                return record, wrapper.copy(), False
            base = B1.acquire_path(native, ctx, d, req, cohort=D.COHORT, index=0, seeds=[11, 12], operators=ops,
                                   blocks=blocks, compile_cached=fake)
            calls.clear(); log = D.c2().Progress(folder / 'progress.jsonl')
            binding = dict(request=req, step_seeds=[[11, 12]], dictionary_identity=dict(artificial=True))
            fast, stats = D.acquire(B1, native, ctx, d, binding, 0, ops, blocks, fake, log)
            self.assertEqual(len(calls), 2)
            for k in ('prices', 'fingerprints', 'event_metadata', 'events', 'scalar', 'seeds', 'checks', 'bit_order'):
                self.assertEqual(base[k], fast[k], k)
            self.assertEqual(len(ctx.preparations[3].deterministic_blocks), 4)
            path = dict(index=0, cohort=D.COHORT, request=req, dictionary_identity=binding['dictionary_identity'], result=fast)
            source = dict(commit='0' * 40, caller='ARTIFICIAL', cell='Q8R660')
            receipts = B1.receipts().compile_receipts([path], source, binding['dictionary_identity'], folder)
            D.check_result_path(path, binding, receipts, source)
            for key in ('axis', 'execution_source', 'seeds'):
                bad = deepcopy(receipts); bad['p0X'][key] = 'WRONG'
                with self.assertRaises(ValueError):
                    D.check_result_path(path, binding, bad, source)
            stages = [json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines()]
            self.assertTrue(any(s['stage'] == 'acquire_path' and s['state'] == 'DONE' for s in stages))

    def test_edges_bodies_coefficients_and_price_exact(self):
        # 3 wires; deterministic entry F (first step) / P (later), suffix S, end E; 2 paths x 8 steps
        F = [[2, 4, 6], [8], [10, 12]]; P = [[2, 4], [8], [10]]; S = [[20, 22, 21], [24, 26], [28]]; E = [[30, 32], [34], []]
        rng = np.random.default_rng(5); counter = iter(range(10 ** 6))
        bodies = {}
        for pid in ('Q8R660/p0', 'Q8R660/p1'):
            # unique first/last body tokens never extend the common entry/suffix; middles mix RZ/non-RZ
            bodies[pid] = [[[10 ** 6 + next(counter)] + [int(x) for x in rng.integers(40, 90, size=int(rng.integers(3, 9)))]
                            + [3 * 10 ** 6 + next(counter)] for _ in range(3)] for _ in range(8)]
        arrays = synthetic_arrays(F, P, S, bodies, E)
        edge, s_len, f_len, p_len = AU.edges(arrays, 8)
        self.assertEqual([s_len[w] for w in range(3)], [3, 2, 1])
        self.assertEqual([f_len[w] for w in range(3)], [3, 1, 2]); self.assertEqual([p_len[w] for w in range(3)], [2, 1, 1])
        const = edge['constants']
        self.assertEqual(const, dict(first_prefix_rz=6, later_prefix_rz=4, suffix_rz=5, end_rz=3))
        segs = AU.bodies(arrays, 8, s_len, f_len, p_len)
        coef = AU.coefficients(const, 2, segs)
        self.assertEqual((coef['A3'], coef['H3']), (6 + 5 + 3 + 2 - 9, 9))
        later = sum(r['body_rz'] for rows in segs.values() for r in rows if r['context'] == 'LATER')
        self.assertEqual(coef['E3'], Fraction(later, 14 * 660)); self.assertGreater(later, 0)
        for pid, rows in segs.items():
            total = sum(int(arrays[pid][w]['segs'][s][1].sum()) for w in range(3) for s in range(8))
            total += sum(int(arrays[pid][w]['end'][1].sum()) for w in range(3)) + 2
            self.assertEqual(coef['A3'] + coef['H3'] * 8 + sum(r['body_rz'] for r in rows), total)
        config = dict(rounds=[dict(m=m, q_PF=2 ** m, R_outer=m + 1, candidate=dict(id='ID%d' % m, q=2 ** m, R_outer=m + 1))
                              for m in range(13)])
        rows, total = AU.price_table(config, dict(A3=10, H3=3, E3=Fraction(1, 2)), vp=[1] * 13)
        self.assertEqual(len(rows), 26)
        self.assertEqual(total, 2 * sum(10 + 3 * 2 ** m + Fraction(1, 2) * 2 ** m * (m + 1) for m in range(13)))
        with self.assertRaises(ValueError):
            AU.coefficients(const, 2, {'Q8R660/p0': segs['Q8R660/p0']})

    def test_observer_records_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = D.c2().Progress(Path(tmp) / 'progress.jsonl')
            stream = (Path(tmp) / 'child.txt').open('w')
            proc = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'], stdout=stream)
            states = D.c2().observe([(0, proc, stream)], log, poll_seconds=.02)
            self.assertEqual(states[0]['exit_code'], 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
