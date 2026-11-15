"""Split calibration (LD5) driver/audit checks: saved metadata, provider toy, LD3 saved tokens.

No H4 build, sampling of H4 events, transpile, QPY read or price of the real LD5 cell.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('split_cal', ROOT / 'scripts/run_pauli_split_cal.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
sys.path[:0] = [str(D.D3.PIN_ROOT / 'scripts'), str(D.D3.PIN_ROOT / 'tests'), str(ROOT / 'scripts')]
import h46_long_time_signal as signal  # noqa: E402  provider root first, as the real child
signal.install()
import check_pauli_ld1_m0_pilot_circuits as T  # noqa: E402
import pauli_split_cal_audit as SA  # noqa: E402
import numpy as np  # noqa: E402

LD3_BINDING = dict(commit='3074433d16dd8bbfa3fbf173e1da3606af6670d2', path='docs/research/a_pauli_ld3_cal_price_20261116/BINDING.json')
LD3_RUN = ROOT / 'docs/research/a_pauli_ld3_cal_price_20261116/audit_run01'


def toy(prefix=5, fragments=7):
    """Toy 4-mode DF Hamiltonian with seven fragments; native prepare(ham, 5); toy Pauli tail."""
    from trotterlib.df_hamiltonian import DFHamiltonian
    from h46_new2_native import OccupationOracle, prepare
    P = T.P; rng = np.random.default_rng(29)
    x = rng.normal(size=(4, 4)); g = []
    for _ in range(fragments):
        y = rng.normal(size=(4, 4)); g.append(.3 * (y + y.T))
    ham = DFHamiltonian(.41, .3 * (x + x.T), np.linspace(.4, .1, fragments), tuple(g), {})
    full = np.zeros(16, complex)
    for i in range(16):
        if bin(i).count('1') == 2:
            full[i] = rng.normal() + 1j * rng.normal()
    full /= np.linalg.norm(full)
    oracle = OccupationOracle(ham, full, prefix=ham.n_blocks)
    qstate = np.zeros(16, complex); qstate[oracle.qindices] = oracle.initial
    ctx = SimpleNamespace(ham=ham, oracle=oracle, qstate=qstate, preparations={prefix: prepare(ham, prefix)})
    a = T.jw_saved(4)
    H1 = T.one_body_saved(ham.one_body, a)
    F = [ham.lambdas[k] * T.one_body_saved(g[k], a) @ T.one_body_saved(g[k], a) for k in range(prefix)]
    sR = ctx.preparations[prefix].extracted_identity_coefficient
    terms = []
    for label, c in (('YXIZ', -.031), ('ZZII', .022), ('IXYI', .017)):
        terms.append(dict(label_qiskit=label, label_in_saved_tensor_order=label[::-1], coefficient=c,
                          coefficient_hex=float(c).hex(), sign=1 if c > 0 else -1))
    lam = math.fsum(abs(t['coefficient']) for t in terms)
    for t in terms:
        t['probability'] = abs(t['coefficient']) / lam
    d = dict(prefix=prefix, lambda_P=lam, saved_DF_scalar=sR, pauli_identity=sR, terms=terms)
    A_s = P.pauli_sum(d, 'saved'); tail = A_s + sR * np.eye(16)
    operators = dict(head=H1 + sum(F), tail=tail, Pauli_nonidentity=A_s)
    request = dict(m=None, prefix=prefix, q=2, R=4, K=2, T=.2, step_time=.1, lambda_P=lam,
                   candidate_id='PAULI_PRICE_CELL:TOY_LD5', physical_cell=dict(profile=dict(compiler='TOY')))
    return ctx, d, operators, np.array([H1, *F, tail]), request


class SplitCalTests(unittest.TestCase):
    def test_metadata_LD5_and_LD3_spec_reproduces_published_binding(self):
        m = D.metadata(D.SPECS[5]); r = m['request']
        self.assertEqual((r['prefix'], r['q'], r['R'], r['T'], r['step_time']), (5, 8, 4, 0.8, 0.1))
        self.assertNotEqual(r['candidate_id'], D.SPECS[5]['m12']['id'])
        self.assertTrue(m['rule_check']['equal'] and m['seed_checks']['unique_new'])
        self.assertEqual(m['seed_checks']['old_seed_count'], 1325 + 16)
        three = D.metadata(D.SPECS[3]); saved = D.read(LD3_BINDING)
        diff = sorted(k for k in set(three) | set(saved) if k not in D.FROZEN_KEYS and three.get(k) != saved.get(k))
        self.assertEqual(diff, ['driver'])
        bad = deepcopy(m); bad['step_seeds'][0][0] += 1
        with self.assertRaises(ValueError):
            D.D3.check_binding(bad, m)

    def test_toy_prefix5_path_through_legacy_core_and_receipts(self):
        import h46_long_time_native as native
        B1 = D.D3.legacy(); ctx, d, ops, blocks, req = toy()
        self.assertEqual(len(ctx.preparations[5].deterministic_blocks), 6)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); native.PRIVATE = folder; cache = folder / 'compiled_cache'; cache.mkdir()
            def fake(wrapper, fp, representative):
                record = dict(fingerprint=fp, compiler='TOY', metrics={k: 1 for k in native.METRICS})
                (cache / (fp + '.json')).write_text(json.dumps(record))
                for suffix in ('_wrapper.qpy.gz', '_compiled.qpy.gz'):
                    (cache / (fp + suffix)).write_bytes(b'ARTIFICIAL')
                return record, wrapper.copy(), False
            P = B1.B0.connection(); log = D.D3.c2().Progress(folder / 'p.jsonl')
            with D.D3.c2().accelerated_observed(P, log):
                r = B1.acquire_path(native, ctx, d, req, cohort='TOY', index=0, seeds=[5, 6], operators=ops,
                                    blocks=blocks, compile_cached=fake)
            binding = dict(request=req, cohort='TOY', step_seeds=[[5, 6]], dictionary_identity=dict(artificial=True))
            path = dict(index=0, cohort='TOY', request=req, dictionary_identity=binding['dictionary_identity'], result=r)
            source = dict(commit='0' * 40, caller='ARTIFICIAL', cell='Q8R4')
            receipts = B1.receipts().compile_receipts([path], source, binding['dictionary_identity'], folder)
            D.check_result_path(path, binding, receipts, source)
            bad = deepcopy(binding); bad['cohort'] = 'OTHER'
            with self.assertRaises(ValueError):
                D.check_result_path(path, bad, receipts, source)

    @unittest.skipUnless((LD3_RUN / 'work' / 'Q8R660_p0_X.npz').exists(), 'LD3 regenerable token arrays not present')
    def test_counting_reproduces_LD3_coefficients_and_saved_totals(self):
        import pauli_ld2_step_audit as A
        A.RUN, A.WORK = LD3_RUN, LD3_RUN / 'work'
        results = {pid: json.loads((LD3_RUN / ('PATH_%s.json' % pid.replace('/', '_'))).read_text())
                   for pid in ('Q8R660/p0', 'Q8R660/p1')}
        arrays = {pid: A.load_x_arrays(pid, 8, r['n_wires']) for pid, r in results.items()}
        edge, S, F, P = SA.AU.edges(arrays, 8)
        coef = SA.coefficients(edge['constants'], 2, SA.AU.bodies(arrays, 8, S, F, P), paths=2, q=8, R=660)
        self.assertEqual((coef['A'], coef['H'], coef['E'], coef['events_in_denominator']), (313, 2685, Fraction(7793, 1155), 9240))
        cfg = A.git_json(SA.LD3_CONFIG['commit'], SA.LD3_CONFIG['path'])
        out = SA.compare(coef, coef, cfg, cfg)   # LD3 against itself: saved totals reproduced, zero difference
        for v in out.values():
            self.assertEqual(v['difference_display'], 0.0)
        with self.assertRaises(ValueError):
            SA.coefficients(edge['constants'], 2, {'Q8R660/p0': []}, paths=2, q=8, R=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
