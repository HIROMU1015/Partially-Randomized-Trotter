"""Synthetic/off-domain checks only; registered tables are not scored by tests."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from itertools import product
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("g2", ROOT/"scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def synthetic():
    xs = "1/3"  # not a registered science input
    cols, raw_rows = [], []
    for name, precision in product(g.ORDER, g.EPS):
        degree, a, b = g.vectors(F(xs))[name]
        ev = []
        for labels in product((0, 1), repeat=degree+int(b != 0)):
            p = F(1)
            for j in labels:
                p *= (F(3, 4), F(1, 4))[j]
            label = str(degree)+":"+str(labels)
            ev.append({"source_label": label, "label_probability": str(p), "word": list(labels[:degree]),
                       "rotation": labels[-1] if b else None, "rotation_sign": 1, "phase_i_power": (-degree)%4,
                       "complement": name == "A1", "native_cost": {
                           "T": 0 if name == "P2" and labels == (0, 0) else 8+sum(labels),
                           "CX": 2, "1Q": 10, "strict_event_error_upper": "1/1000000",
                           "IR_sha256": g.digest([name, labels]), "IR_gate_count": 3}})
        norm = g.sqrt_i(a*a+b*b)
        direction = [(g.I(a)/norm).json() if k == degree else (g.I(b)/norm).json()
                     if k == degree+1 else g.I(0).json() for k in range(4)]
        c = {"id": name+":"+precision, "prototype": name, "epsilon": precision, "degree": degree,
             "saved_ideal_ab_exact": [str(a), str(b)], "D_intervals": [[d['lo'],d['hi']] for d in direction],
             "workspace_peak": 1, "d_upper": "1/500000", "events": ev,
             "costs": {k: str(sum(F(e['label_probability'])*F(e['native_cost'][k]) for e in ev)) for k in g.RESOURCES}}
        c['implementation_identity_sha256'] = g.digest({k: c[k] for k in (
            "D_intervals", "costs", "d_upper", "workspace_peak", "events")})
        cols.append(c)
    by_id = {c['id']: c for c in cols}
    arms = {"ordinary": ('O0','O2'), "PTSC_K0": ('O0','P2','P3'), "A": ('A0','A1','A2')}
    for arm, precision, sigma in product(arms, g.EPS, (1, -1)):
        events = []
        for name in arms[arm]:
            for e in by_id[name+":"+precision]['events']:
                events.append({"label": e['source_label'], **{k: v for k,v in e.items() if k != 'source_label'}})
        raw_rows.append({"context": "distinct_basis", "controlled": True, "x": xs, "arm": arm,
                         "epsilon": precision, "sigma": sigma, "profile": {"events": events}})
    seq = "HTt"
    raw = json.dumps({"runs": 1, "retries": 0, "mandatory_STOP": True, "resource_rows": raw_rows,
                      "synthesis_rows": [{"sequence": seq, "sequence_sha256": g.digest_bytes(seq.encode()),
                         "T_count": 2, "Tdagger_count": 1, "error_pass": True,
                         "strict_operator_error_upper": "1/1000000"}]}).encode()
    return {"source_result_sha256": g.digest_bytes(raw), "tables": {
        xs: {"columns": cols, "distinct_columns": 21, "workspace_exclusions": []}}}, raw


class ExactArithmetic(unittest.TestCase):
    def test_root_encloses_exact_rational(self):
        for v in (F(0), F(4), F(7, 13), F(10**-6)):
            z = g.sqrt_i(v)
            self.assertLessEqual(z.lo*z.lo, v)
            self.assertGreaterEqual(z.hi*z.hi, v)

    def test_root_rejects_negative(self):
        with self.assertRaises(ValueError): g.sqrt_i(-1)

    def test_signed_interval_multiplication(self):
        z = g.I(-3, 2)*g.I(4, 5)
        self.assertEqual((z.lo,z.hi), (F(-15), F(10)))

    def test_zero_crossing_denominator(self):
        with self.assertRaises(ZeroDivisionError): g.I(1)/g.I(-1,1)

    def test_log_enclosure(self):
        from decimal import Decimal, localcontext
        z = g.log_integer(10560)
        with localcontext() as ctx:
            ctx.prec = 100
            value = F(Decimal(10560).ln())
        self.assertLess(z.lo, value)
        self.assertGreater(z.hi, value)

    def test_mean_all_vertices_off_domain(self):
        for x in (F(1,3), F(2), F(7,5)):
            for v in g.vertices(x).values(): g.mean_check(x,v)

    def test_profile_combinatorics_off_domain(self):
        counts = [3**sum(v>0 for v in w.values()) for w in g.vertices(F(1,3)).values()]
        self.assertEqual(counts, [9,27,27,81,81,27])
        self.assertEqual(sum(counts),252)

    def test_mean_tamper(self):
        v = g.vertices(F(1,3))['A'];v['A0'] += 1
        with self.assertRaises(ValueError): g.mean_check(F(1,3),v)

    def test_confidence_bias_exhausted(self):
        z = g.finite_confidence(g.I(1),g.I(1),g.I(0),{'T':g.I(1)},g.I(8),1000)
        self.assertEqual(z['status'],'BIAS_EXHAUSTED_OR_UNRESOLVED')

    def test_confidence_range_and_shot_cap(self):
        a = g.finite_confidence(g.I(1),g.I(1),g.I('1/10'),{'T':g.I(1)},g.I(8),10**9)
        b = g.finite_confidence(g.I(1),g.I(5),g.I('1/10'),{'T':g.I(1)},g.I(8),10**9)
        self.assertGreater(b['shots_per_axis_enclosure'][0], a['shots_per_axis_enclosure'][1])
        z = g.finite_confidence(g.I(1),g.I(1),g.I('1/10'),{'T':g.I(1)},g.I(8),100)
        self.assertEqual(z['status'],'SHOT_CAP')

    def test_zero_mass_nonpositive_stat_does_not_beat_components(self):
        # Supplemental witness for the general proof, not a proof by enumeration.
        k, s = F(3), F(2)
        mixed_k,mixed_s = F(1,2)*k+F(1,2)*4, F(1,2)*s+F(1,2)*(-1)
        self.assertGreater(mixed_k/mixed_s,k/s)

    def test_E_sqrt_is_not_sqrt_E(self):
        h = F(1,2)*g.sqrt_i(1)+F(1,2)*g.sqrt_i(9)
        self.assertLess(h.hi, g.sqrt_i(5).lo)


class SavedIdentity(unittest.TestCase):
    def test_synthetic_saved_identity(self):
        t, raw = synthetic(); self.assertEqual(g.verify_inputs(t, raw)['saved_synthesis_rows_verified'],1)

    def test_raw_sha_tamper(self):
        t, raw = synthetic()
        with self.assertRaises(ValueError): g.verify_inputs(t,raw+b' ')

    def test_column_tamper(self):
        t, raw = synthetic();t['tables']['1/3']['columns'][0]['events'][0]['phase_i_power'] += 1
        with self.assertRaises(ValueError): g.verify_inputs(t,raw)

    def test_rehashed_phase_tamper(self):
        t, raw = synthetic();c=t['tables']['1/3']['columns'][0];c['events'][0]['phase_i_power'] += 1
        c['implementation_identity_sha256']=g.digest({k:c[k] for k in ('D_intervals','costs','d_upper','workspace_peak','events')})
        with self.assertRaises(ValueError): g.verify_inputs(t,raw)

    def test_sequence_count_tamper(self):
        t, raw = synthetic();r=json.loads(raw);r['synthesis_rows'][0]['T_count']=9
        raw=json.dumps(r).encode();t['source_result_sha256']=g.digest_bytes(raw)
        with self.assertRaises(ValueError): g.verify_inputs(t,raw)

    def test_IS_zero_cost_infimum_is_not_finite_law(self):
        t,_=synthetic();cols={c['id']:g.column_data(c) for c in t['tables']['1/3']['columns']}
        w=g.vertices(F(1,3))['PTSC_K0'];p={k:'1e-4' for k,v in w.items() if v}
        r=g.evaluate_profile('1/3','PTSC_K0',w,p,cols,{'epsilon_axis':'1/200','alpha_axis':'1/5280','shot_cap_per_axis':10**9})
        self.assertEqual(r['IS']['T']['attainment'],'UNATTAINED_NET_COST_INFIMUM')
        self.assertEqual(r['IS']['T']['finite_confidence']['status'],'MISSING_FINITE_PROPOSAL_ZERO_COST')
        self.assertEqual(r['IS']['1Q']['finite_confidence']['status'],'CONDITIONAL_ANALYTIC_PASS')
        self.assertFalse(r['IS']['1Q']['finite_confidence']['sampler_built'])


if __name__ == '__main__':
    unittest.main()
