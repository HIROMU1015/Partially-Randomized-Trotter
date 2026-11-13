"""Published metadata, artificial laws/kernels, mock batch; no physical run."""
from copy import deepcopy
from decimal import Decimal,localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
from trottertracks import h4_structural_schedule as s
from trottertracks import h4_structural_kernel as k
from check_h4_structural_kernel import artificial_model


def toy_manifest():
    rows=[]
    for m,R in enumerate(s.EXPECTED_R):
        records=[dict(m=m,axis=a,quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',mean_S=7.,mean_compiled_C=None,fixture='ARTIFICIAL') for a in ('cosine','sine')] if m in s.EXISTING else []
        rows.append(dict(m=m,saved_values=records,identity=dict(fixture='ARTIFICIAL',m=m),law_binding=dict(fixture='ARTIFICIAL'),
            structure=dict(event_count=(2**m)*R),physical_candidate=dict(id='MOCK_'+str(m),q=2**m,R_outer=R,T=.1*(2**m))))
    return dict(configuration_id='MOCK_ONLY',requests=rows,existing_source=dict(fixture='ARTIFICIAL'))


def fake_results(row):
    results={}
    for q in range(9):
        m=k.MeanKernel(('MOCK_FIXED',),np.ones((1,1)),np.array([1.]))
        band=k.Band(m,m.transition,m.transition,m.reward,m.reward)
        results[q]=[dict(kernel=band,entry_indices=[0,0],prefix=[1,0],terminal=np.array([2.]),
            suffix=dict(cosine=3,sine=4),metadata=dict(wire=q,R=row['physical_candidate']['R_outer']))]
    return results


class MockBackend:
    def __init__(self,*args):self.calls=[];self.closed=False;self.fail=None
    def k0(self,row,out):
        self.calls.append(('K0',row['m']))
        return fake_results(row),dict(status=self.fail if self.fail and row['m']==2 else 'PASS')
    def k1(self,row,results):
        assert (next(Path(tempdir).glob('m'+str(row['m'])+'/K0.json'))).exists()
        self.calls.append(('K1',row['m']))
        law=dict(m=row['m'],candidate_id=row['physical_candidate']['id'],q_PF=row['physical_candidate']['q'],R_outer=row['physical_candidate']['R_outer'])
        return k.run_K1(results,[law],probability_enclosure=s.probability_enclosure)
    def close(self):self.closed=True


class Tests(unittest.TestCase):
    def test_real_metadata_13_exact_R_kappa_and_saved_values_no_scientific_entry(self):
        with patch.object(k,'Model',side_effect=AssertionError('No physical model')),patch.object(k,'run_K1',side_effect=AssertionError('No new means')),patch.object(s,'make_laws',side_effect=AssertionError('No physical laws')):
            plan=s.prepare(ROOT)
        self.assertEqual([r['physical_candidate']['R_outer'] for r in plan['requests']],list(s.EXPECTED_R))
        self.assertEqual(plan['remaining_rounds'],list(s.REMAINING));self.assertEqual(plan['M'],12)
        old=s.read(ROOT,s.BASE,s.SAVED+'K1.json')['values']
        copied=[v for r in plan['requests'] for v in r['saved_values']]
        self.assertEqual(copied,old);self.assertTrue(all(r['law_probabilities'] is None for r in plan['requests']))
        self.assertTrue(all(r['mean_S']==dict(cosine=None,sine=None) for r in plan['requests'] if r['K0_required']))
        a=k.c.read(ROOT,'7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75','docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json')
        self.assertEqual(plan['exact_kappa'],a['conditions']['kappa_paper_exact'])

    def test_same_gamma_formula_fast_interval_contains_exact_small_and_large_reference(self):
        gamma=Fraction(16,2**53-16)
        for n in (0,2,8192):
            lo,hi,relative=s.probability_enclosure(3.,5.,n)
            self.assertLessEqual(Fraction.from_float(lo),3*(1-gamma)**n)
            self.assertGreaterEqual(Fraction.from_float(hi),5*(1+gamma)**n)
            self.assertGreaterEqual(Fraction.from_float(relative),(1+gamma)**n-1)
        # Synthetic exponent only, not an H4 mean. No O(n)-bit integer power.
        n=4096*8160;lo,hi,_=s.probability_enclosure(3.,5.,n)
        with localcontext() as ctx:
            ctx.prec=100;g=Decimal(16)/Decimal(2**53-16)
            self.assertLessEqual(Decimal.from_float(lo),3*(1-g)**n)
            self.assertGreaterEqual(Decimal.from_float(hi),5*(1+g)**n)
        _,steps=s.positive_power_interval((1.,1.),n)
        self.assertLessEqual(steps,2*n.bit_length())

    def test_new_law_adapter_only_remaining_11_and_different_angles(self):
        plan=toy_manifest();remaining=[r for r in plan['requests'] if r['m'] not in s.EXISTING];seen=[]
        def law(context,tail,time,R):
            seen.append(R);tau=.75*time/R
            return tau,None,NS(orders=(0,2),unnormalized_order_weights=(1.,tau*tau),order_probabilities=(1/(1+tau*tau),tau*tau/(1+tau*tau)),exact_finite_distribution=1+tau*tau)
        context=NS(ctx=NS(prep=NS(rte_preparation=NS(symbolic_tail='ARTIFICIAL'))))
        with patch.object(k.c,'make_law',side_effect=law):
            laws=s.make_laws(context,remaining)
            with self.assertRaises(ValueError):s.make_laws(context,[plan['requests'][0]])
        self.assertEqual(seen,[s.EXPECTED_R[m] for m in s.REMAINING])
        self.assertEqual(len({v['rotation_angles']['0'] for v in laws}),11)
        self.assertEqual([v['T'] for v in laws],[r['physical_candidate']['T'] for r in remaining])

    def test_total_time_hook_preserves_old_default_and_never_aliases_other_rounds(self):
        self.assertEqual(k.Model.total_time(dict(R_outer=2)),.1)
        self.assertEqual(k.Model.total_time(dict(R_outer=128)),6.4)
        self.assertEqual(k.Model.total_time(dict(R_outer=255,T=12.8)),12.8)
        self.assertEqual(k.Model.total_time(dict(R_outer=8160,T=409.6)),409.6)

    def test_mock_complete_batch_skips_existing_and_saves_pass_before_K1(self):
        global tempdir
        b=MockBackend()
        with tempfile.TemporaryDirectory() as td:
            tempdir=Path(td)/'fresh';s.execute(ROOT,'MOCK',tempdir,backend_factory=lambda *a:b,manifest=toy_manifest())
            data=json.loads((tempdir/'COVERAGE.json').read_text())
            self.assertEqual(len(data['values']),26);self.assertEqual(data['missing_rounds'],[])
            self.assertTrue(b.closed);self.assertIsNone(data['mean_compiled_C'])
            self.assertEqual([m for stage,m in b.calls if stage=='K0'],list(s.REMAINING))
            self.assertFalse((tempdir/'m0').exists());self.assertFalse((tempdir/'m6').exists())

    def test_mock_UNKNOWN_FAIL_stops_next_round_and_no_K1_for_failed_round(self):
        global tempdir
        for status in ('UNKNOWN','FAIL'):
            b=MockBackend();b.fail=status
            with tempfile.TemporaryDirectory() as td:
                tempdir=Path(td)/'fresh';s.execute(ROOT,'MOCK',tempdir,backend_factory=lambda *a:b,manifest=toy_manifest())
                self.assertEqual(b.calls,[('K0',1),('K1',1),('K0',2)])
                self.assertFalse((tempdir/'m2/K1.json').exists());self.assertFalse((tempdir/'m3').exists())
                self.assertEqual(len(json.loads((tempdir/'COVERAGE.json').read_text())['values']),6)
                self.assertIn(status,json.loads((tempdir/'STOP.json').read_text())['status'])

    def test_compiled_quantity_rejected_and_partial_kernel_survives(self):
        global tempdir
        class Wrong(MockBackend):
            def k1(self,row,results):
                values=super().k1(row,results)
                for v in values:v['quantity']='FULL_WRAPPER_MEAN_COMPILED_PRICE'
                return values
        b=Wrong()
        with tempfile.TemporaryDirectory() as td:
            tempdir=Path(td)/'fresh'
            with self.assertRaises(ValueError):s.execute(ROOT,'MOCK',tempdir,backend_factory=lambda *a:b,manifest=toy_manifest())
            self.assertEqual(json.loads((tempdir/'m1/K0.json').read_text())['status'],'PASS')
            self.assertTrue((tempdir/'m1/KERNELS.npz').exists());self.assertFalse((tempdir/'m1/K1.json').exists())
            self.assertEqual(json.loads((tempdir/'STOP.json').read_text())['stage'],'K1')

    def test_native_synthetic_kernel_new_R_T_phase_boundary_and_large_power(self):
        original=artificial_model()
        # Build a new-R *artificial* model using the old fixture's own descriptor.
        context=NS(Accounting=original.A,blocks=[original.blocks],scalar=original.scalar,
            ctx=NS(prep=NS(preparation_hash='ARTIFICIAL',rte_preparation=NS(symbolic_tail=NS(components=original.components)))))
        def native_ops(comp):return [dict(name=o.name,qubits=list(o.qubits),parameters=list(o.parameters)) for o in comp.basis_change_operations]
        old=dict(preparation=dict(hash='ARTIFICIAL',components=[dict(id=c.component_id,coefficient_abs=c.coefficient_abs,probability_hex=c.probability.hex(),support=list(c.diagonal_pauli_support),sign=c.coefficient_sign) for c in original.components],
            basis_registry=[dict(id=key[0],hash=key[1],operations=native_ops(next(c for c in original.components if c.basis_id==key[0]))) for key in original.keys]),
            laws=[dict(original.laws[0],R_outer=255,q_PF=128,m=7,T=12.8,candidate_id='MOCK_NEW_R')])
        model=k.Model(ROOT,context,old)
        wire=k.Wire(model,8);result=wire.build(model.laws[0])
        self.assertTrue(result['metadata']['all_terminal_exits_fixed'])
        for axis in ('cosine','sine'):
            word=model.suffix[255,axis]
            self.assertIn(-model.scalar*12.8,[o['angle'] for o in word if o['name']=='rz'])
        self.assertEqual(s.phase_checks(model,4),3*(3*4+1))
        # Pure one-state mock at large qR tests K1 arithmetic without events.
        row=toy_manifest()['requests'][12];law=dict(m=12,candidate_id='MOCK_LARGE',q_PF=4096,R_outer=8160)
        values=k.run_K1(fake_results(row),[law],probability_enclosure=s.probability_enclosure)
        self.assertEqual(len(values),2);self.assertIsNone(values[0]['mean_compiled_C'])


if __name__=='__main__':unittest.main(verbosity=2)
