"""Artificial fixtures only; no real H4 scalar evaluations."""
import copy
from fractions import Fraction as F
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('h4_cng_private'); pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['h4_cng_private']=pkg
from h4_cng_private import h4_central_cng as A
P,C=A.P,A.C


def fixture():
    rows=[]; samples={c:[] for c in A.CASES}
    for c in A.CASES:
        for i in range(1 if c.endswith('_PF') else 256):
            old=dict(trajectory_id=c+str(i),trajectory_record_hash='toy'+str(i),RZ=dict.fromkeys(A.AXES,100+i))
            samples[c].append(old)
            rows.append(dict(case=c.rsplit('_',1)[1],index=i,trajectory_id=old['trajectory_id'],
                             trajectory_record_hash=old['trajectory_record_hash'],generic_RZ=old['RZ'],
                             cosine=dict(RZ=i+1),sine=dict(RZ=i+2)))
    return rows,samples


def toy_function(case):
    ax=dict(s_exact='1',u_enc_exact='0',rho_guard_exact='1/101',domain_nonempty=True,
            scale=dict(A_lower='10000',A_upper='10000'))
    return dict(case=case,epsilon_sig='0.1',domain_nonempty=True,axes={a:copy.deepcopy(ax) for a in A.AXES})


class CentralTests(unittest.TestCase):
    def test_reference_scope_allows_distinct_represented_H(self):
        original={}
        for rep in ('pauli_jw','df_native'):
            for method in ('PF','PR'):
                original[rep+'_'+method]=dict(state=dict(hash='same-ray'),reference=dict(T=.8,method='saved',hamiltonian_hash=rep,signal=rep))
        refs=A.reference_scopes(original)
        self.assertNotEqual(refs['pauli_jw'],refs['df_native'])
        original['df_native_PR']['reference']['signal']='changed'
        with self.assertRaises(ValueError):A.reference_scopes(original)

    def test_reference_scope_rejects_state_or_time_change(self):
        for field in ('state','time'):
            original={c:dict(state='same',reference=dict(T=.8,method='saved')) for c in (*A.CASES,'df_native_PF','df_native_PR')}
            if field=='state':original['df_native_PF']['state']='wrong'
            else:
                for method in ('PF','PR'):original['df_native_'+method]['reference']['T']=.9
            with self.assertRaises(ValueError):A.reference_scopes(original)

    def test_join_257_artificial_prices(self):
        rows,s=fixture(); p=A.bind_prices(list(reversed(rows)),s)
        self.assertEqual(len(p['pauli_jw_PR']),256)
        self.assertEqual(p['pauli_jw_PF'],[dict(cosine=1,sine=2)])

    def test_corrupt_id_and_hash_rejected(self):
        for key in ('trajectory_id','trajectory_record_hash'):
            r,s=fixture();r[5][key]='corrupt'
            with self.assertRaises(ValueError):A.bind_prices(r,s)

    def test_duplicate_index_rejected(self):
        r,s=fixture();r[-1]['index']=0
        with self.assertRaises(ValueError):A.bind_prices(r,s)

    def test_missing_and_extra_rejected(self):
        r,s=fixture()
        for rr in (r[:-1], r+[dict(case='OTHER')]):
            with self.assertRaises(ValueError):A.bind_prices(rr,s)

    def test_prices_integer_nonnegative(self):
        for value in (-1,1.0,True):
            r,s=fixture();r[0]['sine']['RZ']=value
            with self.assertRaises(ValueError):A.bind_prices(r,s)

    def test_generic_join_rejected(self):
        r,s=fixture();r[0]['generic_RZ']=dict(cosine=0,sine=0)
        with self.assertRaises(ValueError):A.bind_prices(r,s)

    def test_cached_comparison_identical_without_shot_calls(self):
        l,r=toy_function('toy_PF'),toy_function('toy_PR')
        means={'toy_PF':dict.fromkeys(A.AXES,'2'),'toy_PR':dict.fromkeys(A.AXES,'1')}
        expected=P.comparison(l,r,means)
        bounds={}
        for f in (l,r):
            ev={a:dict(rho_zero=P.shot(f['axes'][a],0),guard_face=P.shot(f['axes'][a],F(1,101))) for a in A.AXES}
            bounds[f['case'],'0.1']=A.guard_range(ev,means[f['case']])
        with patch.object(P,'shot',side_effect=AssertionError('DF/duplicate shot forbidden')):
            actual=A.cached_comparison(l,r,bounds)
        self.assertEqual(actual,expected)

    def test_cached_overlap_keeps_ceil_partition(self):
        l,r=toy_function('toy_PF'),toy_function('toy_PR')
        ranges={('toy_PF','0.1'):[F(100),F(102)],('toy_PR','0.1'):[F(101),F(103)]}
        self.assertEqual(A.cached_comparison(l,r,ranges)['empirical_order'],'CEIL_PARTITION_DEPENDENT')

    def test_null_range_and_empty_guard(self):
        l,r=toy_function('toy_PF'),toy_function('toy_PR')
        ranges={('toy_PF','0.1'):None,('toy_PR','0.1'):[F(1),F(2)]}
        self.assertEqual(A.cached_comparison(l,r,ranges)['status'],'NUMERICAL_BOUND_UNRESOLVED')
        l['domain_nonempty']=False
        self.assertEqual(A.cached_comparison(l,r,ranges)['status'],'BASE_GUARD_DOMAIN_EMPTY')

    def test_paired_G_covariance_not_axis_independence(self):
        _,stats,_=P.reuse()
        prices=[dict(cosine=1,sine=1),dict(cosine=3,sine=3)]
        d=P.paired(prices,dict(cosine=2,sine=3),stats)
        self.assertEqual(d['mean_exact'],'10');self.assertAlmostEqual(d['standard_error'],5)
        anti=P.paired([dict(cosine=1,sine=3),dict(cosine=3,sine=1)],dict.fromkeys(A.AXES,2),stats)
        self.assertEqual(anti['standard_error'],0)

    def test_singleton_deterministic(self):
        _,stats,_=P.reuse();d=P.paired([dict(cosine=4,sine=7)],dict(cosine=2,sine=3),stats)
        self.assertEqual(d['mean_exact'],'29');self.assertEqual(d['standard_error'],0);self.assertEqual(d['nominal95_halfwidth'],0)

    def test_native_exit_and_reap_toy(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'native';self.assertEqual(C.witness(p,[sys.executable,'-B','-c','pass']),0)
            self.assertTrue(C.read(p/'STOP.json')['all_owned_reaped'])
            self.assertTrue(C.read(p/'worker_native_exit.json')['reaped'])

    def test_nonrefundable_batch_budget(self):
        with tempfile.TemporaryDirectory() as td:
            b=C.CallBudget(Path(td),dict(toy=1));b.take('toy')
            with self.assertRaises(ValueError):b.take('toy')
            with self.assertRaises(ValueError):b.take('compile')

    def test_scope_and_no_quantum_import(self):
        self.assertEqual(len(A.CASES)*len(A.EPS),10);self.assertEqual(len(A.PAIRS)*len(A.EPS),15)
        self.assertFalse(any(x.startswith(('qiskit','pyscf','openfermion','trotterlib.chemistry')) for x in sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
