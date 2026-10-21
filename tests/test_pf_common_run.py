"""Synthetic matrices, toy Hamiltonian, real fork and grant boundary tests only."""
import copy
import concurrent.futures
import math
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import pf_common_run_contract as c
import pf_common_run_native as native
import run_pf_common_run as run

class FakeStage:
    def __init__(self,out):self.out=Path(out);self.plan=dict(SOURCE='SYNTHETIC',binding=dict(hamiltonian='toy',state='toy'));self.calls=[]
    def take(self,*args):self.calls.append(args)

class Boundaries(unittest.TestCase):
    def test_unapproved_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c.save(p/'plan_APPROVED.json',dict(approval=dict(execution_authorized=False)))
            with self.assertRaisesRegex(ValueError,'APPROVAL'):c.Stage(p,'SIGNAL',dict(base_signal=1))
            self.assertFalse((p/'SIGNAL').exists())
    def test_global_reservations_nonrefundable(self):
        with tempfile.TemporaryDirectory() as d,patch.object(c,'verify'):
            c.save(Path(d)/'plan_APPROVED.json',dict(SOURCE='toy',system='H2'))
            s=c.Stage(d,'SIGNAL',dict(base_signal=1));s.take('base_signal','a')
            with self.assertRaisesRegex(ValueError,'GLOBAL_CAP'):s.take('base_signal','b')
            self.assertEqual(s.total()['base_signal'],1)
            with self.assertRaisesRegex(ValueError,'ALREADY_ENTERED'):c.Stage(d,'SIGNAL',dict(base_signal=1))
    def test_undeclared_call_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(c,'verify'):
            c.save(Path(d)/'plan_APPROVED.json',dict(SOURCE='toy',system='H4'))
            s=c.Stage(d,'QUALIFY',dict(endpoint=1))
            with self.assertRaises(ValueError):s.take('compile')
    def test_resources_no_constant_worker_cap(self):
        self.assertEqual(c.workers(dict(idle_cpu=128,available_ram=1000,free_disk=1000),200,1,1),128)
        self.assertEqual(c.workers(dict(idle_cpu=128,available_ram=7,free_disk=1000),200,1,1),7)
    def test_unknown_prevents_whole_minimum(self):
        rows=[dict(candidate=dict(formula='2nd'),status='UNDETERMINED')]
        self.assertEqual(run.best_complete(rows)['status'],'UNRESOLVED')
    def test_exact_cost_tie_and_ineligible_null(self):
        rows=[dict(candidate=dict(system='H2',method='B0',prefix=p,q=1,formula='2nd'),status='ELIGIBLE',total_RZ_exact='1000000000000000000001') for p in (2,1)]
        best=run.best_complete(rows);self.assertEqual(best['best']['candidate']['prefix'],1)

class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from scipy.linalg import expm
        from h46_new2_core import spectrum
        cls.np=np;cls.expm=staticmethod(expm);cls.spectrum=staticmethod(spectrum)
    def ctx(self):
        np=self.np;m=[np.array([[.2,.1],[.1,-.1]],complex),np.array([[.1,-.12j],[.12j,.3]],complex),np.diag([.04,-.06]).astype(complex)]
        v=np.array([1,1j],complex)/math.sqrt(2);scalar=.13;ref=self.expm(-.8j*sum(m))@v*np.exp(-.8j*scalar)
        return NS(np=np,oracle=NS(initial=v,specs=[self.spectrum(x) for x in m]),reference=ref,scalar=scalar,scalars={0:.03,1:.01,2:0.},preparations={p:NS(preparation_hash='toy'+str(p)) for p in range(3)}),m
    def test_all_formula_q_prefix0_full_independent_forward(self):
        ctx,m=self.ctx()
        with tempfile.TemporaryDirectory() as d:
            stage=FakeStage(d)
            for f in c.FORMULAS:
                for q in (1,2,4,8):
                    for p in (0,1,2):
                        with self.subTest(f=f,q=q,p=p):
                            row=dict(system='H2',method='FULL' if p==2 else 'B0',prefix=p,q=q,formula=f)
                            value=native.new_signal(stage,ctx,row,m)
                            self.assertLess(value['independent_forward_error'],1e-12)
                            if p!=2:
                                overlay=native.overlay(stage,ctx,dict(row,method='B0I'),value)
                                self.assertEqual(overlay['signal']['reference'],value['signal']['reference'])
                                self.assertNotEqual(overlay['signal']['corrected'],value['signal']['corrected'])
    def test_qualification_raises_u_and_exact_integers(self):
        ctx,m=self.ctx()
        with tempfile.TemporaryDirectory() as d:
            row=dict(system='H2',method='FULL',prefix=2,q=4,formula='4th(m5_best)')
            value=native.new_signal(FakeStage(d),ctx,row,m)
            value['old_u']=2e-8;value['old_margin']=1e-10
            ep=run.qualify_record(value,row,.05)
            self.assertGreaterEqual(ep['u'],2e-8);self.assertGreaterEqual(ep['log_B_margin'],1e-10)
            self.assertTrue(ep['decimal_crosscheck'])
            self.assertEqual(ep['status'],'ELIGIBLE');self.assertGreater(ep['N_cos'],0)
    def test_nonfinite_observation_rejected(self):
        value=dict(signal=dict(log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=float('nan'),raw_corrected_vector_closure=0.)),old_u=0.,old_margin=0.)
        with self.assertRaisesRegex(ValueError,'FINITE'):run.qualify_record(value,{},.05)
    def test_real_toy_fork_native_pipeline(self):
        np=self.np
        from trotterlib.df_hamiltonian import DFHamiltonian
        from h46_new2_native import OccupationOracle,prepare
        ham=DFHamiltonian(.17,np.array([[.1,.03j],[-.03j,-.2]],complex),np.array([.13]),(np.array([[.2,.1],[.1,-.3]],complex),),{'kind':'SYNTHETIC'})
        full=np.array([0.,1.,0.,0.],complex);oracle=OccupationOracle(ham,full,prefix=1)
        qstate=np.zeros(4,complex);qstate[oracle.qindices]=oracle.initial
        ctx=NS(np=np,ham=ham,oracle=oracle,qstate=qstate,reference=oracle.initial.copy(),scalar=ham.constant,
            scalars={0:.02,1:0.},preparations={p:prepare(ham,p) for p in (0,1)})
        rows=[dict(system='H2',method='FULL' if p else 'B0I',prefix=p,q=2,formula=f) for f in c.FORMULAS for p in (0,1)]
        import h46_m5_native as old
        from functools import partial
        import trotterlib.rte_compiled_cost as compiler
        from h2_m5_compiler_phase_guard import transpile
        with tempfile.TemporaryDirectory() as d,patch.object(compiler,'transpile',partial(transpile,num_processes=1)):
            native.STAGE=FakeStage(d);native.CTX=ctx;native.SELECTED={}
            for row in rows:
                local=NS(**ctx.__dict__,tail_identity=ctx.scalars[row['prefix']])
                sig=old.signal(local,dict(row,policy=None,r_value=None))
                native.SELECTED[c.cid(row)]=dict(signal=sig)
            with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('fork')) as pool:
                results=list(pool.map(native.cost_one,rows))
            self.assertEqual(len(results),6)
            self.assertTrue(all(r['status']=='PASS' and len(r['archives'])==6 for r in results))
            self.assertTrue(all(max(x['error'] for x in r['checks'])<1e-7 for r in results))
            self.assertTrue(all(r['prices']['cosine']['phase_guard']['unchanged_gate_stream'] for r in results))

if __name__=='__main__':unittest.main()
