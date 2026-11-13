"""Artificial noncommuting H/state only; no saved molecular numerical calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('mock_B_signal',ROOT/'tests/check_pr_rpe_signal_plan.py')
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
import numpy as np
from scipy.linalg import expm
from trottertracks import pr_rpe_signal_provider as P
from trottertracks.pr_rpe_power_mean import kernels

SOURCE=dict(fixture='ARTIFICIAL_ONLY')


def invented(config):
    x=np.array([[0.,1.],[1.,0.]],complex);z=np.diag([1.,-1.]).astype(complex)
    matrices=[.005*(i+1)*(x if i%2 else z)+.0002*i*np.eye(2) for i in range(11)]
    state=np.array([1.,.2j]);state/=np.linalg.norm(state)
    prefix=config['spec']['L_D']
    tails={prefix:dict(scalar=.003*(10-prefix),lambda_R=.5,source=config['spec']['split']['lambda_source'])}
    return P.make_context(matrices,state,.17,tails,binding=config['spec']['binding'],source=SOURCE)


def reference(ctx,job):
    c=job['physical_context']['physical_candidate'];H=sum(ctx.matrices,np.zeros((2,2),complex))+ctx.constant*np.eye(2)
    value=complex(np.vdot(ctx.state,expm(-1j*c['T']*H)@ctx.state));z=dict(real=value.real,imag=value.imag)
    return dict(binding=deepcopy(ctx.binding),T=c['T'],source=SOURCE,
        evidence=dict(reference=z,independent_reference=z,signal_difference=0.))


def setup(**kwargs):
    b,config,physical,_=fixture_module.fixture(**kwargs)
    provider=P.SignalBatch(config,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,
        source=SOURCE,root=ROOT,context_factory=invented,reference_provider=lambda j:reference(invented(config),j))
    return provider,invented(config)


def sequential(ctx,job):
    """Small q/R direct application, independent product-loop reference."""
    from trotterlib.rte import finite_rte_distribution
    c=job['physical_context']['physical_candidate'];st=job['physical_context']['structure']
    tail=ctx.tails[c['prefix']];A=tail['matrix'];v=np.repeat(ctx.state[:,None],3,axis=1);logB=0.
    for _ in range(c['q']):
        for w,r in zip(st['signed_weights'],st['allocation']):
            t=c['T']*w/c['q'];dt=t/r
            heads=[expm(-.5j*t*a) for a in ctx.matrices[:c['prefix']+1]]
            for H in heads:v=H@v
            v[:,0]=expm(-1j*t*A)@v[:,0]
            term=np.eye(2,dtype=complex);poly=term.copy()
            for k in range(1,c['K']+2):term=term@(-1j*dt*A/k);poly+=term
            b=finite_rte_distribution(tail['lambda_R']*dt,c['K']).exact_finite_distribution
            for _ in range(r):
                v[:,1]=poly@v[:,1];v[:,2]=(poly/b)@v[:,2]
            logB+=r*np.log(b)
            for H in reversed(heads):v=H@v
    v*=np.exp(-1j*c['T']*(ctx.constant+tail['scalar']))
    return [complex(np.vdot(ctx.state,v[:,i])) for i in range(3)],logB


class SignalTests(unittest.TestCase):
    def test_noncommuting_prefix_signed_K_and_sequential_reference(self):
        for prefix,formula,K in ((1,'2nd',2),(3,'2nd',4),(2,'4th(m5_best)',6)):
            batch,ctx=setup(prefix=prefix,formula=formula,K=K,M=2,kappa='1/2')
            job=batch.pending['jobs'][-1];out=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
            expected,logB=sequential(ctx,job)
            self.assertEqual(out['status'],'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED')
            for key,value in zip(('exact_tail','corrected','raw'),expected):
                got=out['signal'][key];self.assertLess(abs(complex(got['real'],got['imag'])-value),2e-12)
            self.assertLess(abs(out['signal']['log_B']-logB),1e-12)
            self.assertEqual(sum(out['allocation']),job['physical_context']['physical_candidate']['R_outer'])
            if formula!='2nd':self.assertTrue(any(p['signed_time']<0 for p in out['piece_diagnostics']))

    def test_legacy_finite_RTE_kernel_matches_small_artificial_point(self):
        batch,ctx=setup(prefix=2,formula='4th(m5_best)',K=2,M=1)
        job=batch.pending['jobs'][-1];c=job['physical_context']['physical_candidate'];k=kernels(ROOT)
        specs=tuple(k.spectrum(a) for a in ctx.matrices)
        tail=ctx.tails[c['prefix']];ref=reference(ctx,job)
        original=SimpleNamespace(state=ctx.state,ham=SimpleNamespace(constant=ctx.constant),specs=specs,
            tails={c['prefix']:dict(spec=k.spectrum(tail['matrix']),scalar=tail['scalar'],lam=tail['lambda_R'])},
            refs={c['T']:complex(ref['evidence']['reference']['real'],ref['evidence']['reference']['imag'])})
        prior,_,_=k.evolve(original,[c]);out=P.evaluate(ctx,job,ref,root=ROOT)
        for key in ('raw','corrected','exact_tail'):
            a=prior[0]['signal'][key];b=out['signal'][key]
            self.assertLess(abs(complex(a['real'],a['imag'])-complex(b['real'],b['imag'])),1e-12)

    def test_scalar_applied_once_and_raw_propagates_independently(self):
        batch,ctx=setup(M=0);job=batch.pending['jobs'][0]
        first=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
        changed=deepcopy(ctx);changed.constant+=.31
        second=P.evaluate(changed,job,reference(changed,job),root=ROOT)
        for key in ('raw','corrected','exact_tail'):
            a=complex(first['signal'][key]['real'],first['signal'][key]['imag'])
            b=complex(second['signal'][key]['real'],second['signal'][key]['imag'])
            self.assertLess(abs(b-a*np.exp(-1j*.31*job['physical_context']['physical_candidate']['T'])),1e-13)
        clean=invented(batch.config);actual=np.linalg.matrix_power;count=0
        def corrupt_raw(a,n):
            nonlocal count
            count+=1;value=actual(a,n)
            return value*(1+1e-5) if count==2 else value
        with patch('numpy.linalg.matrix_power',side_effect=corrupt_raw):
            out=P.evaluate(clean,job,reference(clean,job),root=ROOT)
        self.assertEqual(out['status'],'NUMERICAL_FAILURE')
        self.assertIn('RAW_CORRECTED_CLOSURE_GATE',out['numerical_failure_reasons'])
        self.assertGreater(out['numerical']['closure'],1e-9)
        self.assertFalse(out['numerical']['gate_pass'])

    def test_exact_tail_failure_retains_diagnostics_not_corrected_unitarity(self):
        batch,ctx=setup(M=0);job=batch.pending['jobs'][0]
        macro,_=P._macros(ctx,job['physical_context']['physical_candidate'],job['physical_context']['structure'])
        macro['operators'][0]=macro['operators'][0]*(1+1e-6)
        out=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
        self.assertEqual(out['status'],'NUMERICAL_FAILURE')
        self.assertGreater(out['numerical']['power_unitarity_error'],1e-9)
        self.assertIsNotNone(out['signal']);self.assertIsNone(out['scientific_PASS_FAIL'])
        batch,ctx=setup(M=3);job=batch.pending['jobs'][-1]
        out=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
        self.assertTrue(out['numerical']['gate_pass'])
        c=out['signal']['corrected'];self.assertGreater(abs(abs(complex(c['real'],c['imag']))-1.),1e-9)

    def test_complete13_rounds_two_structures_no_optimizer_and_cache(self):
        for prefix,formula in ((1,'2nd'),(3,'4th(m5_best)')):
            batch,ctx=setup(prefix=prefix,formula=formula,M=12)
            batch.factory=lambda c:ctx;batch.reference=lambda j:reference(ctx,j)
            with tempfile.TemporaryDirectory(prefix='pr-mean-artificial-') as tmp:
                out=batch.run(Path(tmp)/'run')
                self.assertEqual(out['status'],'FINITE_MEAN_SCHEDULE_COMPLETE_NOT_RMSE_QUALIFIED')
                self.assertEqual([r['m'] for r in out['records']],list(range(13)))
                self.assertEqual(out['not_started_rounds'],[])
                self.assertIsNone(out['G']);self.assertIsNone(out['global_RMSE_bound'])
                for r in out['records']:
                    self.assertIsNone(r['N_X']);self.assertTrue(r['raw_corrected_independent_propagation'])
            job=batch.pending['jobs'][-1];out=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
            self.assertTrue(out['macro_cache_hit'])

    def test_failure_stops_batch_and_saved_evidence_never_falls_back(self):
        batch,ctx=setup(M=3);seen=[]
        def ref(job):
            seen.append(job['m']);r=reference(ctx,job)
            if job['m']==1:r['source']=None
            return r
        batch.factory=lambda c:ctx;batch.reference=ref
        with tempfile.TemporaryDirectory() as tmp:
            out=batch.run(Path(tmp)/'run')
            self.assertEqual(out['status'],'STOP_INPUT_OR_SOURCE_FAILURE')
            self.assertEqual(len(out['records']),1);self.assertEqual(out['not_started_rounds'],[1,2,3])
            self.assertEqual(seen,[0,1]);self.assertFalse(out['automatic_retry'])
            with self.assertRaises(ValueError):batch.run(Path(tmp)/'run')
        b,c,physical,_=fixture_module.fixture(M=1)
        old=batch.pending['jobs'][0]['physical_context']
        saved=dict(physical_context=old,source=SOURCE,status='NUMERICAL_FAILURE')
        # Use an exact context from this same plan, not a foreign M/source.
        fresh=P.SignalBatch(c,physical,b_validate=b.validate,b_exact_scalar=b.exact_scalar,source=SOURCE,
            root=ROOT,context_factory=lambda c:(_ for _ in ()).throw(AssertionError('factory')),
            reference_provider=lambda j:(_ for _ in ()).throw(AssertionError('reference')))
        saved['physical_context']=fresh.pending['jobs'][0]['physical_context']
        fresh.inventory=[saved];fresh.pending=fresh._plan()
        with tempfile.TemporaryDirectory() as tmp:
            out=fresh.run(Path(tmp)/'run')
            self.assertEqual(out['records'],[]);self.assertEqual(out['status'],'STOP_INPUT_OR_SOURCE_FAILURE')

    def test_input_candidate_lambda_and_reference_misuse_rejected(self):
        batch,ctx=setup(M=0);job=batch.pending['jobs'][0]
        for field in ('id','prefix','K'):
            bad=deepcopy(job);bad['physical_context']['physical_candidate'][field]='FOREIGN'
            with self.assertRaises((ValueError,TypeError)):P.evaluate(ctx,bad,reference(ctx,job),root=ROOT)
        bad=deepcopy(job);bad['B_identity']['configuration']['configuration_id']='FOREIGN'
        with self.assertRaises(ValueError):P.evaluate(ctx,bad,reference(ctx,job),root=ROOT)
        bad=deepcopy(job);bad['physical_context']['split']['lambda_source']={'foreign':'SOURCE'}
        with self.assertRaises(ValueError):P.evaluate(ctx,bad,reference(ctx,job),root=ROOT)
        bad=deepcopy(job);bad['physical_context']['split']['randomized_scope']='FULL_H'
        with self.assertRaises(ValueError):P.evaluate(ctx,bad,reference(ctx,job),root=ROOT)
        ref=reference(ctx,job);ref['T']=.2
        with self.assertRaises(ValueError):P.evaluate(ctx,job,ref,root=ROOT)
        with self.assertRaises(ValueError):P.make_context(ctx.matrices,ctx.state*2,ctx.constant,{},binding=ctx.binding,source=SOURCE)

    def test_normalizer_precision_failure_preserves_observation(self):
        batch,ctx=setup(M=0);job=batch.pending['jobs'][0]
        with patch('trotterlib.rte.finite_rte_distribution',return_value=SimpleNamespace(exact_finite_distribution=1.)):
            out=P.evaluate(ctx,job,reference(ctx,job),root=ROOT)
        self.assertEqual(out['status'],'NUMERICAL_FAILURE')
        self.assertEqual(out['numerical']['normalizer'],1.)
        self.assertIsNone(out['signal']);self.assertIsNone(out['N_X'])

    def test_saved_loader_and_reference_bridge_with_fictional_files_only(self):
        from trottertracks import h4_pr_rmse_completion as completion
        from trottertracks.h4_four_method_screen import PHYS
        batch,ctx=setup(M=0);config=deepcopy(batch.config)
        # A loader unit fixture, not a newly generated scientific B configuration.
        config['spec']['binding']['rank_policy']='MOCK_POLICY'
        binding=config['spec']['binding'];prefix=str(config['spec']['L_D'])
        metadata=dict(binding=binding,prefixes={prefix:dict(lambda_R=.5,scalar=.027)})
        data=json.dumps(metadata).encode();sha=hashlib.sha256(data).hexdigest()
        index=dict(T20_reference_files=dict(H4=dict(path='/FICTIONAL/context.json',sha256=sha)))
        descriptor=dict(index=deepcopy(completion.INDEX),context_sha256=sha,selector=['prefixes',prefix])
        config['spec']['split']['lambda_source']=descriptor
        job=deepcopy(batch.pending['jobs'][0]);job['physical_context']['binding']=binding
        ref=reference(ctx,job)
        physical=dict(binding=binding,rounds=[dict(physical_candidate=job['physical_context']['physical_candidate'],
            diagnostics=dict(reference_pair=ref['evidence']))])
        store=SimpleNamespace(read=lambda r:deepcopy(physical if r==PHYS else index))
        saved=SimpleNamespace(matrices=ctx.matrices,state=ctx.state,ham=SimpleNamespace(constant=ctx.constant))
        with patch.object(P.common,'SavedArtifacts',return_value=store), \
                patch.object(Path,'read_bytes',return_value=data), \
                patch.object(completion,'load_context',return_value=(saved,{'fixture':'STATE_METADATA'})) as loader:
            loaded=P.load_h4_context(ROOT,config)
            self.assertEqual(loaded.tails[1]['source'],descriptor);loader.assert_called_once()
            pair=P.saved_h4_reference(ROOT,job,store=store)
            self.assertEqual(pair['evidence'],ref['evidence'])
            self.assertEqual(pair['binding'],binding)
            wrong=deepcopy(config);wrong['spec']['split']['lambda_source']=SOURCE
            with self.assertRaises(ValueError):P.load_h4_context(ROOT,wrong)
            absent=deepcopy(config);absent['spec']['L_D']=4
            with self.assertRaises(ValueError):P.load_h4_context(ROOT,absent)
            loader.assert_called_once()  # neither bad descriptor reaches array loading

    def test_import_and_construction_do_not_load_arrays_or_science(self):
        code="import sys;sys.path.insert(0,'src');import trottertracks.pr_rpe_signal_provider;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
        with patch.object(P,'load_h4_context',side_effect=AssertionError('real H4')):
            batch,ctx=setup(M=0)
            self.assertEqual(batch.pending['science_calls'],0)
        with self.assertRaises(ValueError):
            b,c,physical,_=fixture_module.fixture(formula='4th')


if __name__=='__main__':unittest.main()
