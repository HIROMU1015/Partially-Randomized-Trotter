"""Synthetic matrices, metadata and child processes; no molecular input restored."""
import ast
import copy
import math
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_m5_signal_contract as c
import h46_m5_signal_adapter as adapter
import run_h46_m5_signal as runner


def toy_record(bias=0.,closure=0.):
    return dict(log_B=0.,total_signed_bias=dict(real=bias,imag=bias),
        allowance_evidence=dict(maximum_intermediate_norm=1.,raw_corrected_vector_closure=closure))


def synthetic_system_worker(cfg,system,threads,stop):
    if cfg.get('failed_system')==system:
        c.save(Path(cfg['output'])/(system+'_STOP.json'),dict(status='STOP',reason='SYNTHETIC_FAILURE'));return
    completed=[]
    for row in c.rows(system):
        c.reserve(cfg,row);new=row['id'] not in cfg['systems'][system]['reuse']
        path=Path(cfg['output'])/(row['id']+'.json')
        c.save(path,dict(status='SIGNAL_DIAGNOSTIC_PASS',candidate=row,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
            new_compile=0,new_sampling=0,cost_accounting=False,ranking=False,empirical_eligibility=c.eligibility(toy_record(),row)))
        completed.append(dict(id=row['id'],file=path.name,sha256=c.sha(path),new_signal=int(new)))
    c.save(Path(cfg['output'])/(system+'_complete.json'),dict(status='SIGNAL_SYSTEM_COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],system=system,
        results=completed,new_signal=sum(x['new_signal'] for x in completed),reused_signal=len(completed)-sum(x['new_signal'] for x in completed)))


class ContractTests(unittest.TestCase):
    def test_original_pilot_is_verified_at_its_sealed_root(self):
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/'original';(original/'scripts').mkdir(parents=True)
            source=original/'scripts/h46_m5_tol_contract.py';source.write_text('# synthetic original contract\n')
            config=Path(d)/'origin.json';payload=dict(SOURCE='a'*40,sealed_files=[dict(path=str(source))]);c.save(config,payload)
            with patch.object(c.subprocess,'check_output',return_value=source.read_bytes()) as git,patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')) as process:
                self.assertEqual(c.verify_original_pilot(config),payload)
                self.assertEqual(git.call_args.kwargs['cwd'],original)
                self.assertEqual(process.call_args.args[0][-2:],[str(original/'scripts'),str(config)])
            with patch.object(c.subprocess,'check_output',return_value=source.read_bytes()),patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='ORIGINAL_SEAL_MISMATCH')):
                with self.assertRaisesRegex(ValueError,'ORIGINAL_PILOT_SEALS'):c.verify_original_pilot(config)

    def test_ambiguous_original_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'origin.json';c.save(config,dict(sealed_files=[dict(path='/a/scripts/h46_m5_tol_contract.py'),dict(path='/b/scripts/h46_m5_tol_contract.py')]))
            with self.assertRaisesRegex(ValueError,'UNIQUE_ORIGINAL_PILOT_CONTRACT'):c.verify_original_pilot(config)

    def test_88_adopted_no_optional(self):
        self.assertEqual(len(c.rows()),88);self.assertEqual(len(c.rows('H4')),40);self.assertEqual(len(c.rows('H6')),48)
        self.assertTrue(all(x['role']=='PRIMARY' for x in c.rows()))
        self.assertEqual(c.scope()['maximum_new_candidate_signal'],82)
        for k in ('new_sampling','new_compile','new_circuit_state_action','SCF_DF_state_generation','new_full_H_reference'):self.assertEqual(c.scope()[k],0)
        self.assertEqual(c.scope()['allowance_policy']['reference_absolute_error'],None)
        self.assertFalse(c.scope()['allowance_policy']['compile_tolerance_used_for_shots'])

    def test_draft_cli_before_molecular_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft.json';c.save(p,dict(authorized=False,grant=dict(valid=False)))
            for mode in ('--execute','--supervise'):
                r=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h46_m5_signal.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_SIGNAL_EXECUTION_APPROVAL_REQUIRED',r.stdout)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_wrong_grant_budget_rejected(self):
        cfg=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64)
        grant=dict(valid=True,grant_id='toy',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],scope_id=c.SCOPE_ID,maximum_new_signal=82,maximum_compile=0,maximum_sampling=0,retry=False,next_phase_authorized=False)
        for key,value in [('maximum_new_signal',88),('maximum_compile',1),('maximum_sampling',1),('retry',True),('next_phase_authorized',True),('scope_id','OLD')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'EXACT_SIGNAL_ONLY_GRANT'):c.authorize(dict(cfg,grant=dict(grant,**{key:value})))

    def test_reservation_exclusive_and_domain_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'ledger').mkdir();cfg=dict(marker=str(r/'CONSUMED.json'),ledger=str(r/'ledger'),grant={},systems={s:dict(reuse={}) for s in ('H4','H6')},SOURCE='a'*40,plan_sha256='b'*64)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            row=c.rows()[0];c.reserve(cfg,row)
            with self.assertRaises(FileExistsError):c.reserve(cfg,row)
            with self.assertRaises(ValueError):c.reserve(cfg,dict(row,q=16))

    def test_allowance_eligibility_not_shots_or_certificate(self):
        row=c.rows()[0];result=c.eligibility(toy_record(),row)
        self.assertEqual(len(result['rows']),10);self.assertTrue(all(x['status_under_proposed_empirical_policy']=='ELIGIBLE' for x in result['rows']))
        self.assertTrue(all(x['shots'] is None and x['cost'] is None and x['strict_bias_certificate'] is None and x['reference_absolute_error'] is None for x in result['rows']))
        self.assertEqual(result['u'],1e-9)
        for record,expected in [(toy_record(1.),'INELIGIBLE'),(toy_record(0.,1e-4),'INELIGIBLE')]:
            self.assertTrue(any(x['status_under_proposed_empirical_policy']==expected for x in c.eligibility(record,row)['rows']))
        # Positive headroom with an empirical allowance above the guard ratio.
        record=toy_record(.0005/math.sqrt(2)-1e-9-1e-8)
        self.assertEqual(c.eligibility(record,row)['rows'][-1]['status_under_proposed_empirical_policy'],'UNDETERMINED')

    def test_nonfinite_allowance_rejected(self):
        for key in ('maximum_intermediate_norm','raw_corrected_vector_closure'):
            x=toy_record();x['allowance_evidence'][key]=float('nan')
            with self.assertRaises(ValueError):c.eligibility(x,c.rows()[0])
        x=toy_record();x['log_B']=float('inf')
        with self.assertRaises(ValueError):c.eligibility(x,c.rows()[0])

    def test_no_compile_sampling_or_state_solver_calls(self):
        tree=ast.parse(Path(adapter.__file__).read_text())
        names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(names&{'transpile','transpile_and_measure_cost','simulate_statevector','eigh','expm','sample','preflight','make_requests','compose'})
        self.assertNotIn('reference_job=True',Path(adapter.__file__).read_text())

    def test_two_system_processes_success_and_failure_isolation(self):
        for failed in (None,'H4'):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'run';systems={}
                for s in ('H4','H6'):
                    reuse={r['id']:{} for r in c.rows(s) if r['kind']=='DET_SIGNAL' and r['q']==2 and r['prefix'] in ((3,10) if s=='H4' else (5,19))}
                    systems[s]=dict(reuse=reuse)
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,systems=systems,failed_system=failed,
                    runtime=dict(python=sys.executable),run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                    resources=dict(system_RSS_proxy_bytes=dict(H4=1,H6=1)))
                fresh=dict(system_workers=2,threads_per_system=2)
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(runner,'fresh_resources',return_value=fresh),patch.object(runner.observed,'available_memory',return_value=10**10):
                    self.assertEqual(runner.execute(cfg,target=synthetic_system_worker),0 if failed is None else 1)
                    result=c.read(root/'completion_witness.json');self.assertTrue(all(x['exit']==0 and x['reaped'] for x in result['workers'].values()))
                    self.assertTrue((root/'output/H6_complete.json').exists())
                    with self.assertRaisesRegex(ValueError,'UNUSED_SIGNAL_RUN_ROOT'):runner.execute(cfg,target=synthetic_system_worker)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,'/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/provider/src')
        import numpy as np
        from h46_new2_core import spectrum
        from scipy.linalg import expm
        from trotterlib.rte import finite_rte_distribution
        cls.np=np;cls.spectrum=staticmethod(spectrum);cls.expm=staticmethod(expm);cls.distribution=staticmethod(finite_rte_distribution)

    def base(self):
        np=self.np
        matrices=[np.array([[.2,.1],[.1,-.1]],complex),np.array([[.1,-.12j],[.12j,.3]],complex),np.diag([.04,-.06]).astype(complex)]
        state=np.array([1,1j],complex)/math.sqrt(2);scalar=.13
        reference=self.expm(-.8j*sum(matrices))@state*np.exp(-.8j*scalar)
        full=SimpleNamespace(preparation_hash='full',deterministic_fragment_indices=(0,1),extracted_identity_coefficient=0.,exact_rte_lambda_r=0.)
        def prep(ham,p):return SimpleNamespace(preparation_hash='prefix'+str(p),deterministic_fragment_indices=tuple(range(p)),extracted_identity_coefficient=.01,exact_rte_lambda_r=.1)
        return SimpleNamespace(np=np,ham=SimpleNamespace(n_blocks=2),oracle=SimpleNamespace(initial=state,specs=[self.spectrum(m) for m in matrices]),matrices=matrices,
            fullprep=full,prepare=prep,reference=reference,scalar=scalar,context_key=('SYNTHETIC','H4','input-A'))

    def context(self,prefix):
        values=(.01,.1) if prefix==1 else (0.,0.)
        with patch('h46_new2_core.matrix_only_tail_coefficients',return_value=values):return adapter.prefix_context(self.base(),prefix)

    def test_prefix_tail_and_cache_identity(self):
        a,b=self.context(1),self.context(2)
        self.assertNotEqual(a.context_key,b.context_key);self.assertNotEqual(a.cut.preparation_hash,b.cut.preparation_hash)
        self.assertLess(self.np.linalg.norm((a.tail_spec[1]*a.tail_spec[0])@a.tail_spec[2]-(a.matrices[2]-.01*self.np.eye(2))),1e-14)
        self.assertEqual(b.lambda_R,0.);self.assertLess(self.np.linalg.norm(b.tail_spec[0]),1e-15)

    def test_full_prefix_uses_empty_tail_without_mocking_helper(self):
        # Real helper rejects prefix == n_blocks: do not mock this boundary.
        from h46_new2_core import matrix_only_tail_coefficients
        base=self.base()
        with self.assertRaisesRegex(ValueError,'INVALID_GENERATION_PREFIX'):
            matrix_only_tail_coefficients(base.ham,base.ham.n_blocks)
        ctx=adapter.prefix_context(base,base.ham.n_blocks)
        self.assertIs(ctx.cut,base.fullprep)
        self.assertEqual(ctx.tail_identity,0.)
        self.assertEqual(ctx.lambda_R,0.)
        self.assertTrue(self.np.array_equal(ctx.tail_spec[0],self.np.zeros(2)))

    def test_full_prefix_never_calls_partial_tail_helper(self):
        with patch('h46_new2_core.matrix_only_tail_coefficients',side_effect=AssertionError('partial-only helper called for full')) as partial:
            adapter.prefix_context(self.base(),2)
        partial.assert_not_called()

    def test_signed_schedules_and_negative_times_R32(self):
        for f,n in [('2nd',1),('4th',3),('4th(m5_best)',11)]:
            steps=c.old.core.schedule(f,8,'ABS_TIME_PER_OUTER',32)
            self.assertEqual(len(steps),8*n);self.assertEqual(sum(x['rte_steps'] for x in steps),256)
            self.assertTrue(all(x['rte_steps']>=1 for x in steps))
            if f!='2nd':self.assertTrue(any(x['signed_time']<0 for x in steps))

    def test_three_formulas_vs_independent_toy_matrix_products(self):
        np=self.np
        for f,q in product(('2nd','4th','4th(m5_best)'),(1,2,4,8)):
            for method in ('B0','B0I','FULL','PR'):
                with self.subTest(formula=f,method=method,q=q):
                    p=2 if method=='FULL' else 1;ctx=self.context(p)
                    row=dict(prefix=p,formula=f,method=method,q=q,r_policy='ABS_TIME_PER_OUTER' if method=='PR' else None,r_value=32 if method=='PR' else None)
                    actual=adapter.evaluate(ctx,row);v=ctx.oracle.initial.copy();logB=0.
                    for step in c.old.core.schedule(f,q,row['r_policy'] or 'PER_PIECE',row['r_value'] or 1):
                        t=step['signed_time']
                        for m in ctx.matrices[:p+1]:v=self.expm(-.5j*t*m)@v
                        if method=='PR':
                            tail=ctx.matrices[2]-.01*np.eye(2);r=step['rte_steps'];x=-1j*t/r*tail
                            polynomial=sum(np.linalg.matrix_power(x,k)/math.factorial(k) for k in range(6))
                            v=np.linalg.matrix_power(polynomial,r)@v
                            logB+=r*math.log(self.distribution(ctx.lambda_R*t/r,4).exact_finite_distribution)
                        for m in reversed(ctx.matrices[:p+1]):v=self.expm(-.5j*t*m)@v
                    v*=np.exp(-.8j*(ctx.scalar+(ctx.tail_identity if method in ('B0I','PR') else 0.)))
                    expected=complex(np.vdot(ctx.oracle.initial,v));observed=complex(actual['corrected']['real'],actual['corrected']['imag'])
                    self.assertLess(abs(observed-expected),2e-13);self.assertLess(abs(actual['log_B']-logB),1e-14)
                    raw=complex(actual['raw']['real'],actual['raw']['imag']);self.assertLess(abs(raw-np.exp(-logB)*observed),2e-13)

    def test_wrong_prefix_refused_before_signal(self):
        with self.assertRaisesRegex(ValueError,'NO_ANCHOR_TAIL_REUSE'):adapter.evaluate(self.context(1),dict(prefix=2))

    def test_saved_raw_signal_reuse_has_no_propagation(self):
        row=next(x for x in c.rows('H4') if x['method']=='FULL' and x['q']==2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'out').mkdir();(root/'ledger').mkdir();p=root/'old.json';c.save(p,toy_record())
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,marker=str(root/'CONSUMED.json'),ledger=str(root/'ledger'),output=str(root/'out'),grant={},systems=dict(H4=dict(reuse={row['id']:dict(path=str(p),sha256=c.sha(p))})))
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            ctx=SimpleNamespace(context_key=('toy',10),cut=SimpleNamespace(preparation_hash='saved'))
            with patch.object(adapter,'evaluate',side_effect=AssertionError('must reuse')):result=adapter.one_candidate(cfg,row,ctx,threading.Event())
            self.assertEqual(result['new_signal'],0);self.assertEqual(c.read(root/'out'/result['file'])['signal'],toy_record())

    def test_system_context_once_and_prefix_sharing_across_threads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'out').mkdir();(root/'ledger').mkdir()
            candidates=c.rows('H4');reuse={}
            for row in candidates:
                if row['kind']=='DET_SIGNAL' and row['q']==2 and row['prefix'] in (3,10):
                    p=root/(row['id']+'.json');c.save(p,toy_record());reuse[row['id']]=dict(path=str(p),sha256=c.sha(p))
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,output=str(root/'out'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},systems=dict(H4=dict(reuse=reuse)))
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            base=SimpleNamespace(fullprep=SimpleNamespace(preparation_hash='SYNTHETIC_FULL'))
            def prefix(base,p):return SimpleNamespace(context_key=('SYNTHETIC',p),cut=SimpleNamespace(preparation_hash='p'+str(p)),lambda_R=.1,tail_identity=.01)
            with patch.object(c,'authorize'),patch.object(adapter,'base_context',return_value=base) as restore,patch.object(adapter,'prefix_context',side_effect=prefix) as cache,patch.object(adapter,'evaluate',return_value=toy_record()) as calculation:
                adapter.system_worker(cfg,'H4',8,threading.Event())
                self.assertEqual(restore.call_count,1);self.assertEqual(cache.call_count,4);self.assertEqual(calculation.call_count,37)
            report=c.read(root/'out/H4_complete.json');self.assertEqual(report['new_signal'],37);self.assertEqual(report['reused_signal'],3);self.assertEqual(len(report['results']),40)


if __name__=='__main__':unittest.main()
