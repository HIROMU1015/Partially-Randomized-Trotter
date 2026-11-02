"""Artificial recovery metadata/coefficients only; never load real H2 arrays."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_d_recovery_contract as c
import h2_d_recovery_science as science
from prepare_h2_panels import row


def record(cell,eligible):
    return dict(cell=cell,precision_rows={str(e):{'1.0':dict(cost_acquisition_eligible=eligible,
        eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE',
        axes={a:dict(shots=3 if a=='real' else 5) for a in ('real','imag')})} for e in c.old.EPS})


class SavedMetadata(unittest.TestCase):
    def setUp(self):
        base=c.PRIVATE/'artificial_d_recovery';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        grid=c.old.registrations()['D'];fpids={x['id'] for x in c.old.registrations()['FP'] if not x['R']}
        selected=fpids|{x['id'] for x in c.cells()}
        records=[record(x,x['id'] in selected) for x in grid]
        self.signals=dict(signals=records,reused_signal_ids=sorted(fpids))
        self.fp=dict(signals=[deepcopy(r) for r in records if r['cell']['id'] in fpids],
            measurements=[dict(cell_id=k) for k in fpids])
        self.preps=dict(binding=c.old.BINDING,rows={x['id'].replace('B0I_','B0_',1):{} for x in c.cells()})
        workers=[];fpraw={x['id'] for x in c.old.registrations()['FP'] if x['method']!='B0I'}
        for r in records:
            x=r['cell']
            if x['method']=='B0I' or x['id'] in fpraw:continue
            d=self.root/x['id'];d.mkdir();c.save(d/'task_result.json',r)
            workers.append(dict(directory=str(d),exit_code=0,reaped=True,result_sha256=c.sha(d/'task_result.json')))
        self.receipt=dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=workers)
    def tearDown(self):self.temp.cleanup()

    def test_saved_grid_and_reaped_workers(self):
        c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)
        self.assertEqual(len(self.receipt['workers']),36)

    def test_exact_new_grid_and_budget(self):
        self.assertEqual(len(c.cells()),28)
        self.assertEqual({(x['method'] in ('B0','B0I'),x['prefix']) for x in c.cells()},{(True,2),(False,3)})
        self.assertEqual((c.CALLS['wrapper_compile'],c.CALLS['control_probe'],c.CALLS['endpoint_accounting_pass']),(56,112,1))

    def test_no_signal_reference_sampling_budget(self):
        for name in ('SCF','DF','saved_state_solver','full_reference_pair','truncated_reference_pair',
                     'primitive_probe','base_signal','B0I_overlay','new_signed_shot_endpoints','trajectory','quantum_shots'):
            self.assertEqual(c.CALLS[name],0)

    def test_missing_reap_is_not_accepted(self):
        self.receipt['workers'][0]['reaped']=False
        with self.assertRaisesRegex(ValueError,'WORKER_IDENTITY'):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_changed_saved_signal_worker_bytes(self):
        p=Path(self.receipt['workers'][0]['directory'])/'task_result.json';p.write_text('{}')
        with self.assertRaises((KeyError,ValueError)):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_changed_signal_grid_rejected(self):
        self.signals['signals'].pop()
        with self.assertRaisesRegex(ValueError,'EXACT_D70'):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_unknown_eligibility_rejected(self):
        ep=self.signals['signals'][0]['precision_rows']['0.05']['1.0'];ep.update(cost_acquisition_eligible=False,eligibility_under_declared_allowance='UNKNOWN')
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_changed_acquisition_partition_rejected(self):
        r=next(r for r in self.signals['signals'] if r['cell']['id'] in {x['id'] for x in c.cells()})
        for v in r['precision_rows'].values():v['1.0'].update(cost_acquisition_eligible=False,eligibility_under_declared_allowance='INELIGIBLE')
        with self.assertRaisesRegex(ValueError,'PARTITION'):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_FP_signals_are_exact_copies(self):
        self.fp['signals'][0]['unregistered']='tamper'
        with self.assertRaisesRegex(ValueError,'FP_SIGNALS_UNCHANGED'):c.verify_saved_signals(self.signals,self.fp,self.preps,self.receipt)

    def test_bound_input_hash_tamper(self):
        p=self.root/'input.json';c.save(p,dict(toy=True));rows=[row(p,'toy')];c.verify_hashes(rows)
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SAVED_BYTES_CHANGED'):c.verify_hashes(rows)

    def test_duplicate_input_roles_rejected(self):
        p=self.root/'input.json';c.save(p,{});r=row(p,'toy')
        with self.assertRaisesRegex(ValueError,'UNIQUE_INPUT_ROLES'):c.verify_hashes([r,r])

    def test_out_of_scope_budget_rejected_before_imports(self):
        spec=dict(scope=c.SCOPE,stage='D',plan=c.plan(),calls=dict(c.CALLS,base_signal=1))
        with self.assertRaisesRegex(ValueError,'FIXED_RECOVERY'):c.verify_static(spec)

    def test_source_closure_requires_new_and_frozen_kernels(self):
        closure=c.source_closure()
        self.assertIn('scripts/run_h2_d_recovery.py',closure)
        self.assertIn('scripts/h2_panels_science.py',closure)
        self.assertIn('scripts/h2_tol_only_input.py',closure)

    def test_invalid_adapter_before_output(self):
        out=self.root/'uncreated'
        with self.assertRaisesRegex(ValueError,'EXPLICIT_D_RECOVERY'):science.execute({},c.invalid_grant(),out)
        self.assertFalse(out.exists())

    def test_invalid_driver_and_worker_without_science_packages(self):
        spec=self.root/'spec.json';grant=self.root/'grant.json';c.save(spec,{});c.save(grant,c.invalid_grant())
        for flags in ([],['--worker']):
            r=subprocess.run([sys.executable,'-B','-S',str(c.ROOT/'scripts/run_h2_d_recovery.py'),
                '--spec',str(spec),'--grant',str(grant),*flags],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('EXPLICIT_D_RECOVERY_APPROVAL_REQUIRED',r.stderr)
            self.assertNotIn('ModuleNotFoundError',r.stderr)

    def test_forbidden_new_call_nonrefundable_budget(self):
        b=c.old.Calls(self.root,c.CALLS)
        with self.assertRaisesRegex(ValueError,'SCIENTIFIC_CALL_BUDGET'):b.take('base_signal')
        b.take('wrapper_compile',56)
        with self.assertRaisesRegex(ValueError,'SCIENTIFIC_CALL_BUDGET'):b.take('wrapper_compile')
        self.assertEqual(b.used['wrapper_compile'],56)

    def test_recovery_partial_cost_pooling_disabled(self):
        self.assertEqual(c.plan()['unconfirmed_D_compile_pairs_reused'],0)
        self.assertEqual(c.plan()['cumulative_compile_reservations_maximum'],112)
        self.assertFalse(c.plan()['disconnect_remediation'])


class ApprovalMetadata(unittest.TestCase):
    def setUp(self):
        base=c.PRIVATE/'artificial_d_recovery';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.mock=patch.object(c,'verify_static');self.mock.start()
        self.spec=dict(scope=c.SCOPE,stage='D',SOURCE='a'*40,status='SEALED_PENDING_APPROVAL',pending=['USER'],plan=c.plan())
        pending=self.root/'pending.json';c.save(pending,self.spec)
        resource=self.root/'resources.json';c.save(resource,dict(admitted=True))
        self.spec.update(status='SEALED_FOR_EXECUTION',pending=[],pending_spec=str(pending),pending_spec_sha256=c.sha(pending),
            worker_cpus=[min(os.sched_getaffinity(0))],resource_receipt=str(resource),resource_receipt_sha256=c.sha(resource))
        approval=self.root/'approval.json';c.save(approval,dict(role='user',stage='D',SOURCE='a'*40,
            plan_digest=c.digest(c.plan()),verbatim_user_message='ARTIFICIAL FIXTURE ONLY'))
        self.grant=dict(scope=c.SCOPE,stage='D',SOURCE='a'*40,spec_digest=c.digest(self.spec),one_shot=True,
            automatic_retry=False,next_stage_authorized=False,approved_by_user=True,runtime_authorization=True,
            approval_receipt=str(approval),approval_receipt_sha256=c.sha(approval))
    def tearDown(self):self.mock.stop();self.temp.cleanup()

    def test_valid_disposable_approval(self):c.authorize(self.spec,self.grant)

    def test_no_automatic_retry_or_next_stage(self):
        for key in ('automatic_retry','next_stage_authorized'):
            with self.assertRaisesRegex(ValueError,'ONE_SHOT_STOP'):c.authorize(self.spec,dict(self.grant,**{key:True}))

    def test_pending_contract_rejected(self):
        self.spec['status']='SEALED_PENDING_APPROVAL'
        with self.assertRaisesRegex(ValueError,'RECOVERY_PENDING'):c.authorize(self.spec,self.grant)

    def test_changed_approval_bytes_rejected(self):
        Path(self.grant['approval_receipt']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'USER_APPROVAL_BYTES'):c.authorize(self.spec,self.grant)

    def test_changed_pending_seal_rejected(self):
        Path(self.spec['pending_spec']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'PENDING_SEAL_PRESERVED'):c.authorize(self.spec,self.grant)

    def test_activation_cannot_change_plan(self):
        self.spec['plan']['new_signal_shot_reference_solver_calls']=1;self.grant['spec_digest']=c.digest(self.spec)
        with self.assertRaisesRegex(ValueError,'ACTIVATION_MAY_NOT_CHANGE_SEAL'):c.authorize(self.spec,self.grant)

    def test_Foreign_stage_rejected(self):
        with self.assertRaisesRegex(ValueError,'D_RECOVERY_ONLY'):c.authorize(self.spec,dict(self.grant,stage='A'))


class ArtificialNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=c.PRIVATE/'artificial_d_recovery';base.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=base);cls.root=Path(cls.temp.name)
        provider=c.PRIVATE/'phase0_preparation01/provider';sys.path.insert(0,str(provider/'src'))
        import numpy as np
        import h4_native_science as h
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        cls.m=h.provider();cls.np=np
        one=np.diag([.13,.13,-.21,-.21]).astype(complex);one[0,2]=one[2,0]=.07;one[1,3]=one[3,1]=.07
        gs=tuple(np.diag(v).astype(complex) for v in ([.1,.1,-.2,-.2],[.3,.3,.1,.1],[-.4,-.4,.2,.2]))
        cls.ham=DFHamiltonian(.031,one,np.array([.2,-.15,.1]),gs,{})
        cls.sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
        cls.state=np.array([1.,0.,0.,0.],dtype=complex);cls.full=np.zeros(16,dtype=complex);cls.full[cls.sector.basis_indices]=cls.state
        expected={True:cls.m._prepare_discard(cls.ham,2),False:cls.m._prepare(cls.ham,3)}
        rows={x['id'].replace('B0I_','B0_',1):dict(preparation_hash=expected[x['method']!='B1'].preparation_hash,
            hamiltonian_hash=expected[x['method']!='B1'].hamiltonian_hash) for x in c.cells()}
        p=cls.root/'saved_preparations.json';c.save(p,dict(rows=rows))
        cls.spec=dict(native_config=dict(provider_root=str(provider),binding=c.old.BINDING,compiler_options=c.old.COMPILER),saved_preparations=str(p))
        cls.loader=staticmethod(lambda config:(cls.ham,cls.sector,cls.full,cls.state,
            dict(hamiltonian_hash=c.old.BINDING['hamiltonian'],state_hash=c.old.BINDING['state'])))
        cls.output=cls.root/'context';cls.output.mkdir();cls.calls=c.old.Calls(cls.output,c.CALLS)
        cls.port,_,cls.compiler=science.context(cls.spec,cls.output,cls.calls,loader=cls.loader)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_exact_two_preparations_no_forward_or_reference(self):
        self.assertEqual(self.calls.used['preparation'],2);self.assertEqual(self.calls.used['saved_input_load'],1)
        for name in ('base_signal','full_reference_pair','primitive_spectrum','new_signed_shot_endpoints'):
            self.assertEqual(self.calls.used[name],0)

    def test_preparation_identity_mismatch_stops(self):
        spec=dict(self.spec);p=self.root/'bad_preps.json';data=c.read(self.spec['saved_preparations'])
        for r in data['rows'].values():r['preparation_hash']='0'*64
        c.save(p,data);spec['saved_preparations']=str(p);out=self.root/'bad_context';out.mkdir()
        with self.assertRaisesRegex(ValueError,'UNCHANGED_NATIVE_PREPARATION'):
            science.context(spec,out,c.old.Calls(out,c.CALLS),loader=self.loader)

    def test_reused_lowering_matches_independent_occupation_expm(self):
        from scipy.linalg import expm
        from fractions import Fraction
        import h6_native_b0i_bridge_v1 as bridge
        from h2_panels_science import compile_one
        np=self.np;m=self.m
        targets=[x for x in c.cells() if (x['method']=='B0I' and x['q']==1) or (x['method']=='B1' and x['q']==2)]
        self.assertEqual(len(targets),4)
        for x in targets:
            p=x['prefix'];basis=self.sector.basis_indices
            matrices=[np.asarray(m.occupation_df_matrix(0.,self.ham.one_body,[],[],basis),dtype=complex)]
            matrices += [np.asarray(m.occupation_df_matrix(0.,np.zeros_like(self.ham.one_body),
                self.ham.lambdas[i:i+1],self.ham.g_matrices[i:i+1],basis),dtype=complex) for i in range(p)]
            state=self.state.copy();schedule=m.primitive_time_schedule(x,T=.8)['ordinary_one_outer_step']
            for _ in range(x['q']):
                for i,t in schedule:state=expm(-1j*t*matrices[i])@state
            scalar=self.ham.constant
            if x['method']=='B0I':scalar+=float(sum((bridge.exact_identity(g,w) for g,w in zip(self.ham.g_matrices[p:],self.ham.lambdas[p:])),Fraction()))
            z=complex(np.vdot(self.state,np.exp(-1j*.8*scalar)*state))
            record=dict(signals=dict(corrected=m.complex_record(z)),scalar_receipt=dict(scalar=scalar))
            out=self.root/('compiled_'+x['id']);out.mkdir()
            pair=compile_one(self.port,m,self.compiler,dict(cell=x,replica=0,expected=record),out,'D')
            c.old.check_pair(pair);self.assertLess(c.read(out/'control.json')['max_error'],1e-9)
