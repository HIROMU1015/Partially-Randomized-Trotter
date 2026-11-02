"""Artificial Hamiltonians/receipts only. Never load real H2 arrays or results."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_panels_contract as c
import h2_panels_science as s
from prepare_h2_panels import freeze_kernel, primitive_count


def toy_record(x,eligible=True):
    return dict(cell=x,precision_rows={str(e):{'1.0':dict(cost_acquisition_eligible=eligible,
        eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE',
        axes={a:dict(shots=3 if a=='real' else 5) for a in ('real','imag')})} for e in c.EPS})


def toy_pair(cid,rep=0,rz=0,stage='D'):
    return dict(cell_id=cid,replica=rep,seed=None,cohort=c.plan(stage)['cohort'],
        axes={a:dict(RZ=rz,compiler_hash=c.BINDING['compiler'],fingerprint='f'*64) for a in ('cosine','sine')})


class MetadataTests(unittest.TestCase):
    def test_unique_h2_grid(self):
        r=c.registrations(); self.assertEqual([len(r[x]) for x in ('FP','D','A')],[8,70,66])
        self.assertEqual({x['prefix'] for x in r['A']},{0,1,2})
        self.assertFalse(any(x['prefix']==3 and x['method']!='B1' for x in r['D']))

    def test_exact_call_maxima(self):
        expected={'FP':(6,2,2,8,16),'D':(36,28,0,64,128),'A':(64,0,528,528,1056)}
        for stage,values in expected.items():
            b=c.budget(stage,123)
            self.assertEqual(tuple(b[k] for k in ('base_signal','B0I_overlay','trajectory','native_evolution','wrapper_compile')),values)
            self.assertEqual(b['SCF']+b['DF']+b['saved_state_solver']+b['quantum_shots'],0)

    def test_seed_namespaces_and_hierarchy(self):
        fp=c.seed_tasks('FP','1'*40); a=c.seed_tasks('A','1'*40)
        self.assertEqual((len(fp),len(a),len(c.seed_values(fp)),len(c.seed_values(a))),(2,528,4,3888))
        self.assertFalse(c.seed_values(fp)&c.seed_values(a))
        self.assertEqual(c.stats.seed_hierarchy(47,1),[47])
        self.assertNotEqual(c.seed_tasks('A','1'*40),c.seed_tasks('A','2'*40))

    def test_seed_duplicate_denied(self):
        tasks=c.seed_tasks('FP','1'*40)
        with self.assertRaisesRegex(ValueError,'COLLISION'): c.seed_values(tasks+tasks)

    def test_empty_tail_explicitly_ineligible(self):
        cell=c.registrations()['A'][0]; d=s.degenerate(cell,0.)
        self.assertEqual(c.selected([d],[cell]),[])
        self.assertEqual(d['status'],'DEGENERATE_EMPTY_TAIL')
        self.assertIsNone(s.degenerate(cell,1.))

    def test_invalid_tail_not_ineligible(self):
        for value in [-1.,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'INVALID_TAIL'): s.degenerate({},value)

    def test_unresolved_is_stop(self):
        cell=c.registrations()['A'][0]; r=toy_record(cell,False)
        r['precision_rows']['0.05']['1.0']['eligibility_under_declared_allowance']='UNKNOWN'
        with self.assertRaisesRegex(ValueError,'UNKNOWN'): c.selected([r],[cell])

    def test_missing_or_duplicate_candidate_rejected(self):
        cells=c.registrations()['D'][:2]
        for records in [[toy_record(cells[0])],[toy_record(cells[0])]*2]:
            with self.assertRaisesRegex(ValueError,'COMPLETE_SIGNAL_GRID'): c.selected(records,cells)

    def test_positive_shot_rule(self):
        cell=c.registrations()['D'][0]; r=toy_record(cell)
        r['precision_rows']['0.05']['1.0']['axes']['real']['shots']=0
        with self.assertRaisesRegex(ValueError,'POSITIVE_SHOTS'): c.selected([r],[cell])

    def test_zero_rz_pf_and_exact_ties(self):
        cells=c.registrations()['D']; result=c.account([toy_record(x) for x in cells],
            {x['id']:[toy_pair(x['id'])] for x in cells},'D')
        self.assertTrue(all(x['G_RZ_exact']=='0' for x in result['winners'].values()))
        self.assertEqual(result['winners']['0.05']['cell_id'],min(x['id'] for x in cells))

    def test_missing_eligible_cost_stops(self):
        with self.assertRaisesRegex(ValueError,'COMPLETE_ELIGIBLE_COST'):
            c.account([toy_record(x) for x in c.registrations()['D']],{},'D')

    def test_a_exact_n8_and_covariance(self):
        d_cells=c.registrations()['D']; d=c.account([toy_record(x) for x in d_cells],
            {x['id']:[toy_pair(x['id'])] for x in d_cells},'D')
        cells=c.registrations()['A']; pairs={x['id']:[toy_pair(x['id'],i,i,stage='A') for i in range(8)] for x in cells}
        result=c.account([toy_record(x) for x in cells],pairs,'A',dict(stage='D',binding=c.BINDING,accounting=d))
        self.assertTrue(all(x['n']==8 and x['G_RZ_exact']=='28' for x in result['winners'].values()))
        self.assertEqual(result['comparisons'][0]['status'],'ZERO_PF_DENOMINATOR')
        self.assertEqual(result['covariances'][cells[0]['id']]['sample_covariance'][0][1],'384')

    def test_fp_cost_not_pooled(self):
        cells=c.registrations()['A']; pairs={x['id']:[toy_pair(x['id'],i,stage='A') for i in range(8)] for x in cells}
        pairs[cells[0]['id']][0]['cohort']='FP_NATIVE_DIAGNOSTIC_N1'
        with self.assertRaisesRegex(ValueError,'FRESH_N8'):
            c.account([toy_record(x) for x in cells],pairs,'A')

    def test_pair_trajectory_mismatch(self):
        p=toy_pair('toy'); p['seed']=12
        with self.assertRaises((KeyError,ValueError)): c.check_pair(p)

    def test_negative_metrics_rejected(self):
        with self.assertRaisesRegex(ValueError,'METRICS'): c.check_pair(toy_pair('toy',rz=-1))

    def test_rare_strata(self):
        for orders,expected in [([0],'S0_ONLY'),([2],'S2_SINGLE'),([2,2],'S2_MULTI'),([0,4],'S4_PRESENT')]:
            self.assertEqual(c.stats.event_stratum([[dict(taylor_order=k) for k in orders]]),expected)

    def test_invalid_direct_execution_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'uncreated'
            for stage in ('FP','D','A'):
                with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT'):
                    s.execute(dict(stage=stage),c.invalid_grant(stage),output)
            self.assertFalse(output.exists())

    def test_driver_and_worker_invalid_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for stage in ('FP','D','A'):
                spec=root/(stage+'_spec.json'); grant=root/(stage+'_grant.json')
                c.save(spec,dict(stage=stage)); c.save(grant,c.invalid_grant(stage))
                for flag in ([],['--worker']):
                    r=subprocess.run([sys.executable,'-B','-S',str(c.ROOT/'scripts/run_h2_panels.py'),
                        '--spec',str(spec),'--grant',str(grant),*flag],capture_output=True,text=True)
                    self.assertNotEqual(r.returncode,0); self.assertIn('SEPARATE_EXPLICIT_STAGE_APPROVAL_REQUIRED',r.stderr)
                    self.assertNotIn('numpy',r.stderr)

    def test_pending_approved_grant_cannot_launch(self):
        grant=dict(scope=c.SCOPE,stage='FP',approved_by_user=True,runtime_authorization=True,
            one_shot=True,automatic_retry=False,next_stage_authorized=False)
        with self.assertRaisesRegex(ValueError,'PENDING_CONTRACT'):
            c.authorize(dict(stage='FP',status='SEALED_PENDING_APPROVAL'),grant)

    def test_foreign_and_retry_grants_rejected(self):
        g=dict(scope=c.SCOPE,stage='FP',approved_by_user=True,runtime_authorization=True,
            one_shot=True,automatic_retry=False,next_stage_authorized=False)
        for change in [dict(stage='D'),dict(automatic_retry=True),dict(next_stage_authorized=True),dict(one_shot=False)]:
            with self.assertRaises(ValueError): c.authorize(dict(stage='FP'),dict(g,**change))

    def test_append_only_nonrefundable_budget(self):
        with tempfile.TemporaryDirectory() as td:
            b=c.Calls(td,dict(compile=2)); b.take('compile',2)
            with self.assertRaisesRegex(ValueError,'CALL_BUDGET'): b.take('compile')
            self.assertEqual(b.used['compile'],2)
            self.assertEqual(len(list(Path(td).glob('reservation*'))),1)


class SealedMetadataTests(unittest.TestCase):
    """Disposable valid fixtures; no grant or execution in a real attempt namespace."""
    def setUp(self):
        base=c.PRIVATE/'artificial_panels'; base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base); self.root=Path(self.temp.name)
        self.store=self.root/'seal'; self.store.mkdir(); self.patches=[]
        for patcher in [patch.object(c,'ROOT',self.root),patch.object(c,'STORE',self.store),
                        patch.object(c.subprocess,'check_output',return_value=b'# toy source\n')]:
            self.patches.append(patcher); patcher.start()
        self.source=self.root/'scripts/toy.py'; self.source.parent.mkdir(); self.source.write_bytes(b'# toy source\n')
        patcher=patch.object(c,'source_closure',return_value={'scripts/toy.py':c.sha(self.source)})
        patcher.start(); self.patches.append(patcher)
        self.provider=self.root/'provider'; self.provider.mkdir(); (self.provider/'toy.py').write_text('# toy\n')
        self.kernel=self.store/'A_kernel/scripts'; self.kernel.mkdir(parents=True); (self.kernel/'toy.py').write_text('# toy\n')
        self.python=self.root/'toy-python'; self.python.write_bytes(b'ARTIFICIAL BINARY')
        self.runtime=self.root/'runtime.json'; c.save(self.runtime,{str(self.python):c.sha(self.python)})
        self.snapshot=self.root/'toy.npz'; self.snapshot.write_bytes(b'NOT A NUMERICAL ARRAY')
        patcher=patch.dict(c.BINDING,dict(snapshot=c.sha(self.snapshot))); patcher.start(); self.patches.append(patcher)
        self.receipt=self.root/'toy_receipt.json'; c.save(self.receipt,{'artificial':True})
        c.save(self.store/'old_seed_values.json',dict(values=[]))
        self.spec,self.grant=self.fixture('FP')

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def fixture(self,stage):
        from prepare_h2_panels import row
        source='a'*40; p=c.plan(stage)
        inputs=[row(self.snapshot,'snapshot'),row(self.receipt,'receipt'),row(self.store/'old_seed_values.json','old_seeds')]
        spec=dict(scope=c.SCOPE,stage=stage,SOURCE=source,plan=p,status='SEALED_PENDING_APPROVAL',
            pending=['ARTIFICIAL_PENDING'],source_hashes=c.source_closure(),inputs=inputs,
            native_config=dict(provider_root=str(self.provider),binding=deepcopy(c.BINDING),compiler_options=c.COMPILER,
                mapping=dict(tol_snapshot=str(self.snapshot),tol_receipt=str(self.receipt))),
            provider_roots=[str(self.provider)],provider_files=[row(self.provider/'toy.py','toy_provider')],
            kernel_files=[row(self.kernel/'toy.py','toy_kernel')],A_kernel_SOURCE=c.A_SOURCE,A_kernel_scripts=str(self.kernel),
            runtime_manifest=str(self.runtime),runtime_manifest_sha256=c.sha(self.runtime),python=str(self.python),
            python_sha256=c.sha(self.python),attempt=str(c.PRIVATE/('attempts/'+stage+'_run01')),
            primitive_probes=0,calls=c.budget(stage,0))
        if stage in ('FP','A'):
            registry=self.root/(stage+'_seeds.json'); tasks=c.seed_tasks(stage,source); c.save(registry,dict(tasks=tasks))
            check=self.root/(stage+'_check.json'); values=c.seed_values(tasks)
            c.save(check,dict(status='COLLISION_ZERO',new_values_digest=c.digest(sorted(values)),
                old_explicit_exclusions_sha256=c.sha(self.store/'old_seed_values.json'),geometry_checks=[dict(collisions=[])]))
            spec.update(seed_registry=str(registry),seed_check=str(check)); inputs.extend([row(registry,'seed_registry'),row(check,'seed_check')])
        pending=self.root/(stage+'_pending.json'); c.save(pending,spec)
        spec=deepcopy(spec); spec.update(status='SEALED_FOR_EXECUTION',pending=[],pending_spec=str(pending),
            pending_spec_sha256=c.sha(pending),worker_cpus=[min(os.sched_getaffinity(0))],dependencies={})
        for name in (['FP'] if stage=='D' else ['FP','D'] if stage=='A' else []):
            result=self.root/(stage+'_'+name+'_result.json'); c.save(result,dict(status='COMPLETE_MANDATORY_STOP',stage=name,binding=c.BINDING))
            accepted=self.root/(stage+'_'+name+'_accepted.json'); c.save(accepted,dict(accepted_by_user=True,result_sha256=c.sha(result),verbatim_user_message='ARTIFICIAL'))
            audit=self.root/(stage+'_'+name+'_audit.json'); c.save(audit,dict(status='COMPLETE_MANDATORY_STOP',result_sha256=c.sha(result)))
            spec['dependencies'][name]=dict(result=str(result),sha256=c.sha(result),acceptance=str(accepted),
                acceptance_sha256=c.sha(accepted),completion_audit=str(audit),completion_audit_sha256=c.sha(audit))
        resource=self.root/(stage+'_resources.json'); c.save(resource,dict(admitted=True))
        spec.update(resource_receipt=str(resource),resource_receipt_sha256=c.sha(resource))
        if stage in ('FP','A'):
            refresh=self.root/(stage+'_refresh.json'); c.save(refresh,dict(status='COLLISION_ZERO',
                new_values_digest=c.digest(sorted(c.seed_values(c.read(spec['seed_registry'])['tasks']))),
                known_campaigns_checked=['ARTIFICIAL'],new_RTE_sampling=0,checked_at_UTC='ARTIFICIAL'))
            spec.update(seed_refresh=str(refresh),seed_refresh_sha256=c.sha(refresh))
        approval=self.root/(stage+'_approval.json'); c.save(approval,dict(role='user',stage=stage,SOURCE=source,
            plan_digest=c.digest(p),verbatim_user_message='ARTIFICIAL FIXTURE ONLY'))
        grant=dict(scope=c.SCOPE,stage=stage,approved_by_user=True,runtime_authorization=True,one_shot=True,
            automatic_retry=False,next_stage_authorized=False,SOURCE=source,spec_digest=c.digest(spec),
            approval_receipt=str(approval),approval_receipt_sha256=c.sha(approval))
        return spec,grant

    def test_valid_artificial_metadata_fp(self): c.authorize(self.spec,self.grant,interpreter=False)

    def test_valid_artificial_d_a_dependencies(self):
        for stage in ('D','A'):
            spec,grant=self.fixture(stage); c.authorize(spec,grant,interpreter=False)

    def test_source_change_denied(self):
        self.source.write_text('# changed\n')
        with self.assertRaises(ValueError): c.authorize(self.spec,self.grant,interpreter=False)

    def test_runtime_change_denied(self):
        runtime_file=self.root/'runtime-lib'; runtime_file.write_text('old')
        self.runtime.write_text(__import__('json').dumps({str(runtime_file):c.sha(runtime_file)}))
        # Rebind metadata only in both artificial seals, then alter runtime bytes.
        self.spec['runtime_manifest_sha256']=c.sha(self.runtime)
        pending=Path(self.spec['pending_spec']); p=c.read(pending); p['runtime_manifest_sha256']=c.sha(self.runtime)
        pending.write_text(__import__('json').dumps(p)); self.spec['pending_spec_sha256']=c.sha(pending)
        self.grant['spec_digest']=c.digest(self.spec); runtime_file.write_text('changed')
        with self.assertRaisesRegex(ValueError,'RUNTIME_CHANGED'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_input_bytes_change_denied(self):
        self.snapshot.write_bytes(b'CHANGED')
        with self.assertRaisesRegex(ValueError,'inputs_CHANGED'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_extra_provider_file_denied(self):
        (self.provider/'unbound.py').write_text('# extra')
        with self.assertRaisesRegex(ValueError,'EXACT_PROVIDER_FILE_SET'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_kernel_file_change_denied(self):
        (self.kernel/'toy.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'kernel_files_CHANGED'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_approval_bytes_change_denied(self):
        Path(self.grant['approval_receipt']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'APPROVAL_RECEIPT_BYTES'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_activation_may_not_change_plan_budget(self):
        self.spec['calls']['wrapper_compile']+=1; self.grant['spec_digest']=c.digest(self.spec)
        with self.assertRaisesRegex(ValueError,'FIXED_CALL_BUDGET'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_missing_dependency_denied(self):
        spec,grant=self.fixture('D'); spec['dependencies']={}; grant['spec_digest']=c.digest(spec)
        with self.assertRaisesRegex(ValueError,'EXACT_ACCEPTED_DEPENDENCIES'): c.authorize(spec,grant,interpreter=False)

    def test_resource_receipt_must_be_admitted(self):
        Path(self.spec['resource_receipt']).write_text('{"admitted":false}')
        self.spec['resource_receipt_sha256']=c.sha(self.spec['resource_receipt']); self.grant['spec_digest']=c.digest(self.spec)
        with self.assertRaisesRegex(ValueError,'RESOURCE_ADMISSION'): c.authorize(self.spec,self.grant,interpreter=False)

    def test_stale_or_missing_seed_refresh_denied(self):
        Path(self.spec['seed_refresh']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'SEED_REFRESH_BYTES'): c.authorize(self.spec,self.grant,interpreter=False)


class NativeArtificialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=c.PRIVATE/'artificial_panels'; base.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=base); cls.root=Path(cls.temp.name)
        cls.provider=c.PRIVATE/'phase0_preparation01/provider'
        cls.kernel=cls.root/'A_kernel'; freeze_kernel(cls.kernel)
        sys.path.insert(0,str(cls.provider/'src')); sys.path.insert(0,str(cls.kernel/'scripts'))
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        # Deliberately invented coefficients and a computational basis state.
        one=np.diag([.13,.13,-.21,-.21]).astype(complex)
        one[0,2]=one[2,0]=.07; one[1,3]=one[3,1]=.07
        gs=[np.diag(v).astype(complex) for v in ([.1,.1,-.2,-.2],[.3,.3,.1,.1],[-.4,-.4,.2,.2])]
        cls.ham=DFHamiltonian(.031,one,np.array([.2,-.15,.1]),tuple(gs),{})
        cls.sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
        cls.state=np.array([1.,0.,0.,0.],dtype=complex); cls.full=np.zeros(16,dtype=complex)
        cls.full[cls.sector.basis_indices]=cls.state
        config=dict(provider_root=str(cls.provider),binding=c.BINDING,compiler_options=c.COMPILER,mapping={})
        cells=[x for x in c.registrations()['FP'] if x['method']!='B0I']
        count=primitive_count(cells,cls.provider)
        cls.spec=dict(stage='FP',native_config=config,A_kernel_scripts=str(cls.kernel/'scripts'),
            primitive_probes=count,calls=c.budget('FP',count))
        cls.output=cls.root/'setup'; cls.output.mkdir(); cls.calls=c.Calls(cls.output,cls.spec['calls'])
        loader=lambda config:(cls.ham,cls.sector,cls.full,cls.state,
            dict(hamiltonian_hash=c.BINDING['hamiltonian'],state_hash=c.BINDING['state']))
        cls.port,cls.m,cls.h,cls.reference=s.setup(cls.spec,cells,cls.output,cls.calls,loader=loader)
        cls.records={}
        for x in cells:
            directory=cls.root/x['id']; directory.mkdir()
            cls.records[x['id']]=s.evaluate(cls.port,dict(cell=x),directory)
        from h4_tol_main_science import overlay
        for x in c.registrations()['FP']:
            if x['method']=='B0I': cls.records[x['id']]=overlay(cls.records[x['id'].replace('B0I_','B0_',1)],cls.port,cls.m)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_native_forward_full_and_truncated_s2_s4(self):
        for r in self.records.values():
            self.assertLess(max(r.get('signal_discrepancies',[0.])),1e-10)
            self.assertEqual(set(r['precision_rows']),{str(e) for e in c.EPS})

    def test_b0i_reference_unrotated(self):
        for f in ('2nd','4th'):
            a=self.records['B0_p1_q1_'+f]; b=self.records['B0I_p1_q1_'+f]
            self.assertEqual(a['reference'],b['reference']); self.assertIn('scalar_receipt',b)

    def test_synthetic_native_wrappers_and_compiler_all_fp(self):
        from h4_dp_science import compiler
        settings=compiler(self.m,self.spec['native_config'])
        seeds={t['cell']['id']:t for t in c.seed_tasks('FP','a'*40)}
        for x in c.registrations()['FP']:
            directory=self.root/('compiled_'+x['id']); directory.mkdir()
            t=dict(id=x['id'],cell=x,replica=0,expected=self.records[x['id']],**
                ({'seed':seeds[x['id']]['seed']} if x['R'] else {}))
            pair=s.compile_one(self.port,self.m,settings,t,directory,'FP')
            c.check_pair(pair); self.assertTrue((directory/'control.json').exists())

    def test_prefix0_s2_s4_native(self):
        import numpy as np
        from trottertracks.resource_applicability.ax2b_native_df_v5 import build_deterministic_native
        prep=self.m._prepare_discard(self.ham,0)
        for formula in ('2nd','4th'):
            evolution=build_deterministic_native(prep.deterministic_blocks,num_system_qubits=4,T=.8,q=1,
                formula=formula,scalar=prep.constant_coefficient,control_policy='symmetric_directional',max_instructions=1000000)
            low=np.concatenate((self.port.qstate,np.zeros_like(self.port.qstate)))
            self.assertLess(np.linalg.norm(self.m.simulate_statevector(evolution.circuit,low)-low),1e-10)

    def test_forced_s2_s4_events_on_artificial_hamiltonian(self):
        # Conditional fixtures are restricted to invented coefficients. They never
        # enter an FP/A registry or stand in for an unconditional cost sample.
        import importlib
        import pr_cost_native as kernel
        from h6_dba_sampling import steps
        from types import SimpleNamespace
        native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
        for x,stratum in zip([x for x in c.registrations()['FP'] if x['R']],['s2_single','s4']):
            requests,receipt=steps(self.port.preps[x['id']],x,stratum,314159)
            self.assertIn(4 if stratum=='s4' else 2,receipt['orders'])
            evolution=native.partial_native_from_step_requests(requests,control_policy='symmetric_directional',max_instructions=1000000)
            directory=self.root/stratum; directory.mkdir()
            kernel.control(evolution,requests,SimpleNamespace(m=self.m,native=native,qstate=self.port.qstate),c.Writer(directory),1e-9)
