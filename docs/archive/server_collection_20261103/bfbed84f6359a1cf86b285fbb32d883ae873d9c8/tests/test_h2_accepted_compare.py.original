"""Artificial data and mock science only; no NumPy/Qiskit imports."""
import ast
import copy
from dataclasses import asdict
from contextlib import ExitStack
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h2_accepted_compare_contract as c
import h2_accepted_diagnostics as a
import h2_diagnostics_binding as b
import h2_common_production_contract as old


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);b.write(p,value)


def npy(shape,dtype,values=None):
    header=repr(dict(descr=dtype,fortran_order=False,shape=shape)).encode()+b'\n'
    n=16 if dtype=='<c16' else 8
    for d in shape:n*=d
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+(values if values is not None else bytes(n))


def archive(path,columns,*,wrong_basis=False):
    with zipfile.ZipFile(path,'w') as z:
        for name in ('primary','independent'):z.writestr(name+'.npy',npy((6,columns),'<c16'))
        basis=(3,5,6,9,10,13) if wrong_basis else (3,5,6,9,10,12)
        z.writestr('occupation_basis.npy',npy((6,),'<i8',struct.pack('<6q',*basis)))
        z.writestr('qindices.npy',npy((6,),'<i8',struct.pack('<6q',12,10,6,9,5,3)))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h2-accepted-compare-artificial-')
        self.root=Path(self.tmp.name);self.stack=ExitStack();self.addCleanup(self.tmp.cleanup);self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(old,'seal_check',return_value=set()))
        self.stack.enter_context(patch.object(a,'origin_source_check'))
        self.stack.enter_context(patch.object(c.Session,'loaded_closure'))

    def origin(self):
        run=self.root/'origin';(run/'results').mkdir(parents=True)
        oldref=self.root/'old_reference.json';write(oldref,{'artificial':True})
        runtime=dict(python_sha256='e'*64,python_target='/ARTIFICIAL/python',qiskit='1.3.0',numpy='1.26.4',
            inventory_sha256='d'*64,sys_path=['/ARTIFICIAL/site'],closure_mode='STATIC_CLOSED_IMPORT_SET_V1')
        cfg=dict(schema='artificial_origin',series=old.SERIES,branch='DIAGNOSTICS',source_commit=a.ORIGIN_SOURCE,
            binding=old.BINDING,physical=c.policy.PHYSICAL,compiler=c.policy.COMPILER,
            candidates=[asdict(x) for x in c.policy.CANDIDATES],native_policy=c.policy.NATIVE_POLICY,
            allowance_policy=c.policy.ALLOWANCE_POLICY,reference_policy='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC',
            runtime=runtime,files=[],source_files=[],inputs={'old_reference':str(oldref)},budget=old.counts('DIAGNOSTICS'),run_root=str(run))
        contract=self.root/'origin_contract.json';write(contract,cfg)
        self.stack.enter_context(patch.object(a,'ORIGIN_CONTRACT',b.digest(cfg)))
        approval=dict(issuer='USER_EXPLICIT',execution_authorized=True,contract_sha256=b.digest(cfg),
            policies_adopted=[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')])
        grant=dict(id='ARTIFICIAL_OLD_GRANT',contract_sha256=b.digest(cfg),approval_sha256=b.digest(approval),
            run_root=str(run),budget=cfg['budget'],state='UNUSED',retry=False)
        write(self.root/'approval.json',approval);write(self.root/'old_grant.json',grant)
        write(run/'CONSUMED.json',dict(grant=grant,state='CONSUMED_NO_RETRY_NO_REFUND',approval_sha256=b.digest(approval)))
        sequence=0
        for kind,amount in cfg['budget'].items():
            if amount:
                sequence+=1;write(run/'ledger'/('%06d.json'%sequence),dict(kind=kind,amount=amount,nonrefundable=True,contract_sha256=b.digest(cfg)))
        for prefix in (1,3):
            operations=[dict(rotation_sha256='f'*64,operations=[])]
            write(run/'results'/('preparation_'+str(prefix)+'.json'),dict(prefix=prefix,
                native_policy=cfg['native_policy'],preparation_sha256=str(prefix)*64,
                operation_stream=operations,operations_sha256=b.digest(operations)))
        for candidate in c.policy.CANDIDATES:
            cid=candidate.id;pr=candidate.method=='PR';ev='results/'+cid+'_diagnostics.json'
            obs=lambda value,unit,scope:c.policy.Observation(value,unit,scope,ev)
            refid='OLD_FP:'+b.sha(oldref)
            inputs=c.policy.AdaptiveInputs(cid,'SAVED_SCIENCE',refid,
                obs(1e-16,'CORRECTED_SIGNAL','H2_REFERENCE_COMBINED_DIAGNOSTIC'),
                tuple(obs(0.,'CORRECTED_SIGNAL','PRIMARY_VS_INDEPENDENT_SIGNAL') for _ in range(2 if pr else 1)),
                tuple(obs(0.,'CORRECTED_STATE_VECTOR','PRIMARY_VS_INDEPENDENT_VECTOR') for _ in range(2 if pr else 1)),
                obs(0.,'CORRECTED_SIGNAL','B_RAW_MINUS_CORRECTED_SIGNAL'),
                obs(0.,'LOG_NORMALIZATION','TWO_PATH_LOG_B_DIFFERENCE'),
                dict(deterministic=2,tail=2 if pr else 0),dict(deterministic=2,tail=2 if pr else 0),
                obs(1.,'CORRECTED_STATE_VECTOR','BOTH_PATHS_ALL_REQUIRED_STAGES'),'BOTH_PATHS_ALL_REQUIRED_STAGES',ev,
                cfg['native_policy'],b.digest(dict(binding=cfg['binding'],files=cfg['files'],series=old.SERIES)))
            archive(run/'results'/(cid+'_vectors.npz'),2 if pr else 1)
            prep=b.read(run/'results'/('preparation_'+str(candidate.prefix)+'.json'))
            row=dict(candidate=cid,native_policy=cfg['native_policy'],strict_diagnostic_limit=1e-9,
                phase_alignment=False,postcompile_correction=False,scalar_applications=1,
                vector_archive=dict(path=cid+'_vectors.npz',sha256=b.sha(run/'results'/(cid+'_vectors.npz'))),
                inputs=asdict(inputs),preparation_sha256=prep['preparation_sha256'],operations_sha256=prep['operations_sha256'],
                signal_columns=['corrected','raw'] if pr else ['corrected'],
                primary_signals=[dict(real=.4,imag=.5)]*(2 if pr else 1),
                independent_signals=[dict(real=.4,imag=.5)]*(2 if pr else 1),
                reference_binding=dict(reference_id=refid,reference_recomputed=False,reference_error_certified=False,
                    bias={'real':.01,'imag':.02}),log_B=0.,schedule=[],scalar=.1)
            write(run/'results'/(cid+'_diagnostics.json'),row)
        ref=dict(contract=str(contract),run_root=str(run),completion_sha256='',approval=str(self.root/'approval.json'),
            grant=str(self.root/'old_grant.json'),source_roots={k:str(self.root/'ARTIFICIAL_SOURCE') for k in ('runner','acquisition','provider')})
        self.refresh_origin(cfg,ref)
        return cfg,ref

    def refresh_origin(self,cfg,ref):
        run=Path(ref['run_root']);rows=[dict(path=str(p.relative_to(run)),sha256=b.sha(p)) for p in sorted((run/'results').iterdir())]
        completion=dict(status='CONTRACT_COMPLETE_NOT_SCIENTIFIC_GO',result={'status':'DIAGNOSTICS_ONLY'},
            contract_sha256=b.digest(cfg),used=cfg['budget'],result_files=rows,marker_sha256=b.sha(run/'CONSUMED.json'),
            **{k:cfg[k] for k in ('source_commit','binding','runtime','physical','native_policy','allowance_policy','reference_policy')})
        write(run/'completion.json',completion);ref['completion_sha256']=b.sha(run/'completion.json')
        self.stack.enter_context(patch.object(a,'ORIGIN_COMPLETION',ref['completion_sha256']))


class OriginTests(Fixture):
    def test_complete_metadata_and_npy_headers_no_vector_evaluation(self):
        cfg,ref=self.origin();plan=a.inspect_origin(ref)
        self.assertEqual(len(plan['records']),4);self.assertFalse(plan['signal_vectors_evaluated'])

    def test_completion_byte_change_rejected(self):
        cfg,ref=self.origin();p=Path(ref['run_root'])/'completion.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(c.Stop,'EXTERNALLY_SEALED'):a.inspect_origin(ref)

    def test_npz_or_json_byte_change_rejected(self):
        cfg,ref=self.origin();p=next((Path(ref['run_root'])/'results').glob('*vectors.npz'));p.write_bytes(b'TAMPERED')
        with self.assertRaisesRegex(c.Stop,'RESULT_HASH'):a.inspect_origin(ref)

    def test_diagnostic_json_byte_change_rejected(self):
        cfg,ref=self.origin();p=next((Path(ref['run_root'])/'results').glob('*diagnostics.json'));p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(c.Stop,'RESULT_HASH'):a.inspect_origin(ref)

    def test_candidate_prefix_preparation_mismatch(self):
        cfg,ref=self.origin();p=Path(ref['run_root'])/'results'/(c.policy.CANDIDATES[0].id+'_diagnostics.json')
        row=b.read(p);row['preparation_sha256']='1'*64;write(p,row);self.refresh_origin(cfg,ref)
        with self.assertRaisesRegex(c.Stop,'PREFIX_PREPARATION'):a.inspect_origin(ref)

    def test_basis_wire_order_mismatch_even_if_archive_hash_refreshed(self):
        cfg,ref=self.origin();cid=c.policy.CANDIDATES[0].id;run=Path(ref['run_root'])
        p=run/'results'/(cid+'_vectors.npz');archive(p,1,wrong_basis=True)
        q=run/'results'/(cid+'_diagnostics.json');row=b.read(q);row['vector_archive']['sha256']=b.sha(p);write(q,row)
        self.refresh_origin(cfg,ref)
        with self.assertRaisesRegex(c.Stop,'BASIS_WIRE_ORDER'):a.inspect_origin(ref)

    def test_wrong_raw_column_semantics_rejected(self):
        cfg,ref=self.origin();cid=c.policy.CANDIDATES[2].id;p=Path(ref['run_root'])/'results'/(cid+'_diagnostics.json')
        row=b.read(p);row['signal_columns']=['raw','corrected'];write(p,row);self.refresh_origin(cfg,ref)
        with self.assertRaisesRegex(c.Stop,'RAW_COLUMNS'):a.inspect_origin(ref)

    def test_native_reference_allowance_consumer_policy_mismatch(self):
        cfg,ref=self.origin()
        for key in ('native_policy','reference_policy','allowance_policy'):
            other=copy.deepcopy(cfg);other[key]='DIFFERENT'
            with self.subTest(key=key):
                with self.assertRaisesRegex(c.Stop,'CONSUMER_ORIGIN'):a.inspect_origin(ref,other)

    def test_origin_runtime_source_and_input_mismatch(self):
        cfg,ref=self.origin();other=copy.deepcopy(cfg);other['runtime']['python_sha256']='0'*64
        with self.assertRaisesRegex(c.Stop,'RUNTIME_IDENTITY'):a.inspect_origin(ref,other)
        other=copy.deepcopy(cfg);other['inputs']={'old_reference':'/DIFFERENT'}
        with self.assertRaisesRegex(c.Stop,'CONSUMER_ORIGIN'):a.inspect_origin(ref,other)

    def test_missing_old_ledger_or_marker_rejected(self):
        cfg,ref=self.origin();p=next((Path(ref['run_root'])/'ledger').glob('*.json'));p.unlink()
        with self.assertRaises(c.Stop):a.inspect_origin(ref)

    def test_missing_consumed_marker_rejected(self):
        cfg,ref=self.origin();(Path(ref['run_root'])/'CONSUMED.json').unlink()
        with self.assertRaises(c.Stop):a.inspect_origin(ref)

    def test_npz_wrong_column_shape_rejected(self):
        cfg,ref=self.origin();cid=c.policy.CANDIDATES[2].id;run=Path(ref['run_root'])
        p=run/'results'/(cid+'_vectors.npz');archive(p,1)
        q=run/'results'/(cid+'_diagnostics.json');row=b.read(q);row['vector_archive']['sha256']=b.sha(p);write(q,row)
        self.refresh_origin(cfg,ref)
        with self.assertRaisesRegex(c.Stop,'SHAPE_DTYPE'):a.inspect_origin(ref)

    def test_partial_stop_not_accepted(self):
        cfg,ref=self.origin();write(Path(ref['run_root'])/'STOP.json',{'partial':True})
        with self.assertRaisesRegex(c.Stop,'PARTIAL'):a.inspect_origin(ref)

    def test_decode_requires_authorized_active_bundle_load(self):
        with self.assertRaisesRegex(c.Stop,'AUTHORIZED_BUNDLE'):
            a.load_for_backend(type('Fake',(),{'s':object()})())

    def test_missing_adaptive_value_not_zero_filled(self):
        cfg,ref=self.origin();cid=c.policy.CANDIDATES[0].id;p=Path(ref['run_root'])/'results'/(cid+'_diagnostics.json')
        row=b.read(p);row['inputs']['maximum_norm']['value']=None;write(p,row);self.refresh_origin(cfg,ref)
        with self.assertRaisesRegex(c.Stop,'MISSING_COMMON_DIAGNOSTICS'):a.inspect_origin(ref)


class MockBackend:
    def __init__(self,s):self.s=s
    def setup(self):pass
    def prepare(self,prefix):pass
    def load_diagnostics(self):return {x.id:{'synthetic':True} for x in c.policy.CANDIDATES}
    def diagnose(self,*args):raise AssertionError('DIAGNOSIS_MUST_NEVER_RUN')
    def reuse(self,*args):return [{'cosine':10,'sine':10}]
    def native(self,candidate,diagnostic):
        self.s.call('compose',lambda:None)
        for _ in range(4 if candidate.q==4 else 6):self.s.call('lowering',lambda:None)
        for _ in range(2):
            self.s.call('wrapper',lambda:None)
            self.s.call('compile',lambda:self.s.call('transpile',lambda:None))
        for _ in range(6):self.s.call('simulate',lambda:None)
        return [{'cosine':10,'sine':10}]
    def allowance(self,*args):return {'ARTIFICIAL':True}
    def account(self,*args):return {'eligible':True,'G':100}
    def rank(self,rows):return {'status':'ARTIFICIAL_ONLY','science_GO':False}


class ControlTests(Fixture):
    def runnable(self):
        _,ref=self.origin();witness=self.root/'artificial_witness.json';write(witness,{'legacy':{}})
        cfg=dict(source_commit='a'*40,run_root=str(self.root/'new_run'),budget=c.counts(),
            native_policy=c.policy.NATIVE_POLICY,allowance_policy=c.policy.ALLOWANCE_POLICY,
            reference_policy='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC',files=[],
            accepted_diagnostics=ref,reuse_witnesses=[str(witness)]*2)
        approval=dict(issuer='USER_EXPLICIT',execution_authorized=True,contract_sha256=b.digest(cfg),
            policies_adopted=[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')])
        grant=dict(id='NEW_ARTIFICIAL_ONLY',contract_sha256=b.digest(cfg),approval_sha256=b.digest(approval),
            budget=cfg['budget'],run_root=cfg['run_root'],state='UNUSED',retry=False)
        self.stack.enter_context(patch.object(c,'preflight',return_value={'execution_ready':True}))
        return cfg,approval,grant

    def test_exact_mock_trace_no_duplicate_diagnosis(self):
        cfg,approval,grant=self.runnable();c.execute(cfg,approval,grant,backend_factory=MockBackend)
        row=b.read(Path(cfg['run_root'])/'completion.json');self.assertEqual(row['used'],c.counts())
        for key in ('diagnose','path_signal','reference_rebind','request_factory','rte_event','sampling'):self.assertEqual(row['used'][key],0)

    def test_gated_bundle_loader_restores_inputs_using_fake_decoder(self):
        cfg,approval,grant=self.runnable();origin=old.read(cfg['accepted_diagnostics']['contract'])
        for key in ('physical','compiler','binding','candidates','inputs','runtime'):cfg[key]=origin[key]
        approval['contract_sha256']=b.digest(cfg);grant['contract_sha256']=b.digest(cfg);grant['approval_sha256']=b.digest(approval)
        class Vector:
            def copy(self):return self
        class Z(dict):
            def __enter__(self):return self
            def __exit__(self,*args):pass
        class FakeNP:
            @staticmethod
            def load(*args,**kwargs):
                return Z(primary=Vector(),independent=Vector(),occupation_basis=(3,5,6,9,10,12),qindices=(12,10,6,9,5,3))
            array_equal=staticmethod(lambda a,b:a==b)
            isfinite=staticmethod(lambda a:SimpleNamespace(all=lambda:True))
        class Loader(MockBackend):
            def __init__(self,s):
                super().__init__(s);self.np=FakeNP();self.preps={};self.operations={}
                self.ctx=SimpleNamespace(oracle=SimpleNamespace(basis=(3,5,6,9,10,12),qindices=(12,10,6,9,5,3)))
            def prepare(self,prefix):
                p=Path(self.s.cfg['accepted_diagnostics']['run_root'])/'results'/('preparation_'+str(prefix)+'.json');row=b.read(p)
                self.preps[prefix]=SimpleNamespace(preparation_hash=row['preparation_sha256']);self.operations[prefix]=row['operations_sha256']
            def load_diagnostics(self):return a.load_for_backend(self)
        c.execute(cfg,approval,grant,backend_factory=Loader)
        reuse=b.read(Path(cfg['run_root'])/'results/accepted_diagnostics_reuse.json')
        self.assertFalse(reuse['path_recomputed']);self.assertFalse(reuse['reference_rebound'])
        self.assertEqual(b.read(Path(cfg['run_root'])/'completion.json')['used'],c.counts())

    def test_old_consumed_grant_id_cannot_authorize_new_execution(self):
        cfg,approval,grant=self.runnable();grant['id']='ARTIFICIAL_OLD_GRANT'
        with self.assertRaisesRegex(c.Stop,'NEW_UNUSED_COMPARE_GRANT'):c.execute(cfg,approval,grant,backend_factory=MockBackend)
        self.assertFalse(Path(cfg['run_root']).exists())

    def test_boolean_grant_id_rejected(self):
        cfg,approval,grant=self.runnable();grant['id']=True
        with self.assertRaisesRegex(c.Stop,'NEW_UNUSED_COMPARE_GRANT'):c.execute(cfg,approval,grant,backend_factory=MockBackend)

    def test_new_grant_reuse_and_existing_root_refused(self):
        cfg,approval,grant=self.runnable();c.execute(cfg,approval,grant,backend_factory=MockBackend)
        # Artificial preflight normally bypasses root checks, so mkdir is the independent exclusive guard.
        with self.assertRaises(FileExistsError):c.execute(cfg,approval,grant,backend_factory=MockBackend)

    def test_unapproved_backend_never_constructed(self):
        cfg,approval,grant=self.runnable();approval['execution_authorized']=False
        with self.assertRaisesRegex(c.Stop,'EXPLICIT_COMPARE_APPROVAL'):c.execute(cfg,approval,grant,backend_factory=lambda s:self.fail('IMPORT'))

    def test_bundle_budget_overrun_stops_nonrefundable(self):
        cfg,approval,grant=self.runnable()
        class Bad(MockBackend):
            def load_diagnostics(self):self.s.take('bundle_load');return super().load_diagnostics()
        with self.assertRaisesRegex(c.Stop,'BUDGET_EXCEEDED'):c.execute(cfg,approval,grant,backend_factory=Bad)
        self.assertEqual(b.read(Path(cfg['run_root'])/'STOP.json')['used']['bundle_load'],1)
        self.assertFalse((Path(cfg['run_root'])/'completion.json').exists())

    def test_sampling_zero_budget_rejects_before_callback(self):
        cfg,approval,grant=self.runnable()
        class Bad(MockBackend):
            def load_diagnostics(inner):
                inner.s.call('sampling',lambda:self.fail('SAMPLING_MUST_NOT_START'))
        with self.assertRaisesRegex(c.Stop,'BUDGET_EXCEEDED:sampling'):
            c.execute(cfg,approval,grant,backend_factory=Bad)
        self.assertEqual(b.read(Path(cfg['run_root'])/'STOP.json')['used']['sampling'],0)
        self.assertFalse((Path(cfg['run_root'])/'completion.json').exists())

    def test_partial_native_failure_never_completes_or_retries(self):
        cfg,approval,grant=self.runnable()
        class Bad(MockBackend):
            def native(self,*args):raise ValueError('ARTIFICIAL_FAILURE')
        with self.assertRaises(ValueError):c.execute(cfg,approval,grant,backend_factory=Bad)
        self.assertTrue((Path(cfg['run_root'])/'CONSUMED.json').exists())
        self.assertFalse((Path(cfg['run_root'])/'completion.json').exists())

    def test_ineligible_account_never_enters_ranking(self):
        cfg,approval,grant=self.runnable()
        class Bad(MockBackend):
            def account(self,*args):return {'eligible':False,'G':None}
            def rank(self,*args):raise AssertionError('MUST_NOT_RANK')
        with self.assertRaisesRegex(c.Stop,'NO_RANK'):c.execute(cfg,approval,grant,backend_factory=Bad)
        self.assertEqual(b.read(Path(cfg['run_root'])/'STOP.json')['used']['ranking'],0)

    def test_pr_witness_missing_blocks_before_bootstrap(self):
        cfg,approval,grant=self.runnable()
        with patch.object(c,'preflight',return_value={'execution_ready':False}):
            with self.assertRaisesRegex(c.Stop,'PR_IDENTITY_BLOCKS'):c.execute(cfg,approval,grant,backend_factory=MockBackend)
        self.assertFalse(Path(cfg['run_root']).exists())

    def test_cli_execute_without_new_external_files_refused_before_read(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h2_accepted_compare.py'),
            '--contract','/ABSENT','--execute'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('NEW_EXPLICIT_COMPARE_APPROVAL',r.stderr)

    def test_numeric_backend_methods_preserved_and_old_diagnostics_absent(self):
        def methods(path):
            t=ast.parse(path.read_text());cl=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Backend')
            return {n.name:ast.dump(n,include_attributes=False) for n in cl.body if isinstance(n,ast.FunctionDef)}
        before=methods(ROOT/'scripts/h2_common_production_backend.py');after=methods(ROOT/'scripts/h2_accepted_compare_backend.py')
        for key in ('setup','prepare','steps','allowance','account','rank'):self.assertEqual(before[key],after[key],key)
        self.assertNotIn('diagnose',after);self.assertNotIn('reference',after)

    def test_fixed_budget_and_old_counts_not_changed(self):
        expected=old.counts('COMPARE_REUSE_PR');expected.update(diagnose=0,path_signal=0,reference_rebind=0,bundle_load=1,sampling=0)
        self.assertEqual(c.counts(),expected);self.assertEqual(old.counts('DIAGNOSTICS')['path_signal'],8)

    def test_no_scientific_module_imported(self):
        self.assertFalse(any(n.split('.')[0] in {'numpy','scipy','qiskit','trotterlib','trottertracks'} for n in sys.modules))

    def test_contract_scope_cannot_be_replaced_with_old_branch(self):
        with self.assertRaisesRegex(c.Stop,'NEW_COMPARE_CONTRACT_SCHEMA'):c.preflight({'branch':'COMPARE_REUSE_PR'})


class WitnessTests(Fixture):
    def bundle(self):
        candidate=c.policy.CANDIDATES[2];count=33
        task_ids=['H2E1__'+candidate.id+'__'+str(i) for i in range(count)]
        event=dict(taylor_order=0,rotation_component_id='ARTIFICIAL_ROTATION',product_component_ids=[],
            selected_component_ids=['ARTIFICIAL_ROTATION'],phase=dict(real=1.,imag=0.),
            application_sequence=[dict(application_index=0,role='rotation',component_id='ARTIFICIAL_ROTATION')])
        streams=[[copy.deepcopy(event) for _ in range(32)] for _ in range(4)]
        event_digest=b.digest(streams);anchors=[];originals=[];costs=[]
        for i,tid in enumerate(task_ids):
            task=dict(id=tid,sample=i,candidate=asdict(candidate))
            axes=[dict(axis=axis,metrics={'rz_count':100+i}) for axis in ('cosine','sine')]
            raw=dict(task=task,event_digest=event_digest,axis_costs=axes)
            p=self.root/'originals'/(str(i)+'.json');write(p,raw)
            originals.append(dict(task=tid,path=str(p),sha256=b.sha(p)))
            anchors.append(dict(task=task,result={'saved':raw},acceptance={'saved':{'result_sha256':b.sha(p)}}))
            costs.append(dict(sample=i,cosine=100+i,sine=100+i,receipt_sha256=b.digest(raw),accepted=True))
        events=self.root/'events.json';cohort=self.root/'cohort.json';correctness=self.root/'correctness.json'
        write(events,dict(candidate=candidate.id,task_event_hashes=[event_digest]*count,
            tasks=[dict(task=tid,sample=i,events=streams,conditional=False,cohort='EXPLORATION') for i,tid in enumerate(task_ids)]))
        write(cohort,dict(candidate=candidate.id,conditional=False,pooling=False,tasks=costs))
        write(correctness,dict(candidate=candidate.id,accepted_tasks=[t['receipt_sha256'] for t in costs],scope='SINGLE_SAVED_INPUT_STATE'))
        proof=dict(events=str(events),correctness=str(correctness),cohort=str(cohort))
        identities={k:b.digest('ARTIFICIAL_IDENTITY:'+k) for k in old.REUSE_FIELDS}
        legacy=dict(candidate=candidate.id,saved=identities,proposed=identities,all_identities_equal=True,tasks=costs,
            correctness_scope='SAVED_COHORT_ACCEPTANCE_WITH_DECLARED_VECTOR_GATE_SCOPE',proof=proof)
        attestations={}
        for key in old.REUSE_FIELDS:
            p=self.root/'attestations'/(key+'.json');attestations[key]=str(p)
            write(p,dict(kind='HISTORICAL_TO_COMMON_BYTE_IDENTITY_V1',field=key,saved_sha256=identities[key],
                current_sha256=identities[key],original_receipt_sha256=[r['sha256'] for r in originals],
                evidence_files=[{'path':r['path'],'sha256':r['sha256']} for r in originals],
                observation='WORKER_OBSERVED_LOAD_MANIFEST',historical_runtime_identity=identities[key]))
        wrapper=self.root/'witness.json';write(wrapper,dict(schema='H2_PR_COMPILED_IDENTITY_BYTE_JOIN_V1',
            legacy=legacy,attestations=attestations,original_results=originals))
        paths=[events,cohort,correctness,wrapper]+[Path(r['path']) for r in originals]+list(map(Path,attestations.values()))
        cfg={'files':[dict(path=str(p),sha256=b.sha(p)) for p in paths]}
        self.stack.enter_context(patch.object(c,'saved_anchors',return_value=anchors))
        self.stack.enter_context(patch.object(old,'saved_cost_tasks',return_value=costs))
        self.stack.enter_context(patch.object(c,'expected_identities',return_value=identities))
        return wrapper,candidate,cfg

    def test_complete_artificial_proofs_can_be_accepted(self):
        p,candidate,cfg=self.bundle();result=c.validate_witness(p,candidate,cfg,{})
        self.assertEqual(len(result['tasks']),33)

    def test_labels_and_prep_hash_only_never_suffice(self):
        p,candidate,cfg=self.bundle();row=b.read(p);row['attestations']={};write(p,row)
        with self.assertRaisesRegex(c.Stop,'ALL_IDENTITY_FIELD_PROOFS'):c.validate_witness(p,candidate,cfg,{})

    def test_missing_historical_original_receipts_rejected(self):
        p,candidate,cfg=self.bundle();row=b.read(p);row['original_results']=[];write(p,row)
        with self.assertRaisesRegex(c.Stop,'ORIGINAL_TASK_RECEIPTS'):c.validate_witness(p,candidate,cfg,{})

    def test_changed_saved_paired_rz_rejected(self):
        p,candidate,cfg=self.bundle();row=b.read(p);row['legacy']['tasks'][0]['cosine']+=1;write(p,row)
        with self.assertRaisesRegex(c.Stop,'PAIRED_COST_PROJECTION'):c.validate_witness(p,candidate,cfg,{})

    def test_missing_field_evidence_file_rejected(self):
        p,candidate,cfg=self.bundle();q=Path(b.read(p)['attestations']['rotation']);row=b.read(q);row['evidence_files']=[];write(q,row)
        with self.assertRaisesRegex(c.Stop,'FIELD_PROOF_EVIDENCE'):c.validate_witness(p,candidate,cfg,{})

    def test_declared_runtime_is_not_actual_worker_runtime(self):
        p,candidate,cfg=self.bundle();q=Path(b.read(p)['attestations']['runtime_closure']);row=b.read(q);row['observation']='PACKAGE_VERSIONS_ONLY';write(q,row)
        with self.assertRaisesRegex(c.Stop,'HISTORICAL_ACTUAL_RUNTIME'):c.validate_witness(p,candidate,cfg,{})

    def test_summary_events_cannot_replace_event_sequence(self):
        p,candidate,cfg=self.bundle();q=Path(b.read(p)['legacy']['proof']['events']);row=b.read(q)
        row['tasks'][0]['events']={'N2':0,'N4':0};write(q,row)
        with self.assertRaisesRegex(c.Stop,'SOURCE_EVENT_DIGEST'):c.validate_witness(p,candidate,cfg,{})

    def test_event_order_mutation_rejected(self):
        p,candidate,cfg=self.bundle();q=Path(b.read(p)['legacy']['proof']['events']);row=b.read(q)
        row['tasks'][0]['events'][0][0]['rotation_component_id']='TAMPERED';write(q,row)
        with self.assertRaisesRegex(c.Stop,'SOURCE_EVENT_DIGEST'):c.validate_witness(p,candidate,cfg,{})

    def test_correctness_scope_cannot_be_silently_promoted(self):
        p,candidate,cfg=self.bundle();q=Path(b.read(p)['legacy']['proof']['correctness']);row=b.read(q);row['scope']='ALL_STATE_CERTIFIED';write(q,row)
        with self.assertRaisesRegex(c.Stop,'CORRECTNESS_SCOPE'):c.validate_witness(p,candidate,cfg,{})


if __name__=='__main__':unittest.main(verbosity=2)
