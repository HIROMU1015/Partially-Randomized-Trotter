"""Artificial data only. Never read a registered event or evaluate a saved key."""
import copy
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from trottertracks.algorithm_codesign.parent_experiment.design import candidates,kkt_candidate,proposals,ledger,comparisons
from trottertracks.algorithm_codesign.parent_experiment.pipeline import expected,freeze_inventory,serial,validate_common,acquire
from trottertracks.algorithm_codesign.parent_experiment.launch import validate_binding,verify_launch,CONTRACT_SHA,CONTRACT
from trottertracks.algorithm_codesign.parent_experiment.runner import resource_reason,final_status,COMPLETE,INCONCLUSIVE
from trottertracks.algorithm_codesign.rte_development.evidence import EvidenceWriter,verify_chunks,atomic_json
from trottertracks.algorithm_codesign.return_aggregation import Interval

POLICY=dict(iterations=128,root_bits=128,share_bits=32)
PREC=dict(root_bits=128,probability_bits=80,rho=F(1,10**10),eta=F(1,10**10))

def events():
    return [dict(coefficient=F(2),proposal=F(1,2),weight=F(4)),dict(coefficient=F(1),proposal=F(1,2),weight=F(2))]

def prices():
    return [dict(T=10,CX=2,**{'1Q':15},strict_native_error_upper=F(1,10**8)),
            dict(T=3,CX=7,**{'1Q':8},strict_native_error_upper=F(1,10**8))]

def config():
    return dict(input=dict(tau='1/3'),epsilon_axis='1/100',alpha_axis='1/40',caps=dict(shot_cap_per_axis=10**9))

def tool():
    return dict(name='pygridsynth',version='mock',source_sha256='1'*64,seed=3,
        tool_identity_sha256='2'*64,runtime_executable_sha256='3'*64,package_lock_sha256='4'*64,
        synthesizer_options=dict(seed=3,up_to_phase=False),interval_dps=50)

class CandidateControls(unittest.TestCase):
    def test_simplex_and_frozen_policy(self):
        r=candidates(F(7),[F(4),F(3)],[F(1),F(3,4)],POLICY)
        self.assertEqual(len(r['candidates']),5)
        for v in r['candidates']:self.assertEqual(sum(v['allocation']),7)
        self.assertFalse(r['new_prices_used']);self.assertFalse(r['replacement_candidates'])
        self.assertGreaterEqual(r['kkt']['L_infinity_distance_upper'],0)
        self.assertEqual(r['kkt']['simplex_residual'],0)
    def test_kkt_known_interior(self):
        # Independent exact witness: raw C=(1,9/16), lambda=3/5,
        # z=(3,4), b=(4,3); sqrt(C)*z/sqrt(z*z+b*b)=3/5.
        r=kkt_candidate(F(7),[F(4),F(3)],[F(1),F(9,16)],**POLICY)
        lo,hi=r['lambda_interval'];self.assertLessEqual(lo,F(3,5));self.assertGreaterEqual(hi,F(3,5))
        z=r['allocation'];self.assertLess(abs(z[0]-3),F(1,10**8));self.assertFalse(r['exact_KKT_optimum'])
    def test_equal_costs_duplicate_not_refilled(self):
        r=candidates(F(2),[F(1),F(1)],[F(4),F(4)],POLICY)
        self.assertEqual(len(r['candidates']),3);self.assertEqual(len(r['dropped']),2)
    def test_zero_proxy_not_interior_negative(self):
        r=candidates(F(3),[F(1),F(2)],[F(0),F(4)],POLICY)
        self.assertEqual(r['kkt']['status'],'MATHEMATICALLY_NOT_APPLICABLE');self.assertEqual(len(r['candidates']),3)
    def test_bad_dimension_and_float(self):
        with self.assertRaises(ValueError):kkt_candidate(1,[1,2],[1],**POLICY)
        with self.assertRaises(TypeError):kkt_candidate(.1,[1],[1],**POLICY)
    def test_fixed_resolution_no_rescue(self):
        with self.assertRaises(ArithmeticError):kkt_candidate(F(7),[F(4),F(3)],[F(1),F(3,4)],iterations=1,root_bits=1,share_bits=2)

class LawLedgerControls(unittest.TestCase):
    def test_shared_scale_and_dyadic_corrected_mean(self):
        laws,dropped=proposals(events(),[F(10),F(3)],F(8),PREC)
        self.assertEqual(len(laws),5)
        self.assertEqual([v['kappa'] for v in laws[2:]],[F(1,2),F(8),F(128)])
        for v in laws:
            self.assertEqual(sum(e['proposal'] for e in v['events']),1)
            for e in v['events']:self.assertEqual(e['proposal']*e['weight'],e['coefficient'])
    def test_zero_cost_preserves_support(self):
        laws,dropped=proposals(events(),[F(0),F(3)],F(8),PREC)
        self.assertEqual(len(laws),4);self.assertEqual(dropped[0]['label'],'analytic_IS')
        for v in laws:self.assertTrue(all(e['proposal']>0 for e in v['events']))
    def test_zero_scale_no_floor(self):
        laws,dropped=proposals(events(),[F(0),F(0)],F(0),PREC)
        self.assertEqual(len(laws),1);self.assertTrue(dropped)
    def test_regularizer_not_charged(self):
        laws,_=proposals(events(),[F(10),F(3)],F(8),PREC)
        for law in laws:
            got=ledger(law['events'],prices(),F(1,10**10),config())
            self.assertEqual(got['per_trial_native']['T'],sum(e['proposal']*p['T'] for e,p in zip(law['events'],prices())))
    def test_common_error_and_sufficient_budget(self):
        got=ledger(events(),prices(),F(1,10**10),config())
        self.assertEqual(got['Taylor_upper'],F(1,3)**6/720)
        self.assertEqual(got['weighted_native_error_upper'],F(3,10**8))
        self.assertEqual(got['probability_rounding_mean_bias'],0)
        self.assertEqual(got['alpha_axis'],F(1,40));self.assertEqual(got['quantum_shots_executed'],0)
        self.assertEqual(got['expected_native_intercept']['T'],2*got['N_per_axis']*F(13,2))
    def test_missing_not_zero(self):
        p=prices();del p[0]['T']
        with self.assertRaises(KeyError):ledger(events(),p,0,config())
    def test_bias_ineligible_distinct(self):
        got=ledger(events(),prices(),F(1),config());self.assertEqual(got['eligibility'],'BIAS_INELIGIBLE')
    def test_shot_cap_no_clipping(self):
        c=config();c['caps']['shot_cap_per_axis']=1;got=ledger(events(),prices(),0,c)
        self.assertEqual(got['eligibility'],'SHOT_CAP_INELIGIBLE');self.assertGreater(got['N_per_axis'],1)
    def test_noncommon_not_inclusion_positive(self):
        l=ledger(events(),prices(),0,config());rows=[dict(id='base',arm='ordinary',candidate='common',ledger=l)]
        self.assertEqual(comparisons(rows)['mechanisms'][0]['accounting_relation'],'NO_COMPARABLE_NONCOMMON')
        rows.append(dict(id='vertex',arm='ordinary',candidate='vertex_0',ledger=copy.deepcopy(l)))
        r=comparisons(rows);self.assertEqual(r['mechanisms'][0]['accounting_relation'],'NO_STRICT_REGISTERED_ALLOCATION_GAIN')
        self.assertFalse(r['scientific_adoption'])
    def test_positive_is_only_fixed_policy_relation(self):
        l=ledger(events(),prices(),0,config());better=copy.deepcopy(l);better['expected_native_intercept']['T']-=1
        r=comparisons([dict(id='a',arm='ordinary',candidate='common',ledger=l),dict(id='b',arm='ordinary',candidate='vertex_0',ledger=better)])
        self.assertLess(r['mechanisms'][0]['difference'],0);self.assertFalse(r['mechanisms'][0]['scientific_GO'])

class AcquisitionControls(unittest.TestCase):
    def c(self):return dict(primitive_error='1/1000000',caps=dict(new_synthesis_keys=18,new_root_events=36,old_unique_guards=36,all_unique_primitive_guards=54))
    def test_key_tool_binding(self):
        a=expected('2/11',tool(),self.c());b=expected('4/22',tool(),self.c());self.assertEqual(a,b)
        t=tool();t['seed']=4;self.assertNotEqual(a['key'],expected('2/11',t,self.c())['key'])
    def test_freeze_old_reuse_and_root_only_new(self):
        cells=[dict(candidate='common',bindings=[dict(native_ir=[('R',0,'2/11',1)],new_root=True)]),
               dict(candidate='vertex',bindings=[dict(native_ir=[('R',0,'3/13',1)],new_root=True)])]
        r=freeze_inventory(cells,{'2/11':{}},tool(),self.c())
        self.assertEqual(r['new_keys'],1);self.assertEqual(r['planned_root_matrix_checks'],4)
    def test_missing_old_baseline_stops(self):
        cells=[dict(candidate='common',bindings=[dict(native_ir=[('R',0,'2/11',1)],new_root=True)])]
        with self.assertRaises(ArithmeticError):freeze_inventory(cells,{},tool(),self.c())
    def test_new_nonroot_forbidden(self):
        cells=[dict(candidate='vertex',bindings=[dict(native_ir=[('R',0,'2/11',1)],new_root=False)])]
        with self.assertRaises(ArithmeticError):freeze_inventory(cells,{},tool(),self.c())
    def test_exact_endpoint_has_no_key(self):
        r=freeze_inventory([dict(candidate='vertex',bindings=[dict(native_ir=[('S',0)],new_root=True)])],{},tool(),self.c())
        self.assertEqual(r['keys'],[]);self.assertEqual(r['new_keys'],0)
    def test_no_worker_needed_for_empty_new_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=EvidenceWriter(tmp,100000);g=Mock();c=self.c();c['caps']['workers']=2
            r=acquire(tmp,c,{},dict(keys=[]),writer,g);self.assertEqual(r,({},[]))
    def test_common_compatibility_failure(self):
        e=dict(word=(),child=0,ratio=F(2,11),phase_i_power=0,coefficient=1,proposal=1,weight=1)
        validate_common([dict(event=e)],[e]);bad=dict(e,phase_i_power=2)
        with self.assertRaises(ArithmeticError):validate_common([dict(event=e)],[bad])

class GateOutputControls(unittest.TestCase):
    def authority(self):
        c=dict(execution=dict(authorization_path='auth.json',optional_receipt_path='receipt.md'))
        a=dict(status='APPROVED_FOR_ONE_PARENT_ALLOCATION_RUN',science_execution_authorized=True,source_commit='1'*40,
               contract_sha256=CONTRACT_SHA,runs=1,retries=0,mandatory_STOP=True,explicit_execution_instruction='artificial approval only')
        return a,c
    def test_clean_direct_child(self):
        a,c=self.authority();validate_binding(a,c,'1'*40,'2'*40,['1'*40],['auth.json'],False)
    def test_source_auth_scope_and_dirty_reject(self):
        a,c=self.authority()
        for parents,changed,dirty in [(['0'*40],['auth.json'],False),(['1'*40],['auth.json','code.py'],False),(['1'*40],['auth.json'],True),(['1'*40],[],False)]:
            with self.assertRaises(PermissionError):validate_binding(a,c,'1'*40,'2'*40,parents,changed,dirty)
    def test_pending_rejected_before_input_and_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'auth.json').write_text('{"science_execution_authorized":false}')
            c=dict(execution=dict(authorization_path='auth.json'))
            with patch('trottertracks.algorithm_codesign.parent_experiment.launch.read_contract',return_value=c),patch('trottertracks.algorithm_codesign.parent_experiment.launch.check_identities') as inputs:
                with self.assertRaises(PermissionError):verify_launch(root,'1'*40)
                inputs.assert_not_called()
    def test_retry_and_instruction_required(self):
        a,c=self.authority()
        for key,value in [('runs',2),('retries',1),('mandatory_STOP',False),('explicit_execution_instruction',None)]:
            bad=dict(a,**{key:value})
            with self.assertRaises(PermissionError):validate_binding(bad,c,'1'*40,'2'*40,['1'*40],['auth.json'],False)
    def test_exclusive_receipt_no_marker_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'marker.json';atomic_json(p,dict(runs=1))
            with self.assertRaises(FileExistsError):atomic_json(p,dict(runs=2))
            self.assertEqual(json.loads(p.read_text())['runs'],1)
    def test_utf8_lossless_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=EvidenceWriter(tmp,100000,chunk_bytes=31);info=w.write('data.json',dict(text='日本語'*100))
            self.assertTrue(verify_chunks(tmp,info));self.assertTrue(all(v['bytes']<=31 for v in info['chunks']))
            part=Path(tmp)/info['chunks'][0]['path'];part.write_text('changed')
            with self.assertRaises(ArithmeticError):verify_chunks(tmp,info)
    def test_output_cap_partial_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):EvidenceWriter(tmp,33000).write('large.json',dict(text='x'*20000))
            self.assertFalse((Path(tmp)/'large.json').exists())
    def test_interval_serialization_is_lossless(self):
        self.assertEqual(serial(dict(v=Interval(F(1,3),F(1,2)))),dict(v=dict(lo=F(1,3),hi=F(1,2))))
    def test_outer_failure_overrides_success_prefix(self):
        c=dict(caps=dict(configurations=15,laws=75,new_synthesis_keys=18));i={'execution_HEAD':'1'*40}
        inner=dict(**i,status=COMPLETE,retries=0,configurations=2,laws=3,new_synthesis_calls=1)
        self.assertEqual(final_status(0,inner,i,None,c),COMPLETE)
        for code,reason in [(1,None),(0,'RSS cap')]:self.assertEqual(final_status(code,inner,i,reason,c),INCONCLUSIVE)
        self.assertEqual(final_status(0,None,i,None,c),INCONCLUSIVE)
    def test_outer_guard_caps_and_no_wall(self):
        caps=dict(total_CPU_seconds=10,aggregate_RSS_MiB=3,parent_RSS_MiB=2,worker_RSS_MiB=1,per_key_CPU_seconds=5,workers=2,output_bytes=100000,per_key_log_bytes=100)
        snap=dict(CPU=1,RSS=10,processes=[dict(key=False,RSS=10,own_CPU=1,completed_child_CPU=0)])
        self.assertIsNone(resource_reason(snap,caps,10,[10]))
        for update,output,logs in [({'CPU':11},10,[10]),({'RSS':4*1048576},10,[10]),({},100000,[10]),({},10,[101])]:
            self.assertIsNotNone(resource_reason(dict(snap,**update),caps,output,logs))
    def test_import_no_numeric_acquisition(self):
        import ast
        root=Path(__file__).resolve().parents[3]
        for p in (root/'src/trottertracks/algorithm_codesign/parent_experiment').glob('*.py'):
            for node in ast.parse(p.read_text()).body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    name=(node.module or '') if isinstance(node,ast.ImportFrom) else ','.join(v.name for v in node.names)
                    self.assertFalse(any(v in name for v in ('numpy','scipy','qiskit','pygridsynth')))


class PipelineConnectionControls(unittest.TestCase):
    def test_artificial_complete_workflow_freezes_before_acquisition(self):
        # Construct only off-domain local metadata. Provider/native matrices and
        # synthesis are mocked; no registered paths are read by read_bound.
        from trottertracks.algorithm_codesign.parent_allocation import RootAllocationGenerator
        from trottertracks.algorithm_codesign.parent_experiment import pipeline as pp
        p=(F(1,7),F(2,7),F(4,7));x=F(1,3)
        ordinary=RootAllocationGenerator.ordinary(p,x,**PREC)
        p5=RootAllocationGenerator.closed_p5(p,x,**PREC)
        def fake_ir(e):
            if e.get('ratio') is None:return [('S',0)]
            return [('R',0,str(e['ratio']),1)]
        source={};cache={};invent=[]
        for arm,gen in [('ordinary',ordinary),('closed_P5_full',p5)]:
            source[arm]=[dict(index=i,event=e,native_ir=fake_ir(e)) for i,e in enumerate(gen.reference_events())]
        for arm in ('partial_return_tail','closed_P3_tail','matched_CTS'):
            source[arm]=copy.deepcopy(source['ordinary'][:3])
        t=tool();c=config();c.update(central_arms=['ordinary','closed_P5_full'],reference_arms=['partial_return_tail','closed_P3_tail','matched_CTS'],
            candidates=dict(KKT=POLICY),precision={k:str(v) if isinstance(v,F) else v for k,v in PREC.items()},primitive_error='1/1000000',matrix_tolerance=1e-10,
            acquisition=dict(tool_identity=dict(path='tool.json',sha256='2'*64),runtime_executable_sha256='3'*64,options=t['synthesizer_options'],interval_dps=50))
        c['input'].update(p=list(map(str,p)))
        c['caps'].update(new_synthesis_keys=18,new_root_events=36,old_unique_guards=36,all_unique_primitive_guards=54,configurations=15,events_per_configuration=273,event_bindings=4096,laws=75,new_root_matrix_checks=72,workers=2)
        for items in source.values():
            for item in items:
                r=str(item['event']['ratio'])
                if r not in cache:
                    cache[r]=dict(T_count=10,one_qubit_count=12)
                    exp=pp.expected(r,t,c)
                    from trottertracks.algorithm_codesign.parent_allocation_native import legacy_spec,sha,canonical
                    invent.append(dict(ratio=r,key=sha(canonical(legacy_spec(exp)))))
        bound={};rows=[]
        for i,(arm,items) in enumerate(source.items()):
            path=f'events/row_{i:02d}.json';key='events_'+str(i)
            bound[key]=dict(path=path,sha256='5'*64);rows.append(dict(steps=1,arm=arm,events=dict(path=path)))
        for name in ('resource_map','cache','inventory','old_contract','guard_audit'):bound[name]=dict(path=name+'.json',sha256='5'*64)
        c['reuse']=dict(inputs=bound,historical_guard_source=dict(sha256='5'*64))
        documents={bound['resource_map']['path']:dict(rows=rows),bound['cache']['path']:cache,
            bound['inventory']['path']:dict(keys=invent),bound['old_contract']['path']:dict(primitive_error=c['primitive_error'],synthesizer_options=c['acquisition']['options'],interval_dps=50),bound['guard_audit']['path']:{}}
        documents.update({f'events/row_{i:02d}.json':items for i,items in enumerate(source.values())})
        class Writer:
            def __init__(self):self.records={}
            def write(self,path,value):
                self.records[path]=value
                return dict(path=path,bytes=1,sha256=hashlib.sha256(path.encode()).hexdigest())
        writer=Writer();new_acquisition=[]
        def acquire_mock(root,c,identity,inventory,w,g):
            self.assertIn('candidate_freeze.json',w.records);self.assertIn('inventory.json',w.records)
            self.assertEqual(identity['candidate_freeze_sha256'],hashlib.sha256(b'candidate_freeze.json').hexdigest())
            fresh={k['ratio']:dict(T_count=11,one_qubit_count=13) for k in inventory['keys'] if k['acquisition']=='NEW_ONCE'}
            new_acquisition.append(fresh);return fresh,[]
        def price(gates,cache):
            value=sum(cache[str(F(g[2]))]['T_count'] for g in gates if g[0]=='R')
            return dict(T=value,CX=2,**{'1Q':20},strict_native_error_upper=F(1,10**8))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'tool.json').write_text('{}')
            with patch.object(pp,'read_bound',side_effect=lambda root,entry:copy.deepcopy(documents[entry['path']])),\
                 patch.object(pp,'tool_binding',return_value=t),patch.object(pp,'digest',return_value='5'*64),\
                 patch.object(pp,'normalize_saved',side_effect=lambda saved,exp,**kw:saved),patch.object(pp,'recheck_entry',side_effect=lambda entry,exp:entry),\
                 patch.object(pp,'native_accounting',side_effect=price),patch.object(pp,'native_ir_g9',side_effect=fake_ir),\
                 patch.object(pp,'acquire',side_effect=acquire_mock),patch.object(pp,'root_validation',return_value=dict(mock=True)),\
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'),\
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.configure'),\
                 patch('trottertracks.algorithm_codesign.rte_development.numeric.NativeReference'):
                result=pp.run_pipeline(tmp,c,{},writer,Mock())
        self.assertLessEqual(result['configurations'],15);self.assertLessEqual(result['laws'],75)
        self.assertEqual(len(new_acquisition),1);self.assertIn('resource_map.json',writer.records)
        self.assertFalse(writer.records['resource_map.json']['scientific_adoption'])

    def test_new_root_independent_controlled_target_off_domain(self):
        import numpy as np
        from trottertracks.algorithm_codesign.parent_experiment.pipeline import root_validation
        from trottertracks.algorithm_codesign.rte_development.numeric import controlled
        X=np.array([[0,1],[1,0]],complex)
        event=dict(word=(),child=0,event_kind='pure_child',ratio=None,rotation_sign=0,phase_i_power=3)
        wanted=controlled(-1j*X)
        class Ref:
            operators=[X]
            def actual(self,gates,realized):return wanted
        got=root_validation(event,[],dict(strict_native_error_upper=F(0)),Ref(),1e-12)
        self.assertTrue(got['independent_target']);self.assertEqual(got['actual_error_diagnostic'],0)
        class Bad(Ref):
            def actual(self,gates,realized):return controlled(X)
        with self.assertRaises(ArithmeticError):root_validation(event,[],dict(strict_native_error_upper=F(0)),Bad(),1e-12)


class AdditionalRuntimeControls(unittest.TestCase):
    def test_manifest_tamper_static_metadata(self):
        from trottertracks.algorithm_codesign.parent_experiment.launch import check_identities
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source.py').write_text('pass\n');h=hashlib.sha256((root/'source.py').read_bytes()).hexdigest()
            (root/'m.json').write_text(json.dumps(dict(focused_tests_passed=True,critical_sha256={'source.py':h})))
            (root/'p.json').write_text('{"sha256":{}}')
            c=dict(execution=dict(source_manifest='m.json',protected_ledger='p.json'),reuse=dict(historical_guard_source=dict(commit='1'*40,path='old.py',bytes=1,sha256=hashlib.sha256(b'x').hexdigest())))
            with patch('subprocess.check_output',return_value=b'x'):
                check_identities(root,c);(root/'source.py').write_text('changed\n')
                with self.assertRaises(PermissionError):check_identities(root,c)
    def test_resource_guard_alarm_conflict(self):
        from trottertracks.algorithm_codesign.rte_development.launch import Guard
        import signal
        c=dict(worker_RSS_MiB=100,parent_RSS_MiB=100,aggregate_RSS_MiB=200,per_key_CPU_seconds=10,total_CPU_seconds=20,virtual_address_MiB=4096,workers=2)
        with patch('signal.getitimer',return_value=(1,1)):
            with self.assertRaises(RuntimeError):Guard(c).__enter__()
    def test_guard_restores_alarm_and_rlimit(self):
        from trottertracks.algorithm_codesign.rte_development.launch import Guard
        import signal,resource
        c=dict(worker_RSS_MiB=100,parent_RSS_MiB=100,aggregate_RSS_MiB=200,per_key_CPU_seconds=10,total_CPU_seconds=20,virtual_address_MiB=4096,workers=2)
        g=Guard(c)
        with patch('signal.getitimer',return_value=(0,0)),patch('signal.getsignal',return_value=signal.SIG_DFL),\
             patch('resource.getrlimit',return_value=(resource.RLIM_INFINITY,resource.RLIM_INFINITY)),patch('resource.setrlimit') as setlimit,\
             patch('signal.signal') as sig,patch('signal.setitimer') as timer,patch.object(g,'check'):
            with self.assertRaises(ValueError):
                with g:raise ValueError('artificial failure')
            self.assertEqual(timer.call_args.args,(signal.ITIMER_REAL,0));self.assertEqual(sig.call_args.args,(signal.SIGALRM,signal.SIG_DFL))
            self.assertEqual(setlimit.call_args.args[1],(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    def test_worker_pending_before_restart_or_synthesis(self):
        from trottertracks.algorithm_codesign.parent_experiment.launch import verify_worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'results';(d/'jobs').mkdir(parents=True)
            (root/'auth.json').write_text('{"science_execution_authorized":false}')
            c=dict(execution=dict(result_directory='results',authorization_path='auth.json'))
            with patch('trottertracks.algorithm_codesign.parent_experiment.launch.read_contract',return_value=c),\
                 patch('trottertracks.algorithm_codesign.parent_experiment.launch.git',return_value='1'*40):
                with self.assertRaises(PermissionError):verify_worker(root,d/'jobs/key_000.json')
    def test_worker_restart_and_marker_mismatch(self):
        from trottertracks.algorithm_codesign.parent_experiment.launch import verify_worker,execution_identity
        from trottertracks.algorithm_codesign.rte_development.evidence import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'results';(d/'jobs').mkdir(parents=True)
            c=dict(execution=dict(result_directory='results',authorization_path='auth.json',source_manifest='m.json',optional_receipt_path='receipt.md'),review=dict(sha256='a'*64),acquisition=dict(tool_identity=dict(sha256='b'*64)))
            a,_=GateOutputControls().authority();(root/'auth.json').write_text(json.dumps(a));(root/'m.json').write_text('{}')
            i=execution_identity(root,c,a,'2'*40);atomic_json(d/'one_shot_consumed.json',i)
            job=dict(**i,parent_pid=123,marker_sha256=digest(d/'one_shot_consumed.json'),key='k',ratio='2/11')
            (d/'candidate_freeze.json').write_text('{}');(d/'inventory.json').write_text(json.dumps(dict(keys=[dict(key='k',ratio='2/11',acquisition='NEW_ONCE')])))
            c['caps']=dict(new_synthesis_keys=18);job.update(candidate_freeze_sha256=digest(d/'candidate_freeze.json'),inventory_sha256=digest(d/'inventory.json'))
            path=d/'jobs/key_000.json';path.write_text(json.dumps(job))
            def fake_git(root,*args):
                if args[:2]==('rev-parse','HEAD'):return '2'*40
                if args[0]=='show':return '1'*40
                if args[0]=='diff':return 'auth.json'
                return ''
            with patch('trottertracks.algorithm_codesign.parent_experiment.launch.read_contract',return_value=c),\
                 patch('trottertracks.algorithm_codesign.parent_experiment.launch.git',side_effect=fake_git),\
                 patch('trottertracks.algorithm_codesign.parent_experiment.launch.check_identities'),patch('os.getppid',return_value=123):
                verify_worker(root,path)
                path.with_suffix('.started.json').write_text('{}')
                with self.assertRaises(FileExistsError):verify_worker(root,path)
                path.with_suffix('.started.json').unlink();job['marker_sha256']='0'*64;path.write_text(json.dumps(job))
                with self.assertRaises(PermissionError):verify_worker(root,path)
    def test_new_worker_calls_once_failure_keeps_raw(self):
        from trottertracks.algorithm_codesign.parent_experiment import worker
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'key_000.json';path.write_text('{}');c=dict(acquisition=dict(runtime_executable_sha256='a'*64,tool_identity={},interval_dps=50,options={}),primitive_error='1/1000000',caps=dict(per_key_CPU_seconds=120,sequence_characters=100))
            job=dict(ratio='2/11');g=Mock();g.__enter__=Mock(return_value=g);g.__exit__=Mock(return_value=False);g.snapshot.return_value={}
            def backend(angle,eps,options,on_synthesis,max_characters):
                on_synthesis();raise ArithmeticError('artificial backend failure')
            with patch.object(worker,'verify_worker',return_value=(c,job,Path(tmp))),patch.object(worker,'digest',return_value='a'*64),\
                 patch.object(worker,'Guard',return_value=g),patch('resource.setrlimit'),\
                 patch.dict('os.environ',{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}),\
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'),\
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.configure'),\
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.synthesize',side_effect=backend) as call:
                self.assertEqual(worker.execute(tmp,path),1);self.assertEqual(call.call_count,1)
            v=json.loads(path.with_suffix('.result.json').read_text());self.assertEqual(v['synthesis_calls'],1)
            self.assertEqual(v['retries'],0);self.assertFalse(v['prefix_scientific_use']);self.assertTrue(path.with_suffix('.backend_call_consumed.json').exists())


class FreshNormalizationControls(unittest.TestCase):
    def test_backend_guard_receipt_rebound_without_second_operator_call(self):
        from trottertracks.algorithm_codesign.parent_experiment.pipeline import normalize_saved,expected
        from trottertracks.algorithm_codesign.parent_allocation_native import checked_entry
        ratio='2/11';eps='1/1000000';seq='HTtSW';angle='atan:'+ratio+':scale:1'
        raw=dict(key=angle+':epsilon:'+eps,angle_key=angle,epsilon=eps,sequence=seq,
            sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),T_count=2,Tdagger_count=1,
            one_qubit_count=4,global_W_count=1,strict_operator_error_upper='1/10000000',error_pass=True)
        c=dict(primitive_error=eps);exp=expected(ratio,tool(),c)
        with patch('trottertracks.algorithm_codesign.parent_allocation_native.strict_primitive_guard',side_effect=AssertionError('second operator call')):
            entry=normalize_saved(raw,exp,old=False,c=c,raw_hashes={});checked_entry(entry)
        self.assertTrue(entry['strict_guard_rechecked']);self.assertEqual(entry['T_count'],2)
        for field,value in [('sequence_sha256','0'*64),('T_count',3),('error_pass',False),('epsilon','1/10000000')]:
            altered=dict(raw,**{field:value})
            with self.assertRaises((PermissionError,ValueError)):normalize_saved(altered,exp,old=False,c=c,raw_hashes={})
    def test_raw_root_identification_not_reduced_word(self):
        from trottertracks.algorithm_codesign.parent_allocation import RootAllocationGenerator
        p=(F(1,7),F(2,7),F(4,7));g=RootAllocationGenerator.ordinary(p,F(1,3),**PREC)
        all_events=list(g.reference_events());root=[e for e in all_events if not e['raw_word']]
        collapsed_rest=[e for e in all_events if e['raw_word'] and not e['word']]
        self.assertEqual(len(root),3);self.assertTrue(collapsed_rest)

class LargeExactFractionControls(unittest.TestCase):
    def test_global_is_large_fraction_decimal_io_and_restore(self):
        import sys
        from trottertracks.algorithm_codesign.parent_allocation import reweight_events,event_statistics
        from trottertracks.algorithm_codesign.parent_experiment.design import integer_io_limit
        # Artificial 273-atom I1 table: no registered scalar, law or forecast.
        e=[dict(coefficient=F(i+1,37401)) for i in range(273)]
        law=reweight_events(e,mode='regularized',costs=[i%17+1 for i in range(273)],regularization=F(2),
                            root_bits=256,probability_bits=160,eta=F(1,10**12),rho=F(1,10**12))
        stat=event_statistics(law);old=sys.get_int_max_str_digits()
        with integer_io_limit(65536):
            text=str(stat['weight_second_moment']);self.assertGreater(len(text),4300)
            self.assertEqual(F(text),stat['weight_second_moment'])
        self.assertEqual(sys.get_int_max_str_digits(),old)

if __name__=='__main__':unittest.main()



