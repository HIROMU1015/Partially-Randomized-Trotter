"""Synthetic-only confirmation checks. No scientific/native packages imported."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h46_confirmation_contract as contract
import h46_confirmation_collect as collector
import h46_confirmation_native as adapter
import launch_h46_confirmation as launcher


def plan_fixture():
    payload={'study_id':'HIGHER_PF_PURPOSE_A_REPLICATION','stage':'CONFIRMATION','SOURCE':'synthetic',
             'groups':list(contract.GROUPS),'n_per_candidate':contract.PROPOSED_N,'input_sha256':contract.INPUT_SHA,
             'selection_evidence_sha256':'synthetic_selection'}
    p={'phase':contract.PHASE,'study_id':payload['study_id'],'SOURCE':'synthetic','status':'DRAFT_UNAPPROVED',
       'sample_size_formally_adopted':False,'groups':list(contract.GROUPS),'n_per_candidate':contract.PROPOSED_N.copy(),
       'comparison_pairs':[list(v) for v in contract.PAIRS],'compiler_settings_hash':contract.PROFILE,'numeric_tolerance':1e-9,
       'fixed_science':{'T':.8,'K':4,'RTE_steps_per_outer':32,'epsilon':.001,'alpha_axis':.025,'allowance':1e-9,'log_B_margin':1e-12,'P':0,'control':'ordinary'},
       'purpose':'INDEPENDENT_POINT_ORDER_REPLICATION_NOT_POPULATION_GUARANTEE',
       'analysis_policy':{'primary':'EXACT_MEAN_DIFFERENCE_SIGN','secondary':'NORMAL_BONFERRONI_APPROX_DIAGNOSTIC','MC_family_alpha':.05,'family_size':2,'population_guarantee':False,'pool_exploration':False},
       'trajectories':103,'sample_axes':206,'extra_scientific_compiles':0,'input_sha256':contract.INPUT_SHA,
       'selection_evidence_sha256':payload['selection_evidence_sha256'],'seed_scope_payload':payload,
       'seed_scope_digest':contract.digest(payload),'retry':False,'mandatory_STOP':True}
    p['seed_namespace']='H6_HIGHER_PF_C_CONFIRMATION:synthetic:'+p['seed_scope_digest']
    p['tasks']=[contract.task_id(p,g,i) for g in contract.GROUPS for i in range(contract.PROPOSED_N[g['id']])]
    return p


def registry_fixture(p):
    tasks={};k=collector.kernel()
    for g in contract.GROUPS:
        for sample in range(contract.PROPOSED_N[g['id']]):
            tasks[contract.task_id(p,g,sample)]={f'{o}:{piece}':k.seed_for(p['seed_namespace'],'CONFIRMATION',g['id'],sample,o,piece,contract.INPUT_SHA)
                for o in range(g['q']) for piece in range(1 if g['label']=='2nd' else 3)}
    return {'stage':'CONFIRMATION','seed_scope_digest':p['seed_scope_digest'],'namespace':p['seed_namespace'],'tasks':tasks}


MOCK_CONFIG=None
def mock_initialize(path,stop):
    global MOCK_CONFIG
    MOCK_CONFIG=json.loads(Path(path).read_text())


def mock_worker(group,sample):
    """Actual spawn-process test backend; never exposed through production CLI."""
    c=MOCK_CONFIG;out=Path(c['output']);gid=group['id']
    if sample>0 and not (out/(gid+'__000_COMPLETE.json')).exists():raise ValueError('SAMPLE0_GATE_BYPASSED')
    if c.get('fail_sample0') and gid=='H6_Y4_q2' and sample==0:raise ValueError('SYNTHETIC_CONTROL_FAIL')
    for axis in ('cosine','sine'):
        contract.write(Path(c['call_ledger'])/(gid+'__%02d__'%sample+axis+'.json'),{'synthetic':True})
    return {'status':'PASS','stage':'CONFIRMATION','group':gid,'sample':sample}


def synthetic_campaign(root):
    p=plan_fixture();c={'SOURCE':'synthetic','provider_SOURCE':'provider','plan':p,'plan_sha256':'plan',
        'seed_registry_sha256':'registry','output':str(root/'out'),'call_ledger':str(root/'ledger'),'group_binding':{},
        'sealed_files':[{'path':'/synthetic/provider.py'}]}
    Path(c['output']).mkdir();Path(c['call_ledger']).mkdir()
    for g in contract.GROUPS:
        gid=g['id'];c['group_binding'][gid]={'preparation_hash':'prep','shots':{'axes':{'real':{'shots':2},'imag':{'shots':3}}}}
        for sample in range(contract.PROPOSED_N[gid]):
            directory=Path(c['output'])/contract.task_id(p,g,sample);directory.mkdir(parents=True)
            events=[[{'taylor_order':2 if sample==1 else 0}]];ed=hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest()
            contract.write(directory/'events_PRIVATE.json',{'events':events,'conditional':False,'seed_namespace':p['seed_namespace']})
            records=[]
            for axis in ('cosine','sine'):
                base=100 if gid=='H6_Y4_q2' else 120
                record={'sample':sample,'axis':axis,'event_digest':ed,'fingerprint':'f'+str(sample),
                        'compiler_settings_hash':contract.PROFILE,'metrics':{'rz_count':base+sample,'cx_count':sample}}
                records.append(record);contract.write(directory/('cost_'+axis+'.json'),record)
                contract.write(Path(c['call_ledger'])/(gid+'__%02d__'%sample+axis+'.json'),
                    {'state':'ENTERED_NO_REFUND','group':gid,'sample':sample,'axis':axis,'event_digest':ed,'fingerprint':record['fingerprint'],'SOURCE':'synthetic'})
            raw={'status':'ACQUIRED_VALIDATED_NO_ACCOUNTING','SOURCE':'synthetic','provider_SOURCE':'provider',
                 'plan_sha256':'plan','seed_registry_sha256':'registry','seed_namespace':p['seed_namespace'],'group':g,'sample':sample,
                 'preparation_hash':'prep','input_sha256':contract.INPUT_SHA,'records':records,
                 'validation_checks':[{'check':str(i),'error':0.} for i in range(12 if sample==0 else 5)],'forced_events_enter_cost_sample':False,
                 'imported_provider_modules':{'trotterlib.synthetic':'/synthetic/provider.py'}}
            contract.write(directory/'result.json',raw)
            contract.write(directory/'confirmation.json',{'status':'CONFIRMATION_COST_VALIDATED','stage':'CONFIRMATION',
                'study_id':p['study_id'],'SOURCE':'synthetic','plan_sha256':'plan','seed_scope_digest':p['seed_scope_digest'],
                'group':g,'sample':sample,'raw_native_result_sha256':contract.sha(directory/'result.json')})
    return c


class ConfirmationTests(unittest.TestCase):
    def test_full_authorization_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir()
            kernel=root/'scripts/h46_acquire_native.py';kernel.write_text('synthetic kernel')
            snapshot=root/'snapshot';snapshot.write_text('synthetic input')
            with patch.object(contract,'INPUT_SHA',contract.sha(snapshot)),patch.object(contract,'KERNEL_SHA',contract.sha(kernel)),patch.object(contract,'ROOT',root),\
                 patch.object(contract.subprocess,'check_output',return_value='synthetic\n'),patch.object(contract.sys,'version_info',(3,12,3)),\
                 patch('importlib.metadata.version',side_effect=lambda name:{'qiskit':'1.3.0','numpy':'1.26.4','scipy':'1.14.1','openfermion':'1.6.1'}[name]):
                p=plan_fixture();p.update(status='SEALED_APPROVED',sample_size_formally_adopted=True)
                r=registry_fixture(p);registry=root/'registry';contract.write(registry,r)
                exclusion=root/'exclusion';contract.write(exclusion,{'values':[17]})
                inputs={'H6_snapshot.npz':{'path':str(snapshot),'sha256':contract.sha(snapshot)}}
                binding={'synthetic':True}
                sealed=[{'path':str(v),'sha256':contract.sha(v)} for v in (kernel,snapshot,registry,exclusion)]
                p.update(closure_digest=contract.digest(sealed),input_bindings_digest=contract.digest(inputs),
                         group_binding_digest=contract.digest(binding),provider_SOURCE='c31ef424175c02379eafebe8fc6c7548c1d9aabe',seed_registry_sha256=contract.sha(registry))
                planpath=root/'plan';contract.write(planpath,p)
                grant={'valid':True,'phase':contract.PHASE,'SOURCE':'synthetic','plan_sha256':contract.sha(planpath),
                       'approval_reference':'ARTIFICIAL_TEST_ONLY','grant_id':'ARTIFICIAL_ONLY',
                       'maximum_trajectories':103,'maximum_cost_axes':206,'maximum_extra_compiles':0,'retry':False,
                       'output':str(root/'out'),'call_ledger':str(root/'ledger'),'grant_marker':str(root/'marker')}
                c={'authorized':True,'plan':p,'grant':grant,'SOURCE':'synthetic','provider_SOURCE':p['provider_SOURCE'],
                   'python':sys.executable,'source_root':str(root),'plan_path':str(planpath),'plan_sha256':contract.sha(planpath),
                   'sealed_files':sealed,'inputs':inputs,'group_binding':binding,'seed_registry_path':str(registry),
                   'seed_registry_sha256':contract.sha(registry),'excluded_seed_path':str(exclusion),
                   **{key:grant[key] for key in ('output','call_ledger','grant_marker')}}
                self.assertEqual(contract.authorize(c)['fresh_values'],914)
                for key,value in [('SOURCE','wrong'),('plan_sha256','wrong'),('group_binding',{}),('sealed_files',[]),('output',str(root/'other'))]:
                    bad=copy.deepcopy(c);bad[key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):contract.authorize(bad)
                kernel.write_text('tampered')
                with self.assertRaisesRegex(ValueError,'HASH'):contract.authorize(c)

    def test_scope_and_high_sample_registration(self):
        p=plan_fixture();self.assertEqual(len(contract.scope(p)),103)
        self.assertEqual(contract.task_id(p,contract.GROUPS[1],44),'H6_Y4_q2/sample_044')
        for value in (-1,45,True):
            with self.assertRaises(ValueError):contract.task_id(p,contract.GROUPS[1],value)

    def test_scope_mutations(self):
        for key,value in [('phase','A_COST'),('sample_axes',208),('retry',True),('mandatory_STOP',False),
                          ('compiler_settings_hash','wrong'),('input_sha256','wrong'),('comparison_pairs',[]),
                          ('purpose','POPULATION_GUARANTEE'),('numeric_tolerance',1e-8)]:
            with self.subTest(key=key):
                p=plan_fixture();p[key]=value
                with self.assertRaises(ValueError):contract.scope(p)

    def test_changed_n_and_extra_task_rejected(self):
        p=plan_fixture();p['n_per_candidate']['H6_Y4_q2']=46
        with self.assertRaises(ValueError):contract.scope(p)
        p=plan_fixture();p['tasks'].append('extra')
        with self.assertRaises(ValueError):contract.scope(p)

    def test_seed_derivation_and_exclusion(self):
        p=plan_fixture();r=registry_fixture(p);audit=contract.registry_audit(p,r,{17})
        self.assertEqual(audit['fresh_values'],914)
        value=next(iter(next(iter(r['tasks'].values())).values()))
        with self.assertRaisesRegex(ValueError,'COLLISION'):contract.registry_audit(p,r,{value})

    def test_seed_stage_namespace_and_derivation_refused(self):
        p=plan_fixture();base=registry_fixture(p)
        for key,value in [('stage','EXPLORATION'),('namespace','A_COST'),('seed_scope_digest','wrong')]:
            r=copy.deepcopy(base);r[key]=value
            with self.assertRaises(ValueError):contract.registry_audit(p,r,set())
        r=copy.deepcopy(base);next(iter(r['tasks'].values()))['0:0']+=1
        with self.assertRaisesRegex(ValueError,'DERIVATION'):contract.registry_audit(p,r,set())

    def test_draft_refused_before_reads(self):
        with patch.object(contract.Path,'read_bytes',side_effect=AssertionError('read before approval')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT'):contract.authorize({'plan':plan_fixture(),'grant':{'valid':False}})

    def test_cli_refusal_creates_no_markers_or_native_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'draft.json';path.write_text(json.dumps({'authorized':False,'plan':plan_fixture(),'grant':{'valid':False}}))
            cmd=[sys.executable,'-B',str(ROOT/'scripts/launch_h46_confirmation.py'),'--execute','--config',str(path)]
            r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT',r.stdout)
            self.assertEqual(list(root.iterdir()),[path])
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/launch_h46_confirmation.py')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0);self.assertIn('SCOPE_ONLY',r.stdout)

    def test_frozen_scientific_kernel_unchanged(self):
        self.assertEqual(contract.sha(ROOT/'scripts/h46_acquire_native.py'),contract.KERNEL_SHA)

    def test_process_local_registration_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.json';p.write_text(json.dumps({'plan':plan_fixture()}))
            old=adapter.frozen.task_id
            try:
                with patch.object(adapter,'authorize'),patch.object(adapter.frozen,'initialize'):
                    adapter.initialize(str(p),None)
                self.assertEqual(adapter.frozen.task_id(contract.GROUPS[2],44),'H6_Y4_q4/sample_044')
            finally:adapter.frozen.task_id=old

    def test_spawn_parallel_sample0_gate_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir();(root/'ledger').mkdir()
            c={'output':str(root/'out'),'call_ledger':str(root/'ledger'),'resources':{'group_estimated_RSS':{g['id']:1 for g in contract.GROUPS}}}
            p=root/'config.json';p.write_text(json.dumps(c))
            w=launcher.run_pool(c,str(p),3,mock_initialize,mock_worker)
            self.assertEqual(w['status'],'COMPLETE_NATIVE_CONFIRMATION');self.assertEqual(w['completed_trajectories'],103)
            self.assertEqual(w['compile_entries'],206);self.assertTrue(w['all_workers_reaped'])
            self.assertTrue(all(p['exit_code']==0 for p in w['workers']))

    def test_spawn_control_failure_stops_and_reaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'out').mkdir();(root/'ledger').mkdir()
            c={'fail_sample0':True,'output':str(root/'out'),'call_ledger':str(root/'ledger'),
               'resources':{'group_estimated_RSS':{g['id']:1 for g in contract.GROUPS}}}
            p=root/'config.json';p.write_text(json.dumps(c))
            w=launcher.run_pool(c,str(p),3,mock_initialize,mock_worker)
            self.assertEqual(w['status'],'STOP');self.assertTrue(w['all_workers_reaped']);self.assertFalse(w['retry'])
            self.assertFalse((root/'ledger/H6_Y4_q2__01__cosine.json').exists())

    def test_collector_full_synthetic_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=synthetic_campaign(Path(tmp));r=collector.collect(c)
            self.assertEqual(r['control_checks'],536);self.assertEqual(r['compiled_axes'],206)
            self.assertFalse(r['exploration_pooled']);self.assertFalse(r['population_ranking_certified'])
            self.assertEqual(r['candidates']['H6_Y4_q2']['summaries']['RZ']['mean'],610)
            self.assertEqual(r['point_order_status'],'BOTH_REPRODUCED')

    def test_collector_rejects_incomplete_pair_and_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=synthetic_campaign(Path(tmp));d=Path(c['output'])/'H6_Y4_q2/sample_044'
            e=json.loads((d/'confirmation.json').read_text());e['stage']='EXPLORATION';(d/'confirmation.json').write_text(json.dumps(e))
            with self.assertRaisesRegex(ValueError,'ENVELOPE'):collector.collect(c)

    def test_collector_rejects_extra_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=synthetic_campaign(Path(tmp));contract.write(Path(c['call_ledger'])/'extra.json',{})
            with self.assertRaisesRegex(ValueError,'BUDGET'):collector.collect(c)

    def test_exact_sign_tie_zero_variance_and_pooling(self):
        k=collector.kernel();s={g:k.sample_summary([100]*n) for g,n in contract.PROPOSED_N.items()}
        rows=collector.compare(s);self.assertTrue(all(r['point_order']=='TIE' for r in rows))
        self.assertTrue(all(r['approx_nominal_interval_excludes_zero_in_expected_direction'] is None for r in rows))
        s['H6_S2_q8']=k.sample_summary([99]*13)
        self.assertEqual(collector.compare(s)[0]['point_order'],'NOT_REPRODUCED')
        s['H6_Y4_q2']=k.sample_summary([100]*53)
        with self.assertRaisesRegex(ValueError,'POOLING'):collector.compare(s)

    def test_axis_reservation_never_refunded(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter.frozen.reserve_axis(tmp,contract.GROUPS[0],12,'cosine',{'synthetic':True})
            with self.assertRaises(FileExistsError):adapter.frozen.reserve_axis(tmp,contract.GROUPS[0],12,'cosine',{'synthetic':True})


if __name__=='__main__':unittest.main()
