"""Phase 1 preparation: artificial/mock only. No real C/S fit/reanalysis."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import numpy as np
from trottertracks import r_price_phase1 as m
from trottertracks import r_price_phase1_runner as run
from trottertracks import r_price_phase1_science as science
from check_pf_pr_rpe_pipeline import fake_shot


def toy_rows(plan,cell,offset=0):
    c=m.row_for(plan,cell)['candidate'];q=c['q'];R=c['R_outer'];scale=1 if cell['system']=='H4' else 10
    correction=np.array([1,q,q*R])@np.array([[1,2],[2,3],[3,4]])*scale
    rows=[]
    for i in range(cell['n']):
        S=[100+i,110+i];noise=i-(cell['n']-1)/2
        C=(np.array(S)+correction+np.array([noise,2*noise])+offset).tolist()
        rows.append(dict(cell_id=cell['id'],candidate_id=c['id'],index=i,cohort=cell['cohort'],conditional=False,
            C=C,S=S,source={'fixture':'MOCK_ONLY'},event_digest='MOCK_EVENTS',compiler=m.p.PROFILE['compiler']))
    return rows


class MockBackend:
    def __init__(self,plan):self.plan=plan;self.calls=[]
    def acquire_many(self,cells,directory):
        self.calls.append([c['id'] for c in cells])
        return {c['id']:toy_rows(self.plan,c,7 if c['role']=='VALIDATION' else 0) for c in cells}
    def rare_pairs(self,directory):
        self.calls.append('rare');return [dict(unit='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT',mock=True) for _ in range(4)]


def commit(root,path):
    subprocess.run(['git','add',str(path.relative_to(root))],cwd=root,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-c','user.name=Mock','-c','user.email=mock@example.invalid','commit','-m','Freeze synthetic model'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # make_plan reads public task metadata only; saved_cell is not called.
        with patch.object(m,'saved_cell',side_effect=AssertionError('No scientific C/S during preparation')):
            cls.plan=m.make_plan(ROOT,m.BASE)

    def calibration(self):
        return {c['id']:toy_rows(self.plan,c) for c in self.plan['cells'] if c['role']=='CALIBRATION'}

    def test_metadata_roles_R_allocation_seed_budget(self):
        m.validate_plan(self.plan);self.assertEqual(len(self.plan['cells']),8)
        self.assertEqual(sum(2*c['n'] for c in self.plan['cells'] if not c['saved']),64)
        self.assertEqual(len(self.plan['rare_groups'])*2*2,16)
        seeds=[v for c in self.plan['cells'] if not c['saved'] for vs in c['seeds'] for v in vs]+[v for g in self.plan['rare_groups'] for v in g['seeds']]
        self.assertEqual(len(seeds),len(set(seeds)))
        for c in self.plan['cells']:
            row=m.row_for(self.plan,c)
            self.assertEqual(row['structure']['event_count'],row['q_PF']*row['candidate']['R_outer'])
            self.assertEqual(c['request']['law_binding']['R_outer'],row['candidate']['R_outer'])
        bad=deepcopy(self.plan);bad['cells'][-1]['role']='CALIBRATION'
        with self.assertRaises(ValueError):m.validate_plan(bad)

    def test_A_B_candidate_identity_law_compatibility(self):
        a=m.fixed_module(ROOT,m.A,'src/trottertracks/pr_r_dependent_n.py','trottertracks._phase1_test_A')
        b=m.fixed_module(ROOT,m.B,'src/trottertracks/rpe_r_selection.py','trottertracks._phase1_test_B')
        for system,mapping in [('H4',{3:8,4:8}),('H6',{4:32,5:32})]:
            task=a.with_round_R(a.saved_task(ROOT,system),mapping,{'fixture':'MOCK_ONLY'})['task']
            for c in self.plan['cells']:
                if c['system']!=system or c['saved']:continue
                row=next(r for r in task['candidates'] if r['candidate']['id']==c['candidate_id'])
                self.assertEqual(m.p.identity(task,row),c['request']['identity'])
                self.assertEqual(b.request_context(self.plan['menus'][system],m.row_for(self.plan,c)),c['request'])

    def test_paired_fit_independent_systems_rank_and_no_validation(self):
        data=self.calibration();snapshot=deepcopy(data)
        frozen=m.calibrate(ROOT,self.plan,data)
        for s,k in [('H4',1),('H6',10)]:
            np.testing.assert_allclose(frozen['models'][s]['coefficients'],np.array([[1,2],[2,3],[3,4]])*k,rtol=1e-10,atol=1e-9)
            self.assertEqual(frozen['models'][s]['rank'],3);self.assertEqual(frozen['models'][s]['cell_level_lack_of_fit_DOF'],0)
            self.assertEqual(np.asarray(frozen['models'][s]['nominal_coefficient_covariance']).shape,(6,6))
            self.assertIsNone(frozen['models'][s]['R_q_systematic'])
        self.assertEqual(data,snapshot)
        invalid=deepcopy(data);c=next(c for c in self.plan['cells'] if c['role']=='VALIDATION');invalid[c['id']]=toy_rows(self.plan,c)
        with self.assertRaisesRegex(ValueError,'NO_HOLDOUT'):m.calibrate(ROOT,self.plan,invalid)
        invalid=deepcopy(data);invalid[next(iter(invalid))][0]['conditional']=True
        with self.assertRaisesRegex(ValueError,'CONDITIONAL'):m.calibrate(ROOT,self.plan,invalid)
        helper=m.helpers(str(ROOT));old=helper['fit']
        with patch.dict(helper,{'fit':Mock(side_effect=ValueError('UNIDENTIFIABLE_MODEL'))}):
            with self.assertRaisesRegex(ValueError,'UNIDENTIFIABLE'):m.calibrate(ROOT,self.plan,data)
        self.assertIs(helper['fit'],old)

    def test_validation_metrics_covariance_and_no_refit(self):
        frozen=m.calibrate(ROOT,self.plan,self.calibration());before=deepcopy(frozen)
        for cell in [c for c in self.plan['cells'] if c['role']=='VALIDATION']:
            with patch.dict(m.helpers(str(ROOT)),{'fit':Mock(side_effect=AssertionError('No validation fit'))}):
                ev,rows=m.evaluate_cell(ROOT,self.plan,cell,toy_rows(self.plan,cell,7),frozen)
            self.assertAlmostEqual(ev['summaries'][0]['metrics']['bias'],7,places=7)
            self.assertGreater(ev['summaries'][0]['metrics']['RMSE'],7)
            self.assertIsNone(ev['total_CI']);self.assertFalse(ev['qualified_for_candidate'])
            self.assertEqual(len(rows),16);self.assertEqual(len(ev['paired_residual_covariance']),2)
        self.assertEqual(frozen,before)
        foreign=deepcopy(cell);foreign['id']='OUTSIDE_PHASE1'
        with self.assertRaisesRegex(ValueError,'NO_EXTRAPOLATION'):m.evaluate_cell(ROOT,self.plan,foreign,toy_rows(self.plan,cell),frozen)

    def test_B_price_records_full_domain_qualified_policy(self):
        frozen=m.calibrate(ROOT,self.plan,self.calibration());b=m.fixed_module(ROOT,m.B,'src/trottertracks/rpe_r_selection.py','trottertracks._phase1_prices_B')
        cell=next(c for c in self.plan['cells'] if c['role']=='VALIDATION');ev,_=m.evaluate_cell(ROOT,self.plan,cell,toy_rows(self.plan,cell),frozen)
        prices=m.price_records(self.plan,cell,ev,frozen,{'fixture':'MOCK_ONLY'});menu=self.plan['menus'][cell['system']]
        row=m.row_for(self.plan,cell);shot=fake_shot(menu,row);shot['law_binding']=deepcopy(cell['request']['law_binding'])
        select={(row['candidate']['id'],a):'DIRECT:'+cell['cohort'] for a in m.p.AXES}
        with patch.object(b,'evaluate_menu',side_effect=AssertionError('No G')):
            result=b.coverage(menu,[shot],prices,cohort_selection=select,price_policy='finite_cohort_scenario')
        self.assertTrue(next(r for r in result['rows'] if r['candidate_id']==cell['candidate_id'])['eligible'])
        for v in prices:
            self.assertEqual(v['law_binding'],cell['request']['law_binding']);self.assertFalse(v['normalization_is_price_multiplier'])
            self.assertIsNone(v['uncertainty']['joint_covariance'])
            if v['model']:
                self.assertEqual(v['model']['domain'],row['candidate']);self.assertFalse(v['model']['qualified_for_candidate'])
        prices[0]['law_binding']['R_outer']+=1
        with self.assertRaisesRegex(ValueError,'law or R'):b.coverage(menu,[shot],prices,cohort_selection=select,price_policy='finite_cohort_scenario')

    def test_one_shot_stage_order_git_freeze_and_final_STOP(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';root.mkdir();subprocess.run(['git','init','-q',str(root)],check=True)
            public=root/'public';private=Path(temp)/'private';backend=MockBackend(self.plan)
            helper=m.helpers(str(ROOT))
            with (patch.object(m,'helpers',return_value=helper),patch.object(run,'check_source'),patch.object(m,'saved_cell',side_effect=lambda root,plan,c,read=None:toy_rows(plan,c)),
                  patch.object(m.p,'evaluate',side_effect=AssertionError('No G'))):
                with self.assertRaisesRegex(ValueError,'EXPLICIT'):run.execute_stage(root,self.plan,public,private,'calibrate',backend=backend)
                self.assertEqual(backend.calls,[])
                run.execute_stage(root,self.plan,public,private,'calibrate',execute=True,backend=backend)
                self.assertTrue(all('q16_R8' not in c and 'q2_R32' not in c for c in backend.calls[0]))
                with self.assertRaisesRegex(ValueError,'COMMITTED_FREEZE'):run.execute_stage(root,self.plan,public,private,'validate',execute=True,backend=backend)
                with self.assertRaises(subprocess.CalledProcessError):run.execute_stage(root,self.plan,public,private,'validate',execute=True,backend=backend,frozen_commit='0'*40)
                self.assertEqual(len(backend.calls),1)
                frozen_commit=commit(root,public/'FROZEN_MODELS.json')
                original=(public/'FROZEN_MODELS.json').read_bytes()
                run.execute_stage(root,self.plan,public,private,'validate',execute=True,backend=backend,frozen_commit=frozen_commit)
                run.execute_stage(root,self.plan,public,private,'rare',execute=True,backend=backend,frozen_commit=frozen_commit)
                self.assertEqual(original,(public/'FROZEN_MODELS.json').read_bytes())
                self.assertEqual(run.read(public/'STOP.json')['status'],'PHASE1_COMPLETE_MANDATORY_STOP')
                with self.assertRaisesRegex(ValueError,'STOP'):run.execute_stage(root,self.plan,public,private,'rare',execute=True,backend=backend,frozen_commit=frozen_commit)
                self.assertEqual(len(backend.calls),3)
                self.assertIsNone(run.read(public/'VALIDATION.json')['full_rpe_resources'])
                self.assertTrue((public/'PATH_RESIDUALS.csv').exists())

    def test_failure_blocks_retry_and_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';root.mkdir();out=root/'public';private=Path(temp)/'raw';backend=MockBackend(self.plan)
            backend.acquire_many=Mock(side_effect=ValueError('MOCK_COMPILE_FAILURE'))
            with patch.object(run,'check_source'),patch.object(m,'saved_cell',side_effect=lambda root,plan,c,read=None:toy_rows(plan,c)):
                with self.assertRaisesRegex(ValueError,'MOCK_COMPILE'):run.execute_stage(root,self.plan,out,private,'calibrate',execute=True,backend=backend)
                stop=(out/'STOP.json').read_bytes()
                with self.assertRaises(ValueError):run.execute_stage(root,self.plan,out,private,'validate',execute=True,backend=backend,frozen_commit='MOCK')
                self.assertEqual(stop,(out/'STOP.json').read_bytes());backend.acquire_many.assert_called_once()
                with self.assertRaises(FileExistsError):run.execute_stage(root,self.plan,out,private,'calibrate',execute=True,backend=backend)

    def test_force_legal_pair_is_separate_and_preserves_order_and_sign(self):
        @dataclass(frozen=True)
        class Occ:events:tuple
        @dataclass(frozen=True)
        class Req:step_time:float;preparation:object;rte_distribution:object;rte_occurrence:object;rte_basis_plan:object=None
        components=(SimpleNamespace(component_id='a',is_identity=False,basis_hash='A'),SimpleNamespace(component_id='b',is_identity=False,basis_hash='B'))
        prep=SimpleNamespace(rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(components=components)))
        def make(indices,components,dist,order_index):return SimpleNamespace(rotation_component_id=components[indices[0]].component_id,taylor_order=2*order_index,applications=tuple(indices[1:])+tuple(indices[:1]))
        original=(Req(.1,prep,'POS',Occ((make((0,),components,None,0),)*2)),Req(-.2,prep,'NEG',Occ((make((0,),components,None,0),)*2)))
        base,rare=science.force_pair(original,1,'H6',make)
        self.assertEqual([r.step_time for r in rare],[.1,-.2]);self.assertEqual(rare[1].rte_occurrence.events[0].taylor_order,2)
        self.assertEqual(len(rare[1].rte_occurrence.events[0].applications),3)
        self.assertTrue(all(e.taylor_order==0 for r in base for e in r.rte_occurrence.events))
        self.assertEqual(original[1].rte_occurrence.events[0].taylor_order,0)

    def test_saved_join_loaders_with_artificial_JSON_only(self):
        cell=next(c for c in self.plan['cells'] if c['saved'] and c['system']=='H4' and m.row_for(self.plan,c)['q_PF']==8)
        rows=toy_rows(self.plan,cell);doc=dict(scientific_binding=self.plan['menus']['H4']['binding'],compiler_hash=m.p.PROFILE['compiler'],rows=[],private_trial_bindings=[])
        for r in rows:
            doc['private_trial_bindings'].append({'index':r['index'],'events_sha256':'MOCK_EVENT'})
            for j,a in enumerate(m.AXES):doc['rows'].append(dict(candidate_id=cell['candidate_id'],cohort=cell['cohort'],path_index=r['index'],axis=a,compiled_rz=r['C'][j],prediction={'structural_rz':r['S'][j]}))
        loaded=m.saved_cell(ROOT,self.plan,cell,read=lambda ref:deepcopy(doc))
        self.assertEqual([r['C'] for r in loaded],[r['C'] for r in rows])
        doc['rows'][-1]['axis']='cosine'
        with self.assertRaisesRegex(ValueError,'PAIRED'):m.saved_cell(ROOT,self.plan,cell,read=lambda ref:deepcopy(doc))

    def test_scientific_import_and_cli_default_do_not_execute(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_r_price_phase1.py'),'calibrate','--plan','DOES_NOT_EXIST'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('separately authorized',result.stderr)
        for path in ['src/trottertracks/pf_pr_rpe_pipeline.py','src/trottertracks/h4_df10_price_provider.py','src/trottertracks/h6_df19_price_provider.py','scripts/h46_long_time_native.py']:
            self.assertEqual((ROOT/path).read_bytes(),m.blob(ROOT,m.BASE,path))


    def test_other_saved_C_S_formats_use_exact_paired_artificial_rows(self):
        for cell in [c for c in self.plan['cells'] if c['saved'] and not (c['system']=='H4' and m.row_for(self.plan,c)['q_PF']==8)]:
            rows=toy_rows(self.plan,cell);binding=self.plan['menus'][cell['system']]['binding'];c=m.row_for(self.plan,cell)['candidate'];docs={}
            if cell['system']=='H4':
                pred={'predictions':[]};result={'trials':[]};prov=dict(scientific_binding=binding,compiler=m.p.PROFILE['compiler'],cohort=cell['cohort'],trial_bindings=[])
                for r in rows:
                    i=r['index'];dig='MOCK'+str(i)
                    prov['trial_bindings'].append(dict(candidate_id=c['id'],path_index=i,canonical_events_sha256=dig))
                    for j,a in enumerate(m.AXES):pred['predictions'].append(dict(candidate_id=c['id'],path_index=i,axis=a,events_sha256=dig,structural_rz=r['S'][j],Mq=999999))
                    result['trials'].append(dict(candidate_id=c['id'],index=i,prices={a:{'rz_count':r['C'][j]} for j,a in enumerate(m.AXES)},wrapper_bindings=[dict(axis=a,compiler=m.p.PROFILE['compiler']) for a in m.AXES]))
                docs={'PREDICTIONS.json':pred,'RESULT.json':result,'PROVENANCE.json':prov}
            elif c['q']==1:
                doc=dict(scientific_binding=binding,compiler_hash=m.p.PROFILE['compiler'],rows=[],private_trial_bindings=[])
                for r in rows:
                    doc['private_trial_bindings'].append(dict(family='PR_B',index=r['index'],events_sha256='MOCK'))
                    for j,a in enumerate(m.AXES):doc['rows'].append(dict(candidate_id=c['id'],cohort=cell['cohort'],path_index=r['index'],axis=a,compiled_rz=r['C'][j],prediction={'structural_rz':r['S'][j]}))
                docs={'DEVELOPMENT.json':doc}
            else:
                doc=dict(compiler_hash=m.p.PROFILE['compiler'],rows=[dict(status='PASS_PRICE_ACQUIRED',candidate=c,cohort=cell['cohort'],index=r['index'],prices={a:{'rz_count':r['C'][j]} for j,a in enumerate(m.AXES)},predictions={a:{'structural_rz':r['S'][j]} for j,a in enumerate(m.AXES)}) for r in rows]);docs={'Q2_DIRECT.json':doc}
            reader=lambda ref:deepcopy(docs[Path(ref['path']).name])
            loaded=m.saved_cell(ROOT,self.plan,cell,read=reader)
            self.assertEqual([r['C'] for r in loaded],[r['C'] for r in rows]);self.assertEqual([r['S'] for r in loaded],[r['S'] for r in rows])
            if cell['system']=='H4':
                pred['predictions'][0]['events_sha256']='FOREIGN'
                with self.assertRaisesRegex(ValueError,'EVENT_C_S_JOIN'):m.saved_cell(ROOT,self.plan,cell,read=reader)

    def test_structural_word_reuses_original_kernel_on_artificial_events(self):
        event=dict(phase={'real':1.,'imag':0.},taylor_order=0,rotation_angle=.01,
            application_sequence=[dict(role='rotation',coefficient_sign=1,application_index=0,is_identity=True)])
        for system in ('H4','H6'):
            module=m.fixed_module(ROOT,m.BASE if system=='H4' else m.C_HELPERS,
                'src/trottertracks/'+('h4_df10_price_v1.py' if system=='H4' else 'h6_df19_price_v1.py'),
                'trottertracks._synthetic_original_'+system)
            q,T=(8,.8) if system=='H4' else (1,1.6)
            allocation=[4] if system=='H4' else list(m.p._schedule().allocation(m.p._schedule().weights('4th(m5_best)'),16))
            events=[[deepcopy(event) for _ in range(r)] for _ in range(q) for r in allocation]
            blocks=[[] for _ in allocation]
            for axis in m.AXES:
                old=(module.predict([],q,.3,T,axis,events) if system=='H4' else module.predict(blocks,q,.3,T,axis,events,allocation))
                new=science.structural(module.Accounting,blocks,{'system':system,'q':q,'T':T},.3,axis,events)
                self.assertEqual(new['structural_rz'],old['structural_rz'])

    def test_science_backend_mock_compile_cache_and_S_checkpoint(self):
        from contextlib import contextmanager
        cell=next(c for c in self.plan['cells'] if not c['saved'] and c['system']=='H4' and c['role']=='CALIBRATION')
        row=m.row_for(self.plan,cell);c=row['candidate'];reqs=[]
        event=SimpleNamespace(to_dict=lambda:{'fixture':'MOCK_EVENT'})
        for _ in range(c['q']):reqs.append(SimpleNamespace(step_time=.1,rte_occurrence=SimpleNamespace(events=[event]*c['R_outer'])))
        @contextmanager
        def fp_context():yield SimpleNamespace(fingerprint=lambda w:'MOCK_'+w.axis)
        with tempfile.TemporaryDirectory() as temp:
            plan=deepcopy(self.plan);plan['runtime']['old_cache']=str(Path(temp)/'cache');cache=Path(plan['runtime']['old_cache']);cache.mkdir()
            (cache/'MOCK_cosine.json').write_text(json.dumps(dict(fingerprint='MOCK_cosine',compiler=m.p.PROFILE['compiler'],metrics={'rz_count':7})))
            original=(cache/'MOCK_cosine.json').read_bytes();directory=Path(temp)/'raw'
            ctx=SimpleNamespace(prep='MOCK_PREPARATION',Accounting=object,blocks=[[]],scalar=0.,fingerprint_context=fp_context,
                h={'compose':Mock(return_value=SimpleNamespace(circuit=SimpleNamespace(data=[1,2])))},
                native=SimpleNamespace(build_native_hadamard_wrapper=Mock(side_effect=lambda evolution,axis,**kw:SimpleNamespace(axis=axis))),CompilerSettings=Mock(return_value='MOCK_SETTINGS'))
            def compile_mock(*args,**kwargs):
                self.assertTrue((directory/'STRUCTURAL.json').exists())
                return SimpleNamespace(compiler_settings_hash=m.p.PROFILE['compiler'],**{k:9 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')})
            ctx.compile=Mock(side_effect=compile_mock);backend=science.Backend(ROOT,plan)
            with patch.object(backend,'context',return_value=ctx),patch.object(science,'structural',side_effect=lambda A,b,c,s,a,e:{'structural_rz':70 if a=='cosine' else 71}):
                result=backend.price_requests(cell,0,directory,reqs,[],[],conditional=False)
            self.assertEqual(result['C'],[7,9]);self.assertEqual(result['S'],[70,71]);ctx.compile.assert_called_once()
            self.assertEqual(result['new_compiles'],1);self.assertEqual(original,(cache/'MOCK_cosine.json').read_bytes())
            cached=json.loads(original);cached['compiler']='FOREIGN';(cache/'MOCK_cosine.json').write_text(json.dumps(cached));ctx.compile.reset_mock()
            with patch.object(backend,'context',return_value=ctx),patch.object(science,'structural',return_value={'structural_rz':70}):
                with self.assertRaisesRegex(ValueError,'CACHE_BINDING'):backend.price_requests(cell,0,Path(temp)/'bad',reqs,[],[],conditional=False)
            ctx.compile.assert_not_called()

    def test_frozen_model_and_source_changes_cannot_silently_continue(self):
        invalid=deepcopy(self.plan);invalid['menus']['H6']['profile']['compiler']='OTHER'
        with self.assertRaisesRegex(ValueError,'TASK_CHANGED'):m.validate_plan(invalid)
        invalid=deepcopy(self.plan);invalid['source_files']=[]
        with self.assertRaisesRegex(ValueError,'SOURCE'):m.validate_plan(invalid)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'FROZEN_MODELS.json').write_text('{}')
            with self.assertRaises(ValueError):run.committed_frozen(ROOT,root,'0'*40)


if __name__=='__main__':unittest.main(verbosity=2)
