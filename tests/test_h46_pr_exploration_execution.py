"""Artificial caching, fresh allocation, authorization and durable controller checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_pr_exploration_contract as c
import h46_pr_exploration_design as d
import prepare_h46_pr_exploration_execution as prep
import launch_h46_pr_exploration_acquisition as launch
import finalize_h46_pr_exploration_acquisition as final

class CacheTests(unittest.TestCase):
    def cfg(self):
        o=d.selected_option(c.read(c.ROOT/d.PROPOSAL),d.RECOMMENDED)
        return dict(stage=d.STAGES[0],decisions=dict(option_id=d.RECOMMENDED,
            n_E_by_candidate={r['id']:o['n_E_by_candidate_proposed'][r['id']] for r in d.groups() if r['system']=='H4'}))
    def test_cache_reuses_generation_and_defensive_copy(self):
        cfg=self.cfg();c._proposal_tasks.cache_clear()
        with patch.object(d,'tasks_for',wraps=d.tasks_for) as generate:
            a=c.proposal_tasks(cfg);a[0]['candidate']['q']=999
            b=c.proposal_tasks(cfg)
            self.assertNotEqual(b[0]['candidate']['q'],999);self.assertEqual(generate.call_count,1)
        self.assertEqual(len(b),793)
    def test_cache_not_authorization_for_changed_counts(self):
        cfg=self.cfg();c.proposal_tasks(cfg)
        key=next(iter(cfg['decisions']['n_E_by_candidate']));cfg['decisions']['n_E_by_candidate'][key]+=1
        with self.assertRaisesRegex(ValueError,'ADOPTED_COUNTS'):c.proposal_tasks(cfg)
    def test_cache_keys_include_input_bytes_and_stage(self):
        cfg=self.cfg();c._proposal_tasks.cache_clear();c.proposal_tasks(cfg)
        raw=(c.ROOT/d.PROPOSAL).read_bytes();domain=(c.ROOT/d.pilot.DOMAIN).read_bytes()
        c._proposal_tasks(raw+b' ',domain,d.RECOMMENDED,cfg['stage'])
        c._proposal_tasks(raw,domain+b' ',d.RECOMMENDED,cfg['stage'])
        self.assertEqual(c._proposal_tasks.cache_info().misses,3)

class AdmissionTests(unittest.TestCase):
    def capacity(self):return dict(concurrent_peak_growth_proxy_bytes=1000000,
        final_growth_proxy_bytes={'H4':10,'H6':10},restoration_reserve_bytes=10,
        worker_RSS_proxy_bytes={'H4':1,'H6':2},raw_pair_proxy_bytes={'H4':1,'H6':2})
    def test_no_historical_worker_ceiling(self):
        r=prep.allocate_existing_plan(dict(disk_free_bytes=1000000,idle_CPU_equivalents=64,
            effective_available_RAM_bytes=1000),self.capacity(),{'H4':100,'H6':100},{'H4':1,'H6':3})
        self.assertEqual(sum(r.values()),64);self.assertGreater(r['H6'],28)
    def test_cpu_ram_and_fixed_scratch_all_respected(self):
        cap=self.capacity();cap['raw_pair_proxy_bytes']={'H4':100000,'H6':200000}
        r=prep.allocate_existing_plan(dict(disk_free_bytes=1000000,idle_CPU_equivalents=64,
            effective_available_RAM_bytes=1000),cap,{'H4':100,'H6':100},{'H4':1,'H6':3})
        self.assertLessEqual(2*(r['H4']*100000+r['H6']*200000),999970)
        with self.assertRaisesRegex(ValueError,'DISK_GATE'):
            prep.allocate_existing_plan(dict(disk_free_bytes=999999),cap,{}, {})
    def test_no_resources_rejects_without_creating_grants(self):
        with self.assertRaisesRegex(ValueError,'CANNOT_ADMIT'):
            prep.allocate_existing_plan(dict(disk_free_bytes=1000000,idle_CPU_equivalents=1,
                effective_available_RAM_bytes=100),self.capacity(),{'H4':10,'H6':10},{'H4':1,'H6':1})
    def test_approved_projection_is_new_and_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);oldapproval=dest/'old.json';c.save(oldapproval,{'execution_authorized':False})
            lease=dest/'lease.json';c.save(lease,dict(workers={'H4':1,'H6':1}))
            cfg=dict(SOURCE='a'*40,stage=d.STAGES[0],cohort='EXPLORATION',decisions={},resources={},tasks=[],
                storage=c.storage.storage_policy(),approval_receipt_path=str(oldapproval),sealed_files=[c.seal(oldapproval,'old')],
                call_scope={},runtime={},campaign_id='SYNTHETIC',registry_path='synthetic',excluded_seeds_path='synthetic',proposal_sha256='b'*64)
            before=copy.deepcopy(cfg)
            with patch.object(c,'authorize'):
                row=prep.approve(cfg,dest/'new',lease,self.capacity(),dest/'old_det')
            v=c.read(row['config']);self.assertEqual(cfg,before);self.assertFalse(c.read(oldapproval)['execution_authorized'])
            self.assertTrue(v['grant']['valid']);self.assertFalse(v['grant']['retry']);self.assertFalse(v['grant']['next_phase_authorized'])
            self.assertFalse(Path(v['run_root']).exists());self.assertTrue(v['decisions']['sample_design_adopted'])

class ControllerTests(unittest.TestCase):
    def test_common_integrity_is_separate_from_local_numeric_or_PSI(self):
        self.assertTrue(launch.common_integrity_failure('SEALED_BYTES:source'))
        self.assertFalse(launch.common_integrity_failure('COMPILED_RAW_GATE_FAIL PSI CPU MEMORY_PRESSURE'))
    def exercise(self,failed):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);configs=[];processes=[]
            class Proc:
                def __init__(self,code):self.code=code;self.pid=123;self.reaped=False;self.signals=[]
                def poll(self):return self.code
                def wait(self):self.reaped=True;return self.code
                def send_signal(self,s):self.signals.append(s)
            for stage in d.STAGES:
                cfg=dict(stage=stage,SOURCE='a'*40,campaign_id='ARTIFICIAL',plan_sha256='b'*64,
                    runtime={'python':sys.executable},run_root=str(dest/stage))
                p=dest/(stage+'.json');c.save(p,cfg);configs.append(p)
                processes.append(Proc(1 if failed and stage==d.STAGES[0] else 0))
            def audit(cfg):
                Path(cfg['run_root']).mkdir();c.save(Path(cfg['run_root'])/'end_audit.json',{'synthetic':True})
                return dict(status='STOP' if failed and cfg['stage']==d.STAGES[0] else 'ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP')
            with patch.object(c,'authorize'),patch.object(launch.subprocess,'Popen',side_effect=processes),\
                 patch('h46_pr_exploration_runtime.get_runtime',return_value=SimpleNamespace(auditor=SimpleNamespace(audit=audit))),\
                 patch.object(final,'finalize') as finish:
                code=launch.launch(configs,dest/'controller')
            self.assertEqual(code,1 if failed else 0);self.assertTrue(all(p.reaped for p in processes))
            self.assertTrue(all(not p.signals for p in processes));finish.assert_called_once()
            r=c.read(dest/'controller'/'CONTROLLER_FINISHED.json')
            self.assertTrue(r['mandatory_STOP']);self.assertFalse(r['statistics']);self.assertFalse(r['next_phase_started'])
    def test_two_successful_stages_reaped(self):self.exercise(False)
    def test_local_failure_does_not_stop_other_stage(self):self.exercise(True)
    def test_partial_projection_cannot_claim_full_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=dict(run_root=str(root),supervisor={'exit':str(root/'exit')},SOURCE='a'*40,
                stage=d.STAGES[0],plan_sha256='b'*64,tasks=[{}],call_scope={'compile_axes':2},decisions={'n_E_by_candidate':{'toy':1}})
            r=final.project(cfg)
            self.assertFalse(r['native_scientific_full_stage_PASS']);self.assertFalse(r['byte_SHA_roundtrip_full_stage_PASS'])
            self.assertIsNone(r['native_exit']);self.assertEqual(r['n_E_adopted'],{'toy':1})

    def test_public_finalization_updates_only_technical_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo=base/'repo';repo.mkdir();(repo/'docs/research').mkdir(parents=True);(repo/'artifacts').mkdir()
            for name in ('PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md'):
                (repo/name).write_text('historical content\n')
            c.save(repo/'artifacts/validation_manifest.json',{'historical':True})
            controller=base/'controller';controller.mkdir();c.save(controller/'CONTROLLER_FINISHED.json',{'synthetic':True})
            cfg=dict(run_root=str(base/'run'),supervisor={'exit':str(base/'exit')},SOURCE='a'*40,
                stage=d.STAGES[0],plan_sha256='b'*64,tasks=[{}],call_scope={'compile_axes':2},
                storage=c.storage.storage_policy(),decisions={'n_E_by_candidate':{'toy':1},'option_id':d.RECOMMENDED})
            added=[]
            def run(args,**kw):
                if args[1]=='add':added.extend(args[3:])
                return SimpleNamespace(returncode=0)
            def output(args,**kw):
                if args[1]=='rev-parse':return 'a'*40+'\n'
                return ('\0'.join(added)+'\0').encode() if '-z' in args else b''
            with patch.object(c,'ROOT',repo),patch.object(final,'_preserve',return_value={'status':'PASS'}),\
                 patch.object(final.subprocess,'run',side_effect=run),patch.object(final.subprocess,'check_output',side_effect=output):
                final.finalize([cfg],controller)
            public=c.read(repo/final.BUNDLE/'run_summary.json')
            self.assertFalse(public['stages'][0]['native_scientific_full_stage_PASS'])
            self.assertNotIn(str(base),(repo/final.BUNDLE/'run_summary.json').read_text())
            self.assertTrue(c.read(repo/'artifacts/validation_manifest.json')['historical'])
            self.assertEqual(len(added),7)

if __name__=='__main__':unittest.main()
