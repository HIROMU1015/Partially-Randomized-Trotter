"""Artificial boundary regressions; no molecular values or compiler imports."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pr_cost_batch as b
import pr_cost_batch_science as science
import prepare_pr_cost_batch as prepare


class Boundaries(unittest.TestCase):
    def setUp(self):
        base=Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/tmp')
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='artificial-',dir=base)
        self.root=Path(self.temp.name)
        self.receipt=self.root/'approval.txt';self.receipt.write_text('ARTIFICIAL ONLY')
        self.spec=dict(scope='H4_D_ACCOUNTING',stage='D_ACCOUNTING',SOURCE='a'*40,
            technical_sealed=True,approved=False,runtime_authorization=False,retry=False,
            next_stage_authorized=False,memory_estimate_per_worker_bytes=1,output_estimate_bytes=1,inode_estimate=1,
            output=str(self.root/'output'),lock=str(self.root/'ONE_SHOT'))
        self.spec['technical_digest']=b.metadata_binding(self.spec)
        self.grant=dict(scope=self.spec['scope'],technical_digest=self.spec['technical_digest'],SOURCE='a'*40,
            approved_by_user=True,runtime_authorization=True,one_shot=True,retry=False,
            next_stage_authorized=False,allowed_cpus=[min(os.sched_getaffinity(0))],
            concurrent_job_cpus=[],quota_policy='USER_ACCEPTS_UNEXPOSED_QUOTA_ENOSPC_FAIL_CLOSED',
            memory_admission_estimate_bytes=1,approval_receipt=str(self.receipt),
            approval_receipt_sha256=b.sha(self.receipt))
    def tearDown(self):self.temp.cleanup()
    def overview(self,scopes):
        value=dict(SOURCE='a'*40,contracts=[dict(scope=s,status='READY') for s in scopes],
            private_specs={s:'not_read_'+s for s in scopes},batch_lock=str(self.root/'BATCH'),
            authorization_directory=str(self.root))
        value['batch_digest']=b.digest(value);return value
    def approval(self,overview,scopes):
        return dict(batch_digest=overview['batch_digest'],approved_by_user=True,runtime_authorization=True,
            retry=False,next_stage_authorized=False,selected_scopes=scopes,
            individual_grants={s:deepcopy(self.grant) for s in scopes})
    def test_seal_does_not_authorize(self):
        grant=dict(self.grant,approved_by_user=False,runtime_authorization=False)
        with patch.object(b,'verify',side_effect=AssertionError('must not touch inputs')):
            with self.assertRaisesRegex(ValueError,'APPROVAL_REQUIRED'):b.run_contract(self.spec,grant)
    def test_tampered_seal_rejected(self):
        self.spec['scope']='H8_A_COST'
        with self.assertRaisesRegex(ValueError,'DIGEST'):b.sealed(self.spec)
    def test_receipt_tamper_before_launch(self):
        self.receipt.write_text('changed')
        with self.assertRaisesRegex(ValueError,'APPROVAL_RECEIPT'):b.authorize(self.spec,self.grant)
    def test_cross_source_grant_rejected(self):
        with self.assertRaisesRegex(ValueError,'SOURCE_RECEIPT'):
            b.authorize(self.spec,dict(self.grant,SOURCE='b'*40))
    def test_memory_estimate_not_zero_or_negative(self):
        with self.assertRaisesRegex(ValueError,'MEMORY_ESTIMATE'):
            b.authorize(self.spec,dict(self.grant,memory_admission_estimate_bytes=0))
    def test_no_automatic_other_stage(self):
        overview=self.overview(list(b.SCOPES));approval=self.approval(overview,['H4_D_ACCOUNTING'])
        self.assertEqual(b.select(overview,approval),['H4_D_ACCOUNTING'])
        approval['individual_grants']['H4_A_COST']={}
        with self.assertRaisesRegex(ValueError,'NO_AUTOMATIC_OTHER_STAGE'):b.select(overview,approval)
    def test_unselected_invalid_does_not_poison_selection(self):
        overview=self.overview(['H4_D_ACCOUNTING','H6_B_ANALYSIS'])
        overview['contracts'][1]['status']='MISSING';overview.pop('batch_digest')
        overview['batch_digest']=b.digest(overview)
        self.assertEqual(b.select(overview,self.approval(overview,['H4_D_ACCOUNTING'])),['H4_D_ACCOUNTING'])
    def test_independent_prestart_failure_leaves_sibling_running(self):
        scopes=['H4_D_ACCOUNTING','H6_D_ACCOUNTING'];overview=self.overview(scopes)
        approval=self.approval(overview,scopes)
        fake=Mock(pid=999999);fake.poll.return_value=0;fake.wait.return_value=0
        specs={overview['private_specs'][s]:dict(self.spec,scope=s,python=sys.executable) for s in scopes}
        def verify(spec):
            if spec['scope']==scopes[0]:raise ValueError('ARTIFICIAL_MISSING')
        with patch.object(b,'read',side_effect=lambda p:specs[p]),patch.object(b,'authorize'),\
             patch.object(b,'verify',side_effect=verify),patch.object(b,'verify_scoped_paths'),\
             patch.object(b.old,'fresh',return_value={'memory_available_bytes':10,'disk_free_bytes':10,'free_inodes':10}),\
             patch.object(b.subprocess,'Popen',return_value=fake) as spawn,\
             patch.object(b,'identity',return_value={'pid':fake.pid}):
            result=b.launch(overview,approval)
        self.assertEqual(result[scopes[0]]['status'],'PRESTART_STOP')
        self.assertTrue(result[scopes[1]]['reaped']);self.assertEqual(spawn.call_count,1)
    def test_unknown_quota_requires_explicit_selected_policy(self):
        grant=dict(self.grant);grant.pop('quota_policy')
        with self.assertRaisesRegex(ValueError,'QUOTA_POLICY'):b.fresh_admission(self.spec,grant)
    def test_parallel_job_cpu_exclusion(self):
        with self.assertRaisesRegex(ValueError,'CONCURRENT_JOB_CPU'):
            b.fresh_admission(self.spec,dict(self.grant,concurrent_job_cpus=self.grant['allowed_cpus']))
    def test_one_shot_cannot_be_claimed_twice(self):
        path=self.root/'marker';b.old.claim_attempt(path,'a'*40,'artificial')
        with self.assertRaises(FileExistsError):b.old.claim_attempt(path,'a'*40,'artificial')
    def test_first_stop_reason_preserved(self):
        b.old.first_stop(self.root,'first');b.old.first_stop(self.root,'second')
        self.assertEqual(b.read(self.root/'STOP.json')['reason'],'first')
    def test_mutated_input_refused(self):
        row=dict(role='fixture',path=str(self.receipt),bytes=self.receipt.stat().st_size,sha256=b.sha(self.receipt))
        self.receipt.write_text('tamper')
        with self.assertRaisesRegex(ValueError,'INPUT_BYTES_CHANGED'):b.old.verify_byte_rows([row])
    def test_unused_output_required(self):
        Path(self.spec['output']).mkdir()
        with self.assertRaisesRegex(ValueError,'UNUSED_ONE_SHOT'):b.verify_scoped_paths(self.spec)
    def test_cleanup_continues_after_non_esrch(self):
        with patch.object(b,'stop_owned',side_effect=[PermissionError('denied'),None,None]) as stop:
            errors=b.cleanup_all({str(i):(Mock(),{}) for i in range(3)})
        self.assertEqual(stop.call_count,3);self.assertEqual(len(errors),1)
    def test_esrch_signal_reaped(self):
        process=Mock(pid=555555);process.poll.return_value=None
        owned=dict(pid=process.pid,starttime=1,uid=os.getuid(),pgid=process.pid)
        with patch.object(b,'identity',return_value=owned),\
             patch.object(b.os,'killpg',side_effect=ProcessLookupError):b.stop_owned(process,owned)
        process.wait.assert_called_once()
    def test_ownership_mismatch_never_signals(self):
        process=Mock(pid=555555);process.poll.return_value=None
        with patch.object(b,'identity',return_value={'changed':True}),patch.object(b.os,'killpg') as send:
            with self.assertRaisesRegex(ValueError,'OWN_SESSION_STOP_ONLY'):b.stop_owned(process,{'changed':False})
            send.assert_not_called()
    def test_owned_minimal_child_term_reap_and_double_cleanup(self):
        process=subprocess.Popen([sys.executable,'-P','-B','-c','import time; time.sleep(30)'],
            start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        owned=b.identity(process.pid)
        try:
            b.stop_owned(process,owned);b.stop_owned(process,owned)
            self.assertIsNotNone(process.returncode)
            self.assertFalse(Path('/proc',str(process.pid)).exists())
        finally:
            if process.poll() is None:process.kill();process.wait()


class ArtificialAccounting(unittest.TestCase):
    def test_h8_saved_usual_alias_uses_only_usual_shots(self):
        key=('ARTIFICIAL',.01,1.)
        usual={key:{'axes':{'real':{'shots':2},'imag':{'shots':3}}}}
        noncancel={key:{'axes':{'real':{'shots':100},'imag':{'shots':200}}}}
        self.assertEqual(prepare.saved_bias_mode('USUAL_SIGNED_SAVED'),'USUAL_SIGNED')
        self.assertEqual(prepare.saved_pr_accounting('USUAL_SIGNED_SAVED',key,usual,noncancel),usual[key])
        self.assertEqual(prepare.saved_pr_accounting('EMPIRICAL_NONCANCELLATION',key,usual,noncancel),noncancel[key])
    def test_missing_usual_shots_cannot_fall_back_to_noncancel(self):
        key=('ARTIFICIAL',.01,1.)
        self.assertIsNone(prepare.saved_pr_accounting('USUAL_SIGNED_SAVED',key,{}, {key:{'shots':999}}))
    def test_unknown_saved_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'REGISTERED_SAVED_BIAS_MODE'):
            prepare.saved_bias_mode('UNKNOWN')
    def test_saved_string_factor_has_same_identity_as_numeric(self):
        for factor in (1.,10.):
            row=dict(cell_id='ARTIFICIAL',epsilon=.01,factor=factor)
            self.assertEqual(prepare.saved_accounting_key(row),
                prepare.saved_accounting_key(dict(row,factor=str(factor))))
    def test_direct_prices_replace_only_saved_deterministic_cost(self):
        pairs=[dict(cell_id='PR',replica=i,seed=i+10,event_digest='event'+str(i),cohort='CONFIRMATION_N32',
            binding={'artificial':True},axes={a:dict(RZ=mult*(i+1),seed=i+10,event_digest='event'+str(i))
                for a,mult in [('cosine',1),('sine',2)]}) for i in range(32)]
        common=dict(eligible=True,bias_mode='USUAL_SIGNED',epsilon=.05,factor=1.,shots=dict(cosine=2,sine=3))
        data=dict(rows=[dict(common,cell_id='D',method='B0',G_exact='100'),
                        dict(common,cell_id='PR',method='B2',G_exact='132')],
            paired_PR={'PR':pairs},measured={'D':{'cosine':{'RZ':10},'sine':{'RZ':20}}},
            deterministic_sets={'old':['D'],'strengthened':['D']},PR_ids=['PR'])
        before=deepcopy(data);result=science.d_accounting(data)
        self.assertEqual(result['rows'][0]['G_exact'],'80')
        self.assertEqual(result['rows'][1]['G_exact'],'132')
        self.assertGreater(result['rows'][1]['G_SE'],0);self.assertEqual(data,before)
    def test_conditional_cost_never_zero_fills_unconditional(self):
        group=dict(stratum='s4',recorded_log_mass=-20.,original_distribution={'synthetic':True},
            paired_costs=[dict(axes={a:dict(RZ=i+10) for a in ('cosine','sine')}) for i in range(8)])
        data=dict(cases=[dict(cell={'id':'ARTIFICIAL','K':4},conditional_groups=[group],ordinary_samples=[])])
        case=science.b_analysis(data)['cases'][0]
        self.assertIsNone(case['unconditional_mean_RZ'])
        self.assertIn('s2_multi',case['missing_conditional_cost_strata'])
    def test_missing_conditional_sample_refused(self):
        group=dict(stratum='s4',recorded_log_mass=-1.,original_distribution={},paired_costs=[])
        with self.assertRaisesRegex(ValueError,'CONDITIONAL_N8'):
            science.b_analysis(dict(cases=[dict(cell={'id':'A','K':4},conditional_groups=[group],ordinary_samples=[])]))


if __name__=='__main__':unittest.main()
