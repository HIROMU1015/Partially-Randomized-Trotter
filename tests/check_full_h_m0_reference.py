"""Fixed metadata and artificial records only; no H4 sampling/build/compile."""
from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_m0_reference as M


class Tests(unittest.TestCase):
    def setUp(self):
        self.job=json.loads((ROOT/M.FROZEN).read_text());self.proposal=M.contract(self.job)

    def artificial_path(self):
        j=self.job;row=j['row'];metrics={k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')}
        return dict(row=row,cohort=j['cohort'],index=0,seeds=j['seeds'],sampling='ORIGINAL_NORMAL_SAMPLER',parents=[],
            events=[[dict(scope='ARTIFICIAL_EVENT') for _ in range(9)]],fingerprints=dict(cosine='ARTIFICIAL_X',sine='ARTIFICIAL_Y'),
            trial=dict(candidate=row,cohort=j['cohort'],index=0,status='PASS',preparation_hash=self.proposal['target_preparation_hash'],
                scalar=self.proposal['scalar']['total'],checks=[dict(check=g,error=0,tolerance=1,phase_alignment=False) for g in M.GATES],
                prices={a:metrics for a in ('cosine','sine')}),
            checks=dict(parity=dict(status='PASS'),scalar_event_phases=dict(status='PASS'),builder_lowering_id=M.P.P1.LOWERING_ID),
            xy_phase_gate=dict(status='PASS'),compile_receipts={a:dict(record=dict(metrics=metrics,compiler=j['profile']['compiler'],
                fingerprint='ARTIFICIAL_'+axis),fingerprint='ARTIFICIAL_'+axis,reused=False,
                locator=dict(path='ARTIFICIAL_RECORD',sha256='a'*64)) for axis,a in (('X','cosine'),('Y','sine'))})

    def test_fixed_rule_admission_seed_and_scope(self):
        for key,value in [('seeds',[0]),('maximum_compiles',4),('independent_paths',2)]:
            bad=deepcopy(self.job);bad[key]=value
            with self.assertRaises(ValueError):M.contract(bad)
        bad=deepcopy(self.job);bad['row']['R_outer']=10
        with self.assertRaises(ValueError):M.contract(bad)

    def test_two_equal_axis_prices_keep_distinct_receipts_and_no_96_multiplier(self):
        rows=M.prices(self.job,self.artificial_path(),dict(commit='ARTIFICIAL'))
        self.assertEqual([r['per_shot_RZ'] for r in rows],[1,1])
        self.assertNotEqual(rows[0]['fingerprint'],rows[1]['fingerprint'])
        self.assertEqual([r['measurement']['axis'] for r in rows],['X','Y'])
        self.assertTrue(all(r['independent_paths']==1 and not r['shot_multiplier_applied'] and r['CI'] is None for r in rows))

    def test_required_gate_or_preparation_failure_is_not_accepted(self):
        for change in ('gate','preparation','sampling'):
            path=self.artificial_path()
            if change=='gate':path['xy_phase_gate']['status']='FAIL'
            elif change=='preparation':path['trial']['preparation_hash']='FOREIGN'
            else:path['sampling']='FORCED'
            with self.assertRaises(ValueError):M.prices(self.job,path,{})

    def test_child_reuses_normal_draw_once_and_refuses_second_draw(self):
        for twice in (False,True):
            with TemporaryDirectory() as temp:
                job=dict(self.job,private=temp,output=str(Path(temp)/'public'))
                f=Path(temp)/'job.json';f.write_text(json.dumps(job))
                original=Mock(return_value='ARTIFICIAL_REQUESTS')
                native=SimpleNamespace(sample_requests=original)
                full=SimpleNamespace(context=lambda _:None,setup=lambda:native)
                def artificial(*args,**kwargs):
                    self.assertEqual(native.sample_requests('ARTIFICIAL_METADATA'),'ARTIFICIAL_REQUESTS')
                    if twice:native.sample_requests('SECOND_FORBIDDEN')
                    return {'fixture':'ARTIFICIAL_NO_H4'}
                with patch.object(M.P.P1,'provider_checks'),patch.object(M.P.P1,'full_h_module',return_value=full), \
                     patch.object(M,'context_check',return_value=dict(status='ARTIFICIAL')), \
                     patch.object(M.A,'accelerated_path',side_effect=artificial):
                    self.assertEqual(M.child(f),2 if twice else 0)
                self.assertEqual(original.call_count,1)
                result=json.loads((Path(temp)/'child.json').read_text())
                self.assertEqual(result['normal_sampler_calls'],1)
                self.assertEqual(result['status'],'STOP' if twice else 'PASS')


if __name__=='__main__':unittest.main(verbosity=2)
