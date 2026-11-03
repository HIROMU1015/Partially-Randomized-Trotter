"""Closed port tests. Same primary agent self-check, no independent reviewer."""
import copy
import importlib
import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S,ports as P,authority as A
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import offline as O,exercise as E
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as V
P.install_guard()
ROOT=Path(__file__).resolve().parents[3]

def guarded_worker(name):
    P.install_guard()
    try:importlib.import_module('trottertracks.algorithm_codesign.'+name)
    except S.Rejected:return not P.forbidden_imports()
    return False

class Ports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=S.candidate(ROOT)
    def endpoint(self,arm='full_return',layer='B',fault=''):
        task=dict(arm=arm,layer=layer,condition='S',phase='profile',pass_name='baseline')
        return P.ArtificialEndpoint(task,fault=fault),P.ArtificialBits(arm,'X')
    def obtain(self,arm='full_return'):
        e,b=self.endpoint(arm)
        for trial in range(16):
            r=e.sample(b,f'profile/baseline/{arm}/B/X/{trial}')
            if r.kind=='EVENT':return e,b,r,P.TrialIdentity('ARTIFICIAL_PORT','S',arm,'B','X','profile','baseline',trial)
        self.fail('artificial fixture lacked event')
    def test_static_inventory_exact_signatures(self):
        inv=S.inventory(ROOT)
        funcs={x['name']:x['signature'] for x in inv['modules'][S.K.SOURCE+'g6_f3_generator.py']['entries'] if 'signature'in x}
        self.assertEqual(funcs['GeneratorSession.sample'],'self, bits=None, *, trial_id')
        self.assertIn('CANONICAL_PREFLIGHT_NOT_IMPLEMENTED',(ROOT/S.K.SOURCE/'g6_f3_generator.py').read_text())
        self.assertFalse(S.binding()['dispatch_implemented'])
    def test_preflight_normal(self):self.assertFalse(S.preflight(self.c,ROOT)['production_ready'])
    def test_prefix_mode_layer_pass(self):
        for arm in S.ARMS:
            for axis in ('X','Y'):
                streams=[P.ArtificialBits(arm,axis)for _ in range(6)]
                values=[[b.bits(160)for _ in range(16)]for b in streams]
                self.assertTrue(all(x==values[0] for x in values))
                ids=[P.TrialIdentity('run',mode,arm,layer,axis,'profile',pa,0)for mode in ('S','P1') for layer in ('A','B')for pa in ('baseline','instrumented')]
                self.assertEqual(len({x.global_id()for x in ids}),8)
                self.assertEqual(len({x.seed_domain()for x in ids}),1)
    def test_domain_separation(self):
        self.assertEqual(len({P.ArtificialBits(arm,axis).bits(160)for arm in S.ARMS for axis in ('X','Y')}),8)
    def test_no_scientific_namespace_bits(self):
        with self.assertRaises(S.Rejected):P.ArtificialBits('ordinary','X',namespace=self.c['proposal']['science_seed_namespace'])
    def test_parent_and_spawn_import_guards(self):
        for name in S.SCIENCE:
            self.assertTrue(guarded_worker(name))
        with mp.get_context('spawn').Pool(4) as pool:self.assertTrue(all(pool.map(guarded_worker,S.SCIENCE)))
    def test_authority_race_and_spent(self):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-race-'))
        au=A.ArtificialAuthority(self.c,root);b=au.expected;a=A.approval(b)
        def attempt(_):
            try:au.claim(b,a);return True
            except RuntimeError:return False
        with ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(attempt,range(32)))
        self.assertEqual(sum(results),1)
        au.stop(b)
        self.assertEqual(au.snapshot()['claim'][-2:],['STOPPED',1])
        with self.assertRaises(RuntimeError):au.claim(b,a)
    def test_commit_ACK_unknown_is_spent(self):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-ack-'))
        au=A.ArtificialAuthority(self.c,root);b=au.expected
        def fail(point):raise OSError('ARTIFICIAL_ACK_UNKNOWN')
        with self.assertRaises(OSError):au.claim(b,A.approval(b),fault=fail)
        self.assertEqual(au.snapshot()['claim'][-2:],['SPENT',1])
        with self.assertRaises(RuntimeError):au.claim(b,A.approval(b))
        au.stop(b)
    def test_wrong_event_identity(self):
        e,b,r,i=self.obtain()
        with self.assertRaises(S.Rejected):e.angle_for(copy.deepcopy(r))
        with self.assertRaises(S.Rejected):e.sample(b,'profile/baseline/full_return/B/X/99')
    def test_wrong_bits_type(self):
        e,b=self.endpoint()
        with self.assertRaises(S.Rejected):e.sample(object(),'profile/baseline/full_return/B/X/0')
    def test_wrong_result_type(self):
        _,_,r,i=self.obtain()
        with self.assertRaises(S.Rejected):P.validate_result(dict(kind='EVENT'),i,r.evidence['consumption'])
    def test_real_bits_and_authority_never_dispatch(self):
        for fn in (S.real_bits,S.real_authority,S.production_entry):
            with self.assertRaises(S.Rejected):fn(fake=True,run_ready=True,grant='fake')
    def test_offline_import_separation(self):
        code="import sys;from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import offline;assert not any(n.rsplit('.',1)[-1] in ('ports','exercise','authority','static','g7_generator','g6_f3_generator','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_bundle_bridge_v1') for n in sys.modules)"
        p=subprocess.run([os.sys.executable,'-c',code],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr.decode())

def add_candidate(name,change):
    def test(self):
        c=copy.deepcopy(self.c);change(c)
        with self.assertRaises((S.Rejected,KeyError)):S.preflight(c,ROOT)
    setattr(Ports,'test_preflight_reject_'+name,test)
for name,change in {
    'source':lambda c:c['source_pins'].__setitem__(next(iter(c['source_pins'])),'0'*64),
    'runtime':lambda c:c['python'].__setitem__('version','MUTATED'),
    'signature':lambda c:c['api_binding']['request'].__setitem__('G7','wrong_signature'),
    'missing_field':lambda c:c.pop('scientific_reference'),
    'contract_digest':lambda c:c.__setitem__('source_release_candidate','0'*64),
    'schema':lambda c:c.__setitem__('schema','unsupported'),
    'input':lambda c:c['scientific_reference']['fixed'].__setitem__('x','1/2'),
    'retry':lambda c:c.__setitem__('retry',1),
    'resume':lambda c:c.__setitem__('resume',True),
    'run_ready':lambda c:c.__setitem__('run_ready',True),
    'content_adopted':lambda c:c.__setitem__('content_adopted',True),
    'science_release':lambda c:c.__setitem__('science_release','fake'),
    'scientific_adoption':lambda c:c.__setitem__('scientific_adoption','fake'),
    'execution_grant':lambda c:c.__setitem__('contract_grants_execution',True),
    'budget':lambda c:c['proposal'].__setitem__('total_calls',16897),
    'seed':lambda c:c.__setitem__('artificial_seed_namespace','REAL')}.items():add_candidate(name,change)

def add_authority(field):
    def test(self):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-binding-'))
        au=A.ArtificialAuthority(self.c,root);b=copy.deepcopy(au.expected)
        b[field]='MUTATED'
        with self.assertRaises(S.Rejected):au.claim(b,A.approval(b))
        self.assertIsNone(au.snapshot()['claim'])
    setattr(Ports,'test_authority_reject_'+field,test)
for field in ('formal_source_release','source_release_candidate','scientific_input_hash','runtime_hash','contract_hash',
              'global_scope','global_claim','call_budget','result_budget','log_budget','seed_namespace','retry','resume',
              'worker_policy','storage_destination','monitoring_STOP_hash','artificial_only','real_grant','real_authority'):
    add_authority(field)

def add_port(name,change):
    def test(self):
        _,_,r,i=self.obtain();r=copy.deepcopy(r);tr=r.evidence['consumption'];change(r,tr)
        with self.assertRaises((S.Rejected,ValueError,RuntimeError)):P.validate_result(r,i,tr)
    setattr(Ports,'test_result_reject_'+name,test)
for name,change in {
    'missing_field':lambda r,t:r.event.pop('weight'),
    'wrong_rational':lambda r,t:r.event.__setitem__('ratio',0.2),
    'weight_identity':lambda r,t:r.event.__setitem__('weight',r.event['weight']*2),
    'word_identity':lambda r,t:r.event.__setitem__('raw_word',(0,0)),
    'phase_identity':lambda r,t:r.event.__setitem__('phase_i_power',99),
    'consumption_count':lambda r,t:t['draws'].pop(),
    'consumption_width':lambda r,t:t['draws'][0].__setitem__('width',159),
    'consumption_counter':lambda r,t:t.__setitem__('counter_after',999),
    'consumption_value':lambda r,t:t['draws'][0].__setitem__('value_hex','0x100000000000000000000000000000000000000000000'),
    'consumption_valid_wrong_value':lambda r,t:t['draws'][0].__setitem__('value_hex','0x1'),
    'seed_domain':lambda r,t:t.__setitem__('domain','REAL/full_return/X'),
    'trace_schema':lambda r,t:t.__setitem__('schema','unknown')}.items():add_port(name,change)

def add_request(name,trial):
    def test(self):
        e,b=self.endpoint()
        with self.assertRaises((S.Rejected,ValueError)):e.sample(b,trial)
    setattr(Ports,'test_request_reject_'+name,test)
for name,trial in dict(arm='profile/baseline/ordinary/B/X/0',layer='profile/baseline/full_return/A/X/0',
                      axis='profile/baseline/full_return/B/Z/0',phase='wrong/baseline/full_return/B/X/0',
                      pass_name='profile/instrumented/full_return/B/X/0',schema='malformed').items():add_request(name,trial)

def add_failure(stage):
    def test(self):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-f01-'))
        task=S.K.plan(16)[0][1]
        row=E.pair((task,self.c,str(root),stage))
        self.assertEqual(row['status'],'FAILED')
        p=V.verify_worker(root/(task['key']+'.jsonl'),task,row)
        self.assertEqual(p['status'],'TECHNICAL_FAILURE',p)
        self.assertFalse(row['forbidden_imports'])
    setattr(Ports,'test_F01_'+stage,test)
for stage in ('generator','audit','angle','requery'):add_failure(stage)

if __name__=='__main__':unittest.main()
