"""Fresh artificial SQLite scopes only. No real authority or old claims."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.abc
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode=True
FORBIDDEN={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision',
           'g6_f3_generator','g6_w2c_b3_driver','g6_w2c_cost_preparation'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.',1)[-1] in FORBIDDEN:raise RuntimeError('FORBIDDEN_LOAD:'+fullname)
sys.meta_path.insert(0,Guard())
sys.path.insert(0,'/tmp/b3-independent-server-r2-20261020/runtime/src')
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore,Denied

def main():
    here=Path(__file__).resolve().parent
    dest=here/'authority_evidence';dest.mkdir(exist_ok=False)
    observations=[]
    for label in ('atomic_race','commit_ack_unknown'):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-independent-'))
        store=FakeStore(root)
        manifest=dict(global_scope='ARTIFICIAL_ONLY/'+root.name,global_claim=root.name+'/CLAIM',retry=0,resume=False,
                      artifact_identity='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb')
        if label=='atomic_race':
            def attempt(index):
                try:store.claim(manifest);return dict(attempt=index,won=True)
                except Denied as exc:return dict(attempt=index,won=False,reason=str(exc))
            with ThreadPoolExecutor(max_workers=32) as pool:attempts=list(pool.map(attempt,range(32)))
            assert sum(x['won'] for x in attempts)==1
            observations.append(dict(id='atomic_race_one_winner',pass_=True,attempts=attempts))
        else:
            def fault(point):
                assert point=='admission.commit_ack'
                raise RuntimeError('INDEPENDENT_ACK_UNKNOWN_AFTER_COMMIT')
            try:store.claim(manifest,fault)
            except RuntimeError as exc:assert str(exc)=='INDEPENDENT_ACK_UNKNOWN_AFTER_COMMIT'
            else:raise AssertionError('ACK_FAULT_NOT_OBSERVED')
            assert store.snapshot(manifest)['claim'][3:]==['SPENT',1]
            observations.append(dict(id='ack_unknown_remains_spent',pass_=True))
        try:store.claim(manifest)
        except Denied as exc:assert str(exc)=='SCOPE_OR_CLAIM_SPENT'
        else:raise AssertionError('DOUBLE_CLAIM')
        observations.append(dict(id=label+'_double_claim_denied',pass_=True))
        changed=deepcopy(manifest);changed['artifact_identity']='0'*64
        try:store.stop(changed)
        except Denied as exc:assert str(exc)=='FAKE_STOP_BINDING'
        else:raise AssertionError('WRONG_STOP_BINDING')
        observations.append(dict(id=label+'_wrong_binding_denied',pass_=True))
        store.stop(manifest)
        assert store.snapshot(manifest)['claim'][3:]==['STOPPED',1]
        observations.append(dict(id=label+'_stop_spent',pass_=True))
        try:store.claim(manifest)
        except Denied:pass
        else:raise AssertionError('STOP_REUSE')
        observations.append(dict(id=label+'_stop_reuse_denied',pass_=True))
        (dest/(label+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        (dest/(label+'_snapshot.json')).write_text(json.dumps(store.snapshot(manifest),indent=2)+'\n')
        shutil.copyfile(root/'FAKE_ONLY.sqlite3',dest/(label+'_FAKE_ONLY.sqlite3'))
    result=dict(tests=len(observations),PASS=len(observations),FAIL=0,results=observations,
                science_modules_loaded=[x for x in sys.modules if x.rsplit('.',1)[-1] in FORBIDDEN],
                real_authority=False,new_fake_scopes=2,old_scope_reuse=0)
    assert result['science_modules_loaded']==[]
    (here/'independent_authority_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))

if __name__=='__main__':main()
