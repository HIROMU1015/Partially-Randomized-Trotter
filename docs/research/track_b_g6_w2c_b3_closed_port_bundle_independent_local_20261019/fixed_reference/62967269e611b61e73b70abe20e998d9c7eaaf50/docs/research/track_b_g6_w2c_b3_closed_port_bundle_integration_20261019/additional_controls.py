"""Extra fault evidence on integrated runs, not added to 134 unit count."""
from pathlib import Path
from copy import deepcopy
import json,shutil,sys,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import offline as O
    from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore,Denied
    from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import contract as K
    from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_scheduler as P
    matrix=json.loads((HERE/'runs_final/matrix.json').read_text());lookup={x['case']:x for x in matrix['cases']}
    c=json.loads((HERE/'runs_final/trusted_candidate.json').read_text());dest=HERE/'additional';dest.mkdir(exist_ok=False)
    results=[]
    for name,base in [('P1_failure_loses_S_receipt','P1_port_angle'),('S_failure_P1_ENTER_forgery','S_failure_blocks_P1'),('mode_in_seed_domain','normal_bundle_small')]:
        source=Path(lookup[base]['root']);target=dest/name/'case';shutil.copytree(source,target);out=target/'output'
        if name=='P1_failure_loses_S_receipt':(out/'S_receipt.json').unlink()
        elif name=='mode_in_seed_domain':
            p=out/'manifest.json';m=O._read(p);m['seed_namespace']+='/S';p.write_bytes(P.safe_json(m)+b'\n')
        else:
            report=O._read(out/'final.json');p=out/'operations.jsonl';rows=[json.loads(json.loads(line)['payload'])for line in p.read_bytes().splitlines()]
            previous=next(x for x in rows if x['op']=='ENTER');bad=dict(previous,condition='P1',task_key='P1__correctness_baseline_ordinary_A')
            bad['global_trial_id']=bad['global_trial_id'].replace('/S/','/P1/');bad['global_offset']=768
            rows.append(bad);lines=[];chain='0'*64
            from hashlib import sha256
            for i,row in enumerate(rows):
                row['seq']=i;raw=P.safe_json(row);h=sha256(chain.encode()+raw).hexdigest();lines.append(P.safe_json(dict(payload=raw.decode(),previous=chain,sha256=h))+b'\n');chain=h
            data=b''.join(lines);p.write_bytes(data);report['evidence_v2']['trace_prefix']=dict(rows=len(rows),bytes=len(data),sha256=sha256(data).hexdigest(),chain=chain)
            final=P.safe_json(report)+b'\n';assert len(final)<=K.FINAL_SLOT;(out/'final.json').write_bytes(final+b' '*(K.FINAL_SLOT-len(final)))
            delivery=O._read(out/'delivery.json');delivery['final_sha256']=O._sha(out/'final.json');data=P.safe_json(delivery)+b'\n';(out/'delivery.json').write_bytes(data+b' '*(4096-len(data)))
            caller=O._read(target/'external_observation.json');caller['delivery_sha256']=O._sha(out/'delivery.json');data=P.safe_json(caller)+b'\n';(target/'external_observation.json').write_bytes(data+b' '*(4096-len(data)))
        result=O.verify(target,c);assert result['status']!='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',result
        observation=dict(name=name,base_case=base,status=result['status'],issues=result['issues'],evidence_gaps=result['evidence_gaps'])
        (target.parent/'observation.json').write_text(json.dumps(observation,indent=2)+'\n');results.append(observation)
    (dest/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
