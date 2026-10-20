"""Semantic mutations on saved artificial compact rows; reseal file/chain."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json,sys,tempfile
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import offline as O
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_result import decode,canonical

def main(root,destination):
    root=Path(root);out=root/'output';c=O.read(out/'candidate.json');m=O.read(out/'manifest.json');report=O.read(out/'final.json')
    r=next(v for v in report['pairs'] if v['task']['phase_index']==1 and v['task']['pair_index']==0)
    name=r['task']['key']+'.jsonl';rows=[decode(json.loads(line)['payload']) for line in (out/name).read_bytes().splitlines()]
    changes=[('retention',lambda p:p.update(retention='DROP')),
        ('mode',lambda p:p.update(artificial_only=False)),('general_guarantee',lambda p:p.update(general_correctness_certified=True)),
        ('original_digest',lambda p:p.update(original_digest='bad')),('codec_digest',lambda p:p.update(exact_codec_sha256='bad')),
        ('evidence_digest',lambda p:p.update(full_evidence_sha256='bad')),('setup',lambda p:p.update(setup_digest='bad')),
        ('domain',lambda p:p['consumption'].update(domain='WRONG')),
        ('counter',lambda p:p['consumption'].update(counter_before=999)),
        ('width',lambda p:p['consumption']['draws'][0].update(width=159)),
        ('draw_value',lambda p:p['consumption']['draws'][0].update(value_hex='0x0')),
        ('trial_id',lambda p:p.update(trial_id=p['trial_id'].replace('/16','/17'))),
        ('global_offset',lambda p:p['connection'].update(global_offset=999)),
        ('worker',lambda p:p['connection'].update(worker_pid=1)),
        ('manifest',lambda p:p['connection'].update(manifest_binding='0'*64)),
        ('runtime',lambda p:p['connection'].update(runtime_binding='0'*64)),
        ('source',lambda p:p['connection'].update(source_binding='0'*64)),
        ('observer',lambda p:p['trial_cost'].update(observer_enabled=True)),
        ('call_count',lambda p:p['boundary_counts'].update(scheduled_sampling=0)),
        ('event_weight',lambda p:p['event'].update(weight=1))]
    results=[]
    with tempfile.TemporaryDirectory(prefix='b3-pilot-mutation-') as td:
        td=Path(td)
        for label,fn in changes:
            candidate=deepcopy(rows);fn(candidate[16]);previous='0'*64;data=[]
            for p in candidate:
                payload=canonical(p);h=sha256((previous+payload).encode()).hexdigest()
                data.append(json.dumps(dict(previous=previous,payload=payload,sha256=h),separators=(',',':')).encode()+b'\n');previous=h
            path=td/name;path.write_bytes(b''.join(data));rr=deepcopy(r);rr['file_sha']=sha256(path.read_bytes()).hexdigest();rr['chain']=previous
            try:O.check_pairs(td,c,m,[rr],artificial=True)
            except Exception as exc:results.append(dict(mutation=label,rejected=True,type=type(exc).__name__,reason=str(exc),chain_and_file_resealed=True))
            else:raise RuntimeError('SEMANTIC_MUTATION_ADMITTED:'+label)
    Path(destination).write_text(json.dumps(dict(status='PASS',cases=results,total=len(results),scientific_imports=0),indent=2)+'\n')
    print('SEMANTIC_MUTATIONS_PASS',len(results))
if __name__=='__main__':main(*sys.argv[1:])
