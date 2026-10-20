"""Read-only supplemental joins on retained records, never science evaluation."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,sys
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import offline as O,spec as S,guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_result import decode

def main(packet):
    packet=Path(packet);c=O.read(packet/'candidate.json');m=O.read(packet/'approved_manifest.json');out=Path(m['storage_destination']);r=O.read(out/'final.json')
    assert r['complete'];parity={};by_phase=Counter();nested=Counter();draws=0;worker_core_sets=[]
    for row in r['pairs']:
        t=row['task'];path=out/(t['key']+'.jsonl');session=None;bound=None
        for line in path.read_bytes().splitlines():
            p=decode(json.loads(line)['payload']);compact=p.get('schema')=='b3_pilot_compact_trial_v1';ev=p if compact else p['evidence']
            a=p if compact else ev['adapted'];connection=ev['connection']
            if session is None:session=connection['session'];bound=connection['route']
            assert connection['session']==session and connection['route']==bound==S.route(t['arm'],t['layer'])
            assert ev['runtime_origin']=='PINNED_G7_F3' and ev['artificial_only'] is False
            if not compact:
                guard=ev['guard'];assert {n.rsplit('.',1)[-1] for n in guard['core_loaded']}==set(G.CORE)
                assert guard['core_import_attempts']==[];worker_core_sets.append(guard['core_loaded'])
            index=int(p['trial_id'].rsplit('/',1)[1]);key=t['arm'],p['axis'],index
            value=C.fingerprint(dict(kind=p['kind'],classification=a['classification'],event=p['event'],consumption=ev['consumption']))
            assert key not in parity or parity[key]==value;parity[key]=value
            by_phase[t['phase_index']]+=1;draws+=len(ev['consumption']['draws'])
        if row['fine']:
            for label,v in row['fine'].items():
                if any(x in label for x in ['event_witness','__init__','prepare','sample','packet','angle_for_event']):nested[label]+=v['calls']
    assert [by_phase[i] for i in range(6)]==[256,4096,4096,256,4096,4096]
    trace=[json.loads(json.loads(line)['payload']) for line in (out/'operations.jsonl').read_bytes().splitlines()]
    phases=[];open_phase=None;completed_phases=[]
    for v in trace:
        if v['op']=='PHASE_BARRIER_BEGIN':
            assert open_phase is None;open_phase=v['phase_index'];assert open_phase==len(completed_phases);phases.append(dict(phase=open_phase,begin_seq=v['seq']))
        elif v['op']=='PHASE_BARRIER_END':
            assert open_phase==v['phase_index'] and v['complete'];phases[-1].update(end_seq=v['seq']);completed_phases.append(open_phase);open_phase=None
        elif v['op']=='ENTER':
            assert open_phase is not None
            assert next(t for t in m['tasks'] if t['key']==v['task_key'])['phase_index']==open_phase
    assert completed_phases==list(range(6)) and open_phase is None
    result=dict(status='PASS',scope='saved values only; no generator, sampler, scientific bitstream or numerical reevaluation',
        SOURCE=m['measured_SOURCE'],manifest_digest=S.digest(m),kind_classification_event_consumption_AB_S_P1_pass_correctness_prefix=True,
        compact_route_and_session_join=True,all_live_origins=True,full_records_worker_guard_verified=len(worker_core_sets),
        actual_phase_counts={str(k):v for k,v in by_phase.items()},actual_bit_draws=draws,strict_phase_barriers=phases,
        instrumented_internal_call_observations=dict(nested),internal_counts_scope='instrumented passes only; includes nested calls and must not be added to boundary counters',
        parent_core_loaded=G.observation()['core_loaded'],original_digest_for_compact='recorded by fixed worker; original full bytes not retained and cannot be independently reconstructed')
    with (packet/'supplemental_saved_checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',draws=draws,phases=completed_phases)))
if __name__=='__main__':main(sys.argv[1])
