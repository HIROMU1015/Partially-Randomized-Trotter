"""Independent artificial global fixtures using unchanged real Stream/run_pair."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import b3_offline_artificial_fixtures as F
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as A


def control_write(path,docs):
    path.write_bytes(b''.join((json.dumps(x,separators=(',',':'),ensure_ascii=True)+'\n').encode()for x in docs))


def bundle(root,n=18,reverse=True):
    root=Path(root);root.mkdir()
    out=root/'output';_,old=F.bundle(out,n)
    (out/'summary.jsonl').unlink();(out/'completion_receipt.json').unlink();(out/'manifest.json').unlink()
    tasks,planned=G.plan(n);rows=[]
    for t,s in zip(tasks,old['pairs']):
        rows.append(dict(task=t,status='OK',outcome={k:s[k]for k in('counts','axes','trace_sha256','shared_identity','population_cost_certificate','periodic_max_observed_RSS_bytes','RSS_sampling_is_not_continuous_peak')},failure=None,pid=100+t['phase_index']*8+t['pair_index'],
                         cpu_ns=s['reconciliation']['outer_total_ns'][0],wall_ns=s['reconciliation']['outer_total_ns'][1],
                         coarse=s['coarse'],reconciliation=s['reconciliation'],fine=s['functions'],fine_reconciliation=s['fine_reconciliation'],
                         memory=s['memory'],memory_baseline=s['memory_baseline'],control_rpc=dict(calls=1,cpu_ns=1,wall_ns=1),io={},
                         output_bytes=s['output_bytes'],output_rows=s['output_rows'],chain=s['output_chain_sha256'],file_sha=s['file_sha256']))
    state=bytes([4])*planned;(out/'trial_states.bin').write_bytes(state)
    settings=dict(workers=8,mode='P1',inflight_cap=2,profile_n=n,workload='mixed',result_bytes=G.BUDGET['result_bytes'],log_bytes=G.BUDGET['log_bytes'],call_limit=G.BUDGET['total'],faults=[],parent_fault='',collision=False)
    docs=[dict(event='PLAN',schema='b3_parallel_artificial_v1',settings=settings,planned=planned,tasks=tasks,state_codes={s:i for i,s in enumerate(G.STATES)},retry=0,resume=False)]
    order=[]
    for phase in range(3):
        docs.append(dict(event='PHASE_BEGIN',phase=phase,time_ns=phase*10))
        group=[r for r in rows if r['task']['phase_index']==phase]
        for r in group:docs.append(dict(event='WORKER_START',key=r['task']['key'],pid=r['pid']))
        for r in reversed(group)if reverse else group:
            docs.append(dict(event='WORKER_END',report=r));order.append(r['task']['key'])
        docs.append(dict(event='PHASE_END',phase=phase,stopped=False))
    control_write(out/'control.jsonl',docs)
    worker=sum(r['cpu_ns']for r in rows)+24000
    report=dict(schema=A.SCHEMA,settings=settings,planned=planned,reserved=planned,started_observed=planned,
                states={s:planned if s=='COMPLETED'else 0 for s in G.STATES},max_inflight=8,complete=True,
                failures=[],representative_failure=None,completion_order=order,contract_order=[t['key']for t in tasks],all_pair_states={t['key']:'OK'for t in tasks},pairs=rows,
                parent_cpu_ns=100000,worker_cpu_ns=worker,total_cpu_ns=worker+100000,wall_ns=500000,
                output=dict(limits=dict(result=settings['result_bytes'],log=settings['log_bytes']),reserved=dict(result=sum(r['output_bytes']for r in rows),log=planned+(out/'control.jsonl').stat().st_size),observed=dict(result=sum(r['output_bytes']for r in rows),log=planned+(out/'control.jsonl').stat().st_size)),
                peak_sampled_rss_bytes=1000000,log_errors=[],state_update_io_bytes=4*planned,worker_launch_cost=dict(cpu_ns=5000,wall_ns=10000,count=24),
                retry=0,resume=False,mandatory_STOP=True,artificial_only=True)
    meta=F.emit_file(out/'summary.jsonl',[report]);docs.append(dict(event='COMPLETION_RECEIPT',summary_chain=meta['output_chain_sha256'],summary_sha=meta['file_sha256'],all_workers_joined=True));control_write(out/'control.jsonl',docs)
    for channel,amount in(('result',sum(r['output_bytes']for r in rows)+meta['output_bytes']),('log',planned+(out/'control.jsonl').stat().st_size)):
        report['output']['reserved'][channel]=report['output']['observed'][channel]=amount
    report.update(host_outer=dict(parent_cpu_ns=110000,worker_cpu_ns=worker,wall_ns=600000),fake_claim_status='STOPPED',state_file_sha256=sha256(state).hexdigest())
    (root/'report.json').write_text(json.dumps(report)+'\n');(root/'metadata.json').write_text('{}\n')
    m=dict(schema='b3_parallel_artificial_input_v1',adapter='PUBLIC_P1_ARTIFICIAL_V1',run_id='INDEPENDENT_ARTIFICIAL',fixture_origin='INDEPENDENT_ARTIFICIAL_FORMAT_COMPATIBLE',artificial_only=True,producer_commit=A.PUBLISHED,output='output',report='report.json',metadata='metadata.json')
    (root/'manifest.json').write_text(json.dumps(m)+'\n')
    return m,report


def operations(tasks,run_id='INDEPENDENT_ARTIFICIAL'):
    seq=0
    def row(op,**kw):
        nonlocal seq
        r=dict(seq=seq,run_id=run_id,op=op,**kw);seq+=1;return r
    for phase in range(3):
        yield row('PHASE_BEGIN',phase=phase)
        for t in tasks:
            if t['phase_index']!=phase:continue
            for axis in('X','Y'):
                for i in range(t['n']):
                    tid=f"{t['phase']}/{t['pass_name']}/{t['arm']}/{t['layer']}/{axis}/{i}"
                    for op in('RESERVE','ENTER','COMPLETE'):yield row(op,task_key=t['key'],trial_id=tid)
        yield row('PHASE_END',phase=phase)
