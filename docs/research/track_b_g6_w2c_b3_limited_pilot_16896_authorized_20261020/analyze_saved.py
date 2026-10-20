"""Read-only adopted pilot cost/outcome summary. No generator or science imports."""
from pathlib import Path
from collections import Counter
import json,statistics,sys,hashlib
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import offline as O
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_result import decode


def describe(values):
    return dict(count=len(values),sum=sum(values),min=min(values) if values else None,
        median=statistics.median(values) if values else None,max=max(values) if values else None)

def analyze(root):
    out=Path(root)/'output';report=O.read(out/'final.json');axes=[];pairs=[];fine=[]
    for r in report['pairs']:
        task=r['task'];identity={k:task[k] for k in ('condition','phase_index','phase','pass_name','arm','layer')}
        coarse=r['coarse'];setup={k:v for k,v in coarse.items() if k.startswith('setup.')}
        pairs.append(dict(**identity,key=task['key'],status=r['status'],pid=r['pid'],cpu_ns=r['cpu_ns'],wall_ns=r['wall_ns'],
            process_END_own_cpu_ns=r['process_cpu_ns'],setup_coarse=setup,coarse=coarse,reconciliation=r['reconciliation'],
            control_rpc=r['control_rpc'],memory_baseline=r['memory_baseline'],memory=r['memory'],io=r['io'],output_bytes=r['output_bytes']))
        fine.append(dict(**identity,functions=r['fine'],reconciliation=r['fine_reconciliation'],coarse_added=False))
        path=out/(task['key']+'.jsonl')
        if not path.exists():continue
        by_axis={'X':[],'Y':[]}
        for line in path.read_bytes().splitlines():
            try:p=decode(json.loads(line)['payload'])
            except Exception:continue
            if 'trial_cost' in p and p.get('axis') in by_axis:by_axis[p['axis']].append(p)
        for axis,rows in by_axis.items():
            classes=Counter(p.get('classification',p.get('evidence',{}).get('adapted',{}).get('classification')) for p in rows)
            blocks=[]
            for size in (16,64,256):
                cpu=[sum(p['trial_cost']['cpu_ns'] for p in rows[i:i+size]) for i in range(0,len(rows)-size+1,size)]
                wall=[sum(p['trial_cost']['wall_ns'] for p in rows[i:i+size]) for i in range(0,len(rows)-size+1,size)]
                blocks.append(dict(block_size=size,complete_blocks=len(cpu),CPU_ns=describe(cpu),wall_ns=describe(wall),
                    cumulative_prefix_CPU_ns=sum(p['trial_cost']['cpu_ns'] for p in rows[:size]) if len(rows)>=size else None,
                    cumulative_prefix_wall_ns=sum(p['trial_cost']['wall_ns'] for p in rows[:size]) if len(rows)>=size else None,
                    independent_repetitions=False))
            stages={}
            for p in rows:
                for name,v in p['trial_cost']['coarse_exclusive'].items():
                    d=stages.setdefault(name,Counter());d.update(v)
            axes.append(dict(**identity,axis=axis,completed=len(rows),classifications=dict(classes),
                online_trial_CPU_ns=sum(p['trial_cost']['cpu_ns'] for p in rows),online_trial_wall_ns=sum(p['trial_cost']['wall_ns'] for p in rows),
                online_stages={k:dict(v) for k,v in stages.items()},block_statistics=blocks,
                setup_assignment='shared pair setup; not arbitrarily split between axes',
                window='generator through retention before stream serialization/write/IPC COMPLETE',
                unobserved_paths=[x for x in ('ACCEPTED','RAW_NONREDUCED','COIN_REJECT') if not classes[x]]))
    return dict(status=report['classification'],completed=report['states']['COMPLETED'],states=report['states'],
        condition_phases=report['phases'],axis_records=axes,pair_records=pairs,fine_records=fine,
        outer=O.read(out/'host_outer.json') if (out/'host_outer.json').exists() else None,
        final_fields={k:v for k,v in report.items() if k not in ('pairs','phases')},
        limitations=['single repetition per condition','fixed S then P1 order','same prefix blocks not independent',
          'no CPU population CI or IID proof','no general arm ranking or Full-N extrapolation',
          'baseline includes coarse timers, gates, serialization, trace and retention',
          'fine observer diagnostic contains profiling overhead; not an additive partition with coarse',
          'pair walls overlap in P1 and must not be summed as global wall',
          'setup CPU shared by X/Y; axis online excludes IPC and stream serialization/write',
          'unisolated observer and IPC components remain null/residual','resource observations are sampled'])

if __name__=='__main__':
    value=analyze(sys.argv[1]);path=Path(sys.argv[2]);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),classification=value['status'],completed=value['completed'],axis_records=len(value['axis_records']))))
