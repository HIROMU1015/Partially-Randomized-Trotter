"""Phase D presentation of this job's own summaries; no new science or ranking."""
import csv
import json
from pathlib import Path
from support import HERE, save

def main():
    terminal=json.loads((HERE/'result/terminal.json').read_text())
    if terminal['status']!='C1_5A_EXPLORATORY_COMPLETE':
        raise RuntimeError('incomplete result cannot be reported as a completed comparison')
    rows=[]
    for method in ('ordinary','full_return','closed_P5_full'):
        s=json.loads((HERE/'result'/method/'summary.json').read_text())
        t=json.loads((HERE/'result'/method/'terminal.json').read_text())
        setup=sum(v['CPU_seconds'] for v in t['setup_stages'].values())
        rows.append(dict(method=method,completed=s['completed'],accepted=s['accepted_paths'],zeros=s['valid_zero_paths'],
            accepted_fraction=s['accepted_fraction_observed'],empirical_M2=s['empirical_weight_second_moment'],
            observed_max_weight=s['maximum_observed_weight'],
            logical_instructions_per_classical_trial=s['logical_per_classical_trial'].get('instructions',0),
            logical_depth_per_classical_trial=s['logical_per_classical_trial'].get('depth',0),
            logical_instructions_per_accepted=s['logical_per_accepted_path'].get('instructions') if s['logical_per_accepted_path'] else None,
            setup_CPU_seconds=setup,online_batch_CPU_seconds=s['worker_batch_CPU_seconds'],
            setup_plus_online_batch_CPU_seconds=setup+s['worker_batch_CPU_seconds'],
            method_hierarchy_CPU_seconds=t['all_method_hierarchy_CPU'],
            online_wall_seconds=t['online_wall_seconds'],method_total_wall_seconds=t['total_wall_seconds'],
            online_CPU_per_classical_trial=s['worker_batch_CPU_seconds']/1024,
            online_CPU_per_accepted=s['worker_batch_CPU_seconds']/s['accepted_paths'] if s['accepted_paths'] else None,
            max_worker_RSS_KiB=s['max_worker_RSS_KiB'],
            saved_method_bytes=sum(p.stat().st_size for p in (HERE/'result'/method).rglob('*') if p.is_file()),
            angle_cache_hits=s['angle_cache_hits'],angle_cache_misses=s['angle_cache_misses']))
    save(HERE/'comparison.json',dict(scope='UNCERTIFIED_EXPLORATORY_TAIL_ONLY_PLUS_SCALAR',rows=rows,
        final_resource_winner=None,sufficient_N=None,native_cost=None,state_preparation=None,readout=None))
    with (HERE/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Ordinary','Full','Closed P5'];fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    fields=[('logical_instructions_per_classical_trial','Raw logical instructions / classical trial'),
            ('setup_plus_online_batch_CPU_seconds','Observed setup + online batch CPU (s)'),
            ('empirical_M2','Observed weight second moment')]
    for ax,(field,title) in zip(axes,fields):
        ax.bar(labels,[r[field] for r in rows]);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=15)
        ax.set_ylim(bottom=0)
    fig.suptitle('H4 R1: 1024 fixed pseudorandom paths/method; tail + scalar only; no native/measurement/N',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
