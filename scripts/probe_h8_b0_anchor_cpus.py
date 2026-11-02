"""Read-only three-window CPU proposal, before any grant. Never launch or retry.

No existing grant is read or modified, no affinity is changed, no scientific
module is imported. This snapshot cannot authorize or replace the fresh gate.
"""
import argparse
import json
import os
from pathlib import Path

import h8_b0_anchor_runtime as r


def inventory(affinity):
    result={}
    for cpu in sorted(affinity):
        p=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        siblings=set()
        for span in (p/'thread_siblings_list').read_text().strip().split(','):
            ends=list(map(int,span.split('-')));siblings.update(range(ends[0],ends[-1]+1))
        result[cpu]=dict(package_id=(p/'physical_package_id').read_text().strip(),
            core_id=(p/'core_id').read_text().strip(),siblings=sorted(siblings))
    return result


def choose(windows,cores):
    if len(windows)!=3:raise ValueError('THREE_FIXED_PROPOSAL_WINDOWS_REQUIRED')
    observed=[{x['cpu']:x for x in w['CPUs']} for w in windows]
    choices={}
    for cpu,core in sorted(cores.items()):
        samples=[table.get(sibling) for table in observed for sibling in core['siblings']]
        if not samples or any(x is None or x['status']!='PASS' for x in samples):continue
        key=(core['package_id'],core['core_id'])
        row=dict(cpu=cpu,**core,maximum_busy_fraction=max(x['busy_fraction'] for x in samples))
        previous=choices.get(key)
        if previous is None or cpu<previous['cpu']:choices[key]=row
    ordered=sorted(choices.values(),key=lambda x:(x['maximum_busy_fraction'],x['cpu']))
    selected=ordered[:6] if len(ordered)>=6 else []
    return dict(status='CPU_PROPOSAL_ONLY_NOT_A_FRESH_GATE' if selected else 'STOP_NO_SIX_STABLE_PHYSICAL_CORES',
        cores=selected,proposed_cpus=[x['cpu'] for x in selected],grant_issued=False,
        CPUs_reassigned=False,execution_authorized=False,science_started=False,
        policy='WORST_BUSY_FRACTION_OVER_THREE_FIXED_ONE_SECOND_WINDOWS_THEN_CPU_ID',
        further_windows_or_retries=False)


def probe():
    cores=inventory(os.sched_getaffinity(0));cpus={s for row in cores.values() for s in row['siblings']}
    windows=[r.measure_idle(cpus) for _ in range(3)]
    return dict(kind='READ_ONLY_PRE_GRANT_CPU_PROPOSAL',windows=windows,**choose(windows,cores))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('NEW_PROPOSAL_OUTPUT_REQUIRED')
    value=probe()
    with args.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='windows'}))


if __name__=='__main__':main()
