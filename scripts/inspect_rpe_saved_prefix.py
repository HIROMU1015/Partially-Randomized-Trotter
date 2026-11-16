"""Metadata-only comparison of saved Full-PF m12=4 and m12=2 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FIRST=dict(commit='b4296b110b129d226dbef2743c14f725a27c3bf6',path='docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json',selector=['cases',1])
SECOND=dict(commit='1b96c8329',path='docs/research/full_pf_s2_m12_two_results_20261116/INPUTS.json',selector=['cases',0])


def read(loc):
    commit=subprocess.check_output(['git','rev-parse',loc['commit']],cwd=ROOT,text=True).strip()
    raw=subprocess.check_output(['git','show',commit+':'+loc['path']],cwd=ROOT)
    value=json.loads(raw);case=value
    for k in loc['selector']:case=case[k]
    return value,case,dict(loc,commit=commit,sha256=hashlib.sha256(raw).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old,x,s1=read(FIRST);new,y,s2=read(SECOND)
    fields=('configuration_id','binding_id','configuration','signal_packet','evaluation_target')
    physical={k:x.get(k)==y.get(k)for k in fields}
    contracts={k:old[k]==new[k]for k in ('statistical_contract','options','adopted_evaluation','kernel_source','acceleration')}
    sx=x['allocation']['shots'];sy=y['allocation']['shots']
    changed=[r['m']for r,t in zip(sx,sy)if r!=t]
    data=dict(scope='SAVED_INPUT_METADATA_ONLY_NO_COUNT_TABLES_NO_RISK_EVALUATION',
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs=[s1,s2],configuration_id=x['configuration_id'],physical_checks=physical,evaluation_checks=contracts,
        changed_rounds=changed,m0_m11_shots_identical=sx[:12]==sy[:12],last_shots=[sx[-1],sy[-1]],
        maximum_frontier_states=new['options']['total_state_refinement']['maximum_frontier_states'],
        maximum_transition_pairs=new['options']['total_state_refinement']['maximum_transition_pairs'],
        checkpoint_compatibility='CONDITIONAL_ON_SAME_VALIDATED_COUNT_PREFIX_AND_RUNTIME',
        historical_frontier_snapshot_available=False,existing_aggregate_DP_result_cannot_reconstruct_states=True,
        old_results_recomputed=False,real_Q3_calls=0)
    assert all(physical.values())and all(contracts.values())and changed==[12]
    a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))


if __name__=='__main__':main()
