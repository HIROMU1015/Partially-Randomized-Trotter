"""Bind the completed new input; use only cores released by sibling stages."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import h4_followup_contract as c
from prepare_h4_followup import grant
from prepare_h4_native import physical_cpus


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--source',required=True)
    a=p.parse_args();config=deepcopy(c.read(a.private/'config_RBTF.json'));parent=a.private/'F_run01'
    end=c.read(parent/'native_exit.json');gate=c.read(parent/'output/final_gate.json')
    c.require(end['exit_code']==0 and end['wait4_complete'] and end['same_parent_reaped'] and gate['status']=='PASS','F_NORMAL_COMPLETION')
    receipt=c.read(parent/'output/tol_only_receipt.json');active=[]
    occupied=set()
    for stage,cpus in config['cpu_selection_observation']['allocation'].items():
        if not (a.private/(stage+'_run01')/'native_exit.json').exists():occupied.update(cpus);active.append(stage)
    cpus=[cpu for score,cpu in physical_cpus() if cpu not in occupied][:36]
    c.require(bool(cpus),'RELEASED_PHYSICAL_CORES_FOR_FP')
    for role,name in [('tol_snapshot','tol_only_snapshot.npz'),('tol_receipt','tol_only_receipt.json')]:
        origin=parent/'output'/name;target=a.private/'inputs'/name
        c.require(not target.exists(),'EXCLUSIVE_INPUT');shutil.copyfile(origin,target)
        config['mapping'][role]=str(target)
        config['inputs'].append(dict(role=role,path=str(target),sha256=c.sha(target),bytes=target.stat().st_size))
    config['binding'].update(hamiltonian=receipt['hamiltonian_hash'],state=receipt['state_hash'],snapshot=receipt['sha256'])
    config['tol_only_parent']=dict(SOURCE=end['SOURCE'],native_exit_sha256=c.sha(parent/'native_exit.json'),final_gate_sha256=c.sha(parent/'output/final_gate.json'))
    name='scripts/prepare_h4_followup_fp.py';config['source_hashes'][name]=c.sha(c.ROOT/name)
    config['FP_cpu_selection']=dict(cpus=cpus,active_siblings=active,occupied_cpus=sorted(occupied),overlap=False)
    c.save(a.private/'config_FP.json',config);g=grant(config,'FP',a.source,a.private/'FP_run01',cpus)
    public=c.ROOT/'artifacts/resource_applicability/track_a_h4_followup/2026-10-18'
    c.save(public/'FP_binding.json',dict(SOURCE=a.source,grant_digest=c.digest(g),config_digest=c.digest(config),input_binding=config['binding'],
        parent=config['tol_only_parent'],workers=len(cpus),CPU_selection=config['FP_cpu_selection'],source_files=config['source_hashes']))
    print('FP prepared on',len(cpus),'released cores; active siblings',active)


if __name__=='__main__':main()
