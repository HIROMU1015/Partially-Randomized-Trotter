"""New stage-specific grants for the user's explicit four-gap H4 instruction."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import h4_followup_contract as c
from prepare_h4_native import physical_cpus


def closure():
    pending=['h4_followup_contract.py','h4_followup_science.py','h4_tol_only_input.py',
             'run_h4_followup.py','h4_followup_witness.py','prepare_h4_followup.py']
    seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=c.ROOT/'scripts'/name;c.require(path.is_file(),'CLOSURE:'+name);seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name.split('.')[0]+'.py' for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.split('.')[0]+'.py']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):names=[node.value]
            pending.extend(n for n in names if Path(n).name==n and (c.ROOT/'scripts'/n).is_file() and n not in seen)
    result={'scripts/'+n:c.sha(c.ROOT/'scripts'/n) for n in sorted(seen)}
    name='src/trotterlib/h8_rare_tail_selection.py';result[name]=c.sha(c.ROOT/name)
    return result


def grant(config,stage,source,run,cpus):
    run.mkdir()
    g=dict(stage=stage,SOURCE=source,approved_by_user=True,one_shot=True,retry=False,next_stage_authorized=False,
        plan_digest=c.digest(c.plan()),config_digest=c.digest(config),output=str(run/'output'),lock=str(run/'one_shot_lock.json'),
        worker_cpus=cpus,memory_admission_estimate_bytes=int((1.5*len(cpus)+1)*2**30),
        approval_text='この未実施についてH4でも行ってほしい',
        approval_scope='r1/q16 signal; conditional rare costs; fixed-delta time panel; tol-only input and fixed representative comparison',
        created_at=datetime.now(timezone.utc).isoformat())
    c.save(run/'grant.json',g)
    return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--tol-completion',type=Path);a=p.parse_args();a.private.mkdir(parents=True,exist_ok=True)
    public=c.ROOT/'artifacts/resource_applicability/track_a_h4_followup/2026-10-18';public.mkdir(parents=True,exist_ok=True)
    if a.tol_completion:
        config=c.read(a.private/'config_RBTF.json');config=deepcopy(config)
        directory=a.tol_completion;end=c.read(directory/'native_exit.json');gate=c.read(directory/'output/final_gate.json')
        c.require(end['exit_code']==0 and end['wait4_complete'] and gate['status']=='PASS','F_NORMAL_COMPLETION')
        receipt=c.read(directory/'output/tol_only_receipt.json')
        for role,name in [('tol_snapshot','tol_only_snapshot.npz'),('tol_receipt','tol_only_receipt.json')]:
            origin=directory/'output'/name;target=a.private/'inputs'/name;c.require(not target.exists(),'EXCLUSIVE_INPUT')
            shutil.copyfile(origin,target);config['mapping'][role]=str(target)
            config['inputs'].append(dict(role=role,path=str(target),sha256=c.sha(target),bytes=target.stat().st_size))
        config['binding'].update(hamiltonian=receipt['hamiltonian_hash'],state=receipt['state_hash'],snapshot=receipt['sha256'])
        config['tol_only_parent']=dict(SOURCE=end['SOURCE'],native_exit_sha256=c.sha(directory/'native_exit.json'),final_gate_sha256=c.sha(directory/'output/final_gate.json'))
        cpus=[cpu for score,cpu in physical_cpus()][:36]
        config['FP_cpu_selection']=cpus;c.save(a.private/'config_FP.json',config)
        g=grant(config,'FP',a.source,a.private/'FP_run01',cpus)
        c.save(public/'FP_binding.json',dict(SOURCE=a.source,grant_digest=c.digest(g),config_digest=c.digest(config),
            input_binding=config['binding'],parent=config['tol_only_parent'],workers=len(cpus)))
        print('FP grant prepared, workers',len(cpus));return
    old=Path('/home/AbeHiromu/projects/h4-native-validation/20261018/private')
    config=c.read(old/'config_SEALED_PRIVATE.json');config['schema']=c.SCOPE
    folder=a.private/'inputs';folder.mkdir();newrows=[];paths={}
    for row in config['inputs']:
        target=folder/(row['role']+'_'+Path(row['path']).name);shutil.copyfile(row['path'],target)
        c.require(c.sha(target)==row['sha256'],'COPIED_INPUT_IDENTITY');paths[row['path']]=str(target)
        newrows.append(dict(row,path=str(target)))
    config['inputs']=newrows
    config['mapping']={k:paths.get(v,v) for k,v in config['mapping'].items()}
    config['font_seed']['path']=paths.get(config['font_seed']['path'],config['font_seed']['path'])
    for role,path in [('saved_D',old/'D_run01/output/result.json'),('saved_D_signals',old/'D_run01/output/signals.json'),
                      ('saved_D_final_gate',old/'D_run01/output/final_gate.json'),('saved_D_native_exit',old/'D_run01/native_exit.json')]:
        target=folder/(role+'.json');shutil.copyfile(path,target);config['mapping'][role]=str(target)
        config['inputs'].append(dict(role=role,path=str(target),sha256=c.sha(target),bytes=target.stat().st_size))
    config['source_hashes']=closure()
    observations=physical_cpus();available=[cpu for score,cpu in observations]
    c.require(len(available)>=4,'FOUR_STAGE_PHYSICAL_CPU_ADMISSION')
    maxima=dict(R=12,B=32,T=36,F=1);allocation={k:[] for k in maxima}
    for stage in allocation:allocation[stage].append(available.pop(0))
    while available and any(len(allocation[k])<maxima[k] for k in allocation):
        for k in allocation:
            if available and len(allocation[k])<maxima[k]:allocation[k].append(available.pop(0))
    config['cpu_selection_observation']=dict(physical_cores=len(observations),allocation=allocation,
        observations=[dict(cpu=cpu,busy_fraction=score) for score,cpu in observations],PSI_STOP=False)
    c.save(a.private/'config_RBTF.json',config)
    bindings=[]
    for stage,cpus in allocation.items():
        g=grant(config,stage,a.source,a.private/(stage+'_run01'),cpus)
        bindings.append(dict(stage=stage,SOURCE=a.source,grant_digest=c.digest(g),config_digest=c.digest(config),workers=len(cpus)))
    c.save(public/'plan_v1.json',c.plan());c.save(public/'source_closure.json',dict(SOURCE=a.source,files=config['source_hashes']))
    c.save(public/'input_binding.json',dict(binding=config['binding'],provider_SOURCE=config['provider_source'],
        inputs=[{k:v for k,v in x.items() if k not in ('path','origin_path')} for x in config['inputs']],
        compiler_options=config['compiler_options'],runtime_files=len(config['runtime_file_hashes']),provider_files=len(config['provider_hashes'])))
    c.save(public/'execution_bindings.json',dict(stages=bindings,FP='separate grant after F normal completion',mandatory_STOP_per_stage=True))
    print(__import__('json').dumps(config['cpu_selection_observation']))


if __name__=='__main__':main()
