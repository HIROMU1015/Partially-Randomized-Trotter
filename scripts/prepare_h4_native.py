"""Bind existing private inputs and the current user's H4 execution approval."""
import argparse
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import h4_native_contract as c


def closure():
    """Follow project-local imports and literal sidecar module references."""
    pending=['h4_native_contract.py','h4_native_science.py','run_h4_native.py','h4_native_witness.py']
    seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=c.ROOT/'scripts'/name
        c.require(path.is_file(),'LOCAL_SOURCE_CLOSURE:'+name);seen.add(name)
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[x.name.split('.')[0]+'.py' for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.split('.')[0]+'.py']
            elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):
                names=[node.value]
            pending.extend(x for x in names if Path(x).name==x and (c.ROOT/'scripts'/x).is_file() and x not in seen)
    return {'scripts/'+name:c.sha(c.ROOT/'scripts'/name) for name in sorted(seen)}


def cpu_observation():
    values={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            v=list(map(int,fields[1:9]));values[int(fields[0][3:])]=(sum(v),v[3]+v[4])
    return values


def physical_cpus():
    a=cpu_observation();time.sleep(.5);b=cpu_observation();cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        key=((root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip())
        total=b[cpu][0]-a[cpu][0];idle=b[cpu][1]-a[cpu][1]
        score=1-idle/total if total else 0.
        if key not in cores or (score,cpu)<cores[key]:cores[key]=(score,cpu)
    return sorted(cores.values())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--source',required=True);a=p.parse_args()
    config=c.read(a.private/'config_prepared.json');config['source_hashes']=closure()
    cpus=physical_cpus();c.require(len(cpus)>=3,'PHYSICAL_CORES_FOR_INDEPENDENT_STAGES')
    # Task counts, fresh available topology and observed contention determine concurrency.
    available=[x[1] for x in cpus];nA=min(16,max(1,(len(available)-1)//2));nD=min(10,len(available)-nA-1)
    allocations=dict(A=available[:nA],D=available[nA:nA+nD],N=available[nA+nD:nA+nD+1])
    config['cpu_selection_observation']=dict(physical_cores=len(cpus),observations=[dict(cpu=cpu,busy_fraction=score) for score,cpu in cpus],
        allocation=allocations,transient_contention_is_STOP=False)
    c.save(a.private/'config_SEALED_PRIVATE.json',config)
    public=c.ROOT/'artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18';public.mkdir(parents=True,exist_ok=True)
    c.save(public/'plan_v1.json',c.plan());c.save(public/'source_closure.json',dict(SOURCE=a.source,files=config['source_hashes']))
    c.save(public/'input_binding.json',dict(binding=config['binding'],provider_SOURCE=config['provider_source'],
        input_files=[{k:v for k,v in x.items() if k not in ('path','origin_path')} for x in config['inputs']],
        runtime_files=len(config['runtime_file_hashes']),provider_files=len(config['provider_hashes']),
        compiler_options=config['compiler_options'],profile_reference=config['profile_reference']))
    approvals=[]
    for stage in ('D','A','N'):
        run=a.private/(stage+'_run01');run.mkdir()
        grant=dict(stage=stage,SOURCE=a.source,approved_by_user=True,one_shot=True,retry=False,
            next_stage_authorized=False,plan_digest=c.digest(c.plan()),config_digest=c.digest(config),
            output=str(run/'output'),lock=str(run/'one_shot_lock.json'),worker_cpus=allocations[stage],
            memory_admission_estimate_bytes=int((1.25*len(allocations[stage])+.25)*2**30),
            approval_text='その方針でよいので、検証を実際に開始するところまで進めてよい',
            authorization_scope='H4 native 1Angstrom D/A/noncancellation only; no geometry, rare pilot or new PR cost sampling',
            created_at=datetime.now(timezone.utc).isoformat())
        c.save(run/'grant.json',grant)
        approvals.append(dict(stage=stage,SOURCE=a.source,grant_digest=c.digest(grant),
            config_digest=c.digest(config),plan_digest=c.digest(c.plan()),workers=len(allocations[stage]),
            authorization_text=grant['approval_text'],new_independent_grant=True))
    c.save(public/'execution_bindings.json',dict(stages=approvals,mandatory_STOP_per_stage=True))
    print(__import__('json').dumps(dict(SOURCE=a.source,source_files=len(config['source_hashes']),allocation=allocations,
        science_started=False)))


if __name__=='__main__':main()
