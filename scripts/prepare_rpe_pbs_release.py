#!/usr/bin/env python3
"""Add a configured PBS target to a new pinned RomuTail release, without a run."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess


def read(path): return json.loads(Path(path).read_text())
def put(path, value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(base_service, profile_path, settings_path, source_root, source, output):
    source_root,output=Path(source_root).resolve(),Path(output).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()!=source:
        raise ValueError('Release checkout HEAD differs')
    if output.exists(): raise ValueError('Use a new release directory; existing releases remain pinned')
    output.mkdir(parents=True)
    service=read(base_service);profile=read(profile_path);settings=read(settings_path)
    if settings['transport_source']!=source or profile['pbs']['worker_source']!=source:
        raise ValueError('Controller/PBS transport pins must match this new release')
    settings['workers']=[dict(host_id=profile['host_id'],profile=profile)]
    put(output/'PBS_PROFILE.json',profile);put(output/'PBS_SETTINGS.json',settings)
    prior_fixed=deepcopy(service.get('fixed_settings')or{})
    base_round=read(service['round']);distributed=base_round['executor']['distributed']
    admission=distributed.get('host_admission')
    policy=read(admission)if isinstance(admission,str)else deepcopy(admission or dict(hosts={}))
    policy.setdefault('hosts',{})[profile['host_id']]=dict(accepting=True,optional=True)
    put(output/'HOST_ADMISSION.json',policy)
    for key,path in [('round',service['round']),('absolute_kappa_route.round',service.get('absolute_kappa_route',{}).get('round'))]:
        if not path: continue
        value=read(path);before=deepcopy(value)
        dist=value['executor']['distributed'];dist['pbs']=deepcopy(settings)
        dist.setdefault('preparation_profiles',{})[profile['host_id']]=profile
        dist['host_admission']=str(output/'HOST_ADMISSION.json')
        dist.setdefault('host_stage_capabilities',{})[profile['host_id']]=['q3','signal','compile','audit']
        groups=dist.setdefault('auto_allocation',{}).setdefault('priority_groups',[['local','server2','server1'],['local0-wsl']])
        if not any(profile['host_id']in group for group in groups):groups[0].append(profile['host_id'])
        assert value.get('science_approved')==before.get('science_approved')
        assert value.get('max_new_q3_per_method')==before.get('max_new_q3_per_method')
        assert dist.get('preparation')==before['executor']['distributed'].get('preparation')
        target=output/('ROUND.default.json'if key=='round'else'ROUND.absolute-kappa.json')
        put(target,value)
        if key=='round':service['round']=str(target)
        else:service['absolute_kappa_route']['round']=str(target)
        prior_fixed[str(target)]=sha(target)
    for path in (output/'PBS_PROFILE.json',output/'PBS_SETTINGS.json',output/'HOST_ADMISSION.json'):
        prior_fixed[str(path)]=sha(path)
    service.update(code_root=str(source_root),code_source=source,config_path=str(output/'SERVICE.json'),fixed_settings=prior_fixed)
    put(output/'SERVICE.json',service)
    put(output/'DEPLOYMENT.json',dict(base_service=str(Path(base_service).resolve()),controller_source=source,
        pbs_transport_source=source,computation_source=service['computation_source'],
        service=str(output/'SERVICE.json'),remote_root=settings['remote_root'],ssh_alias=settings['ssh_alias'],
        runtime=settings['python'],resources=settings['resources'],scientific_execution=False,
        task_seconds=settings.get('task_seconds',{}),next_scientific_instruction_required=True))
    return service


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-service','profile','settings','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True);a=p.parse_args()
    result=prepare(a.base_service,a.profile,a.settings,a.source_root,a.source,a.output)
    print(json.dumps(dict(service=result['config_path'],started=False)))


if __name__=='__main__':main()
