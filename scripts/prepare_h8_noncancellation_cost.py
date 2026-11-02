"""Metadata evidence audit and UNSEALED cost-only config/INVALID grant only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_noncancellation_cost_contract as r


def prepare(source,mapping,directory):
    r.require(subprocess.check_output([*r.GIT,'rev-parse','HEAD'],cwd=r.ROOT).decode().strip()==source,'FIXED_COST_SOURCE_HEAD')
    files=r.pinned_inputs(r.read(mapping));audit,_,_=r.proof_metadata(files)
    target=r.original.home_path(directory);r.require(not target.exists(),'EXCLUSIVE_PREPARATION_DIRECTORY')
    config=dict(schema='H8_NONCANCELLATION_COST_CONFIG_V3',SOURCE=source,approved=False,
        runtime_authorization=False,execution_sealed=False,plan_sha256=r.c.digest(r.plan()),
        source_hashes={name:r.sha(r.ROOT/name) for name in r.SOURCES},python=r.PYTHON,python_sha256=r.sha(r.PYTHON),
        environment=r.original.environment(),inputs=files,resource_policy=r.RESOURCE_POLICY,
        protected_files=[],protected_worktrees=[],output=str(target/'noncancellation_cost_output01'),
        control=str(target/'control01'),lock=str(target/'NC_COST_ONE_SHOT'))
    r.path_set(config);r.verify(config)
    grant=dict(schema='H8_NONCANCELLATION_COST_GRANT_V3',scope=r.SCOPE,SOURCE=source,
        approved=False,runtime_authorization=False,execution_sealed=False,valid=False,approval_text=None,
        authorized_stages=['COST_COMPARISON'],plan_sha256=config['plan_sha256'],config_sha256=r.c.digest(config),
        input_contract_sha256=r.sha(r.ROOT/r.INPUT_CONTRACT),resource_policy=r.RESOURCE_POLICY,call_caps=r.CAPS,
        allowed_cpus=[],one_shot=True,automatic_retry=False,new_shot_calculation=False,next_stage_authorized=False,
        absolute_launch_command=None)
    target.mkdir(mode=0o700)
    for name,value in [('config_UNAPPROVED_v3.json',config),('grant_INVALID_DRAFT_v3.json',grant),
                       ('evidence_binding_v3.json',audit),('execution_plan_UNAPPROVED_v3.json',r.plan())]:
        with (target/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return dict(status='READY_TECHNICALLY_UNSEALED_UNAPPROVED',SOURCE=source,plan_sha256=config['plan_sha256'],
        config_sha256=r.c.digest(config),grant_sha256=r.c.digest(grant),new_H8_cost_calls=0,new_valid_grants=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    for k in ('source','input-mapping','directory'):p.add_argument('--'+k)
    a=p.parse_args()
    if not a.prepare:
        if a.source or a.input_mapping or a.directory:p.error('--prepare required')
        print(json.dumps(r.plan(),indent=2));return
    if not all((a.source,a.input_mapping,a.directory)):p.error('all metadata arguments required')
    print(json.dumps(prepare(a.source,a.input_mapping,a.directory)))


if __name__=='__main__':main()
