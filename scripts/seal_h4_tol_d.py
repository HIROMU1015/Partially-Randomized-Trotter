"""Seal only H4 DF10 D metadata. This tool cannot grant or launch science."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h4_tol_main_contract as c

BASE='80652ab6b42cf0c11ee97485214027f507b1bf0a'
SCIENCE_SOURCE='84db3909639b34ec521004c3d8f7ef9b5e6ad851'
PHASE0=c.ROOT/'artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19'
OLD_PRIVATE=c.ROOT/'.h4_tol_main_private'
PRIVATE=OLD_PRIVATE/'D_seal01_20261019'
PUBLIC=c.ROOT/'artifacts/resource_applicability/track_a_h4_tol_d_seal/2026-10-19'
IMMUTABLE=('SOURCE','binding','plan','source_hashes','inputs','runtime_hashes','provider_hashes',
    'provider_roots','python','python_sha256','native_config','FP_evidence','saved_reference',
    'font_seed','import_paths')


def check_scope(contract, draft):
    c.require(contract['stage']==draft['stage']=='D','D_ONLY_NO_A_ACTIVATION')
    c.require(contract['SOURCE']==SCIENCE_SOURCE and contract['plan']==c.plan('D'),'FIXED_D_SOURCE_PLAN')
    c.same_binding(contract['binding'])
    c.require(all(contract[name]==draft[name] for name in IMMUTABLE),'UNCHANGED_PHASE0_SCIENTIFIC_CONTRACT')
    c.require(contract['approved_by_user'] is False and contract['runtime_authorization'] is False
        and contract['actual_scientific_calls']==0 and not contract['next_stage_authorized'], 'NO_SCIENTIFIC_APPROVAL')
    c.require(contract['automatic_retry'] is False,'NO_AUTOMATIC_RETRY')
    for key in ('output','lock'):
        c.require(not Path(contract[key]).exists(),'UNUSED_D_NAMESPACE')


def verify_bytes(contract):
    c.require(contract['source_hashes']==c.source_closure(),'COMPLETE_SOURCE_CLOSURE')
    for name,expected in contract['source_hashes'].items():
        blob=subprocess.check_output(['git','show',SCIENCE_SOURCE+':'+name],cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==expected,'FIXED_GIT_SOURCE')
    for row in contract['inputs']:
        c.require(c.sha(row['path'])==row['sha256'],'INPUT_BYTES:'+row['role'])
    for group in ('provider_hashes','runtime_hashes'):
        for path,expected in contract[group].items():
            c.require(c.sha(path)==expected,group+'_BYTES')
    c.verify_provider_tree(contract)
    c.require(c.sha(sys.executable)==contract['python_sha256'],'FIXED_PYTHON')
    bound={row['path'] for row in contract['inputs']}
    c.require({contract['native_config'],contract['FP_evidence'],contract['saved_reference'],
        contract['font_seed']['path']}<=bound,'ACTUALLY_USED_METADATA_BOUND')
    fp=c.read(contract['FP_evidence']); cells={x['id']:x for x in c.registrations()['D']}
    for row in fp['measurements']:
        if row['cell_id'] in c.FP_DETERMINISTIC:
            c.reuse_measurement(row,c.BINDING,cells[row['cell_id']],
                accepted_result_sha256=fp['result_sha256'],source=fp['SOURCE'])
    c.require({r['cell']['id'] for r in fp['signals']}>=set(c.FP_SIGNALS_D),'SAVED_SIX_D_SIGNALS')
    c.require({r['cell_id'] for r in fp['measurements'] if r['cell_id'] in c.FP_DETERMINISTIC}
        ==set(c.FP_DETERMINISTIC),'SAVED_FOUR_D_PAIRED_COSTS')
    return dict(SOURCE_files=len(contract['source_hashes']),inputs=len(contract['inputs']),
        provider_files=len(contract['provider_hashes']),runtime_files=len(contract['runtime_hashes']),
        saved_signal_records=6,saved_configurations=4,saved_compile_axes=8)


def resource_snapshot():
    # Observation only. Admission and assignments must be refreshed after approval.
    def read(path):
        try: return Path(path).read_text().strip()
        except OSError as error: return dict(unavailable=type(error).__name__)
    cpus=sorted(os.sched_getaffinity(0)); topology={}
    for cpu in cpus:
        path=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        topology[str(cpu)]={name:read(path/name) for name in ('physical_package_id','core_id','thread_siblings_list')}
    group=Path('/sys/fs/cgroup')/read('/proc/self/cgroup').split('::')[-1].lstrip('/')
    groups={str(p):{name:read(p/name) for name in ('cpu.max','memory.max','memory.current','cpuset.cpus.effective')}
        for p in [group,*group.parents] if str(p).startswith('/sys/fs/cgroup')}
    return dict(time=time.time(),affinity=cpus,topology=topology,cgroup=groups,
        meminfo=read('/proc/meminfo'),loadavg=read('/proc/loadavg'),
        PSI={name:read('/proc/pressure/'+name) for name in ('cpu','memory','io')},
        disk_free_bytes=shutil.disk_usage(c.ROOT).free,admitted_for_execution=False,
        diagnostic_only=True,agent_imposed_CPU_RAM_worker_wall_caps=None,
        fresh_admission_required_after_explicit_approval=True,shared_environment_changes=False)


def refuse(contract_path, grant_path, directory):
    guard=directory/'refusal_guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text("import sys\ndef guard(event,args):\n if event=='import' and (args[0].startswith(('numpy','scipy','qiskit','trottertracks')) or args[0]=='h4_tol_main_science'): raise AssertionError('SCIENCE_IMPORT')\nsys.addaudithook(guard)\n")
    rows=[]
    for witness in (False,True):
        command=[sys.executable,str(c.ROOT/'scripts/run_h4_tol_main.py'),'--stage','D',
            '--contract',str(contract_path),'--grant',str(grant_path)]+(['--witness'] if witness else [])
        result=subprocess.run(command,capture_output=True,text=True,
            env=dict(os.environ,PYTHONPATH=str(guard),PYTHONDONTWRITEBYTECODE='1'))
        c.require(result.returncode!=0 and 'SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED' in result.stderr
            and 'SCIENCE_IMPORT' not in result.stderr,'BEFORE_SCIENCE_REFUSAL')
        prefix='witness' if witness else 'direct'
        (directory/(prefix+'.stderr')).write_text(result.stderr)
        rows.append(dict(witness=witness,exit_code=result.returncode,
            stderr_sha256=c.sha(directory/(prefix+'.stderr')),refused_before_science=True))
    return rows


def seal(private=PRIVATE,public=PUBLIC):
    c.require(not private.exists() and not public.exists(),'NEW_D_TECHNICAL_SEAL_NAMESPACE')
    old=c.read(OLD_PRIVATE/'D_contract_DRAFT_PRIVATE.json')
    artifact_manifest=c.read(PHASE0/'artifact_manifest.json')
    for name,expected in artifact_manifest['files'].items():
        path=PHASE0/name
        c.require(c.sha(path)==expected,'PRESERVED_PHASE0_ARTIFACT')
        blob=subprocess.check_output(['git','show',BASE+':'+str(path.relative_to(c.ROOT))],cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==expected,'FIXED_PHASE0_COMMIT')
    records=c.read(OLD_PRIVATE/'preservation_manifest_PRIVATE.json')
    for row in records:
        for path in {row['path'],row['copy']}:
            c.require(c.sha(path)==row['sha256'],'PRIOR_RECORD_PRESERVATION')
    contract=deepcopy(old)
    contract.update(status='TECHNICALLY_SEALED_PENDING_EXPLICIT_D_APPROVAL',
        pending=['SEPARATE_EXPLICIT_D_SCIENTIFIC_APPROVAL','FRESH_RESOURCE_ADMISSION_AT_ACTIVATION'],
        output=str(private/'run01/output'),lock=str(private/'run01/ONE_SHOT.json'),
        fixed_Phase0_commit=BASE,scientific_conditions_changed=False,
        saved_state_solver_normalization_difference_preserved=True)
    check_scope(contract,old); counts=verify_bytes(contract)
    resource=resource_snapshot()
    private.mkdir(parents=True);public.mkdir(parents=True)
    c.save(private/'resource_observation_PRIVATE.json',resource)
    c.save(private/'D_contract_SEALED_PENDING_APPROVAL_PRIVATE.json',contract)
    grant=c.invalid_grant('D',contract);c.save(private/'D_grant_INVALID_PRIVATE.json',grant)
    refusal=refuse(private/'D_contract_SEALED_PENDING_APPROVAL_PRIVATE.json',
        private/'D_grant_INVALID_PRIVATE.json',private)
    check_scope(contract,old)
    c.require(not Path(contract['output']).parent.exists(),'NO_SCIENTIFIC_ATTEMPT_CREATED')
    c.save(public/'D_contract_SEALED_PENDING_APPROVAL.json',dict(
        status=contract['status'],SOURCE=SCIENCE_SOURCE,fixed_Phase0_commit=BASE,
        binding=contract['binding'],plan_digest=c.digest(contract['plan']),
        registration='Phase0 candidate_registrations.json D unchanged',registered_candidates=190,
        calls=contract['plan']['calls'],counts=counts,contract_digest=c.digest(contract),
        pending=contract['pending'],approved_by_user=False,runtime_authorization=False,
        source_hashes=contract['source_hashes'],input_binding_digest=c.digest(contract['inputs']),
        provider_binding_digest=c.digest(contract['provider_hashes']),runtime_binding_digest=c.digest(contract['runtime_hashes']),
        python_sha256=contract['python_sha256'],one_shot=True,automatic_retry=False,next_stage_authorized=False,
        scientific_conditions_changed=False,saved_state_solver_normalization_difference_preserved=True))
    c.save(public/'D_grant_INVALID.json',grant)
    c.save(public/'resource_observation.json',dict(allowed_CPUs_observed=len(resource['affinity']),
        proposed_pool='all freshly admitted physical CPUs within existing affinity/quota; at most stage tasks',
        actual_worker_assignment=None,admitted_for_execution=False,
        private_observation_sha256=c.sha(private/'resource_observation_PRIVATE.json'),
        agent_imposed_CPU_RAM_worker_wall_caps=None,fresh_admission_required_after_explicit_approval=True,
        PSI_diagnostic_only=True,shared_environment_changes=False))
    c.save(public/'refusal_report.json',dict(cases=2,rows=refusal,all_refused_before_science=True,
        scientific_output=False,one_shot_marker=False,native_driver_started=False))
    audit=dict(status='D_TECHNICAL_SEAL_COMPLETE_PENDING_USER_APPROVAL',SOURCE=SCIENCE_SOURCE,
        technical_preparer_sha256=c.sha(__file__),
        technical_preparer_commit=subprocess.check_output(['git','log','-1','--format=%H','--',
            'scripts/seal_h4_tol_d.py'],cwd=c.ROOT,text=True).strip(),
        fixed_Phase0_commit=BASE,counts=counts,preserved_records=len(records),
        old_grants_ledgers_markers_STOP_unchanged=True,scientific_calls=0,grant_valid=False,
        new_scientific_result=None,A_status='PHASE0_ONLY_PENDING_FORMAL_D_ACCEPTANCE_AND_NEXT_INSTRUCTION',
        A_seeds_assigned=False,A_science_authorized=False,mandatory_STOP=True,
        next_stage_authorized=False,automatic_retry=False,AI_subagents_used=False)
    c.save(public/'completion_audit.json',audit)
    c.save(public/'artifact_manifest.json',dict(files={p.name:c.sha(p) for p in sorted(public.glob('*.json'))
        if p.name!='artifact_manifest.json'},scientific_results_generated=False))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(seal()))
