"""H8 DBA contracts. H6 helpers are used only for byte I/O and exact arithmetic."""
from h6_dba_contract import ROOT, require, digest, sha, read, save, Writer, axis_check, fresh
import hashlib
import os
from pathlib import Path
import subprocess

SCOPE='H8_DBA_20261018'
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROFILE='622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
PROVIDER='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
MODEL='8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
CAPS=dict(snapshot_load=1,saved_preparation_decode=1,native_evolution_build=1,
          control_probe=4,measured_wrapper_build=2,probe_wrapper_copy=2,qpy_dump=2,
          qpy_load=2,compile=2,model_error_check=2)


def plan():
    return dict(scope=SCOPE,system='linear_H8_1Angstrom_STO3G_tol_only_DF28_sector4900',T=.8,P=0,
        D=[dict(id='B0_p15_q8_2nd',method='B0',prefix=15,q=8,formula='2nd'),
           dict(id='B0I_p16_q4_4th',method='B0I',prefix=16,q=4,formula='4th')],
        axes=['cosine','sine'],calls_per_case=CAPS,per_axis_STOP_at_or_above='1/50',
        relative_error_denominator='measured',case_parallel=True,axis_compile='cosine PASS then sine',
        B=dict(prefix=7,r=4,K=4,q=[4,8],strata=['s2_multi','s4'],n=8,max_axes=64,
               status='ADAPTER_NOT_YET_SEALED'),
        A=dict(signal_cells=16,status='ADAPTER_NOT_YET_SEALED',cost_stage_automatic=False),
        refit=False,retry=False,cost_reaccounting=False,new_SCF_DF_state=False,
        mandatory_STOP=True,transient_PSI_action='observe_only',unrelated_worktree_changes='observe_only')


def verify(config,grant):
    require(grant.get('scope')==SCOPE and grant.get('stage')=='D','H8_D_SCOPE')
    require(grant.get('approved_by_user') is True and grant.get('one_shot') is True
            and grant.get('retry') is False and grant.get('next_stage_authorized') is False,
            'EXPLICIT_SCOPED_EXECUTION_APPROVAL_REQUIRED')
    require(grant['plan_digest']==digest(plan()) and grant['config_digest']==digest(config),'PLAN_CONFIG_BINDING')
    case=next((x for x in plan()['D'] if x['id']==grant['case_id']),None)
    require(case is not None,'FIXED_D_CASE')
    for name,h in config['source_hashes'].items():
        require(sha(ROOT/name)==h,'SOURCE_CHANGED:'+name)
        raw=subprocess.check_output(['git','show',grant['SOURCE']+':'+name],cwd=ROOT)
        require(hashlib.sha256(raw).hexdigest()==h,'GIT_SOURCE_BINDING:'+name)
    for row in config['inputs']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],
                'INPUT_CHANGED:'+row['role'])
    require(config['provider_source']==PROVIDER and config['profile_sha256']==PROFILE
            and config['compiler_hash']==COMPILER,'PROVIDER_PROFILE_COMPILER')
    for name,h in config['provider_hashes'].items():
        require(sha(Path(config['provider_root'])/name)==h,'PROVIDER_CHANGED:'+name)
    for name,h in config['runtime_file_hashes'].items():require(sha(name)==h,'RUNTIME_CHANGED:'+name)
    cpus=grant['worker_cpus']
    require(len(cpus)==2 and len(set(cpus))==2 and set(cpus)<=os.sched_getaffinity(0),'WORKER_CPUS')
    observed=fresh()
    require(observed['available_memory_bytes']>grant['memory_admission_estimate_bytes'],'MEMORY_ADMISSION')
    require(observed['disk_free_bytes']>2**30,'OUTPUT_SPACE_ADMISSION')
    return dict(status='PASS',SOURCE=grant['SOURCE'],case_id=case['id'],inputs=len(config['inputs']),
        provider_files=len(config['provider_hashes']),runtime_files=len(config['runtime_file_hashes']),fresh=observed)


class Permit:
    def __init__(self,config,grant):
        self.config,self.grant,self.output=config,grant,Path(grant['output'])
        self.calls={k:0 for k in CAPS}
    def require(self,scope):
        # Narrow utility capability translation only; no legacy grant is used.
        import h8_key_compile_contract as old
        import h8_b0_anchor_contract as anchor
        require(scope in (SCOPE,old.SCOPE,anchor.SCOPE),'UNEXPECTED_HELPER_CAPABILITY')
    def take(self,name):
        require(name in CAPS and self.calls[name]<CAPS[name],'CALL_BUDGET:'+name)
        self.calls[name]+=1
        save(self.output/f'reservation_{name}_{self.calls[name]}.json',dict(
            case_id=self.grant['case_id'],call=name,nonrefundable=True,SOURCE=self.grant['SOURCE']))
