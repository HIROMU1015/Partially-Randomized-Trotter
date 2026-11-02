"""Four explicitly requested H4 gaps; metadata only, no import-time science."""
from copy import deepcopy
from pathlib import Path
import subprocess
from h4_native_contract import cell, digest, sha, read, save, require, Writer, fresh

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'H4_FOLLOWUP_RARE_TIME_DF_20261018'


def panel(q, rank=12, *, policy=False):
    import math
    p, tail = math.ceil(3*rank/4), math.ceil(rank/4)
    if policy:
        return [cell('B0I',p,q,f) for f in ('2nd','4th')] + [cell('B1',rank,q,f) for f in ('2nd','4th')] + [cell('B2',tail,q,r=4,K=4)]
    return [cell('B0',p,q),cell('B0I',p,q)] + [cell('B1',rank,q,f) for f in ('2nd','4th')] + [cell('B2',tail,q,r=4,K=2)]


def plan():
    r = [cell('B2',3,q,r=1,K=k) for q in (1,2,4,8) for k in (2,4)]
    r += [cell('B2',3,16,r=rr,K=k) for rr in (2,4) for k in (2,4)]
    b = [dict(id=f'q{q}_{s}',cell=cell('B2',3,q,r=4,K=4),stratum=s) for q in (4,8) for s in ('s2_multi','s4')]
    return dict(scope=SCOPE,system='linear_H4_1Angstrom_STO3G',
        R=dict(cells=r,signal=12,compile=0,sampling=0),
        B=dict(groups=b,n=8,trajectories=32,compile=64,control_probe=16,conditional_only=True),
        T=dict(times=[dict(T=t,q=q,delta=.2,cells=panel(q),reuse=(t==.8)) for t,q in ((.4,2),(.8,4),(1.6,8))],
               n=32,new_trajectories=64,new_compile=144,new_signal_recurrences=8,analytic_overlays=2),
        F=dict(geometry=[['H',[0.,0.,float(i)]] for i in range(4)],basis='sto-3g',charge=0,multiplicity=1,
               scf_conv_tol=1e-9,scf_max_cycle=50,df_tol=1e-8,final_rank_supplied=False,cutoff=0.,
               coefficient_order='decomposer_generation_order',signed_weights=True,
               projection='weighted Hermitian projection; full-Fock N=8 budget 9.9e-11 Hartree; engineering only',
               state='lowest sector eigenvector, canonical phase; no certification',integral_build=1,df_decomposition=1),
        FP=dict(T=.8,q=4,n=32,rank_rule='actual tol-only returned rank',
                cells_rule='ceil(3rank/4) B0I S2/S4; full rank S2/S4; ceil(rank/4) PR r4 K4',
                compile=72,signal_recurrences=5,legacy_panel_reused=True,global_reoptimization=False),
        epsilons=[.05,.01,.005,.001,.0005],allowance_factors=[1.,10.],
        bias_modes=['USUAL_SIGNED','EMPIRICAL_NONCANCELLATION'],
        strict_bias_certificate=None,population_mean_certified=False,
        retry=False,refit=False,geometry_series=False,mandatory_STOP_per_stage=True,
        stage_grants_independent=True,old_STOP_modified=False,
        scientific_STOP='nonfinite, source/input/runtime/paired/control/forward mismatch; native exit not zero',
        resources='fresh physical cores and available RAM; single-thread workers; PSI observation only')


def seed(stage, task, replica, binding):
    require(stage in ('B','T','FP') and type(replica) is int and 0<=replica<32,'SEED_SCOPE')
    return int(digest([SCOPE,stage,task,replica,binding])[:16],16)


def verify(config,grant,stage,*,runtime=True):
    require(stage in ('R','B','T','F','FP') and grant['stage']==stage,'STAGE_BINDING')
    require(grant.get('approved_by_user') is True and grant.get('one_shot') is True
        and grant.get('retry') is False and grant.get('next_stage_authorized') is False,'EXPLICIT_H4_FOLLOWUP_APPROVAL')
    require(grant['plan_digest']==digest(plan()) and grant['config_digest']==digest(config),'PLAN_CONFIG_CHANGED')
    source=grant['SOURCE'];require(len(source)==40,'SEALED_SOURCE')
    for name,h in config['source_hashes'].items():
        require(sha(ROOT/name)==h,'SOURCE_CHANGED:'+name)
        blob=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        require(__import__('hashlib').sha256(blob).hexdigest()==h,'SOURCE_GIT_BINDING:'+name)
    for row in config['inputs']:
        path=Path(row['path']);require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'INPUT_CHANGED:'+row['role'])
    for name,h in config['provider_hashes'].items():require(sha(Path(config['provider_root'])/name)==h,'PROVIDER_CHANGED:'+name)
    if runtime:
        for name,h in config['runtime_file_hashes'].items():require(sha(name)==h,'RUNTIME_CHANGED:'+name)
    f=fresh();require(set(grant['worker_cpus'])<=set(f['affinity']),'CPU_AVAILABILITY')
    require(f['available_memory_bytes']>grant['memory_admission_estimate_bytes'],'INSUFFICIENT_MEMORY_TO_START')
    require(f['disk_free_bytes']>2**30,'OUTPUT_SPACE')
    if stage=='FP':require('tol_snapshot' in config['mapping'],'SEALED_TOL_ONLY_INPUT_REQUIRED')
    return dict(status='PASS',SOURCE=source,stage=stage,inputs=len(config['inputs']),
        provider_files=len(config['provider_hashes']),runtime_files=len(config['runtime_file_hashes']) if runtime else 0,fresh=f)
