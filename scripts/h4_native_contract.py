"""Fixed H4 native D/A/N scope. Metadata only; importing never starts science."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
from h6_dba_contract import digest, sha, read, save, require, Writer, fresh

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'H4_NATIVE_DAN_20261018'
EPS = [.05, .01, .005, .001, .0005]
FACTORS = [1., 10.]


def cell(method, prefix, q, formula='2nd', r=None, K=None):
    return dict(id=f'{method}_p{prefix}_q{q}_{formula}' + (f'_r{r}_K{K}' if r else ''),
                method=method, prefix=prefix, q=q, formula=formula, r=r, K=K, R=q*r if r else None)


def plan():
    a = [cell('B2', p, q, r=4, K=k) for p in (2,4) for q in (1,4,8) for k in (2,4)]
    a += [cell('B2',3,1,r=2,K=2), cell('B2',3,4,r=8,K=4)]
    a += [cell('B2',3,8,r=4,K=k) for k in (2,4)]
    d = [cell(method,p,q,f) for p in (2,3,4,6,9,11) for q in (1,2,4,8)
         for f in ('2nd','4th') for method in ('B0','B0I')]
    return dict(scope=SCOPE, target='linear_H4_1Angstrom_STO3G_legacy_DF12_sector36', T=.8,
        epsilons=EPS, allowance_factors=FACTORS, bias_modes=['USUAL_SIGNED','EMPIRICAL_NONCANCELLATION'],
        D=dict(cells=d, base_B0_evaluations=48, analytic_B0I_overlays=48,
            maximum_direct_configurations=10, maximum_compile=20, maximum_control_probes=40,
            selection='union of proxy minima over 5 epsilon x 2 bias, allowance factor 1; lexical ties',
            proxy='saved full-PF same q/formula measured RZ per axis * (prefix+1)/13 times saved shots',
            proxy_is_cost_model=False, proxy_error_threshold=None,
            unmeasured_domain_optimum_certified=False, first_cosine_before_sine=True),
        A=dict(cells=a, signal=16, compile=0, sampling=0, q16='DEFERRED', cost_stage_auto_started=False),
        N=dict(candidate_scope='ORIGINAL_NATIVE_FIXED16_ONLY', usual_reproduction_gate=128,
               new_signal=0, sampling=0, compile=0, strict_bias_certificate=None),
        new_SCF_DF_state=False, new_full_H_reference=False, geometry_campaign=False,
        rare_pilot=False, new_PR_cost_sampling=False, retry=False, refit=False,
        mandatory_STOP_per_stage=True, population_mean_certified=False,
        resource_policy='fresh available physical cores/RAM; transient PSI is observation, no unrelated worktree freeze')


def noncancel(record):
    out = deepcopy(record)
    parts = record['error_decomposition']['signed_parts']
    require(bool(parts), 'MISSING_ERROR_COMPONENTS')
    for axis in ('real','imag'):
        values = [v[axis] for v in parts.values()]
        total = record['error_decomposition']['total_signed'][axis]
        require(all(math.isfinite(v) for v in values+[total]), 'NONFINITE_COMPONENT')
        require(math.isclose(math.fsum(values), total, rel_tol=1e-12, abs_tol=1e-12), 'COMPONENT_CLOSURE')
        out['error_decomposition']['total_signed'][axis] = math.fsum(abs(v) for v in values)
    return out


def select_direct(records, old_costs):
    """Predeclared screening only. It never declares an unmeasured optimum."""
    selected = set(); evidence = []
    for mode, field in [('USUAL_SIGNED','precision_rows'),
                        ('EMPIRICAL_NONCANCELLATION','noncancellation_precision_rows')]:
        for eps in EPS:
            choices = []
            for record in records:
                a = record[field][str(eps)]['1.0']; cell_ = record['cell']
                if not a['cost_acquisition_eligible']: continue
                anchor = f'B1_p12_q{cell_["q"]}_{cell_["formula"]}'
                require(anchor in old_costs, 'MISSING_FULL_PF_SCREENING_ANCHOR')
                score = sum(Fraction(old_costs[anchor][axis]) * Fraction(cell_['prefix']+1,13)
                            * a['axes'][channel]['shots']
                            for axis,channel in [('cosine','real'),('sine','imag')])
                choices.append((score,cell_['id']))
            if choices:
                score,id_ = min(choices); selected.add(id_)
                evidence.append(dict(bias_mode=mode,epsilon=eps,cell_id=id_,screening_score=str(score)))
            else: evidence.append(dict(bias_mode=mode,epsilon=eps,cell_id=None))
    require(len(selected)<=10,'REGISTERED_DIRECT_COMPILE_COUNT')
    return sorted(selected), evidence


def verify(config, grant, stage, *, runtime=True):
    require(stage in ('D','A','N') and grant.get('stage')==stage, 'STAGE_BINDING')
    require(grant.get('approved_by_user') is True and grant.get('one_shot') is True
            and grant.get('retry') is False and grant.get('next_stage_authorized') is False,
            'EXPLICIT_H4_EXECUTION_APPROVAL_REQUIRED')
    require(grant['plan_digest']==digest(plan()) and grant['config_digest']==digest(config),'PLAN_CONFIG_CHANGED')
    source=grant['SOURCE'];require(len(source)==40,'SEALED_SOURCE')
    for name,h in config['source_hashes'].items():
        require(sha(ROOT/name)==h,'SOURCE_CHANGED:'+name)
        blob=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        require(__import__('hashlib').sha256(blob).hexdigest()==h,'SOURCE_GIT_BINDING:'+name)
    for row in config['inputs']:
        p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'INPUT_CHANGED:'+row['role'])
    for name,h in config['provider_hashes'].items():
        require(sha(Path(config['provider_root'])/name)==h,'PROVIDER_CHANGED:'+name)
    if runtime:
        for name,h in config['runtime_file_hashes'].items(): require(sha(name)==h,'RUNTIME_CHANGED:'+name)
    f=fresh()
    require(set(grant['worker_cpus'])<=set(f['affinity']),'AVAILABLE_CPUS')
    require(f['available_memory_bytes']>grant['memory_admission_estimate_bytes'],'INSUFFICIENT_MEMORY_TO_START')
    require(f['disk_free_bytes']>2**30,'OUTPUT_SPACE')
    return dict(status='PASS',SOURCE=source,stage=stage,inputs=len(config['inputs']),
                provider_files=len(config['provider_hashes']),runtime_files=len(config['runtime_file_hashes']) if runtime else 0,
                fresh=f, old_grants_ledgers_STOP_modified=False)
