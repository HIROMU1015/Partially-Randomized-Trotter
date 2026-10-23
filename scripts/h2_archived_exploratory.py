"""One-shot saved-value exploratory comparison; control plane is stdlib only."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import sys
import h2_common_production_contract as old
import h2_diagnostics_binding as seal
import h2_accepted_diagnostics as accepted
import h2_pr_identity_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
SERIES = 'H2_FOUR_METHOD_ARCHIVED_COST_EXPLORATORY_V1'
BASE = '041511aa567c288bcead32d0bf7579fd7f434e44'
B = 'ca29dd9ddc8a09cace84310f0dfdf7124652e07a'
B_DIR = 'artifacts/resource_applicability/track_a_h2_runtime_cost_reuse/2026-10-23/'
B2_DIR = 'artifacts/resource_applicability/track_a_h2_saved_cost_static_join_b2/2026-10-23/'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
ASSETS = ROOT.parent/'h2-archived-exploratory-assets-20261023'
OUT_DIR = 'artifacts/resource_applicability/track_a_h2_archived_exploratory_ready/2026-10-23'
NEW = ('scripts/h2_archived_exploratory.py','scripts/h2_archived_exploratory_backend.py',
       'scripts/prepare_h2_archived_exploratory.py','scripts/launch_h2_archived_exploratory.py')
REUSED = ('scripts/h2_accepted_compare_backend.py','scripts/h2_accepted_compare_contract.py',
          'scripts/h2_accepted_diagnostics.py','scripts/h2_common_production_contract.py',
          'scripts/h2_diagnostics_binding.py','scripts/h2_pr_identity_recovery.py',
          'src/trotterlib/h2_common_native_policy.py')
PROVIDER_PATHS = ('src/trottertracks/__init__.py',
 'src/trottertracks/resource_applicability/__init__.py',
 'src/trottertracks/resource_applicability/h6_matched_accounting_v1.py',
 'src/trottertracks/resource_applicability/ax2b_numerical_accounting.py',
 'src/trottertracks/resource_applicability/ax2a_preparation.py')
KINDS = ('bootstrap','bundle_load','allowance','empirical_allowance','accounting',
         'u_aware_shots','axis_headroom','cost_statistics','resource_row','metric_summary','ranking',
         'sampling','circuit_build','lowering','compile','transpile','simulate','signal',
         'reference_recompute','prepare','setup','primitive_evaluation')
LIMITS = ('Full_complete_parameter_stream','historical_actual_load',
          'provider_dynamic_noninterference','Py312_vs_Py311_output_equivalence',
          'A_independent_absolute_diagonal','single_state_not_operator',
          'finite_sample_not_population_mean','empirical_allowance_not_certificate')
FULL = {'B1_p3_q4_2nd':('H2_FULL_3_4_2nd',1655), 'B1_p3_q2_4th':('H2_FULL_3_2_4th',2463)}
policy, require, Stop = old.policy, old.require, old.Stop
digest, sha, save = policy.digest, old.sha, old.save

def read(path):return recovery.loads(Path(path).read_bytes())

def counts():
    result=dict.fromkeys(KINDS,0)
    result.update(bootstrap=1,bundle_load=1,allowance=4,empirical_allowance=4,accounting=4,
        u_aware_shots=4,axis_headroom=8,cost_statistics=4,resource_row=4,metric_summary=8,ranking=1)
    return result


def validate_costs(pr, full):
    require(len(pr)==134 and len(full)==4,'FIXED_138_COST_AXES')
    keys={(x['task'],x['axis']) for x in pr};require(len(keys)==134,'DUPLICATE_PR_AXIS')
    output={}
    for candidate,n in zip(policy.CANDIDATES[2:],(33,34)):
        rows=[x for x in pr if x['candidate']==candidate.id]
        require(len(rows)==2*n and len({x['task'] for x in rows})==n,'PR_CANDIDATE_COHORT')
        for row in rows:
            require(row['cohort']=='EXPLORATION' and row['runtime_id']==
                'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f' and
                row['unit']=='compiled_RZ_per_full_Hadamard_wrapper_axis' and
                type(row['RZ']) is int and row['RZ']>0,'SAVED_PR_COST_SCOPE')
        costs=[]
        for i,task in enumerate(dict.fromkeys(x['task'] for x in rows)):
            pair=[x for x in rows if x['task']==task]
            require(len(pair)==2 and {x['axis'] for x in pair}=={'cosine','sine'},'PAIRED_PR_AXES')
            costs.append(dict(sample=i,task=task,**{x['axis']:x['RZ'] for x in pair}))
        output[candidate.id]=costs
    for candidate in policy.CANDIDATES[:2]:
        saved,rz=FULL[candidate.id];rows=[x for x in full if x['saved_candidate']==saved]
        require(len(rows)==2 and {x['axis'] for x in rows}=={'cosine','sine'},'FIXED_FULL_AXIS_PAIR')
        for row in rows:
            require(row['requested_candidate']==candidate.id and row['RZ']==rz and type(row['RZ']) is int and
                row['cohort']=='COMMON_PF_DETERMINISTIC_20261022' and
                row['runtime_id']==pr[0]['runtime_id'] and
                row['unit']=='compiled_RZ_per_full_Hadamard_wrapper_axis','COMMON_FULL_SAVED_COST_ONLY')
        output[candidate.id]=[dict(sample=0,task=saved,**{x['axis']:x['RZ'] for x in rows})]
    return output


def saved_inputs(cfg):
    manifest=read(cfg['manifest'])
    require(manifest['schema']==SERIES+'_INPUTS' and manifest['B2_identity_commit']==BASE,'FIXED_B2_INPUTS')
    require(manifest['physical']==policy.PHYSICAL and manifest['candidates']==[asdict(c) for c in policy.CANDIDATES],
            'FIXED_SCIENTIFIC_TARGET')
    costs={}
    for name in ('PR_costs','Full_costs'):
        row=manifest['cost_files'][name]
        require(row['commit']==B and row['path']==B_DIR+('saved_PR_cost_objects_134.json' if name=='PR_costs'
                else 'saved_common_PF_cost_objects_4.json'),'FIXED_SAVED_COST_INPUT_SOURCE')
        raw=recovery.git_bytes(row['commit'],row['path'])
        require(sha(row['local_path'])==row['sha256'] and Path(row['local_path']).read_bytes()==raw,'SAVED_COST_INPUT_BYTES')
        costs[name]=read(row['local_path'])['rows']
    paired=validate_costs(costs['PR_costs'],costs['Full_costs'])
    origin=read(cfg['origin_contract'])
    require(digest(origin)==accepted.ORIGIN_CONTRACT,'ACCEPTED_DIAGNOSTICS_CONTRACT')
    receipt=seal.accepted_diagnostics_receipt(cfg['origin_run_root'],origin,completion_sha256=accepted.ORIGIN_COMPLETION)
    require(manifest['origin_completion']==accepted.ORIGIN_COMPLETION,'ACCEPTED_COMPLETION_BINDING')
    diagnostics={};reference=None
    for candidate in policy.CANDIDATES:
        cid=candidate.id;row=read(Path(cfg['origin_run_root'])/'results'/(cid+'_diagnostics.json'))
        inputs=accepted.adaptive(row['inputs']);inputs.validate(inputs.reference_id)
        require(inputs.domain=='SAVED_SCIENCE' and inputs.candidate_id==cid,'SAVED_DIAGNOSTIC_DOMAIN')
        require(inputs.input_binding_sha256==digest(dict(binding=origin['binding'],files=origin['files'],series=old.SERIES)),
                'SAVED_DIAGNOSTIC_INPUT_BINDING')
        if reference is None:reference=inputs.reference_id
        require(reference==inputs.reference_id and row['reference_binding']['reference_id']==reference,'COMMON_REFERENCE')
        require(row['reference_binding']['reference_recomputed'] is False,'NO_NEW_REFERENCE')
        require(set(row['reference_binding']['bias'])=={'real','imag'},'SAVED_BIAS_REQUIRED')
        for v in row['reference_binding']['bias'].values():policy.number(v,'saved_bias',nonnegative=False)
        policy.number(row['log_B'],'saved_log_B')
        prep=read(Path(cfg['origin_run_root'])/'results'/('preparation_'+str(candidate.prefix)+'.json'))
        require(row['preparation_sha256']==prep['preparation_sha256'] and
                row['operations_sha256']==prep['operations_sha256'],'PREFIX_PREPARATION_BINDING')
        diagnostics[cid]=dict(inputs=inputs,primary=dict(log_B=row['log_B']),
                              bias=row['reference_binding']['bias'])
    return dict(costs=paired,diagnostics=diagnostics,reference_id=reference,origin_receipt=receipt)


def verify(cfg, *, runtime=True):
    require(set(cfg)=={'schema','source_commit','physical','candidates','budget','limitations','files','source_files',
        'manifest','origin_contract','origin_run_root','runtime','run_root','retry','mandatory_STOP'},'EXPLORATORY_CONTRACT_SCHEMA')
    require(cfg['schema']==SERIES and policy.strict_equal(cfg['physical'],policy.PHYSICAL) and
            policy.strict_equal(cfg['candidates'],[asdict(x) for x in policy.CANDIDATES]),'FIXED_EXPLORATORY_SCOPE')
    require(policy.strict_equal(cfg['budget'],counts()) and tuple(cfg['limitations'])==LIMITS and
            cfg['retry'] is False and cfg['mandatory_STOP'] is True,'EXACT_BUDGET_LIMITS_STOP')
    require(Path(cfg['run_root'])==ASSETS/'EXPLORATORY_run01' and not Path(cfg['run_root']).exists(),'NEW_UNUSED_RUN_ROOT')
    require(str(Path(cfg['manifest']).resolve())==cfg['manifest'] and
            Path(cfg['manifest']).is_relative_to(ASSETS),'DEDICATED_PRIVATE_MANIFEST')
    seen=old.seal_check(cfg)
    for role,commit in (('runner',cfg['source_commit']),('provider',PROVIDER)):
        rows=[r for r in cfg['source_files'] if r['root']==role]
        expected=set(NEW+REUSED) if role=='runner' else set(PROVIDER_PATHS)
        require({r['path'] for r in rows}==expected,'EXACT_SOURCE_SET')
        for row,(_,raw) in zip(rows,seal.git_batch(commit,[r['path'] for r in rows])):
            p=Path(row['local_path']);require(p in seen and sha(p)==row['sha256'] and
                p.read_bytes()==raw,'PINNED_SOURCE_BYTES')
    if runtime:
        rt=cfg['runtime']
        require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_I_S_B_REQUIRED')
        require(str(Path(sys.executable).resolve())==rt['python_target'] and sha(Path(sys.executable).resolve())==rt['python_sha256'],
                'SEALED_ANALYSIS_PYTHON')
        require(sys.path==[str(ROOT/'scripts')]+rt['sys_path'],'SEALED_IMPORT_PATH')
        require(not os.environ.get('LD_LIBRARY_PATH') and not os.environ.get('LD_PRELOAD'),'CLEAN_LOADER_ENVIRONMENT')
        require(not Path(rt['cache_prefix']).exists(),'NO_BYTECODE_CACHE')
        seal.verify_mapped_runtime(cfg)
    return saved_inputs(cfg)


def preflight(cfg):
    data=verify(cfg)
    require(not any(n.split('.')[0] in {'numpy','qiskit','scipy','symengine','trottertracks'} for n in sys.modules),
            'PREFLIGHT_NO_NUMERICAL_KERNEL_IMPORT')
    return dict(status='READY_AWAITING_EXPLICIT_SCIENTIFIC_APPROVAL',contract_sha256=digest(cfg),
        input_manifest_sha256=sha(cfg['manifest']),source_commit=cfg['source_commit'],
        fixed_cost_axes=dict(PR=134,Full=4),saved_diagnostics_available=4,
        technical_ready=True,execution_authorized=False,actual_scientific_calls=dict.fromkeys(KINDS,0),
        strict_COMPARE='BLOCKED',limitations=list(LIMITS),mandatory_STOP=True)


def authorize(cfg, approval, grant):
    require(set(approval)=={'issuer','execution_authorized','contract_sha256','accepted_limitations'} and
        approval['issuer']=='USER_EXPLICIT' and approval['execution_authorized'] is True and
        approval['contract_sha256']==digest(cfg) and approval['accepted_limitations']==list(LIMITS),
        'SEPARATE_EXPLICIT_EXPLORATORY_APPROVAL_REQUIRED')
    preflight(cfg)
    require(set(grant)=={'id','purpose','state','contract_sha256','approval_sha256','run_root','budget','retry'} and
        grant['purpose']==SERIES and grant['state']=='UNUSED' and grant['retry'] is False and
        type(grant['id']) is str and grant['id'].startswith(SERIES+':') and
        len(grant['id'])>len(SERIES)+10 and grant['contract_sha256']==digest(cfg) and
        grant['approval_sha256']==digest(approval) and grant['run_root']==cfg['run_root'] and
        policy.strict_equal(grant['budget'],counts()),'NEW_NAMESPACE_UNUSED_ONE_SHOT_GRANT_REQUIRED')
    prior=read(Path(cfg['origin_run_root'])/'CONSUMED.json')['grant']
    require(grant['id']!=prior['id'],'OLD_GRANT_REUSE_FORBIDDEN')


class Session:
    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant);self.cfg=cfg;self.root=Path(cfg['run_root'])
        self.root.mkdir(exist_ok=False);self.used=dict.fromkeys(KINDS,0);self.sequence=0;self.active=None
        save(self.root/'CONSUMED.json',dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=grant,
             approval_sha256=digest(approval),contract_sha256=digest(cfg)))
        self.marker_sha=sha(self.root/'CONSUMED.json')
        for name in ('results','ledger','delegated_receipts'): (self.root/name).mkdir()

    def take(self,kind):
        require(kind in KINDS and self.used[kind]<self.cfg['budget'][kind],'UNAPPROVED_OR_EXCESS_CALL:'+kind)
        self.sequence+=1;save(self.root/'ledger'/('%06d.json'%self.sequence),
            dict(kind=kind,amount=1,nonrefundable=True,contract_sha256=digest(self.cfg)))
        self.used[kind]+=1

    def call(self,kind,fn,*args,**kwargs):
        self.take(kind);previous=self.active;self.active=kind
        try:return fn(*args,**kwargs)
        finally:self.active=previous


def pipeline(session,backend):
    data=session.call('bundle_load',backend.load)
    rows=[];details=[]
    for candidate in policy.CANDIDATES:
        d=data['diagnostics'][candidate.id];costs=data['costs'][candidate.id]
        u=session.call('allowance',backend.allowance,candidate,d)
        row=session.call('accounting',backend.account,candidate,d,u,costs)
        detail=session.call('cost_statistics',backend.statistics,candidate,row,costs)
        rows.append(row);details.append(detail)
    result=session.call('ranking',backend.rank,rows)
    return dict(schema=SERIES+'_RESULT',exploratory_only=True,strict_COMPARE='BLOCKED',
                delegated_comparison=result,methods=details,limitations=list(LIMITS),science_GO=False)


def execute(cfg,approval,grant):
    session=Session(cfg,approval,grant);guard=old.ImportSeal(cfg);backend=None
    try:
        guard.install();session.take('bootstrap')
        from h2_archived_exploratory_backend import Backend
        backend=Backend(session);result=pipeline(session,backend)
        require(session.used==cfg['budget'],'INCOMPLETE_EXACT_LEDGER')
        # Input/source checks after the run; output namespace now exists by design.
        for row in cfg['files']:require(sha(row['path'])==row['sha256'],'END_INPUT_SOURCE_RUNTIME_DRIFT')
        seal.verify_mapped_runtime(cfg)
        save(session.root/'results/comparison.json',result)
        save(session.root/'completion.json',dict(status='EXPLORATORY_COMPLETE_NOT_FORMAL_COMPARE',
            contract_sha256=digest(cfg),used=session.used,result_sha256=sha(session.root/'results/comparison.json'),
            marker_sha256=session.marker_sha,source_commit=cfg['source_commit'],retry=False,mandatory_STOP=True))
        save(session.root/'STOP.json',dict(status='MANDATORY_STOP_EXPLORATORY_COMPLETE',used=session.used,
            strict_COMPARE='BLOCKED',mandatory_STOP=True))
        return result
    except BaseException as exc:
        save(session.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=session.used,
             retry=False,mandatory_STOP=True));raise
    finally:
        if backend is not None:backend.close()
        guard.remove()
