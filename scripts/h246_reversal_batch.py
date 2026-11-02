"""One approved saved-value batch; metadata preflight never imports the backend."""
from pathlib import Path
from types import FunctionType
import os
import sys
import h246_order_cost_analysis as prior

ROOT=Path(__file__).resolve().parents[1]
BASE='364970df33859e108d90b6ceaacde467db969fb8'
SERIES='H246_REVERSAL_SAME_Q_H2_PRECISION_V1'
ASSETS=ROOT.parent/'h246-reversal-batch-analysis-assets-20261024'
OUT=ROOT/'artifacts/resource_applicability/track_a_h246_reversal_batch_ready/2026-10-24'
EPS=[.05,.01,.005,.001]
PHYSICAL=prior.PHYSICAL
LIMITS=prior.LIMITS+['same_q_not_pure_order_causality','H2_new_endpoint_G_SE_not_computed',
 'independent_standard_Y4_not_acquired','no_interpolation_or_candidate_reselection']
NEW=('scripts/h246_reversal_batch.py','scripts/h246_reversal_batch_backend.py',
 'scripts/prepare_h246_reversal_batch.py','scripts/launch_h246_reversal_batch.py',
 'tests/test_h246_reversal_batch.py')
REUSED=tuple(dict.fromkeys(prior.NEW+prior.REUSED))
PROVIDER=prior.old.PROVIDER
PROVIDER_PATHS=('src/trottertracks/__init__.py','src/trottertracks/resource_applicability/__init__.py',
 'src/trottertracks/resource_applicability/ax2b_numerical_accounting.py',
 'src/trottertracks/resource_applicability/ax2a_preparation.py')
KINDS=('bootstrap','bundle_load','A_decomposition','B_account','u_aware_shots','axis_headroom',
 'B_decomposition','integration_summary','sampling','circuit_build','lowering','compile','transpile',
 'simulation','signal','reference_recompute','Hamiltonian_recompute','allowance','new_statistics','ranking')
require,Stop,read,save,sha,digest=prior.require,prior.Stop,prior.read,prior.save,prior.sha,prior.digest
PR_IDS={('H4',4):('M5_MIN_46221d66178510d9','M5_MIN_ab57ad5bd40dc583'),
 ('H4',8):('M5_MIN_72b305020499dd68','M5_MIN_40f4f541e6e162d2'),
 ('H6',4):('M5_MIN_645cd1aeaf52123d','M5_MIN_37ff3428bb58f5cd'),
 ('H6',8):('M5_MIN_a08e8bfb6642eb93','M5_MIN_1e215b9d2d877998')}
PR_N={'M5_MIN_46221d66178510d9':62,'M5_MIN_ab57ad5bd40dc583':33,
 'M5_MIN_72b305020499dd68':48,'M5_MIN_40f4f541e6e162d2':33,
 'M5_MIN_645cd1aeaf52123d':33,'M5_MIN_37ff3428bb58f5cd':33,
 'M5_MIN_a08e8bfb6642eb93':33,'M5_MIN_1e215b9d2d877998':33}
A_TARGETS=[]
for system,rank,fullprefix,prprefix in [('H4',10,10,3),('H6',19,19,5)]:
    for method in ('FULL','PR'):
        for q in (4,8):
            for i,formula in enumerate(('2nd','4th')):
                cid=f'{system}_FULL_{fullprefix}_{q}_{formula}' if method=='FULL' else PR_IDS[system,q][i]
                A_TARGETS.append(dict(id=cid,system=system,rank=rank,method=method,formula=formula,
                    prefix=fullprefix if method=='FULL' else prprefix,q=q,n=1 if method=='FULL' else PR_N[cid]))
B_TARGETS=prior.TARGETS[:4]


def counts():
    d=dict.fromkeys(KINDS,0)
    d.update(bootstrap=1,bundle_load=1,A_decomposition=8,B_account=16,u_aware_shots=16,
             axis_headroom=32,B_decomposition=8,integration_summary=1)
    return d


def validate(data,*,artificial=False):
    require(data['schema']==SERIES+'_INPUTS' and data['kind']==('ARTIFICIAL' if artificial else 'SAVED_REAL_VALUES'), 'DATA_KIND')
    require(data['physical']==PHYSICAL and data['A_targets']==A_TARGETS and data['B_targets']==B_TARGETS
            and data['B_epsilons']==EPS and data['limitations']==LIMITS,'FIXED_A_B_SCOPE')
    require(len(data['A_rows'])==16 and len(data['B_bases'])==4 and len(data['quoted_main_comparisons'])==22,
            'FIXED_ROWS_AND_LITERAL_OLD_COMPARISONS')
    for rows,targets,scope in ((data['A_rows'],A_TARGETS,'A_SAME_Q'),(data['B_bases'],B_TARGETS,'B_H2_PRECISION')):
        require([r['candidate'] for r in rows]==[t['id'] for t in targets],'FIXED_CANDIDATE_ORDER')
        for r,t in zip(rows,targets):
            require(r['target']==t and r['system']==t['system'] and r['subscope']==scope,'Q_PREFIX_FORMULA_SUBSCOPE')
            require(r['binding']==data['bindings'][t['system']] and r['binding']['rank']==t['rank']
                    and r['binding'].get('artificial',False) is artificial,'SYSTEM_RANK_INPUT_BINDING')
            require(r['epsilon']==(.001 if scope=='A_SAME_Q' else .0005) and r['policy_binding']==data['policy_bindings'][t['id']],
                    'EPSILON_AND_ACCEPTED_POLICY_BINDING')
            cohort='COMMON_PF_DETERMINISTIC' if t['method']=='FULL' else 'EXPLORATION'
            require(r['cohort']==cohort and r['unit']==prior.UNIT and r['bias_mode']=='USUAL_SIGNED','COHORT_UNIT_BIAS')
            require(len(r['pairs'])==t['n'] and len({p['pair_id'] for p in r['pairs']})==t['n'],'ORIGINAL_PAIRED_SAMPLE_COUNT')
            for p in r['pairs']:
                require(p['candidate']==t['id'] and p['cohort']==cohort and set(p['axes'])=={'cosine','sine'}
                        and all(type(v) is int and v>0 for v in p['axes'].values()),'PAIRED_AXES_SCOPE')
            require(set(r['C'])==set(r['N'])=={'cosine','sine'} and r['status'] in ('ELIGIBLE','INELIGIBLE','UNDETERMINED'), 'AXIS_STATUS')
            require(all(type(v) in (int,float) and v>0 for v in r['C'].values()) and all(v is None or type(v) is int and v>0 for v in r['N'].values()),'NO_ZERO_PRICE_OR_SHOT')
            require(r['status']!='ELIGIBLE' or r['G_saved'] is not None,'MISSING_G_NOT_ZERO')
            if scope=='B_H2_PRECISION':
                require(r['account_inputs']==data['account_bindings'][t['id']], 'ACCEPTED_DIAGNOSTIC_ALLOWANCE_REFERENCE')
    for a,b in zip(data['A_rows'][::2],data['A_rows'][1::2]):
        require(a['target']['q']==b['target']['q'] and a['target']['prefix']==b['target']['prefix'] and
                a['target']['formula']=='2nd' and b['target']['formula']=='4th','SAME_Q_ONLY')
    require(data['cohort_pooling'] is False and data['recompute_main_0005'] is False,'NO_POOLING_OR_OLD_REACCOUNT')
    return data


def verify(cfg,*,runtime=True,unused=True):
    require(cfg['schema']==SERIES and cfg['A_targets']==A_TARGETS and cfg['B_targets']==B_TARGETS
            and cfg['B_epsilons']==EPS and cfg['physical']==PHYSICAL and cfg['limitations']==LIMITS
            and cfg['budget']==counts() and cfg['retry'] is False and cfg['mandatory_STOP'] is True,'FIXED_BATCH_CONTRACT')
    require(Path(cfg['run_root'])==ASSETS/'BATCH_run01' and (not unused or not Path(cfg['run_root']).exists()),'NEW_UNSPENT_ROOT')
    sealed=prior.old.old.seal_check(cfg)
    require(Path(cfg['manifest'])==ASSETS/'inputs_PRIVATE.json' and sha(cfg['manifest'])==cfg['manifest_sha256'], 'INPUT_MANIFEST')
    require(not any(p.is_relative_to(Path(cfg['run_root'])) for p in sealed),'DISJOINT_INPUT_OUTPUT')
    for role,paths,commit,root in [('runner',NEW+REUSED,cfg['source_commit'],ROOT),('provider',PROVIDER_PATHS,PROVIDER,ASSETS/'provider')]:
        rows=[r for r in cfg['source_files'] if r['root']==role]
        require({r['path'] for r in rows}==set(paths),'SOURCE_CLOSURE')
        for r in rows:
            p=root/r['path'];require(r['commit']==commit and Path(r['local_path'])==p and p in sealed
                and p.read_bytes()==prior.old.recovery.git_bytes(commit,r['path']),'COMMITTED_SOURCE_BINDING')
    if runtime:
        require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_RUNTIME')
        require(str(Path(sys.executable).resolve())==cfg['runtime']['python_target'] and sha(Path(sys.executable).resolve())==cfg['runtime']['python_sha256'],'SEALED_PYTHON')
        require(sys.path==[str(ROOT/'scripts'),str(ROOT/'src/trotterlib'),str(ASSETS/'provider/src')]+cfg['runtime']['sys_path'],'SEALED_IMPORT_PATH')
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_LIBRARY_PATH'),'CLEAN_LOADER')
        prior.old.seal.verify_mapped_runtime(cfg)
    return validate(read(cfg['manifest']))


def preflight(cfg):
    d=verify(cfg)
    return dict(status='METADATA_ONLY_PASS',SOURCE=cfg['source_commit'],contract_sha256=digest(cfg),
        input_manifest_sha256=cfg['manifest_sha256'],A_comparisons=8,B_new_endpoints=16,B_comparisons=8,
        quoted_old_comparisons=22,exact_calls=counts(),exact_ledger_events=sum(counts().values()),
        new_analysis_calls=0,execution_authorized=False,valid_grants=0,mandatory_STOP=True)


def authorize(cfg,approval,grant):
    require(approval.get('issuer')=='USER_EXPLICIT' and approval.get('execution_authorized') is True
        and approval.get('contract_sha256')==digest(cfg) and approval.get('accepted_limitations')==LIMITS,
        'EXPLICIT_FIXED_BATCH_APPROVAL_REQUIRED')
    require(grant.get('purpose')==SERIES and grant.get('state')=='UNUSED' and grant.get('retry') is False
        and isinstance(grant.get('id'),str) and grant['id'].startswith(SERIES+':')
        and grant.get('contract_sha256')==digest(cfg) and grant.get('approval_sha256')==digest(approval)
        and grant.get('run_root')==cfg['run_root'] and grant.get('budget')==counts(),'NEW_UNUSED_BATCH_GRANT_REQUIRED')
    verify(cfg)


class Session:
    take=FunctionType(prior.old.Session.take.__code__,dict(prior.old.Session.take.__globals__,KINDS=KINDS),'take')
    call=prior.old.Session.call

    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant);self.cfg=cfg;self.root=Path(cfg['run_root']);self.root.mkdir(exist_ok=False)
        self.used=dict.fromkeys(KINDS,0);self.sequence=0;self.active=None
        for name in ('ledger','results'):(self.root/name).mkdir()
        save(self.root/'CONSUMED.json',dict(grant=grant,approval_sha256=digest(approval),retry=False))
        self.marker_sha=sha(self.root/'CONSUMED.json')


def execute(cfg,approval,grant):
    s=Session(cfg,approval,grant);guard=prior.old.old.ImportSeal(cfg)
    try:
        guard.install();s.take('bootstrap')
        from h246_reversal_batch_backend import pipeline
        d=s.call('bundle_load',lambda:validate(read(cfg['manifest'])))
        result=pipeline(s,d)
        require(s.used==counts(),'EXACT_BATCH_CALL_COUNTS')
        prior.old.old.Session.verify(s);prior.old.old.Session.loaded_closure(s);verify(cfg,unused=False)
        ledger=sorted((s.root/'ledger').glob('*.json'));seen=dict.fromkeys(KINDS,0)
        for i,p in enumerate(ledger,1):
            row=read(p);require(p.stem=='%06d'%i and row['amount']==1 and row['nonrefundable'] is True
                and row['contract_sha256']==digest(cfg),'LEDGER_IDENTITY');seen[row['kind']]+=1
        require(seen==s.used and len(ledger)==s.sequence,'LEDGER_COMPLETE')
        save(s.root/'results/analysis.json',result)
        save(s.root/'completion.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=cfg['source_commit'],
            contract_sha256=digest(cfg),used=s.used,result_sha256=sha(s.root/'results/analysis.json')))
        save(s.root/'STOP.json',dict(status='MANDATORY_STOP',used=s.used,retry=False))
        return result
    except BaseException as exc:
        save(s.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=s.used,retry=False));raise
    finally:guard.remove()
