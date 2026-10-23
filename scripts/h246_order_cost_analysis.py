"""Metadata and explicit one-shot authorization for the fixed H2/H4/H6 analysis."""
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
import math
import os
import sys
import h6_saved_four_method as old6

old = old6.h4.h2
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent/'h246-order-cost-analysis-assets-20261024'
OUT = ROOT/'artifacts/resource_applicability/track_a_h246_order_cost_analysis_ready/2026-10-24'
BASE = '77ea542152de1c9ff796aefaf9be91c68c7ff1cd'
REFERENCE_SOURCE = '8613b3ecda2030a677b1690fe7aa3da23dec4873'
REFERENCE_PATHS = ('src/trotterlib/h46_m5_plan.py','scripts/h46_new2_core.py',
                   'scripts/h246_saved_cost_evaluation.py','scripts/h46_m5_native.py','scripts/h46_m5_cost_native.py')
SERIES = 'H246_FIXED_ORDER_COST_DECOMPOSITION_V1'
EPS = [.05,.01,.005,.001,.0005]
UNIT = 'compiled_RZ_per_full_Hadamard_wrapper_axis'
TARGETS = []
for system,rank,items in [('H2',3,[asdict(x) for x in old.policy.CANDIDATES]),
                         ('H4',10,old6.h4.candidates()),('H6',19,old6.CANDIDATES)]:
    for row in items:
        TARGETS.append(dict(system=system,rank=rank,**{k:row[k] for k in ('id','method','formula','prefix','q')},
                            n=row.get('n',33 if row['method']=='PR' else 1)))
TARGETS[2]['n']=33;TARGETS[3]['n']=34
PHYSICAL = dict(geometry='linear_1Angstrom',basis='STO-3G',rank_policy='tol_only_1e-8',T=.8,
                alpha_axis=.025,P=0,control='symmetric_directional',bias='USUAL_SIGNED',unit=UNIT)
LIMITS = ['empirical_allowance_not_certificate','conditional_saved_reference_accuracy',
          'different_formula_cohorts_not_paired','paired_SE_is_within_trajectory_cost_only',
          'fixed_post_selected_candidates_not_epsilon_reoptimization',
          'q_and_PF_order_change_together_not_causal_isolation',
          'unobserved_events_not_zero','H2_saved_other_epsilon_allowance_policy_mismatch',
          'H6_original_parent_exit_reap_unknown','strict_COMPARE_remains_BLOCKED']
KINDS = ('bootstrap','bundle_load','cost_decomposition','explanation_side','candidate_structure',
         'precision_summary','sampling','compile','simulation','circuit_build','reference_recompute',
         'Hamiltonian_recompute','u_aware_shots','allowance','new_statistics','ranking')
NEW = ('scripts/h246_order_cost_analysis.py','scripts/h246_order_cost_analysis_backend.py',
       'scripts/prepare_h246_order_cost_analysis.py','scripts/launch_h246_order_cost_analysis.py',
       'tests/test_h246_order_cost_analysis.py')
REUSED = tuple(dict.fromkeys(old6.NEW+old6.REUSED+('src/trotterlib/mechanism_analysis.py',)))
read,save,sha,digest,require,Stop = old.read,old.save,old.sha,old.digest,old.require,old.Stop


def counts():
    d=dict.fromkeys(KINDS,0)
    d.update(bootstrap=1,bundle_load=1,cost_decomposition=22,explanation_side=44,
             candidate_structure=12,precision_summary=6)
    return d


def validate(data, *, artificial=False):
    require(data['schema']==SERIES+'_INPUTS' and data['kind']==('ARTIFICIAL' if artificial else 'SAVED_REAL_VALUES'),
            'DATA_KIND_AND_SCHEMA')
    require(old.policy.strict_equal(data['physical'],PHYSICAL) and old.policy.strict_equal(data['targets'],TARGETS)
            and old.policy.strict_equal(data['epsilons'],EPS),'FIXED_TWELVE_TARGETS')
    require(data['limitations']==LIMITS and data['cohort_pooling'] is False,'LIMITS_NO_POOLING')
    require(len(data['rows'])==44 and len(data['availability'])==60,'EXACT_ENDPOINT_COUNTS')
    require([m['target'] for m in data['structures']]==TARGETS,'EXACT_STRUCTURE_TARGETS')
    for m in data['structures']:
        require(m['binding']==data['bindings'][m['target']['system']] and
                m['binding'].get('artificial',False) is artificial,'STRUCTURE_INPUT_BINDING')
        if m['target']['method']=='PR':
            require(bool(m['representative_event_file']) and bool(m['representative_event_digest']),
                    'ORIGINAL_REPRESENTATIVE_EVENT_WITNESS')
    keys=set()
    for r in data['rows']:
        t=next((t for t in TARGETS if t['id']==r['candidate']),None)
        require(t is not None and r['target']==t and r['epsilon'] in EPS,'CANDIDATE_Q_PREFIX_FORMULA_RANK')
        key=(r['candidate'],r['epsilon']);require(key not in keys,'DUPLICATE_ENDPOINT');keys.add(key)
        require(r['system']==t['system'] and r['unit']==UNIT and r['cohort']==
                ('EXPLORATION' if t['method']=='PR' else 'COMMON_PF_DETERMINISTIC'),'SYSTEM_UNIT_COHORT')
        require(r['binding']==data['bindings'][t['system']] and
                r['binding'].get('artificial',False) is artificial,'INPUT_BINDING')
        require(r['status'] in ('ELIGIBLE','INELIGIBLE','UNDETERMINED') and r['bias_mode']=='USUAL_SIGNED',
                'SAVED_ELIGIBILITY_POLICY')
        require(set(r['N'])=={'cosine','sine'} and set(r['C'])==set(r['N']),'PAIRED_AXIS_SCHEMA')
        for axis in r['N']:
            require(r['N'][axis] is None or type(r['N'][axis]) is int and r['N'][axis]>0,'SHOT_INTEGER_NOT_ZERO')
            v=r['C'][axis];require(type(v) in (int,float) and math.isfinite(v) and v>0,'COMPILED_RZ_UNIT')
        require(len(r['pairs'])==t['n'] and len({p['pair_id'] for p in r['pairs']})==t['n'],'ORIGINAL_COHORT_PAIRING')
        for pair in r['pairs']:
            require(pair['candidate']==t['id'] and pair['cohort']==r['cohort'] and set(pair['axes'])==set(r['N']),
                    'NO_CROSS_CANDIDATE_AXIS_PAIRING')
            require(all(type(v) is int and v>0 for v in pair['axes'].values()),'SAVED_INTEGER_PRICE')
        require(r['G_saved'] is not None or r['status']!='ELIGIBLE','MISSING_G_NOT_ZERO')
        require(r['uncertainty']['interpretation']=='SAVED_WITHIN_TRAJECTORY_COST_VARIABILITY',
                'SE_NOT_QUANTUM_SHOT_OR_POPULATION_BOUND')
        if t['method']=='FULL':require(r['uncertainty']['G_SE'] is None,'FULL_SE_NOT_BORROWED_FROM_PR')
    expected={(t['id'],e) for t in TARGETS for e in EPS if t['system']!='H2' or e==.0005}
    require(keys==expected,'POLICY_MATCHED_ENDPOINT_SCOPE')
    available={(x['candidate'],x['epsilon']) for x in data['availability'] if x['available']}
    require(available==keys and {(x['candidate'],x['epsilon']) for x in data['availability']}==
            {(t['id'],e) for t in TARGETS for e in EPS},
            'AVAILABILITY_MATRIX')
    return data


def verify(cfg, *, runtime=True,unused=True):
    require(cfg['schema']==SERIES and cfg['targets']==TARGETS and cfg['physical']==PHYSICAL and
            old.policy.strict_equal(cfg['budget'],counts()) and cfg['limitations']==LIMITS and cfg['retry'] is False and
            cfg['mandatory_STOP'] is True,'FIXED_ANALYSIS_CONTRACT')
    require(Path(cfg['run_root'])==ASSETS/'ANALYSIS_run01' and
            (not unused or not Path(cfg['run_root']).exists()),'UNUSED_SEPARATE_ANALYSIS_ROOT')
    indexed=old.old.seal_check(cfg)
    require(Path(cfg['manifest'])==ASSETS/'inputs_PRIVATE.json' and sha(cfg['manifest'])==cfg['manifest_sha256'],
            'FIXED_INPUT_MANIFEST')
    require(not any(p.is_relative_to(Path(cfg['run_root'])) for p in indexed),'DISJOINT_INPUT_OUTPUT')
    require({r['path'] for r in cfg['source_files'] if r['root']=='runner'}==set(NEW+REUSED),'SOURCE_SCOPE')
    require({r['path'] for r in cfg['source_files'] if r['root']=='reference'}==set(REFERENCE_PATHS),
            'REFERENCE_SOURCE_SCOPE')
    for r in cfg['source_files']:
        require(r['root'] in ('runner','reference') and r['commit']==
                (cfg['source_commit'] if r['root']=='runner' else REFERENCE_SOURCE),'SOURCE_COMMIT_SCOPE')
        expected=(ROOT/r['path'] if r['root']=='runner' else ASSETS/'fixed'/REFERENCE_SOURCE/r['path'])
        require(Path(r['local_path'])==expected,'SOURCE_LOCATION')
        raw=old.recovery.git_bytes(r['commit'],r['path'])
        require(Path(r['local_path']) in indexed and Path(r['local_path']).read_bytes()==raw,'COMMITTED_SOURCE_BYTES')
    require(Path(cfg['schedule_source'])==ASSETS/'fixed'/REFERENCE_SOURCE/'src/trotterlib/h46_m5_plan.py',
            'SCHEDULE_SOURCE_BOUND')
    if runtime:
        require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_RUNTIME')
        require(str(Path(sys.executable).resolve())==cfg['runtime']['python_target'] and
                sha(Path(sys.executable).resolve())==cfg['runtime']['python_sha256'],'SEALED_PYTHON')
        require(sys.path==[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]+cfg['runtime']['sys_path'],'IMPORT_PATH')
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_LIBRARY_PATH'),'CLEAN_LOADER')
        old.seal.verify_mapped_runtime(cfg)
    return validate(read(cfg['manifest']))


def preflight(cfg):
    d=verify(cfg)
    return dict(status='METADATA_PREFLIGHT_PASS',execution_authorized=False,contract_sha256=digest(cfg),
                source_commit=cfg['source_commit'],manifest_sha256=cfg['manifest_sha256'],
                endpoints=len(d['rows']),missing_endpoints=16,budget=counts(),actual_analysis_calls=0,
                grant_issued=False,mandatory_STOP=True)


def authorize(cfg,approval,grant):
    require(approval.get('issuer')=='USER_EXPLICIT' and approval.get('execution_authorized') is True and
            approval.get('contract_sha256')==digest(cfg) and approval.get('accepted_limitations')==LIMITS,
            'EXPLICIT_FIXED_H246_ANALYSIS_APPROVAL_REQUIRED')
    require(grant.get('purpose')==SERIES and grant.get('state')=='UNUSED' and
            isinstance(grant.get('id'),str) and grant['id'].startswith(SERIES+':') and
            grant.get('contract_sha256')==digest(cfg) and grant.get('approval_sha256')==digest(approval) and
            grant.get('run_root')==cfg['run_root'] and grant.get('budget')==counts() and grant.get('retry') is False,
            'NEW_UNSPENT_ANALYSIS_GRANT_REQUIRED')
    verify(cfg)


class Session:
    take=FunctionType(old.Session.take.__code__,dict(old.Session.take.__globals__,KINDS=KINDS),
                      'take',old.Session.take.__defaults__)
    call=old.Session.call

    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant);self.cfg=cfg;self.root=Path(cfg['run_root']);self.root.mkdir(exist_ok=False)
        self.used=dict.fromkeys(KINDS,0);self.sequence=0;self.active=None
        for name in ('ledger','results'):(self.root/name).mkdir()
        save(self.root/'CONSUMED.json',dict(grant=grant,approval_sha256=digest(approval),retry=False))
        self.marker_sha=sha(self.root/'CONSUMED.json')


def execute(cfg,approval,grant):
    s=Session(cfg,approval,grant);guard=old.old.ImportSeal(cfg)
    try:
        guard.install();s.take('bootstrap')
        from h246_order_cost_analysis_backend import pipeline
        data=s.call('bundle_load',lambda:validate(read(cfg['manifest'])))
        result=pipeline(s,data,cfg)
        require(s.used==counts(),'EXACT_ANALYSIS_BUDGET')
        old.old.Session.verify(s);old.old.Session.loaded_closure(s);verify(cfg,unused=False)
        ledger=sorted((s.root/'ledger').glob('*.json'));seen=dict.fromkeys(KINDS,0)
        for i,p in enumerate(ledger,1):
            row=read(p);require(p.stem=='%06d'%i and row['contract_sha256']==digest(cfg) and
                row['amount']==1 and row['nonrefundable'] is True,'LEDGER_IDENTITY');seen[row['kind']]+=1
        require(seen==s.used and len(ledger)==s.sequence,'LEDGER_COMPLETE')
        save(s.root/'results/analysis.json',result)
        save(s.root/'completion.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=cfg['source_commit'],
             contract_sha256=digest(cfg),used=s.used,result_sha256=sha(s.root/'results/analysis.json')))
        save(s.root/'STOP.json',dict(status='MANDATORY_STOP',used=s.used,retry=False))
        return result
    except BaseException as exc:
        save(s.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=s.used,retry=False));raise
    finally:guard.remove()
