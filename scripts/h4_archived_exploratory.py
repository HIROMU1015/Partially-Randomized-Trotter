"""H4 fixed saved-cost comparison: metadata before an explicit one-shot grant."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import sys
import h2_archived_exploratory as h2

ROOT = Path(__file__).resolve().parents[1]
SERIES = 'H4_FOUR_METHOD_ARCHIVED_COST_EXPLORATORY_V1'
ASSETS = ROOT.parent/'h4-four-method-exploratory-assets-20261024'
OUT = ROOT/'artifacts/resource_applicability/track_a_h4_four_method_exploratory_ready/2026-10-24'
ADOPTED = 'f75a7c25d584b69d90de454731a7f343a421e059'
SAVED = '8613b3ecda2030a677b1690fe7aa3da23dec4873'
H2_RESULT = '9e5ea6b235d7beecc02651b1393b561948e85789'
H2_SOURCE = 'aca689060d302b4e1f37d32d7075c9266b637f56'
ADOPTED_DIR = 'artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/'
SAVED_DIR = 'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
NATIVE = 'M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'
BINDING = dict(hamiltonian='c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0',
 state='9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b',
 snapshot='12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96')
PHYSICAL = dict(system='H4',geometry='linear_1Angstrom',basis='STO-3G',rank=10,
 rank_policy='tol_only_1e-8',T=.8,epsilon_signal=.0005,alpha_axis=.025,P=0,
 system_qubits=8,ancilla=8,control='symmetric_directional',bias='USUAL_SIGNED',
 unit='compiled_RZ_per_full_Hadamard_wrapper_axis')
LIMITS = ('empirical_allowance_not_certificate','conditional_saved_reference_accuracy',
 'finite_exploration_not_population_mean','post_selection_fixed_candidates',
 'historical_runtime_not_complete_dynamic_trace','single_state_not_operator',
 'H2_H4_different_allowance_inputs_and_cohorts','H4_saved_reference_binary64_variants_not_bit_identical',
 'no_global_optimum_or_causal_attribution')
NEW = ('scripts/h4_archived_exploratory.py','scripts/h4_archived_exploratory_backend.py',
 'scripts/prepare_h4_archived_exploratory.py','scripts/launch_h4_archived_exploratory.py')
REUSED = tuple(dict.fromkeys(h2.NEW+h2.REUSED))
KINDS = h2.KINDS
read, sha, save, digest, require, Stop = h2.read, h2.sha, h2.save, h2.digest, h2.require, h2.Stop


@dataclass(frozen=True)
class Candidate:
    id: str
    method: str
    formula: str
    prefix: int
    q: int
    n: int
    K: int | None
    r_policy: str | None
    r_value: int | None


CANDIDATES = (
 Candidate('H4_FULL_10_8_2nd','FULL','2nd',10,8,1,None,None,None),
 Candidate('H4_FULL_10_4_4th','FULL','4th',10,4,1,None,None,None),
 Candidate('M5_MIN_72b305020499dd68','PR','2nd',3,8,48,4,'ABS_TIME_PER_OUTER',32),
 Candidate('M5_MIN_ab57ad5bd40dc583','PR','4th',3,4,33,4,'ABS_TIME_PER_OUTER',32))
# Only metadata constants change in rebound methods. Scientific code objects stay intact.
from types import SimpleNamespace
policy = SimpleNamespace(CANDIDATES=CANDIDATES,number=h2.policy.number,
 ALLOWANCE_POLICY='H4_SAVED_EMPIRICAL_U_FLOOR_1E_MINUS_9_NEVER_REDUCED')


def counts():
    result = dict.fromkeys(KINDS,0)
    result.update(bootstrap=1,bundle_load=1,allowance=4,accounting=4,u_aware_shots=4,
                  axis_headroom=8,cost_statistics=4,resource_row=4,metric_summary=8,ranking=1)
    return result


def candidates():
    return [asdict(c) for c in CANDIDATES]


def validate_bundle(data):
    """Shape, original values and pairing only: no allowance/shot/G arithmetic."""
    require(data['schema']==SERIES+'_INPUTS' and data['physical']==PHYSICAL,'H4_DF10_SCOPE')
    require(data['binding']==BINDING and data['candidates']==candidates(),'FIXED_H4_FOUR_CANDIDATES')
    require(data['compiler']==COMPILER and data['native']==NATIVE,'SAVED_COMPILER_NATIVE')
    require(data['reference_policy']=='CONDITIONAL_SAVED_H4_REFERENCE_NOT_ZERO_ERROR_BOUND',
            'REFERENCE_POLICY')
    require(data['adopted_commit']==ADOPTED and data['saved_commit']==SAVED,'FIXED_RESULTS')
    require(set(data['methods'])=={c.id for c in CANDIDATES},'FOUR_METHOD_INPUTS')
    references = {}
    for c in CANDIDATES:
        d=data['methods'][c.id]
        require(d['candidate']==asdict(c),'CANDIDATE_FORMULA_Q_PREFIX_LAW')
        require(d['binding']==BINDING and d['unit']==PHYSICAL['unit'],'METHOD_INPUT_AND_UNIT')
        require(d['cohort']==('EXPLORATION' if c.method=='PR' else 'COMMON_PF_DETERMINISTIC'),
                'COHORT_SCOPE')
        require(d['compiler']==COMPILER and d['native']==NATIVE,'METHOD_COMPILER_NATIVE')
        require(d['saved_reference'] and set(d['saved_reference'])=={'real','imag'},'REFERENCE_PRESENT')
        if c.method not in references:references[c.method]=d['saved_reference']
        require(d['saved_reference']==references[c.method],'H4_WITHIN_METHOD_SAVED_REFERENCE')
        for field in ('bias','saved_reference'):
            require(set(d[field])=={'real','imag'},'AXIS_SCHEMA')
            for v in d[field].values():require(type(v) in (int,float) and math.isfinite(v),'FINITE_SAVED_SCALAR')
        for k in ('u','log_B_margin','log_B'):
            v=d[k];require(type(v) in (int,float) and math.isfinite(v) and v>=0,'SAVED_ALLOWANCE_NORMALIZATION')
        require(d['u']>=1e-9 and d['allowance_recomputed'] is False,'H4_ALLOWANCE_NEVER_REDUCED')
        require(d['reference_error_bound'] is None,'NO_ZERO_REFERENCE_ERROR_SUBSTITUTION')
        expected=d['saved_shots']
        require(set(expected)=={'real','imag'} and all(v is None or type(v) is int and v>0
                for v in expected.values()),'SAVED_ANALYTIC_SHOTS')
        require(d['preparation_hash'] and len(d['preparation_hash'])==64,'PREPARATION_BINDING')
        pairs=d['costs'];require(len(pairs)==c.n,'ORIGINAL_SAMPLE_COUNT')
        require(len({p['task'] for p in pairs})==c.n,'UNIQUE_TRAJECTORY_PAIR')
        for pair in pairs:
            require(set(pair)=={'task','cosine','sine','receipt_sha256'},'EXACT_AXIS_PAIR')
            require(all(type(pair[a]) is int and pair[a]>0 for a in ('cosine','sine')),'POSITIVE_COMPILED_RZ')
            h2.policy.sha(pair['receipt_sha256'],'saved_task_receipt')
    return data


def saved_inputs(cfg):
    p=Path(cfg['manifest'])
    require(sha(p)==cfg['manifest_sha256'],'INPUT_MANIFEST_SHA')
    return validate_bundle(read(p))


def verify(cfg, *, runtime=True, unused=True):
    require(cfg['schema']==SERIES and cfg['physical']==PHYSICAL and cfg['candidates']==candidates(),
            'H4_EXACT_EXECUTION_TARGET')
    require(h2.policy.strict_equal(cfg['budget'],counts()) and cfg['limitations']==list(LIMITS) and
            cfg['retry'] is False and cfg['mandatory_STOP'] is True,'EXACT_CALLS_AND_STOP')
    require(Path(cfg['run_root'])==ASSETS/'EXPLORATORY_run01','DEDICATED_H4_RUN_ROOT')
    require(not unused or not Path(cfg['run_root']).exists(),'UNUSED_H4_RUN_ROOT')
    require(Path(cfg['manifest']).is_relative_to(ASSETS),'DEDICATED_INPUT_MANIFEST')
    indexed=h2.old.seal_check(cfg)
    require(not any(p.is_relative_to(Path(cfg['run_root'])) for p in indexed),'INPUT_OUTPUT_DISJOINT')
    rows=cfg['source_files']
    require({r['path'] for r in rows if r['root']=='runner'}==set(NEW+REUSED),'EXACT_RUNNER_SOURCE')
    require({r['path'] for r in rows if r['root']=='provider'}==set(h2.PROVIDER_PATHS),'EXACT_PROVIDER_SOURCE')
    for role,commit in (('runner',cfg['source_commit']),('provider',h2.PROVIDER)):
        selected=[r for r in rows if r['root']==role]
        for row,(_,raw) in zip(selected,h2.seal.git_batch(commit,[r['path'] for r in selected])):
            p=Path(row['local_path'])
            require(p in indexed and p.read_bytes()==raw and sha(p)==row['sha256'],'COMMITTED_EXECUTION_SOURCE')
    if runtime:
        rt=cfg['runtime']
        require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_I_S_B')
        require(str(Path(sys.executable).resolve())==rt['python_target'] and
                sha(Path(sys.executable).resolve())==rt['python_sha256'],'ANALYSIS_PYTHON_SEAL')
        require(sys.path==[str(ROOT/'scripts')]+rt['sys_path'],'EXACT_IMPORT_PATH')
        require(not os.environ.get('LD_LIBRARY_PATH') and not os.environ.get('LD_PRELOAD'),'CLEAN_LOADER')
        h2.seal.verify_mapped_runtime(cfg)
    return saved_inputs(cfg)


def preflight(cfg):
    data=verify(cfg)
    return dict(status='H4_METADATA_PREFLIGHT_PASS',technical_ready=True,execution_authorized=False,
        source_commit=cfg['source_commit'],contract_sha256=digest(cfg),manifest_sha256=cfg['manifest_sha256'],
        candidates=candidates(),saved_axes=166,missing_scientific_inputs=[],new_scientific_calls=0,
        cost_pairs={c.id:len(data['methods'][c.id]['costs']) for c in CANDIDATES},
        budget=counts(),kernel_ledger_events=sum(counts().values()),strict_COMPARE='BLOCKED',mandatory_STOP=True)


def authorize(cfg,approval,grant):
    # Explicit authorization is checked before importing a numerical backend or evaluating real inputs.
    require(approval.get('issuer')=='USER_EXPLICIT' and approval.get('execution_authorized') is True and
        approval.get('contract_sha256')==digest(cfg) and approval.get('accepted_limitations')==list(LIMITS),
        'SEPARATE_H4_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED')
    require(grant.get('purpose')==SERIES and grant.get('state')=='UNUSED' and
        isinstance(grant.get('id'),str) and grant['id'].startswith(SERIES+':') and
        grant.get('retry') is False and grant.get('contract_sha256')==digest(cfg) and
        grant.get('approval_sha256')==digest(approval) and grant.get('run_root')==cfg['run_root'] and
        grant.get('budget')==counts(),'NEW_H4_ONE_SHOT_GRANT_REQUIRED')
    verify(cfg)


class Session(h2.Session):
    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant)
        self.cfg=cfg;self.root=Path(cfg['run_root']);self.root.mkdir(exist_ok=False)
        self.used=dict.fromkeys(KINDS,0);self.sequence=0;self.active=None
        save(self.root/'CONSUMED.json',dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=grant,
            approval_sha256=digest(approval),contract_sha256=digest(cfg)))
        self.marker_sha=sha(self.root/'CONSUMED.json')
        for name in ('results','ledger','delegated_receipts'):(self.root/name).mkdir()


def pipeline(session,backend):
    data=session.call('bundle_load',backend.load);rows=[];details=[]
    for c in CANDIDATES:
        d=data['methods'][c.id];u=session.call('allowance',backend.allowance,c,d)
        row=session.call('accounting',backend.account,c,d,u,d['costs'])
        details.append(session.call('cost_statistics',backend.statistics,c,row,d['costs']));rows.append(row)
    result=session.call('ranking',backend.rank,rows)
    return dict(schema=SERIES+'_RESULT',exploratory_only=True,methods=details,
        delegated_comparison=result,limitations=list(LIMITS),science_GO=False,strict_COMPARE='BLOCKED')


def execute(cfg,approval,grant):
    session=Session(cfg,approval,grant);guard=h2.old.ImportSeal(cfg);backend=None
    try:
        guard.install();session.take('bootstrap')
        from h4_archived_exploratory_backend import Backend
        backend=Backend(session);result=pipeline(session,backend)
        h2.audit_end(session);verify(cfg,unused=False);h2.seal.verify_mapped_runtime(cfg)
        save(session.root/'results/comparison.json',result)
        files=[dict(path=str(p.relative_to(session.root)),sha256=sha(p))
            for folder in ('results','delegated_receipts') for p in sorted((session.root/folder).rglob('*')) if p.is_file()]
        save(session.root/'completion.json',dict(status='EXPLORATORY_COMPLETE_NOT_FORMAL_COMPARE',
            result_files=files,contract_sha256=digest(cfg),source_commit=cfg['source_commit'],
            marker_sha256=session.marker_sha,used=session.used,retry=False,mandatory_STOP=True))
        save(session.root/'STOP.json',dict(status='MANDATORY_STOP_H4_EXPLORATORY_COMPLETE',
            used=session.used,strict_COMPARE='BLOCKED',mandatory_STOP=True))
        return result
    except BaseException as exc:
        save(session.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),
            used=session.used,retry=False,mandatory_STOP=True));raise
    finally:
        if backend is not None:backend.close()
        guard.remove()
