"""Unapproved cost-only v3 contract. No valid grant is issued by preparation."""
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import h8_noncancellation_cost as numeric
import h8_noncancellation_contract as original
import h8_phase_c_runtime as common

c = numeric
require = c.require
read, loads, sha, identity = common.read, common.loads, common.sha, common.identity
GIT, THREAD_KEYS = common.GIT, common.THREAD_KEYS
SCOPE = numeric.SCOPE
PYTHON = '/usr/bin/python3.12'
INPUT_CONTRACT = 'references/h8_noncancellation_cost_inputs_v3.json'
REGISTRY = 'references/h8_phase_c_registration_v1.json'
SOURCES = tuple(dict.fromkeys((*original.SOURCE_PATHS, *common.SOURCES,
    'src/trotterlib/h8_noncancellation_cost.py', 'scripts/h8_noncancellation_cost_contract.py',
    'scripts/run_h8_noncancellation_cost.py', 'scripts/prepare_h8_noncancellation_cost.py',
    'tests/test_h8_noncancellation_cost.py', INPUT_CONTRACT)))
RESOURCE_POLICY = dict(agent_imposed_CPU_RAM_worker_wall_caps=None,
    processes=1, supervision='DIRECT_CHILD_WAIT4', internal_threads=1,
    estimates=dict(peak_RSS_bytes=256*2**20, output_bytes=64*2**20, output_inodes=32, wall='seconds'),
    PSI='OBSERVATION_ONLY_NO_STOP_FROM_OTHER_JOB_PROGRESS',
    own_OOM_or_integrity_failure='STOP_WITH_NATIVE_RECEIPT_NO_RETRY')
CAPS = dict(reaccounting_pass=1, saved_row=1900, comparison=20, model_axis_estimate=352,
    PR_weighted_cost_and_SE=60, new_shots=0, new_signal=0, compile=0, sampling=0, fit=0,
    NONCANCELLATION_SHOTS=0, ordinary_Phase_C=0)
_AUTHORIZED = object()


class Permit:
    def __init__(self, synthetic=False, *, _authorization=None):
        require(synthetic is True or _authorization is _AUTHORIZED, 'SEPARATE_COST_APPROVAL_REQUIRED')
        self.synthetic = synthetic

    def require(self, kind):
        require(kind == ('ARTIFICIAL' if self.synthetic else 'SAVED_H8'), 'COST_DATA_SCOPE')


def plan():
    return dict(schema='H8_NONCANCELLATION_COST_PLAN_V3', scope=SCOPE,
        authorized_stages=['COST_COMPARISON'], approved=False, runtime_authorization=False, execution_sealed=False,
        old_v2_plan_preserved='002f9edbf35506fca256431df641552d7c319581f306f13a316b736e07897e6d',
        completed_shot_result=c.SHOT_RESULT, completed_shot_SOURCE=c.SHOT_SOURCE,
        shot_rule_reexecution=False, completed_1900_shots_reused=True,
        completed_shot_plan=c.SHOT_PLAN, completed_shot_grant=c.SHOT_GRANT, completed_shot_input=c.SHOT_INPUT,
        usual_result=c.USUAL_RESULT, usual_SOURCE=c.USUAL_SOURCE, usual_recalculated=False,
        B2_result=c.B2_RESULT, B2_status=c.phase_c.B2_STATUS,
        B2_per_axis_sources=[dict(case_id=a, axis=b, SOURCE=s) for a,b,s,_ in c.phase_c.B2_AXES],
        science=dict(original.SCIENCE), accounting=dict(original.ACCOUNTING), bias_definitions=dict(original.BIAS_KINDS),
        bias_namespace=c.BIAS, usual_namespace='USUAL_SIGNED_SAVED', rigorous_bias_guarantee=None,
        model_sha256=c.MODEL_SHA, compiler_hash=c.phase_c.COMPILER, profile_sha256=c.phase_c.PROFILE,
        model_extrapolation='PREFIX_OUTSIDE_14_TO_21_FLAGGED', S4_validation='FOUR_REGISTERED_AXES_ONLY',
        cost_priority=['B2_REGISTERED_S4_MEASURED','B1_REGISTERED_ANCHOR','LEGACY_FULL_PF_SAVED_N1','FIXED_B1_MODEL'],
        PR_policy='CONFIRMATION_UNION6_N32_PAIRED_NO_EXPLORATION_POOLING',
        PR_SE='FROZEN_D0_WEIGHTED_METRIC_REWEIGHT_PAIRED_SAMPLES_NOT_COPY_USUAL_SE',
        PR_saved_metric_reproduction='REQUIRED_IN_COST_STAGE_BEFORE_NEW_COSTS',
        selection='EXACT_RATIONAL_COST_THEN_LEXICAL_ID', missing_or_unresolved='NULL_AND_STOP_NO_FULL_DOMAIN_CLAIM',
        differences='SAME_EPSILON_FACTOR_SCOPE; REDUCTION_DIFFERENCE_IN_PERCENTAGE_POINTS',
        input_contract_sha256=sha(ROOT/INPUT_CONTRACT), source_paths=list(SOURCES),
        analysis_environment=original.environment(), resource_policy=RESOURCE_POLICY, call_caps=CAPS,
        fresh_gate='SOURCE_INPUT_ENV_EFFECTIVE_MEMORY_CPU_FS_INODES_QUOTA_OWN_LAUNCHER',
        protection='OWN_DEPENDENCIES_STRICT_OTHER_WORKTREES_OBSERVATION_ONLY',
        cleanup='OWN_UNREAPED_DIRECT_CHILD_ONLY_PIDFD_TERM_KILL_WAIT4_FIRST_REASON_RETAINED',
        one_shot=True, retry=False, next_stage_authorized=False, mandatory_STOP=True)


def authorize(config, grant):
    require(config.get('execution_sealed') is True and grant.get('schema') == 'H8_NONCANCELLATION_COST_GRANT_V3' and
        grant.get('approved') is True and grant.get('runtime_authorization') is True and
        grant.get('execution_sealed') is True and grant.get('approval_text') and
        grant.get('scope') == SCOPE and grant.get('authorized_stages') == ['COST_COMPARISON'] and
        grant.get('SOURCE') == config.get('SOURCE') and original.hexdigest(config.get('SOURCE'),40) and
        grant.get('plan_sha256') == config.get('plan_sha256') == c.digest(plan()) and
        grant.get('config_sha256') == c.digest(config) and grant.get('input_contract_sha256') == sha(ROOT/INPUT_CONTRACT) and
        grant.get('call_caps') == CAPS and grant.get('resource_policy') == RESOURCE_POLICY and
        grant.get('one_shot') is True and grant.get('automatic_retry') is False and
        grant.get('next_stage_authorized') is False and grant.get('new_shot_calculation') is False,
        'NEW_EXPLICIT_COST_ONLY_SEALED_GRANT_REQUIRED')
    cpus=grant.get('allowed_cpus')
    require(type(cpus) is list and cpus and len(set(cpus))==len(cpus) and
            all(type(x) is int and x>=0 for x in cpus) and set(cpus)<=os.sched_getaffinity(0), 'FRESH_OWN_CPU_PLACEMENT')
    return cpus


def pinned_inputs(mapping, *, git=True):
    specs=read(ROOT/INPUT_CONTRACT)['files']
    require(set(mapping)=={r['role'] for r in specs}, 'EXACT_COST_INPUT_ROLES')
    rows=[]
    for spec in specs:
        p=original.home_path(mapping[spec['role']]); data=original.read_verified(p,spec)
        if git and spec.get('commit'):
            require(hashlib.sha256(subprocess.check_output([*GIT,'show',spec['commit']+':'+spec['repository_path']],cwd=ROOT)).hexdigest()==spec['sha256'],
                    'COST_INPUT_GIT_OBJECT')
        rows.append(dict(role=spec['role'],path=str(p),bytes=len(data),sha256=spec['sha256']))
    return rows


def verify(config, *, git=True, protection=True):
    require(config['schema']=='H8_NONCANCELLATION_COST_CONFIG_V3' and config['plan_sha256']==c.digest(plan()) and
            config['environment']==original.environment() and config['python']==PYTHON and
            config['python_sha256']==sha(PYTHON), 'COST_PLAN_ENVIRONMENT_BINDING')
    require(set(config['source_hashes'])==set(SOURCES), 'EXACT_COST_SOURCE_CLOSURE')
    if git:
        require(subprocess.check_output([*GIT,'rev-parse','HEAD'],cwd=ROOT).decode().strip()==config['SOURCE'], 'COST_SOURCE_HEAD')
    for name,h in config['source_hashes'].items():
        require(sha(ROOT/name)==h,'COST_SOURCE_BYTES_CHANGED:'+name)
        if git:
            require(hashlib.sha256(subprocess.check_output([*GIT,'show',config['SOURCE']+':'+name],cwd=ROOT)).hexdigest()==h,
                    'COST_SOURCE_GIT_BYTES')
    by=phase_index(config['inputs'])
    specs=read(ROOT/INPUT_CONTRACT)['files']
    require(set(by)=={x['role'] for x in specs},'EXACT_COST_INPUTS')
    for spec in specs:
        row=by[spec['role']]
        require((row['bytes'],row['sha256'])==(spec['bytes'],spec['sha256']), 'COST_PINNED_INPUT_RECEIPT')
        original.read_verified(original.home_path(row['path']),row)
    return common.protected(config) if protection else None


def phase_index(rows):
    return c.phase_c.unique(rows,lambda x:x['role'])


def proof_metadata(inputs):
    """Metadata and already saved field equality only; no cost/shot kernel call."""
    by=phase_index(inputs)
    def raw(role):return original.read_verified(Path(by[role]['path']),by[role])
    def get(role):return loads(raw(role))
    b=get('shot:execution_binding_v1.json')
    for role,size,h in [('raw_result',b['raw_result_bytes'],b['raw_result_sha256']),
                        ('raw_terminal',b['terminal_bytes'],b['terminal_sha256'])]:
        row=by['shot:'+role]; require((row['bytes'],row['sha256'])==(size,h),'RAW_SHOT_RESULT_RECEIPT')
    require(by['shot:original_grant_PRIVATE']['sha256']==b['grant']['file_sha256'],'OLD_GRANT_RECEIPT_NOT_REUSED')
    p=dict(binding=b,summary=get('shot:summary_v1.json'),raw=get('shot:raw_result'),terminal=get('shot:raw_terminal'),
        native=get('shot:native_exit_receipt_v1.json'),csv=raw('shot:accounting_rows_v1.csv'),
        axis_csv=raw('shot:axis_envelopes_v1.csv'),PR_csv=raw('shot:PR_component_checks_v1.csv'))
    registry=read(ROOT/REGISTRY);registry=dict(registry)
    c.completed_shots(p,read(ROOT/original.MANIFEST),registry)
    b2=dict(binding=get('B2:binding'),receipts=get('B2:receipts')); c.phase_c.b2_gate(b2)
    ub=get('usual:execution_binding_v1.json');terminal=get('usual:terminal_v1.json');gate=get('usual:final_gate_v1.json')
    for item in get('usual:file_receipts_v1.json')['files']:
        role='usual:'+item['name']
        if role in by:
            require((by[role]['bytes'],by[role]['sha256'])==(item['bytes'],item['sha256']), 'USUAL_PUBLICATION_RECEIPT')
    require(ub['SOURCE']==terminal['SOURCE']==gate['SOURCE']==c.USUAL_SOURCE and
        ub['scientific_attempts']==1 and ub['source_checksum_after']==ub['input_checksum_after']=='PASS' and
        ub['model_sha256']==c.MODEL_SHA and ub['profile_sha256']==c.phase_c.PROFILE and
        ub['compiler_hash']==c.phase_c.COMPILER and terminal['status']=='COMPLETE_MANDATORY_STOP' and
        gate['status']=='PASS_MANDATORY_STOP' and gate['next_stage_authorized'] is False, 'USUAL_COMPLETED_BINDING')
    native=get('usual:native_completion_v1.json')
    require(native['all_owned_reaped'] is True and native['remaining_owned_processes']==0 and
        native['driver']['native_exit_code']==0 and native['launcher']['native']['native_exit_code']==0,
        'USUAL_NATIVE_COMPLETION')
    aliases=dict(D0='original:D0_accounting',D0_terminal='original:D0_terminal',
        B0I_accounting='original:B0I_accounting',B0I_file_receipts='original:B0I_file_receipts',
        Phase_A='original:A_signals',model='original:B1_models',B1_terminal='original:B1_terminal',
        B1_launcher_native_exit='original:B1_launcher_native_exit',B1_anchors='original:B1_anchors',
        B2_binding='B2:binding',B2_execution_receipts='B2:receipts')
    receipts=phase_index(ub['input_receipts'])
    require(set(receipts)==set(aliases)|{'registry'},'USUAL_INPUT_RECEIPT_COVERAGE')
    for role,other in aliases.items():
        require((receipts[role]['bytes'],receipts[role]['sha256'])==(by[other]['bytes'],by[other]['sha256']),
                'SHARED_INPUT_JOIN_WITH_USUAL:'+role)
    require(receipts['registry']['sha256']==sha(ROOT/REGISTRY),'USUAL_REGISTRY_BINDING')
    return dict(status='PASS_SAVED_EVIDENCE_BINDINGS_NO_NEW_COSTS',B2_axes=4,shot_rows=1900,shot_axes=380,
        SOURCE_of_shots=c.SHOT_SOURCE,usual_result=c.USUAL_RESULT,new_H8_cost_calls=0),p,b2


def load_saved(config):
    """Called only inside the driver after sealed authorization and fresh gate."""
    by=phase_index(config['inputs']);_,shot,b2=proof_metadata(config['inputs'])
    manifest=read(ROOT/original.MANIFEST)
    roots={Path(by['original:'+x['role']]['path']).parent.parent for x in manifest['files']}
    require(len(roots)==1,'ORIGINAL_SNAPSHOT_SINGLE_ROOT')
    permit=Permit(_authorization=_AUTHORIZED)
    data,registry,_=original.load_saved(manifest,roots.pop(),permit)
    saved_registry=read(ROOT/REGISTRY)
    require(registry['old']==saved_registry['old'] and registry['additional']==saved_registry['additional'] and
            data['confirmation']['selected_ids']==saved_registry['confirmation_ids'],'SAME_REGISTERED_184_AND_UNION6')
    registry=saved_registry
    def get(role):return read(by[role]['path'])
    gz=original.read_verified(Path(by['usual:accounting_rows_v1.jsonl.gz']['path']),by['usual:accounting_rows_v1.jsonl.gz'])
    receipt=get('usual:accounting_rows_receipt_v1.json'); decoded=gzip.decompress(gz)
    require(len(decoded)==receipt['original']['bytes'] and hashlib.sha256(decoded).hexdigest()==receipt['original']['sha256'],
            'SAVED_PHASE_C_ROWS_GZIP_RECEIPT')
    data.update(B2=b2,completed_shots=shot,shot_manifest=manifest,
                usual_result=get('usual:comparisons_v1.json'),usual_rows=[loads(line) for line in decoded.splitlines()])
    return data,registry,permit


def path_set(config, *, supervisor_control_exists=False):
    paths=common.path_set(config,supervisor_control_exists=supervisor_control_exists)
    require(all(not p.is_relative_to(ROOT) and not ROOT.is_relative_to(p) for p in paths),
            'COST_OUTPUT_OUTSIDE_SOURCE_CHECKOUT')
    for row in config.get('inputs',[]):
        source=Path(row['path']).resolve()
        require(all(not p.is_relative_to(source) and not source.is_relative_to(p) for p in paths),
                'COST_OUTPUT_INPUT_OVERLAP')
    return paths


fresh_binding=common.fresh_binding
