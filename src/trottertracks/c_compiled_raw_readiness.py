"""Passive default and dormant six-unit local-support saved-pair pilot.

The real H4 pilot needs separate permission. Calibration remains nonexecutable.
Existing source, statistical policy and prior consumed namespaces are untouched.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import time

BASE='325f07ae02d8872d4fe322d6ad6e08c256e0bef7'
FAMILY='artifacts/research_c_compiled_raw_acceleration/2026-11-05'
PILOT='docs/research/research_c_compiled_raw_local_pilot_unapproved_20261105.json'
FAILURE='docs/research/research_c_compiler_abort_scope_unapproved_20261105.json'
SCOPE='C_LOCAL_SUPPORT_COMPILED_RAW_PILOT_V1'
LIMITS=dict(records_load=1,identity_gate=1,structural_plan=1,local_certificate=1,comparison=1,summary=1)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root, contract=PILOT):
    root=Path(root).resolve();c=read(root/contract)
    expected=dict(scope=SCOPE,base_commit=BASE,executable=True,maximum_logical_units=6,
        call_limits=LIMITS,precision_bits=192,controlled_dimension=512,commute_CX=False,
        retry=0,resume=0,mandatory_stop=True,new_sampling=0,new_build=0,new_compile=0,
        new_literal_reference=0,Stage_C_executable=False)
    from .c_per_word_range_schema import same_json
    if not same_json({k:c.get(k) for k in expected},expected): raise ValueError('FIXED_LOCAL_PILOT_SCHEMA')
    if sha(root/c['source_manifest'])!=c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    for group in (read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or sha(p)!=want: raise ValueError('SOURCE_INPUT:'+name)
    for role in ('compiled','raw','old_certificate','old_perword'):
        if c['roles'][role] not in c['saved_inputs']: raise ValueError('UNPINNED_RECORD_ROLE')
    parent=read(root/c['parent_contract'])
    if not same_json(c['identity'],parent['identity']) or not same_json(c['word'],parent['word']):
        raise ValueError('FIXED_SOURCE_MODEL_WORD_COMPILER')
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want: raise ValueError('PACKAGE:'+name)
    if c['fresh_output']!=FAMILY+'/local-pilot01' or (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    f=read(root/FAILURE)
    fixed=dict(executable=False,failure_policy=None,N=None,delta_compile=None,delta_cal=None,
        operational_abort_allowance=None,cache_policy=None,nontermination_policy=None,
        maximum_logical_units=None,measurement_alpha='1/100',joint_001_policy_adopted=False,
        retry=0,resume=0,mandatory_stop=True)
    if not same_json({k:f.get(k) for k in fixed},fixed): raise ValueError('UNADOPTED_FAILURE_SCOPE')
    if f['source_manifest']!=c['source_manifest'] or f['source_manifest_sha256']!=c['source_manifest_sha256']:
        raise ValueError('FAILURE_SCOPE_SOURCE')
    return dict(status='PASSIVE_LOCAL_PILOT_READY_FOR_PERMISSION',scientific_calls=0,
        arrays_decoded=0,native_records_decoded=0,maximum_local_pilot_calls=6,
        actual_H4_correspondence_checked=False,actual_H4_new_upper=None,
        calibration_ready=False,failure_aware_runner_ready=False,Stage_C_ready=False,mandatory_stop=True)


def run_pilot(root, *, execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root);c=read(root/PILOT)
    from . import c_compiled_raw_local as local
    from .c1_stage3_integrated import Calls
    from . import c1_stage1_execution as io
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model();out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    calls=Calls(out,LIMITS,6);io.save(out/'contract.json',c);io.save(out/'SOURCE.json',read(root/c['source_manifest']))
    try:
        values=calls('records_load','saved C/raw/old bound/word;no compile',
            lambda:{k:read(root/v) for k,v in c['roles'].items()})
        def identity():
            C,R=values['compiled'],values['raw'];old=values['old_certificate'];word=values['old_perword']
            local.validate(C);local.validate(R)
            if (C['qubits']!=9 or R['qubits']!=9 or old['left_id']!=local._digest(C)
                    or old['right_id']!=local._digest(R) or word['compiled_id']!=local._digest(C)
                    or word['raw_id']!=local._digest(R) or [x[0] for x in word['binding']['word']]!=c['word']):
                raise ValueError('RECORD_PHASE_SOURCE_WORD_IDENTITY')
            return True
        calls('identity_gate','original digests and fixed32-event word',identity)
        def build_plan():
            tick=time.perf_counter();p=local.plan(values['compiled'],values['raw'],commute_CX=False)
            return dict(plan=p,wall_seconds=time.perf_counter()-tick)
        planned=calls('structural_plan','local candidate expansions;fixed disjoint-only normalization',build_plan)
        p=planned['plan'];io.save(out/'structural_plan.json',planned)
        if not p['correspondence']:
            result=calls('summary','registered early STOP;no bound and no dense fallback',lambda:dict(
                status='NOT_APPLICABLE_NO_SHARED_CX_SKELETON',upper_exact=None,
                local_Arb_evaluations=0,calibration_ready=False,Stage_C_ready=False,mandatory_stop=True))
            io.save(out/'result.json',result);return result
        def evaluate():
            tick=time.perf_counter();value=local.evaluate(p,parallel=True)
            return dict(value,wall_seconds=time.perf_counter()-tick)
        cert=calls('local_certificate','local matrices + exact unitary telescope;no global columns',evaluate)
        io.save(out/'compiled_raw.json',cert)
        def compare():
            local.check_binding(cert,values['compiled'],values['raw'])
            old=values['old_certificate'];upper=F(cert['upper_exact'])
            if upper<F(old['lower_exact']): raise ValueError('ENCLOSURE_CONTRADICTION')
            return dict(status='BOUND_NO_LOOSER_THAN_SAVED' if upper<=F(old['upper_exact']) else 'VALID_BUT_LOOSER',
                old_upper_exact=old['upper_exact'],new_upper_exact=cert['upper_exact'],
                old_wall_seconds=old['wall_seconds'],new_evaluator_wall_seconds=cert['wall_seconds'],
                new_expansion_and_plan_wall_seconds=planned['wall_seconds'],
                new_total_CR_wall_seconds=planned['wall_seconds']+cert['wall_seconds'],
                fresh_control_timing_run=False,scientific_accuracy_conditions_changed=False)
        comparison=calls('comparison','registered tightness comparator;no new baseline run',compare)
        io.save(out/'comparison.json',comparison)
        result=calls('summary','fixed C/R only;no R/D,all-law,calibration or CA',lambda:dict(
            status='LOCAL_SUPPORT_FIXED_PAIR_COMPLETE',certificate=cert,comparison=comparison,
            calibration_ready=False,Stage_C_ready=False,actual_CA_achieved=None,mandatory_stop=True))
        io.save(out/'result.json',result);return result
    except BaseException as exc:
        calls.stop.set()
        if calls.first_failure is None:calls.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(scientific_units=sum(calls.counts.values()),call_counts=calls.counts,
            failed=calls.first_failure,retry=0,resume=0,mandatory_stop=True,Stage_C_executed=False))
