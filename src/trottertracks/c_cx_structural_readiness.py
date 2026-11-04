"""Passive default; separate permission required for the new six-unit pair pilot."""
from pathlib import Path
from fractions import Fraction as F
import importlib.metadata
import time
from .c_compiled_raw_readiness import read,sha

BASE='f6b1516116d40c3d8d1467e476abb032a0f562b6'
FAMILY='artifacts/research_c_cx_structural_blocks/2026-11-05'
CONTRACT='docs/research/research_c_cx_structural_block_contract_unapproved_20261105.json'
SCOPE='C_CX_INTERNAL_BLOCK_SAVED_PAIR_PILOT_V1'
LIMITS=dict(records_load=1,identity_gate=1,structural_plan=1,local_certificate=1,comparison=1,summary=1)
POLICY=dict(max_support=4,guard=1,exact_clifford=True)


def preflight(root):
    root=Path(root).resolve();c=read(root/CONTRACT)
    from .c_per_word_range_schema import same_json
    expected=dict(scope=SCOPE,base_commit=BASE,executable=True,approval_status='UNAPPROVED',
        maximum_logical_units=6,call_limits=LIMITS,block_policy=POLICY,precision_bits=192,
        controlled_dimension=512,retry=0,resume=0,mandatory_stop=True,new_sampling=0,new_build=0,
        new_compile=0,new_literal_reference=0,Stage_C_executable=False)
    if not same_json({k:c.get(k) for k in expected},expected):raise ValueError('FIXED_BLOCK_PILOT_SCHEMA')
    if sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in group.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or sha(path)!=want:raise ValueError('SOURCE_INPUT:'+name)
    if set(c['roles'])!=set(('compiled','raw','old_certificate','old_perword','saved_plan')):
        raise ValueError('FIXED_RECORD_ROLES')
    if any(path not in c['saved_inputs'] for path in c['roles'].values()):raise ValueError('UNPINNED_INPUT')
    parent=read(root/c['parent_contract'])
    if not same_json(c['identity'],parent['identity']) or not same_json(c['word'],parent['word']):
        raise ValueError('FIXED_WORD_MODEL_COMPILER')
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('PACKAGE:'+name)
    if c['fresh_output']!=FAMILY+'/block-pilot01' or (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_BLOCK_PILOT_READY_FOR_SEPARATE_PERMISSION',scientific_calls=0,
        native_records_decoded=0,new_H4_Arb_evaluations=0,maximum_calls=6,
        real_H4_new_upper=None,calibration_ready=False,Stage_C_ready=False,mandatory_stop=True)


def run_pilot(root,*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root);c=read(root/CONTRACT)
    from . import c_cx_structural_blocks as blocks
    from .c1_stage3_integrated import Calls
    from . import c1_stage1_execution as io
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model();out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    calls=Calls(out,LIMITS,6)
    io.save(out/'contract.json',c);io.save(out/'SOURCE.json',read(root/c['source_manifest']))
    try:
        values=calls('records_load','fixed C/raw/saved primitive plan/old bound/word;no build or compile',
            lambda:{k:read(root/v) for k,v in c['roles'].items()})
        def identity():
            C,R=values['compiled'],values['raw'];old=values['old_certificate'];word=values['old_perword']
            blocks.verify_saved_plan(values['saved_plan']['plan'],C,R)
            if (C['qubits']!=9 or R['qubits']!=9 or old['left_id']!=blocks._digest(C)
                    or old['right_id']!=blocks._digest(R) or word['compiled_id']!=blocks._digest(C)
                    or word['raw_id']!=blocks._digest(R) or [x[0] for x in word['binding']['word']]!=c['word']):
                raise ValueError('FIXED_PAIR_WORD_OPERATOR_IDENTITY')
            return True
        calls('identity_gate','source/record/phase/word and complete saved definition replay',identity)
        saved=values['saved_plan']['plan']
        def plan():
            tick=time.perf_counter();value=blocks.partition(saved,**POLICY)
            return dict(plan=value,wall_seconds=time.perf_counter()-tick)
        planned=calls('structural_plan','fixed exact rewrite and complete internal support partition',plan)
        io.save(out/'structural_plan.json',planned)
        if not planned['plan']['applicable']:
            result=calls('summary','registered support STOP;no bound,dense fallback or unused call transfer',lambda:dict(
                status='NOT_APPLICABLE_BLOCK_SUPPORT',upper_exact=None,new_H4_Arb_evaluations=0,
                unsupported=planned['plan']['unsupported'],mandatory_stop=True))
            io.save(out/'result.json',result);return result
        def evaluate():
            tick=time.perf_counter();value=blocks.evaluate(saved,planned['plan'],parallel=True)
            return dict(value,wall_seconds=time.perf_counter()-tick)
        cert=calls('local_certificate','local primitive/oneq/internal Gram enclosures and unitary telescope',evaluate)
        io.save(out/'compiled_raw.json',cert)
        def compare():
            blocks.check_binding(cert,values['compiled'],values['raw']);old=values['old_certificate']
            if F(cert['upper_exact'])<F(old['lower_exact']):raise ValueError('ENCLOSURE_CONTRADICTION')
            return dict(status='BOUND_NO_LOOSER_THAN_SAVED' if F(cert['upper_exact'])<=F(old['upper_exact']) else 'VALID_BUT_LOOSER',
                old_upper_exact=old['upper_exact'],new_upper_exact=cert['upper_exact'],
                old_wall_seconds=old['wall_seconds'],new_certificate_wall_seconds=cert['wall_seconds'],
                new_plan_wall_seconds=planned['wall_seconds'],
                new_plan_and_certificate_wall_seconds=cert['wall_seconds']+planned['wall_seconds'],
                historical_control_only=True,other_timing_in_call_journal=True,
                downstream_bias_qualification=False)
        comparison=calls('comparison','tightness and historical timing;no fresh baseline or condition tuning',compare)
        io.save(out/'comparison.json',comparison)
        result=calls('summary','fixed pair only,no literal target/all-law/calibration/Stage C/CA',lambda:dict(
            status='INTERNAL_BLOCK_FIXED_PAIR_COMPLETE',comparison=comparison,upper_exact=cert['upper_exact'],
            full_operator_coverage=cert['full_operator_coverage'],calibration_ready=False,Stage_C_ready=False,
            actual_CA_achieved=None,mandatory_stop=True))
        io.save(out/'result.json',result);return result
    except BaseException as exc:
        calls.stop.set()
        if calls.first_failure is None:calls.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(scientific_units=sum(calls.counts.values()),call_counts=calls.counts,
            failed=calls.first_failure,retry=0,resume=0,mandatory_stop=True,Stage_C_executed=False))
