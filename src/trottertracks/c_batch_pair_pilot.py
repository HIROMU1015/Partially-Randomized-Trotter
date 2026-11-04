"""Dormant seven-unit saved-record validation of the new native-pair engine."""
from pathlib import Path
from fractions import Fraction as F
import importlib.metadata
import time
from . import c_totality_calibration as integration
from . import c_arb_column_batch as batch
from . import c_stage_p_completion as prior

io=prior.io
SCOPE='C_FIXED_P_NATIVE_PAIR_BATCH_PILOT_V1'
CONTRACT='docs/research/research_c_fixed_pair_batch_pilot_contract_unapproved_20261104.json'
LIMITS=dict(evidence_metadata=1,records_load=1,identity_gate=1,accelerated_certificate=2,
            compatibility_report=1,summary=1)
MAXIMUM=7


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    expected=dict(base_commit=integration.BASE,scope=SCOPE,maximum_logical_units=7,call_limits=LIMITS,
        executable=True,method='fixed_saved_native_pair_validation',samples=1,statistical_delta=None,
        precision_bits=192,batch_size=16,retry=0,resume=0,mandatory_stop=True,
        new_sampling=0,new_build=0,new_compile=0,new_literal_reference=0,Stage_C_executable=False)
    if not prior.prior.schema.same_json({k:c[k] for k in expected},expected):raise ValueError('FIXED_BATCH_PILOT')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=h:raise ValueError('SOURCE_INPUT:'+p)
    for role in ('compiled','raw','compiled_inverse','old_compiled_raw','old_compiled_inverse','old_perword','population'):
        if c['roles'][role] not in c['saved_inputs']:raise ValueError('UNPINNED_ROLE')
    parent=io.read(root/c['parent_contract'])
    if not prior.prior.schema.same_json(c['identity'],parent['identity']) or c['word']!=parent['word']:
        raise ValueError('CANDIDATE_MODEL_COMPILER_WORD_POLICY')
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('PACKAGE:'+name)
    if c['fresh_output']!=integration.FAMILY+'/fixed-pair-pilot01' or (root/c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_FIXED_PAIR_READY_FOR_PERMISSION',scientific_calls=0,arrays_decoded=0,
        maximum_logical_units=7,calibration_ready=False,Stage_C_ready=False)


def run(root,path,*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_PAIR_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root,path);c=io.read(path)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model()
    from .c1_stage3_integrated import Calls
    from . import c_pr_aware_df_fixed_certificate as fixed
    from .c_per_word_population_range import _digest
    out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False);call=Calls(out,LIMITS,MAXIMUM)
    io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
    try:
        e=call('evidence_metadata','old completed certificates and population;no recalculation',lambda:{k:io.read(root/c['roles'][k]) for k in ('old_compiled_raw','old_compiled_inverse','old_perword','population')})
        records=call('records_load','three saved native records',lambda:{k:io.read(root/c['roles'][k]) for k in ('compiled','raw','compiled_inverse')})
        def identity():
            C,R=records['compiled'],records['raw'];old=e['old_compiled_raw']
            for value in records.values():
                fixed._validate(value)
                if value['qubits']!=9:raise ValueError('FULL512_TARGET')
            if (old['left_id']!=_digest(C) or old['right_id']!=_digest(R)
                    or e['old_compiled_inverse']['same_realized_forward_record']!=_digest(C)
                    or e['old_perword']['compiled_id']!=_digest(C)
                    or [x[0] for x in e['old_perword']['binding']['word']]!=c['word']):
                raise ValueError('SAVED_RECORD_PHASE_WORD_IDENTITY')
            return True
        call('identity_gate','fixed original record digests,phase,word',identity)
        results={}
        for name,left,right in [('compiled_raw',records['compiled'],records['raw']),
                                ('compiled_inverse',records['compiled_inverse'],fixed.adjoint(records['compiled']))]:
            def evaluate(left=left,right=right):
                tick=time.monotonic();value=batch.compare(left,right,batch_size=16,parallel=True)
                return dict(value,wall_seconds=time.monotonic()-tick)
            results[name]=call('accelerated_certificate',name,evaluate);io.save(out/(name+'.json'),results[name])
        def compatible():
            pairs=[(results['compiled_raw'],e['old_compiled_raw']),(results['compiled_inverse'],e['old_compiled_inverse'])]
            for new,old in pairs:
                if F(new['upper_exact'])<F(old['lower_exact']) or F(old['upper_exact'])<F(new['lower_exact']):
                    raise ValueError('ENCLOSURES_CONTRADICT')
            return dict(status='ENCLOSURES_NONCONTRADICTORY_SAME_TARGET',
                note='interval overlap is a consistency check,not the proof;target bindings/unitary reduction/Arb supply the argument',
                tightness_changed_allowed=True,speedup_on_unseen_words=None)
        check=call('compatibility_report','old certificates are not rerun',compatible);io.save(out/'compatibility.json',check)
        result=call('summary','fixed-pair only;no calibration',lambda:dict(status='FIXED_PAIR_BATCH_PILOT_COMPLETE',
            results=results,compatibility=check,all_law_qualified=False,calibration_ready=False,Stage_C_ready=False,actual_CA_achieved=None))
        io.save(out/'result.json',result);return result
    except BaseException as exc:
        call.stop.set()
        if call.first_failure is None:call.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(scientific_units=sum(call.counts.values()),call_counts=call.counts,
            failed=call.first_failure,retry=0,resume=0,mandatory_stop=True,Stage_C_executed=False))
