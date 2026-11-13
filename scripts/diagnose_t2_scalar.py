#!/usr/bin/env python3
"""One saved-input restore and one unchanged context call. No sampling/build.

Observe the original guard exception frame; never replace its scalar or retry.
The old T2 preflight is reused only for read-only SOURCE/input validation.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import ld2_m12_t2 as T

BASE='752b9747a363eb4f1e510092230ef8dc243187ca'
T2_SOURCE='8291f16ceb6b88cc8a43dbe508db419b9a26f373'
DIRECTORY='docs/research/d_t2_scalar_diagnostic_20261113'
GUARD='Prepared scalar differs from the requested total scalar'


def number(x):
    x=float(x)
    if not math.isfinite(x):raise ValueError('Finite scalar required')
    return dict(value=x,decimal=repr(x),decimal_exact=str(Decimal.from_float(x)),hex=x.hex())


def rational(x):
    return dict(numerator=x.numerator,denominator=x.denominator)


def ordered(x):
    u=struct.unpack('>Q',struct.pack('>d',float(x)))[0]
    return (~u & ((1<<64)-1)) if u>>63 else (u | (1<<63))


def difference(actual,expected):
    a,b=float(actual),float(expected)
    exact=Fraction.from_float(a)-Fraction.from_float(b)
    return dict(binary64_subtraction=number(a-b),exact_input_difference=rational(exact),
        signed_adjacent_binary64_steps=ordered(a)-ordered(b),
        ulp_at_expected=number(math.ulp(b)),
        exact_difference_in_expected_ULPs=rational(exact/Fraction.from_float(math.ulp(b))))


def observe_context(runtime,request):
    """Capture locals from this ONE call, without touching science code or order."""
    try:
        ctx=runtime.context(request)
    except BaseException as error:
        record=dict(context_returned=False,error_type=type(error).__name__,error=str(error),
                    traceback=traceback.format_exc(),guard_stop=str(error)==GUARD)
        node=error.__traceback__;found=None
        code=runtime.context.__func__.__code__
        while node:
            if node.tb_frame.f_code is code:
                local=node.tb_frame.f_locals
                if local.get('request') is request and all(k in local for k in ('prep','scalar','lam')):
                    found=dict(prep=local['prep'],scalar=local['scalar'],lam=local['lam'])
            node=node.tb_next
        if found is None:return record,None
        return record,found
    return dict(context_returned=True,guard_stop=False),dict(prep=ctx.prep,scalar=ctx.scalar,
        lam=ctx.prep.rte_preparation.symbolic_tail.lambda_r)


def inventory(prep):
    """Only prepared metadata; no event, circuit, basis-unitary or new eigensolve."""
    tail=prep.rte_preparation.symbolic_tail
    components=[asdict(c) for c in tail.components]
    bases=[asdict(b) for b in tail.basis_definitions]
    head=[{k:getattr(b,k) for k in ('block_id','block_kind','original_fragment_index',
          'order_index','basis_id','basis_hash')} for b in prep.deterministic_blocks]
    return dict(hamiltonian_hash=prep.hamiltonian_hash,partition_hash=prep.partition_hash,
        preparation_hash=prep.preparation_hash,tail_hash=tail.tail_hash,
        deterministic_fragments=list(prep.deterministic_fragment_indices),
        randomized_fragments=list(prep.randomized_block_indices),
        identity_policy=prep.identity_policy,coefficient_atol=prep.coefficient_atol,
        diagonal_sort=prep.diagonal_sort,basis_construction_policy=prep.basis_construction_policy,
        component_count=len(components),ordered_component_metadata_sha256=T.D._key(components),
        component_hash_definition='candidate_price_adapter._key([dataclasses.asdict(component), ...])',
        component_ids=[c['component_id'] for c in components],
        head_basis_metadata=head,basis_definitions=bases,basis_metadata_sha256=T.D._key(bases),
        extraction_metadata=asdict(tail.extraction_metadata),
        normalization_metadata=asdict(tail.normalization_metadata))


def run_once(root,source,request,inputs,runtime_plan,out,*,descriptor,
             restore=T.restore_h4,runtime_class=None):
    """Only scientific call sites here: restore once, context once; then STOP."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();stages={};counts=dict(input_restore=0,context=0,
        sampling=0,event_path_generation=0,native_evolution_build=0,wrapper_build=0,compile=0,T2_retry=0)
    record=dict(source=source,base=BASE,fixed_T2_execution_source=T2_SOURCE,
        provider_source=T.D.zz.PROVIDER,input_files=inputs,request=request,A_descriptor=descriptor,
        counts=counts,mean_compiled_price=None,G=None,mandatory_stop=True)
    try:
        T.check_scope(request)
        before=time.perf_counter();counts['input_restore']+=1
        ham,md=restore(root,request,inputs,runtime_plan)
        stages['input_restore_seconds']=time.perf_counter()-before
        runtime_class=runtime_class or T.D.NativeRuntime
        runtime=runtime_class(root,ham,binding=request['physical_context']['binding'],
            input_source=request['physical_context']['input_source'],execution_source={'commit':source})
        before=time.perf_counter();counts['context']+=1
        outcome,found=observe_context(runtime,request)
        stages['context_seconds']=time.perf_counter()-before
        record.update(input_metadata=md,context_outcome=outcome,
                      preparation_cache_entries=len(runtime.preparations))
        if found is None:
            record['status']='DIAGNOSTIC_STOP_NO_SCALAR_LOCALS'
        else:
            prep=found['prep'];constant=float(ham.constant)
            extracted=float(prep.extracted_identity_coefficient)
            actual=float(found['scalar']);expected=float(request['physical_context']['expected_scalar'])
            descriptor_scalar=float(descriptor['scalar'])
            status=('DIAGNOSTIC_COMPLETE_GUARD_STOP' if outcome['guard_stop'] else
                    'DIAGNOSTIC_COMPLETE_CONTEXT_RETURNED' if outcome['context_returned'] else
                    'DIAGNOSTIC_STOP_OTHER_EXCEPTION_WITH_SCALAR')
            record.update(status=status,
                scalars=dict(ham_constant=number(constant),extracted_identity=number(extracted),
                    total_from_original_context=number(actual),request_expected=number(expected),
                    A_saved_tail_descriptor=number(descriptor_scalar),
                    preparation_constant=number(prep.constant_coefficient)),
                actual_minus_request=difference(actual,expected),
                extracted_minus_A_descriptor=difference(extracted,descriptor_scalar),
                original_addition_roundoff_exact=rational(Fraction.from_float(actual)-
                    (Fraction.from_float(constant)+Fraction.from_float(extracted))),
                diagnostic_checks=dict(
                    extracted_equals_descriptor_hex=extracted.hex()==descriptor_scalar.hex(),
                    requested_equals_descriptor_hex=expected.hex()==descriptor_scalar.hex(),
                    descriptor_plus_constant_equals_actual_hex=(constant+descriptor_scalar).hex()==actual.hex()),
                lambda_R=number(found['lam']))
            # This is a diagnostic expression, never a replacement request/scalar.
            record['descriptor_plus_constant_diagnostic_only']=number(constant+descriptor_scalar)
            metadata=inventory(prep)
            json.dumps(metadata,allow_nan=False)  # optional metadata cannot lose captured scalars
            record['preparation']=metadata
    except BaseException:
        record.update(status='DIAGNOSTIC_STOP_EXCEPTION',error=traceback.format_exc())
    finally:
        record.update(stages_seconds=stages,total_wall_seconds=time.perf_counter()-start,
                      peak_rss_bytes=T.rss(),input_hashes_after={k:T.sha(v['path']) for k,v in inputs.items()})
        T.write(out/'DIAGNOSTIC_RESULT.json',record)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--source',required=True)
    args=ap.parse_args()
    if not args.execute:ap.error('--execute required for the one preparation diagnostic')
    source=args.source
    own='scripts/diagnose_t2_scalar.py'
    if (ROOT/own).read_bytes()!=T.P._blob(ROOT,source,own):raise ValueError('DIAGNOSTIC_SOURCE_DIFFERS')
    for name in T.FILES:
        if (ROOT/name).read_bytes()!=T.P._blob(ROOT,T2_SOURCE,name):raise ValueError('FIXED_T2_SOURCE_CHANGED:'+name)
    fixed=json.loads(T.P._blob(ROOT,BASE,'docs/research/d_ld2_m12_t2_20261113/PLAN.json'))
    request,inputs=T.preflight(ROOT,source,fixed)  # passive only; no old T2 runner
    provider=Path(fixed['runtime']['provider_root'])
    for name in ('df_rte_tail.py','df_trotter/decompose.py','pr2_matched_accuracy_m1_execution.py'):
        path='src/trotterlib/'+name
        if (provider/path).read_bytes()!=T.P._blob(ROOT,T.D.zz.PROVIDER,path):raise ValueError('PREPARATION_PROVIDER_SOURCE:'+path)
    tail_source=request['physical_context']['input_source']['tail']
    index=T.D.p.SavedArtifacts(ROOT).read(tail_source['index'])
    item=index['T20_reference_files']['H4']
    if T.sha(item['path'])!=item['sha256'] or item['sha256']!=tail_source['context_sha256']:
        raise ValueError('A_SAVED_DESCRIPTOR_SOURCE')
    descriptor=json.loads(Path(item['path']).read_text())['prefixes']['2']
    if descriptor['scalar']!=request['physical_context']['expected_scalar']:raise ValueError('A_DESCRIPTOR_REQUEST_DIFFERS')
    plan=json.loads((ROOT/DIRECTORY/'PLAN.json').read_text())
    if plan['context_calls']!=1 or any(plan[k]!=0 for k in ('sampling','event_path_generation','native_evolution_build','wrapper_build','compile','T2_retry')):
        raise ValueError('DIAGNOSTIC_SCOPE')
    if (ROOT/DIRECTORY/'PLAN.json').read_bytes()!=T.P._blob(ROOT,source,DIRECTORY+'/PLAN.json'):
        raise ValueError('DIAGNOSTIC_PLAN_SOURCE')
    result=run_once(ROOT,source,request,inputs,fixed['runtime'],ROOT/DIRECTORY/'run01',
                    descriptor=dict(descriptor,source=tail_source,path=item['path']))
    print(result['status'],result['counts'],flush=True)
    if not result['status'].startswith('DIAGNOSTIC_COMPLETE'):raise SystemExit(1)


if __name__=='__main__':main()
