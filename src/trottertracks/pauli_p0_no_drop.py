"""P0-local nonzero coefficient accumulation and the unchanged P0 gates.

Only this dictionary builder uses local operator subclasses. OpenFermion's
global classes/tolerance, saved factors, multiplication and the fixed P0 module
are unchanged. No positive final cutoff is applied to the collected terms.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from openfermion import FermionOperator, QubitOperator
from openfermion.config import EQ_TOLERANCE

from . import df_head_pauli_tail_p0 as P0
from . import pauli_p0_factor_trace as fixed_trace

MODULE='src/trottertracks/pauli_p0_no_drop.py'
BUILDER_ID='P0_NO_INTERMEDIATE_DROP_V1'
P0_CODE='fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f'
TRACE_CODE='655e742121f0ae52083402ceac4cdf32afeff618'
P0_MODULE='src/trottertracks/df_head_pauli_tail_p0.py'
HISTORICAL_CONTROL=dict(commit='4220f3006d0d8d413e195ed948e250578513b557',
    path='docs/research/df_head_pauli_tail_p0_results_20261114/run01/result.json')


def provenance(source):
    return dict(id=BUILDER_ID,source=dict(commit=source,path=MODULE),
        one_body='FIXED_M1_ONE_BODY_NONZERO_COMPONENTS',
        factor_order='SAVED_SNAPSHOT_ORDER',weights='SAVED_SIGNED_WEIGHTS_UNCHANGED',
        multiplication='EXISTING_OPENFERMION_PRODUCT_AND_SCALAR_MULTIPLICATION',
        accumulation='KEEP_NONZERO_PARTIAL_SUMS_IN_ONE_BODY_FERMION_AND_JW',
        exact_zero_terms_removed=True,intermediate_coefficient_cutoff=0.,final_coefficient_cutoff=0.,
        global_EQ_TOLERANCE_changed=False,
        scalar='KEEP_SAVED_sR_AND_c_ONCE',mapping='EXISTING_JW_AND_REVERSE_BITS')


def _operator_class(base):
    class LocalNoDrop(base):
        def __iadd__(self, addend):
            if not isinstance(addend,type(self)):
                raise TypeError('Local P0 accumulation requires the same operator type')
            for term,coefficient in addend.terms.items():
                value=self.terms.get(term,0.)+coefficient
                if value!=0:self.terms[term]=value
                else:self.terms.pop(term,None)
            return self
    return LocalNoDrop


def square_factor_dictionary(root,ham,prefix):
    """Same fixed M1/JW path, retaining every nonzero intermediate coefficient."""
    if type(prefix)is not int or not 0<=prefix<len(ham.lambdas):
        raise ValueError('NONEMPTY_EXPLICIT_FACTOR_PREFIX_REQUIRED')
    if len(ham.lambdas)!=len(ham.g_matrices):
        raise ValueError('FACTOR_WEIGHT_COUNT_DIFFERS')
    local_f=_operator_class(FermionOperator);local_q=_operator_class(QubitOperator)
    one=fixed_trace.one_body(root,local_f)
    residual=local_f()
    for weight,g in zip(ham.lambdas[prefix:],ham.g_matrices[prefix:],strict=True):
        if not np.isfinite(weight) or not np.isfinite(g).all():
            raise ValueError('NONFINITE_SAVED_FACTOR')
        a=one(g)
        residual+=float(weight)*(a*a)
    op=fixed_trace.jw_function(local_q)(residual)
    return fixed_trace.to_base(op,QubitOperator)


def diagnose(root,ham,sector,full_state,state,meta,result,save,*,prefix,source):
    """All fixed common/state/operator/scalar/sector P0 checks for one split.

    The fixed function has only its literal prefix loop changed, its local
    square-factor dependency replaced, and a provenance hook before the payload
    hash. No gate or matrix formula is modified.
    """
    if type(prefix)is not int or not 1<=prefix<ham.n_blocks:
        raise ValueError('PREFIX_OUTSIDE_NONEMPTY_TAIL')
    builder=provenance(source)
    node=next(n for n in ast.parse(P0.blob(root,P0_CODE,P0_MODULE)).body
              if isinstance(n,ast.FunctionDef)and n.name=='diagnose')
    loops=[n for n in ast.walk(node)if isinstance(n,ast.For)
           and isinstance(n.target,ast.Name)and n.target.id=='prefix']
    if len(loops)!=1 or not isinstance(loops[0].iter,ast.Tuple)or ast.literal_eval(loops[0].iter)!=(1,2):
        raise ValueError('FIXED_P0_PREFIX_LOOP_NOT_FOUND')
    loops[0].iter=ast.copy_location(ast.Name(id='REQUESTED_PREFIXES',ctx=ast.Load()),loops[0].iter)
    payloads=[(body,index,n)for body in (loops[0].body,)for index,n in enumerate(body)
              if isinstance(n,ast.Assign)and len(n.targets)==1
              and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='payload']
    if len(payloads)!=1:raise ValueError('FIXED_P0_PAYLOAD_HOOK_NOT_FOUND')
    body,index,assignment=payloads[0]
    hook=ast.Expr(value=ast.Call(func=ast.Name(id='STAMP_BUILDER',ctx=ast.Load()),
        args=[ast.Name(id='payload',ctx=ast.Load()),ast.Name(id='record',ctx=ast.Load()),
              ast.Name(id='tail',ctx=ast.Load())],keywords=[]))
    body.insert(index+1,ast.copy_location(hook,assignment))
    def stamp(payload,record,tail):
        payload['builder']=deepcopy(builder)
        payload['builder_source']=deepcopy(builder['source'])
        record['builder']=deepcopy(builder)
        record['operator_scale']=dict(tail_frobenius=float(np.linalg.norm(tail)))
    scope=dict(vars(P0),REQUESTED_PREFIXES=(prefix,),square_factor_dictionary=square_factor_dictionary,
               STAMP_BUILDER=stamp)
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,P0_CODE+':'+P0_MODULE+':'+BUILDER_ID,'exec'),scope)
    try:
        return scope['diagnose'](root,ham,sector,full_state,state,meta,result,save)
    finally:
        for record in result.get('prefixes',[]):
            scale=record.get('operator_scale',{}).get('tail_frobenius')
            residual=record.get('identity_metrics',{}).get('Pauli_tail_full_fro')
            if scale is not None and residual is not None:
                record['relative_full_residual']=residual/scale if scale else None
            record['dictionary_eligible_for_configuration']='GATES_PASS' in record.get('status','')


def success(prefix):
    return 'P0_LD%d_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'%prefix


def source_records(root,source):
    """Pin used code; permit subsequent commits containing only own results."""
    out=[]
    for commit,path in ((source,MODULE),(TRACE_CODE,'src/trottertracks/pauli_p0_factor_trace.py'),
                        (P0_CODE,P0_MODULE),
                        (P0.PLAN,'src/trotterlib/pauli_signal_adapter.py'),
                        (P0.PLAN,'src/trotterlib/df_hamiltonian.py'),
                        (P0.PLAN,'src/trotterlib/df_partial_s2.py'),
                        (P0.PLAN,'src/trotterlib/pr2_s0_s1_validation.py')):
        raw=P0.blob(root,commit,path)
        if (Path(root)/path).read_bytes()!=raw:raise ValueError('USED_SOURCE_CHANGED:'+path)
        out.append(dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest()))
    for commit,path in ((P0.SAVED,'scripts/h46_m5_tol_saved_input.py'),
                        (P0.SAVED,'scripts/h46_new2_core.py'),
                        (P0.PLAN,'src/trottertracks/h6_df19_rpe_d2.py'),
                        (P0.M1,'src/trotterlib/o1_m1_matched_split.py')):
        out.append(dict(commit=commit,path=path,sha256=hashlib.sha256(P0.blob(root,commit,path)).hexdigest()))
    return out


def execute_p0(root,out,source,prefix,descriptor,descriptor_source):
    """Fresh scoped scientific leaf called by the ordinary preparation entry.

    Original P0 output directories and STOPs are not reopened. The caller owns
    the subsequent artifact commit and canonical P1 metadata generation.
    """
    import openfermion
    import scipy
    import run_df_head_pauli_tail_ld3 as R3
    root=Path(root).resolve();out=Path(out)
    if out.exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    save=R3.writer(out);start=time.monotonic();builder=provenance(source)
    result=dict(status='STARTED',stage='P0_LD%d'%prefix,fixed_P0_source=P0_CODE,
        scientific_source=source,builder=builder,builder_policy_adopted=True,
        prefixes_run=[prefix],new_prefix=prefix,prefixes=[],checks=[],
        baseline_control=dict(status='HISTORICAL_ONLY_NOT_RECOMPUTED_FOR_NEW_BUILDER',source=HISTORICAL_CONTROL),
        started_UTC=datetime.now(timezone.utc).isoformat(),
        scope=dict(saved_input_loads=0,full_Fock_extensions=0,Pauli_dictionaries=0,
            mean_evaluations=0,reference_actions=0,eigensolves=0,finite_normalizers=0,
            sampling=0,circuit_build_compile=0,QPY_reads=0,price_Q3_shot_allocations=0),
        sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,
        scientific_dictionary_adopted=False,automatic_retry=False,mandatory_stop=True,
        dictionary_eligible_for_configuration=False,
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
                     scipy=scipy.__version__,openfermion=openfermion.__version__),
        coefficient_policy=dict(builder_id=BUILDER_ID,adapter_extra_cutoff=0.,
            collector_imaginary_tolerance=1e-12,openfermion_EQ_TOLERANCE=float(EQ_TOLERANCE),
            intermediate_coefficient_cutoff=0.,final_coefficient_cutoff=0.,
            global_tolerance_changed=False,rigorous_operator_certificate=False))
    save('EXECUTION.json',result)
    def log(stage,event,**fields):
        row=dict(stage=stage,event=event,elapsed_seconds=time.monotonic()-start,
                 UTC=datetime.now(timezone.utc).isoformat(),**fields)
        print(json.dumps(row),flush=True)
        with(out/'PROGRESS.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    try:
        log('sources_and_metadata','START')
        result['code_sources']=source_records(root,source)
        if EQ_TOLERANCE!=1e-8:raise ValueError('OPENFERMION_RUNTIME_TOLERANCE_DIFFERS')
        meta=P0.metadata(root)
        if not isinstance(descriptor,dict)or any(not math.isfinite(descriptor[k])for k in('scalar','lambda_R')):
            raise ValueError('SAVED_DF_DESCRIPTOR_REQUIRED')
        meta.update(saved_tail_descriptors={str(prefix):deepcopy(descriptor)},tail_prefixes=[prefix],
            tail_descriptor_sources={str(prefix):deepcopy(descriptor_source)},builder=builder)
        save('FIXED_INPUTS.json',meta);result['binding']=meta['binding']
        log('sources_and_metadata','DONE')
        log('saved_input_load','START')
        ham,sector,full_state,state,observed=P0.load_saved(root,meta)
        result['scope']['saved_input_loads']=1;result['saved_loader_observed']=observed
        log('saved_input_load','DONE')
        log('full_P0_checks','START')
        diagnose(root,ham,sector,full_state,state,meta,result,save,prefix=prefix,source=source)
        result['scope']['Pauli_dictionaries']=1
        for item in[*meta['inputs'].values(),meta['context_source']]:
            if P0.sha(item['path'])!=item['sha256']:raise ValueError('USED_INPUT_BYTES_CHANGED_DURING_P0')
        if openfermion.config.EQ_TOLERANCE!=EQ_TOLERANCE:raise ValueError('GLOBAL_TOLERANCE_CHANGED')
        result['status']=success(prefix)
        result['dictionary_eligible_for_configuration']=True
        log('full_P0_checks','DONE',checks=len(result['checks']))
    except Exception as error:
        result.update(status='STOP_P0_LD%d_INPUT_OR_NUMERICAL_FAILURE'%prefix,
            error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        result['scope']['Pauli_dictionaries']=int((out/('LD%d_DICTIONARY.json'%prefix)).exists())
        log('P0','STOP',error=str(error))
    result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds']=time.monotonic()-start
    save('result.json',result)
    return result
