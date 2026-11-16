"""Approved LD9 -> LD8 -> LD7 diagnostic, separate from original P0 runs.

Default is metadata preflight. --execute uses a fixed scientific SOURCE and a
fresh output. No library-global tolerance change or production dictionary save.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_p0 as P0
from trottertracks import pauli_p0_operator_analysis as A

SAVED='9fa9525b3'
BASE='docs/research/b2_pauli_split_p0_20261116/'
PLAN=dict(commit='49a6d7adeed6c5dd3b2624db1cf7ea397f8e7050',
    path='docs/research/pauli_p0_gate_diagnosis_20261116/NEXT_CHECKS.md')
CODE=('scripts/run_pauli_p0_factor_trace.py','src/trottertracks/pauli_p0_factor_trace.py',
      'src/trottertracks/pauli_p0_operator_analysis.py')


def read(path):return json.loads(P0.blob(ROOT,SAVED,BASE+path))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:json.dump(value,f,indent=2,allow_nan=False)


def preflight():
    rows=[];binding=None
    for ld in (9,8,7):
        folder='LD%d/'%ld;result=read(folder+'result.json');meta=read(folder+'FIXED_INPUTS.json')
        dictionary=read(folder+'LD%d_DICTIONARY.json'%ld);jw=read(folder+'LD%d_JW_COEFFICIENTS.json'%ld)
        if binding is None:binding=meta['binding']
        A.require(binding==result['binding']==dictionary['binding']==meta['binding'],'SAVED_BINDING_DIFFERS')
        A.require(result['error']['message']=='P0_NUMERICAL_GATE: LD%d_Pauli_tail_full_fro'%ld,'ORIGINAL_STOP_SCOPE_DIFFERS')
        A.require(result['mandatory_stop'] and not result['automatic_retry'],'ORIGINAL_STOP_NOT_PRESERVED')
        A.require(meta['operator_gate']==A.TOL==P0.TOL==1e-9,'ORIGINAL_GATE_DIFFERS')
        for item in [*meta['inputs'].values(),meta['context_source']]:
            A.require(P0.sha(item['path'])==item['sha256'],'SAVED_INPUT_BYTES_DIFFER:'+item['path'])
        # Raw-JW coefficients and dictionary must still be the same producer.
        record=next(r for r in result['prefixes'] if r['prefix']==ld)
        raw=json.dumps(dictionary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        A.require(hashlib.sha256(raw).hexdigest()==record['dictionary_sha256'],'SAVED_DICTIONARY_HASH_DIFFERS')
        rows.append(dict(L_D=ld,folder=BASE+folder,scientific_source=result['scientific_source'],
            operator_source=dict(commit=SAVED,path=BASE+folder+'LD%d_OPERATORS.npz'%ld),
            raw_JW_source=dict(commit=SAVED,path=BASE+folder+'LD%d_JW_COEFFICIENTS.json'%ld),
            old_result_source=dict(commit=SAVED,path=BASE+folder+'result.json'),
            operator_gate=meta['operator_gate'],original_error=result['error'],
            original_stage1_source=result['stage1_extension']))
    for commit,path in (( 'fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f','src/trottertracks/df_head_pauli_tail_p0.py'),
                        ('75fb0ba114ac518bfe5d65d0164cee07d7469bb4','src/trotterlib/pauli_signal_adapter.py')):
        A.require((ROOT/path).read_bytes()==P0.blob(ROOT,commit,path),'FIXED_HELPER_DIFFERS:'+path)
    return dict(plan=PLAN,saved_carrier=SAVED,binding=binding,round_order=[9,8,7],rows=rows,
        reused='All saved tail/nonidentity operators, original JW, scalar, residual and fixed input bindings',
        new='Three saved-operator analyses and three split coefficient traces, default/shadow/bridge',
        no_new_reference_or_signal=True,no_production_dictionary_adoption=True,gate=A.TOL)


def qubit_operator(rows):
    from openfermion import QubitOperator
    op=QubitOperator()
    for row in rows:
        op.terms[tuple((q,axis) for q,axis in row['term'])]=complex(row['real'],row['imag'])
    return op


def matrix(op,n):
    import numpy as np
    model=P0.pauli_api().PauliHamiltonian.from_openfermion(op,num_qubits=n)
    perm=P0.reverse_bits(n)
    return model.sparse_operator().toarray()[np.ix_(perm,perm)],model


def raw_coefficients(op):
    return [dict(term=[[int(q),axis] for q,axis in term],real=float(complex(c).real),imag=float(complex(c).imag))
            for term,c in op.terms.items()]


def execute(source,out):
    import numpy as np
    import scipy
    import openfermion
    from openfermion.config import EQ_TOLERANCE
    from trottertracks import pauli_p0_factor_trace as F
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    A.require(source==head,'EXECUTION_REQUIRES_FIXED_CURRENT_SOURCE')
    for path in CODE:A.require((ROOT/path).read_bytes()==P0.blob(ROOT,source,path),'RUNNING_SOURCE_DIFFERS:'+path)
    A.require(not out.exists(),'FRESH_DIAGNOSTIC_OUTPUT_REQUIRED_NO_RETRY')
    out.mkdir(parents=True)
    start=time.monotonic();result=dict(status='STARTED',scientific_source=source,plan=PLAN,
        started_UTC=datetime.now(timezone.utc).isoformat(),gate=A.TOL,LD_order=[9,8,7],
        saved_analyses=[],factor_traces=[],scope=dict(saved_operator_analyses=0,saved_input_restores=0,split_traces=0,
            new_state_or_H_generation=0,reference_actions=0,mean_calls=0,normalizers=0,sampling=0,compile=0,Q3=0),
        production_dictionary_adopted=False,original_STOPs_preserved=True,automatic_retry=False,mandatory_stop=True)
    result['runtime']=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
                           scipy=scipy.__version__,openfermion=openfermion.__version__,EQ_TOLERANCE=EQ_TOLERANCE)
    write(out/'EXECUTION.json',result)
    def log(stage,event,**kwargs):
        value=dict(stage=stage,event=event,elapsed_seconds=time.monotonic()-start,
                   UTC=datetime.now(timezone.utc).isoformat(),**kwargs)
        print(json.dumps(value),flush=True)
        with (out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    @contextmanager
    def stage(name):
        before=time.monotonic();log(name,'START')
        try:yield
        except BaseException:
            log(name,'FAIL',stage_seconds=time.monotonic()-before);raise
        else:log(name,'DONE',stage_seconds=time.monotonic()-before)
    try:
        with stage('source_and_input_preflight'):
            facts=preflight();write(out/'PREFLIGHT.json',facts)
            A.require(EQ_TOLERANCE==1e-8,'OBSERVED_LIBRARY_TOLERANCE_DIFFERS')
        operators={}
        for ld in (9,8,7):
            with stage('LD%d_saved_operator_analysis'%ld):
                folder='LD%d/'%ld;old=read(folder+'result.json')
                rec=next(r for r in old['prefixes'] if r['prefix']==ld)
                saved_jw=read(folder+'LD%d_JW_COEFFICIENTS.json'%ld)
                with np.load(io.BytesIO(P0.blob(ROOT,SAVED,BASE+folder+'LD%d_OPERATORS.npz'%ld)),allow_pickle=False) as z:
                    ops={k:z[k] for k in z.files}
                A.require(ops['tail'].shape==ops['Pauli_nonidentity'].shape==(256,256),'FULL256_LAYOUT_DIFFERS')
                direct,model=matrix(qubit_operator(saved_jw),8)
                eye=np.eye(256)
                # Construct nonidentity separately to avoid diagonal cancellation from scalar subtraction.
                perm=P0.reverse_bits(8)
                nonidentity=model.sparse_operator(constant=0.).toarray()[np.ix_(perm,perm)]
                rebuild=float(np.linalg.norm(nonidentity-ops['Pauli_nonidentity']))
                A.require(rebuild<=A.TOL,'SAVED_JW_RECONSTRUCTION_DIFFERS')
                A.require(model.constant==rec['pauli_identity'],'SAVED_JW_SCALAR_DIFFERS')
                analysis=A.saved_analysis(ops['tail'],ops['Pauli_nonidentity'],model.constant,rec['saved_DF_descriptor']['scalar'])
                analysis.update(L_D=ld,source=facts['rows'][(9,8,7).index(ld)],
                    reconstructed_JW_vs_saved_A_frobenius=rebuild,
                    reconstructed_JW_vs_saved_A_bitwise=bool(np.array_equal(nonidentity,ops['Pauli_nonidentity'])))
                for name,field in [('using_JW_identity','Pauli_tail_full_fro'),('using_saved_DF_scalar','Pauli_tail_saved_scalar_fro')]:
                    discrepancy=abs(analysis[name]['frobenius']-rec['identity_metrics'][field])
                    A.require(discrepancy<=A.TOL,'SAVED_ABSOLUTE_RESIDUAL_DIFFERS')
                    analysis[name]['old_saved_absolute']=rec['identity_metrics'][field]
                    analysis[name]['recomputed_vs_saved_difference']=discrepancy
                write(out/('LD%d'%ld)/'SAVED_OPERATOR_ANALYSIS.json',analysis)
                result['saved_analyses'].append(analysis);result['scope']['saved_operator_analyses']+=1
                operators[ld]=dict(ops=ops,default_matrix=nonidentity+model.constant*eye,saved_jw=saved_jw)
                write(out/'RESULT.json',result)
        with stage('restore_saved_factors_once'):
            meta=read('LD9/FIXED_INPUTS.json')
            ham,sector,full_state,state,observed=P0.load_saved(ROOT,meta)
            result['scope']['saved_input_restores']=1;result['saved_loader_observed']=observed
            write(out/'SAVED_INPUT_RESTORE.json',dict(binding=meta['binding'],observed=observed))
        for ld in (9,8,7):
            with stage('LD%d_factor_trace'%ld):
                traced=F.trace(ROOT,ham,ld)
                detail=traced['evidence']
                with gzip.open(out/('LD%d'%ld)/'COEFFICIENT_TRACE.json.gz','wt',compresslevel=1) as f:
                    json.dump(detail,f,separators=(',',':'),allow_nan=False)
                original=raw_coefficients(traced['default_jw'])
                write(out/('LD%d'%ld)/'DIAGNOSTIC_JW.json',dict(default=original,
                    shadow_no_intermediate_drop=raw_coefficients(traced['shadow_jw']),
                    no_filter_JW_of_default_Fermion=raw_coefficients(traced['bridge_jw']),
                    diagnostic_only_not_a_production_dictionary=True))
                A.require(original==operators[ld]['saved_jw'],'TRACED_DEFAULT_DIFFERS_FROM_SAVED_RAW_JW')
                shadow,_=matrix(traced['shadow_jw'],ham.n_qubits)
                bridge,_=matrix(traced['bridge_jw'],ham.n_qubits)
                cause=A.attribute(operators[ld]['default_matrix'],bridge,shadow,operators[ld]['ops']['tail'])
                cause.update(L_D=ld,default_JW_exact_saved_match=True,
                    trace_file='LD%d/COEFFICIENT_TRACE.json.gz'%ld,
                    output_files='LD%d/DIAGNOSTIC_JW.json'%ld)
                write(out/('LD%d'%ld)/'ATTRIBUTION.json',cause)
                result['factor_traces'].append(cause);result['scope']['split_traces']+=1
                write(out/'RESULT.json',result)
                A.require(cause['shadow']['within_original_absolute_gate'],'SHADOW_DOES_NOT_MATCH_SAVED_DF_TAIL')
                A.require(openfermion.config.EQ_TOLERANCE==EQ_TOLERANCE==1e-8,'GLOBAL_TOLERANCE_CHANGED')
        result['status']='DIAGNOSTIC_COMPLETE_ORIGINAL_P0_STOPS_UNCHANGED'
    except BaseException as exc:
        result.update(status='STOP_DIAGNOSTIC',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        log('diagnostic','STOP',error=str(exc))
    result['elapsed_seconds']=time.monotonic()-start;result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    write(out/'RESULT.json',result);return 0 if result['status'].startswith('DIAGNOSTIC_COMPLETE') else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--source');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.execute:
        if not args.source or not args.out:p.error('--execute requires --source and a fresh --out')
        return execute(args.source,args.out)
    print(json.dumps(preflight(),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
