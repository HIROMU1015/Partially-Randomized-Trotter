"""Lazy exact deterministic signal and paired native cost, after sealed grant."""
from pathlib import Path
from types import SimpleNamespace
import math
import time
import pf_common_run_contract as c

CTX=None
STAGE=None
SELECTED=None

def install(plan):
    import sys
    sys.path.insert(0,str(Path(plan['provider'])/'src'))
    import numpy as np,qiskit
    c.require(np.__version__=='1.26.4' and qiskit.__version__=='1.3.0','PINNED_NUMPY_QISKIT')
    from functools import partial
    import trotterlib.rte_compiled_cost as compiler
    if plan['system']=='H2':
        from h2_m5_compiler_phase_guard import transpile
    else:transpile=qiskit.transpile
    compiler.transpile=partial(transpile,num_processes=1)

def array_receipt_matches(value,record,*,original_shape):
    """H6 receipt uses NPY shape; migrated H2/H4 receipt used contiguous shape."""
    import numpy as np,hashlib
    v=np.ascontiguousarray(value)
    shape=np.asarray(value).shape if original_shape else v.shape
    return (list(shape)==record['shape'] and v.dtype.str==record['dtype']
        and hashlib.sha256(v.tobytes()).hexdigest()==record['sha256'])

def read_snapshot_arrays(snapshot,receipt,*,system):
    """Load original or migrated arrays under their own receipt convention."""
    import numpy as np
    with np.load(snapshot,allow_pickle=False) as z:
        c.require(set(z.files)==set(receipt['arrays']),'EXACT_SCIENTIFIC_ARRAY_NAMES')
        arrays={name:z[name] for name in z.files}
    for name,value in arrays.items():
        c.require(array_receipt_matches(value,receipt['arrays'][name],original_shape=system=='H6'),
            'EXACT_SCIENTIFIC_ARRAY:'+name)
    return arrays

def restore(stage,prefixes):
    plan=stage.plan;install(plan)
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _state_hash
    from h46_new2_native import OccupationOracle,prepare
    from h46_new2_core import matrix_only_tail_coefficients
    inp=plan['input_paths'];r=c.read(inp['receipt'])
    stage.take('snapshot_restore')
    a=read_snapshot_arrays(inp['snapshot'],r,system=plan['system'])
    if plan['system']=='H6':
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,_,_,_,_=load_h6_snapshot(Path(inp['snapshot']),r,c.read(inp['df_receipt']))
    else:ham=DFHamiltonian(float(a['constant']),a['one_body'],a['lambdas'],tuple(a['g_matrices']),r['hamiltonian_metadata'])
    full,state=a['state_vector'],a['sector_state_vector'];bind=plan['binding']
    c.require(df_hamiltonian_hash(ham)==bind['hamiltonian'] and _state_hash(full,state)==bind['state'] and ham.n_blocks==bind['rank'],'EXACT_OLD_PR_HAMILTONIAN_STATE')
    stage.take('occupation_oracle');oracle=OccupationOracle(ham,full,prefix=ham.n_blocks)
    qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
    stage.take('saved_reference_read')
    with np.load(inp['reference'],allow_pickle=False) as z:
        reference=z['reference'].copy();c.require(np.array_equal(z['qstate'],qstate),'SAVED_REFERENCE_SAME_STATE')
    receipt=c.read(inp['reference_receipt']);stage.take('full_preparation');fullprep=prepare(ham,ham.n_blocks)
    c.require(fullprep.preparation_hash==receipt['full_preparation'],'EXACT_SAVED_FULL_PREPARATION')
    preparations={ham.n_blocks:fullprep};scalars={ham.n_blocks:0.}
    for p in prefixes:
        if p==ham.n_blocks:continue
        stage.take('prefix_preparation',str(p));cut=prepare(ham,p)
        scalar,lam=matrix_only_tail_coefficients(ham,p)
        c.require(abs(cut.extracted_identity_coefficient-scalar)<1e-9 and abs(cut.exact_rte_lambda_r-lam)<1e-9,'PREFIX_SCALAR_LAMBDA')
        preparations[p]=cut;scalars[p]=scalar
    return SimpleNamespace(np=np,ham=ham,oracle=oracle,qstate=qstate,reference=reference,
        fullprep=fullprep,preparations=preparations,scalars=scalars,scalar=ham.constant,plan=plan)

def forward_matrices(ctx):
    from h46_new2_core import occupation_one_body
    layout=SimpleNamespace(basis_indices=ctx.oracle.basis)
    matrices=[occupation_one_body(ctx.ham.one_body,layout)]
    for w,g in zip(ctx.ham.lambdas,ctx.ham.g_matrices):
        v=occupation_one_body(g,layout);matrices.append(w*(v@v))
    return matrices

def forward_propagator(stage,ctx,index,time,matrix,task):
    from scipy.linalg import expm
    if not hasattr(ctx,'forward_cache'):return expm(-.5j*time*matrix)
    key=(index,float(time).hex())
    with ctx.forward_cache_lock:
        if key not in ctx.forward_cache:
            stage.take('forward_expm',str(index)+'_'+float(time).hex())
            ctx.forward_cache[key]=expm(-.5j*time*matrix)
        return ctx.forward_cache[key]

def new_signal(stage,ctx,row,matrices):
    from scipy.linalg import expm
    from h46_m5_native import signal
    import h46_m5_native as old
    from h46_new2_core import complex_record
    identity=c.cid(row);p=row['prefix'];n=c.PIECES[row['formula']]*row['q'];actions=2*(p+1)*n
    stage.take('base_signal',identity);stage.take('spectral_piece',identity,n);stage.take('spectral_block_action',identity,actions)
    local=SimpleNamespace(**ctx.__dict__,tail_identity=ctx.scalars[p])
    record=signal(local,dict(row,policy=None,r_value=None))
    stage.take('independent_forward',identity);stage.take('forward_block_action',identity,actions)
    v=ctx.oracle.initial.copy()
    for step in old.c.core.schedule(row['formula'],row['q'],'PER_PIECE',1):
        for i,matrix in enumerate(matrices[:p+1]):v=forward_propagator(stage,ctx,i,step['signed_time'],matrix,identity)@v
        for i in reversed(range(p+1)):v=forward_propagator(stage,ctx,i,step['signed_time'],matrices[i],identity)@v
    v*=ctx.np.exp(-.8j*ctx.scalar)
    expected=complex(ctx.np.vdot(ctx.oracle.initial,v));actual=complex(record['corrected']['real'],record['corrected']['imag'])
    err=abs(expected-actual);c.require(math.isfinite(err) and err<1e-9,'INDEPENDENT_FORWARD_SIGNAL')
    return dict(signal=record,old_u=0.,old_margin=0.,preparation_hash=ctx.preparations[p].preparation_hash,
        independent_forward_signal=complex_record(expected),independent_forward_error=err,
        provenance='NEW_COMMON_DOMAIN_SIGNAL',candidate=row)

def overlay(stage,ctx,row,base):
    import copy
    from h46_new2_core import complex_record
    identity=c.cid(row);stage.take('analytic_overlay',identity)
    r=copy.deepcopy(base);sig=r['signal'];phase=ctx.np.exp(-.8j*ctx.scalars[row['prefix']])
    for k in ('corrected','raw','exact_tail'):
        if k in sig:sig[k]=complex_record(phase*complex(sig[k]['real'],sig[k]['imag']))
    corrected=complex(sig['corrected']['real'],sig['corrected']['imag']);ref=complex(sig['reference']['real'],sig['reference']['imag'])
    sig['total_signed_bias']=complex_record(corrected-ref)
    sig['outer_PF_or_discard_bias']=dict(sig['total_signed_bias']);sig['finite_RTE_bias']=dict(real=0.,imag=0.)
    sig['allowance_evidence']['maximum_intermediate_norm']=max(sig['allowance_evidence']['maximum_intermediate_norm'],
        abs(phase)*sig['allowance_evidence']['maximum_intermediate_norm'])
    r.update(candidate=row,provenance='NEW_ANALYTIC_SCALAR_OVERLAY',base_signal=c.cid(dict(row,method='B0')),
        tail_identity=ctx.scalars[row['prefix']],reference_rotated=False,preparation_hash=ctx.preparations[row['prefix']].preparation_hash)
    return r

def cost_one(row):
    """Forked worker. Explicit top-level identity avoids rebound pickle failures."""
    import os,resource,gzip,hashlib,io
    import numpy as np
    from qiskit import QuantumCircuit,qpy
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from trotterlib.df_trotter.circuit import simulate_statevector
    import h46_m5_native as old
    stage,ctx=STAGE,CTX;identity=c.cid(row);out=stage.out/identity;out.mkdir(exist_ok=False);started=time.monotonic()
    def take(kind,n=1):stage.take(kind,identity,n)
    def gate(name,error,tol):
        value=float(error);c.require(math.isfinite(value) and 0<=value<tol,'NATIVE_GATE:'+name+':'+str(value))
        return dict(check=name,error=value,strictly_below=tol,PASS=True,phase_alignment=False)
    def archive(name,data):
        rawsha=hashlib.sha256(data).hexdigest();blob=gzip.compress(data,compresslevel=1,mtime=0)
        c.require(hashlib.sha256(gzip.decompress(blob)).hexdigest()==rawsha,'RAW_BYTE_ROUNDTRIP')
        path=out/(name+'.gz')
        with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        return dict(path=path.name,original_bytes=len(data),original_sha256=rawsha,archive_sha256=hashlib.sha256(blob).hexdigest(),compressed_bytes=len(blob),restoration_PASS=True)
    try:
        p=row['prefix'];prep=ctx.preparations[p];scalar=ctx.scalar+(ctx.scalars[p] if row['method']=='B0I' else 0.)
        with use_fingerprint_reuse() as fp:
            take('native_piece',c.PIECES[row['formula']]*row['q']);take('native_evolution')
            evolution=old.compose(dict(row,policy=None,r_value=None),prep,(),scalar,native,QuantumCircuit,fp.fingerprint)
            take('independent_occupation');v=ctx.oracle.initial.copy()
            for s in old.c.core.schedule(row['formula'],row['q'],'PER_PIECE',1):
                for spec in list(ctx.oracle.specs[:p+1])+list(reversed(ctx.oracle.specs[:p+1])):
                    v=spec[1]@(np.exp(-.5j*s['signed_time']*spec[0])*(spec[2]@v))
            v*=np.exp(-.8j*scalar);expected=np.zeros_like(ctx.qstate);expected[ctx.oracle.qindices]=v
            low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)];high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
            take('native_low');lo=simulate_statevector(evolution.circuit,low)
            checks=[gate('low_control',np.linalg.norm(lo-low),1e-9)]
            take('native_high');hi=simulate_statevector(evolution.circuit,high)
            checks.append(gate('high_leakage',np.linalg.norm(hi[:len(expected)]),1e-9))
            checks.append(gate('independent_occupation',np.linalg.norm(hi[len(expected):]-expected),1e-9))
            sig=SELECTED[identity]['signal'];z=complex(np.vdot(ctx.qstate,expected))
            checks.append(gate('saved_signal',abs(z-complex(sig['corrected']['real'],sig['corrected']['imag'])),1e-9))
            prices={};archives=[]
            for ax in ('cosine','sine'):
                take('wrapper_build');wrapper=native.build_native_hadamard_wrapper(evolution,axis=ax,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
                fingerprint=fp.fingerprint(wrapper);take('uncompiled_action');before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                observed=float(np.vdot(before[:len(expected)],before[:len(expected)]).real-np.vdot(before[len(expected):],before[len(expected):]).real)
                checks.append(gate('hadamard_'+ax,abs(observed-(z.real if ax=='cosine' else z.imag)),1e-9))
                take('wrapper_compile');compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
                cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
                c.require(cost.compiler_settings_hash==c.COMPILER and fp.fingerprint(wrapper)==fingerprint,'COMPILER_INPUT_FINGERPRINT')
                take('compiled_action');after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                checks.append(gate('compiled_vector_'+ax,np.linalg.norm(after-before),1e-7))
                for name,qc in [('wrapper',wrapper),('compiled',cost.transpiled_circuit)]:
                    f=io.BytesIO();qpy.dump(qc,f);archives.append(archive(ax+'_'+name+'.qpy',f.getvalue()))
                f=io.BytesIO();np.savez(f,low=low,before=before,after=after);archives.append(archive(ax+'_vectors.npz',f.getvalue()))
                metrics={k:getattr(cost,k) for k in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}
                c.require(all(type(x) is int and x>=0 for x in metrics.values()) and metrics['rz_count']>0,'INTEGER_COMPILED_COST')
                prices[ax]=dict(metrics=metrics,fingerprint=fingerprint,compiler=c.COMPILER,preparation_hash=prep.preparation_hash,
                    phase_guard=cost.transpiled_circuit.metadata.get('H2_phase_guard') if cost.transpiled_circuit.metadata else None)
            result=dict(status='PASS',candidate=row,SOURCE=stage.plan['SOURCE'],prices=prices,checks=checks,archives=archives,
                scalar=scalar,preparation_hash=prep.preparation_hash,hamiltonian=stage.plan['binding']['hamiltonian'],state=stage.plan['binding']['state'],
                native_policy=c.NATIVE,wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                pid=os.getpid(),retry=False)
            c.save(out/'result.json',result);return result
    except BaseException as e:
        import traceback
        c.save(out/'STOP.json',dict(reason=str(e),traceback=traceback.format_exc(),retry=False));raise
