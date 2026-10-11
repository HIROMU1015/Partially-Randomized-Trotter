"""Small synthetic source bridge and native-cost audit; no production input.

Four fixed central compilation calls plus one Gaussian embedding call. Arbitrary
RZ counts are not Clifford+T counts. This script never imports the supplied ZIP.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['QISKIT_PARALLEL']='FALSE'
os.environ['MPLCONFIGDIR']='/tmp/prt-df-correction-audit-mpl'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from audit_df_sparse_correction import primitive, comm, random_unitary

CHECKS=[]
def check(name,ok,details=None):
    CHECKS.append({'name':name,'pass':bool(ok),'details':details})
    if not ok: raise AssertionError(name)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1536*1024:
        raise MemoryError('Synthetic audit RSS guard: 1536 MiB')

def one_particle_from_ops(n,ops):
    """Recover actual source convention using local one-excitation blocks.

    Only matrices of dimension <=4 and n*n are used. Gate-local number
    conservation and the Gaussian two-particle determinant relation are checked.
    Arbitrary basis-registry operations are not assumed Gaussian.
    """
    import numpy as np
    from qiskit.quantum_info import Operator
    w=np.eye(n,dtype=complex); vacuum=1+0j
    for gate,indices in ops:
        k=len(indices)
        if k not in (1,2): raise ValueError('Nonlocal/non-Gaussian gate access unresolved')
        u=Operator(gate).data
        for a in range(1<<k):
            for b in range(1<<k):
                if a.bit_count()!=b.bit_count() and abs(u[a,b])>1e-11:
                    raise ValueError('Number-nonconserving basis is outside DF primitive class')
        phase=u[0,0]
        one=u[np.ix_([1<<j for j in range(k)],[1<<j for j in range(k)])]/phase
        if k==2 and abs(u[3,3]/phase-np.linalg.det(one))>1e-11:
            raise ValueError('Number-conserving but non-Gaussian local gate')
        embedded=np.eye(n,dtype=complex);embedded[np.ix_(indices,indices)]=one
        w=embedded@w;vacuum*=phase
    return w,vacuum

def main():
    import numpy as np
    import scipy
    from scipy.linalg import expm, norm, block_diag
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate, RYGate
    from qiskit.quantum_info import Operator
    from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw_givens
    from trotterlib.df_rte_tail import DFBasisRegistry,basis_change_unitary,exact_df_diagonal_coefficients
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    start=time.monotonic();cpu=time.process_time()
    report={'scope':'Fixed synthetic source bridge and five representative compilations',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'production_runner_calls':0,'molecule_NPZ_GPU_calls':0,
            'trajectory_or_quantum_shots':0,'Clifford_T_synthesis_calls':0}
    try:
        rng=np.random.default_rng(7101107);n=4
        supplied=random_unitary(n,rng)
        ops=U_to_qiskit_ops_jw_givens(supplied)
        w,vacuum=one_particle_from_ops(n,ops)
        registry=DFBasisRegistry();definition=registry.register(ops,num_system_qubits=n)
        g=basis_change_unitary(definition,max_dense_qubits=n)
        residuals=[]
        for support,sign in [((0,),1),((0,2),-1),((),-1)]:
            diagonal=np.diag([sign*(-1)**sum((s>>j)&1 for j in support) for s in range(1<<n)])
            err=float(norm(g@diagonal@g.conj().T-primitive(w[:,support],sign),2))
            check(f'source.Q.{support}',err<2e-11,err);residuals.append(err)
        one=g[np.ix_([1<<j for j in range(n)],[1<<j for j in range(n)])]/g[0,0]
        check('source.one_particle_local_access',norm(w-one,2)<2e-11)
        try: one_particle_from_ops(1,[(RYGate(.123),(0,))])
        except ValueError: check('source.nonGaussian_registry_rejected',True)
        else: check('source.nonGaussian_registry_rejected',False)
        eta=np.array([.7,-.3,.2,.1]);lam=-.8
        coefficients=dict(exact_df_diagonal_coefficients(eta,lam))
        rebuilt=np.zeros((1<<n,1<<n),complex)
        for support,c in coefficients.items():
            rebuilt+=c*np.diag([(-1)**sum((s>>j)&1 for j in support) for s in range(1<<n)])
        ideal=np.diag([lam*sum(eta[j]*((s>>j)&1) for j in range(n))**2 for s in range(1<<n)])
        check('source.signed_DF_coefficients',norm(rebuilt-ideal,2)<2e-12)
        convention={name:float(norm(w-value,2)) for name,value in
                    [('U',supplied),('U_T',supplied.T),('U_conjugate',supplied.conj()),('U_dagger',supplied.conj().T)]}
        report['DF_access']={'n':n,'one_particle_access':'local gate blocks; no full Fock required',
                            'convention_residuals':convention,'Q_residuals':residuals,
                            'vacuum_phase':[float(vacuum.real),float(vacuum.imag)],
                            'thin_isometry_compiler':'NOT_IMPLEMENTED; source accepts square U'}
        compiler=CompilerSettings(basis_gates=('rz','sx','x','cx'),backend_name=None,
                 coupling_map=None,optimization_level=0,layout_method=None,routing_method=None,
                 transpiler_seed=20261011,qiskit_version=qiskit.__version__)
        compiled=[]
        for r in (3,6):
            frames=[random_unitary(r,rng)[:,:2] for _ in range(3)]
            qs=[primitive(v,s) for v,s in zip(frames,(1,-1,1))]
            t=comm(qs[0],comm(qs[1],qs[2]));u=expm(1j*.03125*t/4)
            check(f'central.{r}.nonzero',norm(t,2)>.01)
            for controlled in (False,True):
                target=block_diag(np.eye(1<<r),u) if controlled else u
                circuit=QuantumCircuit(r+int(controlled))
                circuit.append(UnitaryGate(target),circuit.qubits)
                cost=transpile_and_measure_cost(circuit,compiler)
                check(f'compile.{r}.{controlled}.size_guard',cost.circuit_size<200000)
                err=float(norm(Operator(cost.transpiled_circuit).data-target,2))
                check(f'compile.{r}.{controlled}.strict_phase',err<5e-9,err)
                counts=dict(cost.posttranspile_gate_counts)
                compiled.append({'active_modes':r,'controlled':controlled,
                    'ancilla':int(controlled),'workspace':0,'theta':.03125,'kappa':4,
                    'gate_counts':counts,'RZ':cost.rz_count,'CX':cost.cx_count,
                    '1Q':sum(v for k,v in counts.items() if k!='cx'),
                    'depth':cost.total_depth,'strict_operator_residual':err,
                    'circuit_fingerprint':cost.actual_circuit_fingerprint,
                    'compiler_settings_hash':cost.compiler_settings_hash,
                    'T_count':'UNRESOLVED; arbitrary RZ synthesis/error allocation not performed'})
        # Source Gaussian + controlled centre + exact inverse, small representative.
        small_frames=[random_unitary(n,rng)[:,:1] for _ in range(3)]
        local=[primitive(v) for v in small_frames]
        t=comm(local[0],comm(local[1],local[2]));u=expm(1j*.03125*t/4)
        uc=block_diag(np.eye(1<<n),u)
        gc=QuantumCircuit(n)
        for gate,indices in ops:gc.append(gate,indices)
        full=QuantumCircuit(n+1);full.compose(gc.inverse(),qubits=range(n),inplace=True)
        full.append(UnitaryGate(uc),full.qubits);full.compose(gc,qubits=range(n),inplace=True)
        lifted=g@u@g.conj().T;expected=block_diag(np.eye(1<<n),lifted)
        check('embedding.control_zero_relative_phase',norm(Operator(full).data-expected,2)<3e-11)
        cost=transpile_and_measure_cost(full,compiler)
        err=float(norm(Operator(cost.transpiled_circuit).data-expected,2))
        check('embedding.compiled_strict_phase',err<5e-9,err)
        report['full_embedding']={'system_modes':n,'active_used':n,'ancilla':1,'workspace':0,
                                 'gate_counts':dict(cost.posttranspile_gate_counts),'depth':cost.total_depth,
                                 'strict_operator_residual':err,'T_count':'UNRESOLVED'}
        report['central_compilations']=compiled
        report['compile_calls']=5
        report['compiler']=compiler.__dict__
        report['status']='CONDITIONAL_PASS_NATIVE_T_AND_PRODUCTION_ACCESS_UNRESOLVED'
    except Exception as exc:
        report['status']='AUDIT_STOP';report['reason']=str(exc);report['exception_type']=type(exc).__name__
        report['mandatory_STOP']=True
    report['checks']=CHECKS;report['checks_passed']=sum(c['pass'] for c in CHECKS)
    report['wall_seconds']=time.monotonic()-start;report['CPU_seconds']=time.process_time()-cpu
    report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report['runtime']={'python':platform.python_version(),'numpy':np.__version__,
                       'scipy':scipy.__version__,'qiskit':qiskit.__version__,'threads':1}
    output=ROOT/'artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json'
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','central_compilations')}))
    return int(report['status']=='AUDIT_STOP')

if __name__=='__main__':raise SystemExit(main())
