"""P3 Stage 1 operators only. No evolution, law, circuit, or factorization."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from .df_hamiltonian import (DFHamiltonian, PhysicalSector, df_linear_operator,
                             _build_excitation_table, _linear_operator_to_dense)
from .h2_non_df_pilot import reverse_bits
from .pauli_signal_adapter import PauliHamiltonian

POLICY = dict(dtype='complex128', norm='spectral_SVD_gesdd',
    epsilon=float(np.finfo(np.float64).eps), gamma_multiplier=16,
    scale='max(1, spectral norm of each compared operand)',
    dimension='full Fock dimension, also for inherited scalar/tensor errors',
    PASS='residual <= tolerance/4', UNRESOLVED='tolerance/4 < residual <= 4*tolerance',
    FAIL='residual > 4*tolerance',
    uncertainty='engineering consistency resolution, not certified rounding/DF error bound',
    precision_escalation=False, automatic_retry=False)


def correction_from_two_body(two_body):
    """Reuse the original reorder/contraction helper; never factorize."""
    from openfermion.circuits.low_rank import get_chemist_two_body_coefficients
    correction,_ = get_chemist_two_body_coefficients(two_body,spin_basis=True)
    return np.asarray(correction,dtype=np.complex128)


def jw_matrix(one,two,constant=0.):
    from openfermion import InteractionOperator,get_fermion_operator,jordan_wigner
    fermion=get_fermion_operator(InteractionOperator(constant,one,two))
    model=PauliHamiltonian.from_openfermion(jordan_wigner(fermion),num_qubits=len(one))
    return model.sparse_operator().toarray(),model.constant


def pauli_route(data):
    """Original Pauli coefficients + bare W/D: no saved DF effective matrix."""
    one=data['spin_one'];two=data['spin_two'];d=correction_from_two_body(two)
    zero=np.zeros_like(two);matrices={};identity={}
    saved=data['pauli']
    model=PauliHamiltonian(len(one),saved['constant'],tuple(map(tuple,saved['terms'])),'{}')
    matrices['H_P']=model.sparse_operator().toarray();identity['H_P']=model.constant
    for name,o,w,c in [('T_P',one+d,zero,0.),('V_P',-d,two,0.),
                       ('D_P',d,zero,0.),('V_normal_P',np.zeros_like(one),two,0.),
                       ('H_bare_JW',one,two,data['constant'])]:
        matrices[name],identity[name]=jw_matrix(o,w,c)
    return matrices,dict(correction_tensor=d,identity=identity,
                         full_Fock_matrix_constructions=6,
                         path='saved Pauli / W-contraction / fermion-JW-Qiskit; no saved h_eff in T/V')


def _native_one_body(matrix,table,dimension):
    """Vectorize the existing native excitation-table action, independently of JW."""
    values=matrix[table.p_indices,table.q_indices]*table.signs
    return csr_matrix((values,(table.target_positions,table.source_positions)),
                       shape=(dimension,dimension),dtype=np.complex128)


def df_route(data):
    """Saved effective one-body/factors in native fermionic occupation order."""
    one=data['df_one_body'];n=len(one);dimension=1<<n
    space=PhysicalSector(n_qubits=n,basis_indices=np.arange(dimension,dtype=np.int64))
    table=_build_excitation_table(space);perm=reverse_bits(n)
    convert=lambda x:x[np.ix_(perm,perm)]
    t=_native_one_body(one,table,dimension)
    v=csr_matrix((dimension,dimension),dtype=np.complex128)
    fragment_identity=[]
    for weight,g in zip(data['lambdas'],data['g_matrices'],strict=True):
        a=_native_one_body(g,table,dimension)
        v=v+float(weight)*(a@a)
        # Basis-independent version of exact_df_diagonal_coefficients' I term.
        fragment_identity.append(float((weight/4*(np.trace(g)**2+np.trace(g@g))).real))
    d=one-data['spin_one']
    d_operator=_native_one_body(d,table,dimension)
    ham=DFHamiltonian(data['constant'],one,data['lambdas'],tuple(data['g_matrices']),{})
    native,counter=df_linear_operator(ham,space,backend='python')
    # Full is constructed separately by the original native action on basis columns.
    full=_linear_operator_to_dense(native)
    matrices=dict(H_DF=convert(full),T_DF=convert(t.toarray()),
                  V_DF=convert(v.toarray()),D_DF=convert(d_operator.toarray()))
    return matrices,dict(correction_tensor=d,fragment_identity=fragment_identity,
        one_body_identity=float(np.trace(one).real/2),
        full_Fock_matrix_constructions=4,native_operator_basis_column_actions=counter['count'],
        physical_sector_projection=False,
        path='saved h_eff/lambda/G; native excitation signs, CSR squares; original full native action')


def spectral(matrix):
    matrix=np.asarray(matrix,dtype=np.complex128)
    if matrix.size==0:return 0.
    return float(svd(matrix,compute_uv=False,lapack_driver='gesdd',check_finite=True)[0])


def tolerance(dimension,scale):
    e=POLICY['epsilon'];gamma=dimension*e/(1-dimension*e)
    return POLICY['gamma_multiplier']*gamma*max(1.,float(scale))


def classify(residual,tol):
    if not math.isfinite(residual) or not math.isfinite(tol):raise ValueError('NONFINITE_METRIC')
    if residual<=tol/4:return 'PASS'
    if residual<=4*tol:return 'UNRESOLVED'
    return 'FAIL'


def compare(left,right,dimension,*,scale=None):
    difference=left-right
    scale=max(1.,spectral(left),spectral(right)) if scale is None else max(1.,scale)
    error=spectral(difference);tol=tolerance(dimension,scale)
    return dict(spectral_norm=error,frobenius_norm=float(np.linalg.norm(difference,'fro')),
        maximum_entry=float(np.max(np.abs(difference))),scale=scale,tolerance=tol,
        pass_threshold=tol/4,fail_threshold=4*tol,status=classify(error,tol))


def diagnose(data,matrices,pauli_info,df_info,*,workers):
    n=len(data['spin_one']);dimension=1<<n;eye=np.eye(dimension,dtype=complex)
    scalar=data['constant'];jobs={}
    def add(name,left,right,scale=None):jobs[name]=(left,right,scale)
    for name in ('H','T','V','D'):
        add(name+'_Pauli_vs_DF',matrices[name+'_P'],matrices[name+'_DF'])
    add('D_tensor_from_W_vs_saved',pauli_info['correction_tensor'],df_info['correction_tensor'])
    add('Pauli_decomposition_closure',matrices['H_P'],scalar*eye+matrices['T_P']+matrices['V_P'])
    add('DF_decomposition_closure',matrices['H_DF'],scalar*eye+matrices['T_DF']+matrices['V_DF'])
    add('Pauli_bare_tensor_vs_saved_coefficients',matrices['H_bare_JW'],matrices['H_P'])
    add('Pauli_normal_order_closure',matrices['V_normal_P'],matrices['V_P']+matrices['D_P'])
    for name in ('H_P','H_DF','T_P','T_DF','V_P','V_DF'):
        op=matrices[name]
        add(name+'_Hermiticity',op,op.conj().T)
    inside=np.asarray(data['sector_indices'],dtype=int)
    outside=np.setdiff1d(np.arange(dimension),inside)
    # Use the parent operator's norm, never a sector-projected agreement metric.
    for name in ('H_P','H_DF','T_P','T_DF','V_P','V_DF'):
        op=matrices[name];scale=max(1.,spectral(op))
        for direction,rows,cols in [('out',outside,inside),('in',inside,outside)]:
            part=op[np.ix_(rows,cols)]
            add(name+'_sector_'+direction,part,np.zeros_like(part),scale)
    for name in ('H_P','H_DF'):
        add(name+'_vs_saved_snapshot',matrices[name],data['snapshot_'+name])
    identity=dict(pauli_JW=pauli_info['identity'],
        DF_one_body=df_info['one_body_identity'],
        DF_square_fragments=df_info['fragment_identity'],nuclear=scalar,
        full_matrix_trace={k:float(np.trace(matrices[k]).real/dimension)
                           for k in ('H_P','T_P','V_P','H_DF','T_DF','V_DF')})
    df_v_identity=math.fsum(df_info['fragment_identity'])
    for key,value,reference in [
        ('identity_T',pauli_info['identity']['T_P'],df_info['one_body_identity']),
        ('identity_V',pauli_info['identity']['V_P'],df_v_identity),
        ('identity_Pauli_full_closure',pauli_info['identity']['H_P'],
            scalar+pauli_info['identity']['T_P']+pauli_info['identity']['V_P']),
        ('identity_DF_full_closure',identity['full_matrix_trace']['H_DF'],
            scalar+df_info['one_body_identity']+df_v_identity),
        ('identity_DF_square_trace',identity['full_matrix_trace']['V_DF'],df_v_identity)]:
        add(key,np.array([[value]]),np.array([[reference]]))
    def job(item):
        name,(left,right,scale)=item
        return name,compare(left,right,dimension,scale=scale)
    with ThreadPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
        results=dict(pool.map(job,jobs.items()))
    identity['DF_square_total']=df_v_identity
    overall='FAIL' if any(r['status']=='FAIL' for r in results.values()) else (
        'UNRESOLVED' if any(r['status']=='UNRESOLVED' for r in results.values()) else 'PASS')
    return dict(overall=overall,comparisons=results,scalar_identity=identity,
                policy=POLICY,physical_dimension=dimension,metric_jobs=len(jobs))


def matrix_fingerprint(matrix):
    a=np.ascontiguousarray(matrix,dtype='<c16')
    return dict(shape=list(a.shape),dtype='<c16',sha256=hashlib.sha256(a.tobytes()).hexdigest())
