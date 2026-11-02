"""Fixed matched native-cost pilot; no signal, shot or effective-law mean."""
from dataclasses import asdict
from collections import Counter
from fractions import Fraction as F
from math import cos, sin, fsum, sqrt
from types import SimpleNamespace as NS
from threading import Lock

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .matched_partition_stage1 import spectral, compare
from .matched_partition_stage2 import gamma, native_context
from .df_trotter.decompose import diag_hermitian
from .df_trotter.ops import (U_to_qiskit_ops_jw_givens,one_body_diagonal_primitives,
    append_diagonal_primitives)
from .df_rte_tail import DFBasisRegistry
from .df_rte_circuit import DFRTEComponentCircuitSpec, DFRTEEventSequenceCircuitRequest
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .rte import RTEComponent, finite_rte_distribution, sample_rte_events
from .pauli_signal_adapter import PauliHamiltonian, append_pauli_event
from .h4_d2_pauli_control import central_control_body
from .fixed_signal_evaluation import hadamard_wrapper,default_compiler
from .rte_compiled_cost import (canonical_qiskit_circuit_fingerprint,
    transpile_and_measure_cost)

TIME=F.from_float(.8); Q=8; R=4; K=4
SEEDS={'P':(310201,310202,310203,310204),'DF':(310301,310302,310303,310304)}
OPERATIONS=Counter();_LOCK=Lock()


def record(name):
    with _LOCK:OPERATIONS[name]+=1


def require_pass(checks):
    bad={k:v for k,v in checks.items() if v['status']!='PASS'}
    if bad:
        error=ValueError('PREFLIGHT_NONPASS:'+','.join(bad));error.checks=checks;raise error


def norm(a):return spectral(a)


def fro(a):return float(np.linalg.norm(a,'fro'))


def controlled(a):
    d=len(a);out=np.eye(2*d,dtype=complex);out[d:,d:]=a;return out


def classification(residual,u):
    if not np.isfinite([residual,u]).all() or u<0:raise ValueError('INVALID_ERROR_BUDGET')
    return 'PASS' if residual<=u else 'UNRESOLVED' if residual<=4*u else 'FAIL'


def checked_product(a,b,ea,eb,nb):
    """Contract product propagation, with measured left norm and carried right norm."""
    na=norm(a)
    rounding=gamma(8*a.shape[1])*fro(np.abs(a)@np.abs(b))
    return a@b,na*eb+nb*ea+ea*eb+rounding,na*nb


def circuit_matrix(circuit):
    """Full native operator plus local-dot-product engineering roundoff estimate.

    Local gate matrices are cached; no per-library-call audit log is produced.
    Full operator is evaluated by Qiskit, independently of the Hamiltonian oracle.
    """
    d=1<<circuit.num_qubits;error=0.;bound=1.;cache={}
    for item in circuit.data:
        op=item.operation
        key=(op.name,op.num_qubits,tuple(map(float,op.params)))
        if key not in cache:
            g=np.asarray(Operator(op).data);w=len(g)
            ng=norm(g);ng_abs=norm(np.abs(g));eg=gamma(16*w)*max(1.,ng)
            cache[key]=(w,ng,ng_abs,eg)
        w,ng,ng_abs,eg=cache[key]
        # || |g| |B| ||_F <= || |g| ||_2 ||B||_F for the local tensor action.
        rounding=gamma(8*w)*ng_abs*sqrt(d)*bound
        error=ng*error+eg*sqrt(d)*bound+eg*error+rounding
        bound*=ng
    phase=float(circuit.global_phase)
    error+=gamma(8)*(1+abs(phase))*sqrt(d)*bound
    actual=np.asarray(Operator(circuit).data)
    record('native_operator_evaluations')
    if not np.isfinite(actual).all():raise ValueError('NONFINITE_NATIVE_OPERATOR')
    return actual,error


def basis_from_saved(u):
    """No eigensolver or dense fallback. Actual inverse is the gate adjoint."""
    n=len(u);ops=tuple(U_to_qiskit_ops_jw_givens(u))
    record('native_basis_lowerings')
    if any(g.name=='unitary' or g.num_qubits>2 for g,qs in ops):
        raise ValueError('GAUSSIAN_NATIVE_FAMILY_CHANGED')
    circuit=QuantumCircuit(n)
    for g,qs in ops:circuit.append(g,qs)
    inverse=circuit.inverse()
    m,em=circuit_matrix(circuit);mi,ei=circuit_matrix(inverse)
    indices=[1<<q for q in range(n)];vacuum=m[0,0]
    if abs(vacuum)==0:raise ValueError('GAUSSIAN_VACUUM_PHASE')
    checks=dict(one_particle=compare(m[np.ix_(indices,indices)]/vacuum,u,n),
        actual_adjoint=compare(mi,m.conj().T,1<<n),
        inverse_composition=compare(mi@m,np.eye(1<<n),1<<n),
        unitarity=compare(m.conj().T@m,np.eye(1<<n),1<<n))
    checks['vacuum_leakage']=compare(m[:,[0]],np.eye(1<<n)[:,[0]]*vacuum,1<<n)
    require_pass(checks)
    return NS(u=u,ops=ops,circuit=circuit,inverse=inverse,matrix=m,inverse_matrix=mi,
        error=em,inverse_error=ei,checks=checks,
        metadata=dict(native_family='BogoliubovTransform(U.T), givens, no dense fallback',
            vacuum_phase=dict(real=float(vacuum.real),imag=float(vacuum.imag)),
            fingerprint=canonical_qiskit_circuit_fingerprint(circuit),
            inverse_fingerprint=canonical_qiskit_circuit_fingerprint(inverse),
            gate_histogram=dict(circuit.count_ops())))


def half_from_basis(basis,eps,tau,*,control):
    n=len(eps);c=QuantumCircuit(n+int(control))
    c.compose(basis.inverse,qubits=range(n),inplace=True)
    append_diagonal_primitives(c,one_body_diagonal_primitives(eps,tau),
        controlled=control,ancilla_qubit=n if control else None)
    c.compose(basis.circuit,qubits=range(n),inplace=True)
    return c


def retained_preflight(one_body,saved_t,*,time=TIME,q=Q):
    n=len(one_body);d=1<<n;checks={'h_Hermiticity':compare(one_body,one_body.conj().T,n)}
    require_pass(checks)
    u,eps=diag_hermitian(one_body,sort='descending_abs',assume_hermitian=True)
    record('T_eigendecompositions')
    basis=basis_from_saved(u)
    checks.update(eigen_reconstruction=compare((u*eps)@u.conj().T,one_body,n),
        eigen_orthogonality=compare(u.conj().T@u,np.eye(n),n))
    native=native_context(n)(one_body).toarray()
    checks['saved_T']=compare(native,saved_t,d)
    require_pass(checks)
    tau=float(time/q/2);reference=expm(-1j*tau*saved_t)
    record('reference_half_exponentials')
    half=half_from_basis(basis,eps,tau,control=False)
    actual_inverse=half.inverse();chalf=half_from_basis(basis,eps,tau,control=True)
    mh,eh=circuit_matrix(half);minv,einv=circuit_matrix(actual_inverse)
    mc,ec=circuit_matrix(chalf)
    checks.update(half=compare(mh,reference,d),actual_inverse=compare(minv,reference.conj().T,d),
        controlled_half=compare(mc,controlled(reference),2*d))
    require_pass(checks)
    # Expm and binding diagnostics; this is an engineering estimate, not a certificate.
    reference_error=norm(mh-reference)+eh+gamma(8*d)*(1+tau*norm(saved_t))
    return NS(basis=basis,eps=eps,half=half,inverse=actual_inverse,controlled_half=chalf,
        reference=reference,reference_error=reference_error,checks=checks,
        errors=dict(system_half=eh,actual_inverse=einv,controlled_half=ec),q=q,
        time=time,tau=tau,n=n)


def component_inputs(name,descriptors,normalization,bases,n):
    lam=normalization['lambda_R'];rows=descriptors['components']
    if fsum(abs(x['coefficient']) for x in rows)!=lam:raise ValueError('SAVED_LAMBDA_BINDING')
    components=[];specs=[];registry=DFBasisRegistry();definitions={}
    if name=='DF':
        for index,basis in sorted(bases.items()):
            definitions[index]=registry.register(basis.ops,num_system_qubits=n,basis_id=f'p3-df-basis:{index}')
    for row in rows:
        c=row['coefficient'];kwargs={}
        if c==0 or not np.isfinite(c):raise ValueError('COMPONENT_COEFFICIENT')
        if name=='DF':
            definition=definitions[row['basis']];support=tuple(row['support'])
            kwargs=dict(df_fragment_id=str(row['fragment']),basis_id=definition.basis_id,
                basis_hash=definition.basis_hash,diagonal_pauli_support=support,
                basis_change_operations=definition.metadata.operations)
            specs.append(DFRTEComponentCircuitSpec(component_id=row['id'],coefficient_abs=abs(c),
                coefficient_sign=1 if c>0 else -1,num_system_qubits=n,**kwargs))
        components.append(RTEComponent(row['id'],abs(c)/lam,abs(c),1 if c>0 else -1,**kwargs))
    dist=finite_rte_distribution(normalization['dimensionless_step_time'],K)
    for field,key in [('unnormalized_order_weights','weights'),('order_probabilities','order_probabilities')]:
        if list(getattr(dist,field))!=normalization[key]:raise ValueError('SAVED_DISTRIBUTION_BINDING')
    if dist.exact_finite_distribution!=normalization['beta']:raise ValueError('SAVED_BETA_BINDING')
    return NS(name=name,components=tuple(components),specs=tuple(specs),registry=registry,
        distribution=dist,rows={x['id']:x for x in rows},n=n,
        labels={x['id']:x['label'] for x in rows} if name=='P' else {})


def residual_sequence(inputs,events):
    n=inputs.n
    if inputs.name=='P':
        source=QuantumCircuit(n)
        for event in events:append_pauli_event(source,event,inputs.labels)
        built=central_control_body(source)
        return built.body,dict(control='D2 Pauli residual only',phase=built.sidecar['aggregate_source_global_phase'])
    request=DFRTEEventSequenceCircuitRequest(events=tuple(events),component_specs=inputs.specs,
        controlled=True,ancilla_qubit=n,cancel_adjacent_equal_bases=True,occurrence_rte_steps=len(events))
    built=QiskitDFRTEEventCircuitBuilder(basis_registry=inputs.registry).build_sequence(request)
    return built.circuit,dict(control='DF central diagonal only',basis_reuse_policy=built.basis_reuse_policy,
        cancelled_basis_change_pairs=built.cancelled_basis_change_pairs,basis_switches=built.basis_switch_count,
        relative_phase=built.relative_ancilla_phase)


def scalar_angle(nuclear,vbar,time=TIME):return F.from_float(nuclear)*time+F.from_float(vbar)*time


def shared_parts(retained,angle):
    n=retained.n;part=QuantumCircuit(n+1)
    for _ in range(retained.q):
        part.compose(retained.controlled_half,inplace=True)
        part.compose(retained.controlled_half,inplace=True)
    part.p(-float(angle),n)
    return {'R':part,'Empty':QuantumCircuit(n+1)}


def trajectory_parts(inputs,events,retained,angle,*,r=R):
    if len(events)!=retained.q*r:raise ValueError('EVENT_CARDINALITY')
    full=QuantumCircuit(inputs.n+1);only=QuantumCircuit(inputs.n+1);sidecar=[]
    for outer in range(retained.q):
        residual,meta=residual_sequence(inputs,events[outer*r:(outer+1)*r]);sidecar.append(meta)
        full.compose(retained.controlled_half,inplace=True)
        full.compose(residual,inplace=True)
        full.compose(retained.controlled_half,inplace=True)
        only.compose(residual,inplace=True)
    full.p(-float(angle),inputs.n)
    return {'F':full,'E':only},sidecar


class DictionaryOracle:
    """Independent native occupation/JW matrices, never the Gaussian gate list."""
    def __init__(self,inputs,descriptors,n):
        self.inputs=inputs;self.n=n;self.d=1<<n;self.saved=descriptors;self.z={};self.cache={}
        self.native=native_context(n)

    def component(self,identifier):
        if identifier in self.cache:return self.cache[identifier]
        row=self.inputs.rows[identifier];sign=1 if row['coefficient']>0 else -1
        if self.inputs.name=='P':
            a=PauliHamiltonian.from_terms([(row['label'],1.)],num_qubits=self.n).sparse_operator().toarray()
            error=0.
        else:
            index=row['basis']
            if index not in self.z:
                b=self.saved['basis'][index];u=np.empty((self.n,self.n),dtype=complex)
                u.real=b['U_real'];u.imag=b['U_imag']
                self.z[index]=[np.eye(self.d)-2*self.native(np.outer(u[:,i],u[:,i].conj())).toarray()
                               for i in range(self.n)]
            zs=self.z[index];support=row['support'];a=zs[support[0]]
            error=gamma(16*self.n)*max(1.,fro(a))
            if len(support)==2:
                b=zs[support[1]]
                a,error,_=checked_product(a,b,error,error,norm(b))
        self.cache[identifier]=(sign*a,error)
        return self.cache[identifier]

    def trajectory(self,events,retained,angle,*,part,r=R):
        d=self.d;result=np.eye(d,dtype=complex);error=0.;nb=1.
        def left(a,ea):
            nonlocal result,error,nb
            result,error,nb=checked_product(a,result,ea,error,nb)
        for outer in range(retained.q):
            if part in ('F','R'):left(retained.reference,retained.reference_error)
            if part in ('F','E'):
                for event in events[outer*r:(outer+1)*r]:
                    for identifier in event.product_component_ids:
                        a,ea=self.component(identifier);left(a,ea)
                    a,ea=self.component(event.rotation_component_id);theta=event.rotation_angle
                    rotation=cos(theta)*np.eye(d)-1j*sin(theta)*a
                    er=abs(sin(theta))*ea+gamma(16)*(sqrt(d)*abs(cos(theta))+fro(a)*abs(sin(theta)))
                    left(event.phase*rotation,er)
            if part in ('F','R'):left(retained.reference,retained.reference_error)
        if part in ('F','R'):
            phase=np.exp(-1j*float(angle));error+=(float(abs(F(float(angle))-angle))+gamma(8)*(1+abs(float(angle))))*fro(result)
            result=phase*result
        return controlled(result),error


def program_check(circuit,expected,oracle_error):
    actual,native_error=circuit_matrix(circuit)
    residual=norm(actual-expected);u=native_error+oracle_error
    return dict(status=classification(residual,u),spectral_difference=residual,
        frobenius_difference=fro(actual-expected),native_uncertainty=native_error,
        oracle_uncertainty=oracle_error,uncertainty=u,dimension=len(actual),
        phase_alignment=False,precision='complex128 engineering, not rigorous')


def price(circuit):
    compiler=default_compiler()
    cost=transpile_and_measure_cost(circuit,compiler)
    return cost.transpiled_circuit,dict(RZ=cost.rz_count,CX=cost.cx_count,depth=cost.total_depth,
        rz_depth=cost.rz_depth,cx_depth=cost.cx_depth,size=cost.circuit_size,
        input_histogram=dict(cost.pretranspile_gate_counts),output_histogram=dict(cost.posttranspile_gate_counts),
        compiler=asdict(compiler),compiler_hash=cost.compiler_settings_hash,
        input_fingerprint=cost.actual_circuit_fingerprint,
        output_fingerprint=canonical_qiskit_circuit_fingerprint(cost.transpiled_circuit),
        global_phase=float(cost.global_phase))


def wrappers(body):
    return {a:hadamard_wrapper(body,a,already_controlled=True) for a in ('cosine','sine')}
