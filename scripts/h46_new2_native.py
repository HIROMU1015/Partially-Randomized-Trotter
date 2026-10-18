"""Isolated new_2 diagnostic adapter; never changes the historical provider.

Adjacent complex QR follows the recorded H4 Gaussian construction at
5aa6c3685afb9c3afcfbdc36f80244da0a7fe525, extended explicitly to 12 modes.
This is a new compiled-cost series, not the legacy dense-Gaussian series.
"""
import math
from dataclasses import replace
import numpy as np
from scipy.sparse import coo_matrix
from h46_new2_core import weights, allocation


def gaussian_operations(rotation):
    from qiskit.circuit.library import PhaseGate, UnitaryGate
    u = np.asarray(rotation, dtype=complex)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or not 1 <= len(u) <= 12:
        raise ValueError('GAUSSIAN_SHAPE')
    if not np.isfinite(u).all() or np.max(abs(u.conj().T@u-np.eye(len(u)))) > 1e-12:
        raise ValueError('GAUSSIAN_UNITARITY')
    current=u.copy(); eliminations=[]
    for column in range(len(u)-1):
        for lower in range(len(u)-1,column,-1):
            upper=lower-1; a,b=current[upper,column],current[lower,column]
            if b == 0:
                continue
            magnitude=math.hypot(abs(a),abs(b))
            g=np.array([[a.conjugate(),b.conjugate()],[-b,a]],complex)/magnitude
            current[[upper,lower],:]=g@current[[upper,lower],:]
            eliminations.append((upper,lower,g.conj().T))
    diagonal=np.diag(current); phases=np.angle(diagonal)
    if np.max(abs(current-np.diag(diagonal))) > 1e-12 or np.max(abs(abs(diagonal)-1)) > 1e-12:
        raise ValueError('GAUSSIAN_QR_RESIDUAL')
    reconstructed=np.diag(np.exp(1j*phases)); operations=[]
    for mode,phase in enumerate(phases):
        if phase != 0:
            operations.append((PhaseGate(float(phase)),(mode,)))
    for upper,lower,g in reversed(eliminations):
        reconstructed[[upper,lower],:]=g@reconstructed[[upper,lower],:]
        fock=np.zeros((4,4),complex);fock[0,0]=1
        fock[1:3,1:3]=g;fock[3,3]=np.linalg.det(g)
        operations.append((UnitaryGate(fock),(upper,lower)))
    if np.max(abs(reconstructed-u)) > 1e-12:
        raise ValueError('GAUSSIAN_RECONSTRUCTION')
    return operations


def prepare(ham,prefix):
    """Select the lowering only inside this private process and restore it."""
    from trotterlib.df_trotter import ops
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
    previous=ops.U_to_qiskit_ops_jw
    try:
        ops.U_to_qiskit_ops_jw=gaussian_operations
        return _prepare(ham,prefix)
    finally:
        ops.U_to_qiskit_ops_jw=previous


def make_requests(preparation,label,q,*,T,total_steps,seed_for):
    from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
    counts=allocation(weights(label),total_steps)
    requests=[]
    for outer in range(q):
        for piece,(weight,steps) in enumerate(zip(weights(label),counts)):
            requests.append(make_df_partial_s2_step_request(preparation,
                step_time=T*weight/q,rte_steps=steps,truncation_tolerance=1e-10,
                finite_taylor_order=4,seed=seed_for(outer,piece),controlled=True,
                ancilla_qubit=preparation.num_system_qubits,pf_label='2nd'))
    return tuple(requests)


def native_evolution(requests,label,q,T):
    from qiskit import QuantumCircuit
    from trottertracks.resource_applicability.ax2b_native_df_v5 import partial_native_from_step_requests,NativeEvolution
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
    if not requests or len(requests)!=q*len(weights(label)):
        raise ValueError('SCHEDULE_SIZE')
    prep=requests[0].preparation;n=prep.num_system_qubits
    expected=[T*w/q for _ in range(q) for w in weights(label)]
    if any(r.preparation.preparation_hash!=prep.preparation_hash or r.step_time!=t
           for r,t in zip(requests,expected)):
        raise ValueError('SCHEDULE_IDENTITY')
    qc=QuantumCircuit(n+1,name='new2_diagnostic_'+label)
    bound=0
    for request in requests:
        # The old adapter requires equal times; each signed occurrence is
        # lowered separately. No time homogenization or event reordering.
        part=partial_native_from_step_requests([request],control_policy='ordinary',
            max_instructions=10**7)
        qc.compose(part.circuit,inplace=True);bound+=part.instruction_upper_bound
    return NativeEvolution(qc,n,T,q,'partial_'+label,'ordinary',bound,
        canonical_qiskit_circuit_fingerprint(qc))


class OccupationOracle:
    """Independent ladder-algebra oracle in the whole particle-number sector.

    Keeping all spin sectors avoids projecting any sampled intermediate away.
    Orbital eigenvectors define each rotated Z. No Qiskit circuit is used.
    """
    def __init__(self,ham,full_state,prefix=None):
        from scipy.linalg import eigh
        from trotterlib.df_trotter.decompose import diag_hermitian
        self.n=ham.n_qubits
        nonzero=np.flatnonzero(abs(full_state)>0)
        counts={int(i).bit_count() for i in nonzero}
        if len(counts)!=1:
            raise ValueError('SAVED_PARTICLE_NUMBER')
        particles=counts.pop()
        self.basis=np.array([i for i in range(1<<self.n) if i.bit_count()==particles])
        self.initial=np.asarray(full_state)[self.basis]
        self.qindices=np.array([int(format(int(i),f'0{self.n}b')[::-1],2) for i in self.basis])
        self.pairs={};positions={int(b):i for i,b in enumerate(self.basis)}
        for p in range(self.n):
            bp=1<<(self.n-1-p)
            for q in range(self.n):
                bq=1<<(self.n-1-q);rows=[];cols=[];values=[]
                for column,index in enumerate(self.basis):
                    index=int(index)
                    if not index&bq:
                        continue
                    removed=index^bq
                    if removed&bp:
                        continue
                    target=removed^bp
                    sign=(-1)**((index>>(self.n-q)).bit_count()+(removed>>(self.n-p)).bit_count())
                    rows.append(positions[target]);cols.append(column);values.append(sign)
                self.pairs[p,q]=(np.array(rows),np.array(cols),np.array(values))
        self.rotations={i:diag_hermitian(g,sort='descending_abs',assume_hermitian=True)[0]
                        for i,g in enumerate(ham.g_matrices)}
        self.specs=[]
        one=self.one_body(ham.one_body).toarray()
        vals,vecs=eigh(one);self.specs.append((vals,vecs,vecs.conj().T))
        for weight,g in list(zip(ham.lambdas,ham.g_matrices))[:prefix]:
            vals,vecs=eigh(self.one_body(g).toarray())
            self.specs.append((weight*vals**2,vecs,vecs.conj().T))
        self.numbers={}

    def one_body(self,matrix):
        rows=[];cols=[];values=[]
        for (p,q),(r,c,v) in self.pairs.items():
            if matrix[p,q] != 0:
                rows.extend(r);cols.extend(c);values.extend(matrix[p,q]*v)
        d=len(self.basis)
        return coo_matrix((values,(rows,cols)),shape=(d,d)).tocsr()

    def component(self,component,vector):
        index=int(component.df_fragment_id.rsplit('-',1)[1])
        value=vector
        for mode in component.diagonal_pauli_support:
            key=index,mode
            if key not in self.numbers:
                column=self.rotations[index][:,mode]
                self.numbers[key]=self.one_body(np.outer(column,column.conj()))
            value=value-2*(self.numbers[key]@value)
        return component.coefficient_sign*value

    def event(self,event,components,vector):
        v=vector
        for cid in event.product_component_ids:
            v=self.component(components[cid],v)
        return event.phase*(math.cos(event.rotation_angle)*v
            -1j*math.sin(event.rotation_angle)*self.component(components[event.rotation_component_id],v))

    def trajectory(self,requests):
        v=self.initial.copy()
        prep=requests[0].preparation
        comps={c.component_id:c for c in prep.rte_preparation.symbolic_tail.components}
        for request in requests:
            for spec in self.specs[:len(prep.deterministic_blocks)]:
                v=spec[1]@(np.exp(-.5j*request.step_time*spec[0])*(spec[2]@v))
            for event in request.rte_occurrence.events:
                v=self.event(event,comps,v)
            for spec in reversed(self.specs[:len(prep.deterministic_blocks)]):
                v=spec[1]@(np.exp(-.5j*request.step_time*spec[0])*(spec[2]@v))
            v*=np.exp(-1j*(prep.constant_coefficient+prep.extracted_identity_coefficient)*request.step_time)
        result=np.zeros(1<<self.n,complex);result[self.qindices]=v
        return result


def forced_event_request(request,order,*,mixed=True):
    """Conditional correctness fixture; NEVER enters the random cost sample."""
    from trotterlib.rte import _make_event
    comps=request.preparation.rte_preparation.symbolic_tail.components
    first=max(range(len(comps)),key=lambda i:comps[i].probability)
    second=next(i for i,c in enumerate(comps) if c.df_fragment_id!=comps[first].df_fragment_id)
    indices=[first]+[second if mixed and j%2==0 else first for j in range(order)]
    event=_make_event(indices,comps,request.rte_distribution,
        request.rte_distribution.orders.index(order))
    return event
