"""Limited source-bound construction and equal-signal-precision development study.

Constructors use coefficient algebra, orbital eigensystems and graph support.
Dense exponentials are diagnostic references, never efficient input oracles.
The existing paired finite-RTE distribution and return semantics are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import Operator, Pauli
from scipy.linalg import expm

from trotterlib.df_trotter.model import DFModel
from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw_givens, build_df_blocks_givens
from trotterlib.rte import (InvolutoryTailTerm, event_unitary, finite_rte_distribution,
                           finite_taylor_operator, normalize_involutory_tail, sample_rte_events)
from .mechanisms import norm, matrix, second_quantize_one_body

SEED = 20261010
ATOL = 2e-10
EPSILONS = (.05, .02)
CONFIDENCE_FAILURE = .05
METRICS = ("rz", "cx", "sx", "x", "size", "depth")
PAULI_PRODUCT = {
    ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y"),
}


def padd(*dictionaries):
    out = {}
    for terms in dictionaries:
        for label, coeff in terms.items():
            out[label] = out.get(label, 0j) + coeff
    return {p: c for p, c in out.items() if c != 0}


def pscale(terms, coefficient):
    return {p: coefficient*c for p,c in terms.items() if coefficient*c != 0}


def pmul(left, right):
    out = {}
    for p,c in left.items():
        for q,d in right.items():
            phase, chars = 1, []
            for a,b in zip(p,q,strict=True):
                if a == "I": chars.append(b)
                elif b == "I": chars.append(a)
                elif a == b: chars.append("I")
                else:
                    ph,ch = PAULI_PRODUCT[a,b]
                    phase *= ph
                    chars.append(ch)
            label = "".join(chars)
            out[label] = out.get(label,0j)+phase*c*d
    return {p:c for p,c in out.items() if c != 0}


def annihilator_paulis(n, mode):
    base = ["I"]*n
    for j in range(mode): base[n-1-j] = "Z"
    base[n-1-mode] = "X"
    px = "".join(base)
    base[n-1-mode] = "Y"
    return {px:.5, "".join(base):.5j}


def orbital_paulis(g):
    """Polynomial coefficient algebra for dGamma(g), no Fock diagonalization."""
    n = len(g)
    if g.shape != (n,n) or not np.allclose(g,g.conj().T,atol=1e-12,rtol=0):
        raise ValueError("Hermitian orbital matrix required")
    a = [annihilator_paulis(n,j) for j in range(n)]
    out = {}
    for i,j in itertools.product(range(n),repeat=2):
        if g[i,j] != 0:
            out = padd(out,pscale(pmul({p:c.conjugate() for p,c in a[i].items()},a[j]),g[i,j]))
    return out


def real_terms(terms):
    imaginary_l1 = math.fsum(abs(complex(c).imag) for c in terms.values())
    if imaginary_l1 > 1e-11:
        raise ValueError("Non-Hermitian coefficient dictionary")
    # Record/verify this roundoff in construction residuals; no real threshold.
    return {p:float(complex(c).real) for p,c in sorted(terms.items()) if complex(c).real != 0}


def pdense(terms, n):
    return sum((c*Pauli(p).to_matrix() for p,c in terms.items()),np.zeros((2**n,2**n),complex))


def gaussian_circuit(v):
    qc = QuantumCircuit(len(v))
    # Complex dtype avoids the known real-array casting warning path.
    for gate, qubits in U_to_qiskit_ops_jw_givens(np.asarray(v,dtype=complex)):
        if gate.name == "unitary":
            raise ValueError("Dense Gaussian fallback is forbidden")
        qc.append(gate,qubits)
    return qc


def apply_pauli(qc, label, *, angle=None, control=None):
    """Uncontrolled Clifford/parity frame; control only its central action."""
    support = [(i,p) for i,p in enumerate(reversed(label)) if p != "I"]
    if not support:
        phase = -angle if angle is not None else 0.
        if control is None: qc.global_phase += phase
        else: qc.p(phase,control)
        return
    for i,p in support:
        if p == "X": qc.h(i)
        elif p == "Y": qc.sdg(i); qc.h(i)
    target = support[-1][0]
    for i,_ in support[:-1]: qc.cx(i,target)
    if angle is None:
        if control is None: qc.z(target)
        else: qc.h(target); qc.cx(control,target); qc.h(target)
    elif control is None:
        qc.rz(2*angle,target)
    else:
        # Explicit CRZ decomposition retains absolute/control-relative phase.
        qc.rz(angle,target); qc.cx(control,target)
        qc.rz(-angle,target); qc.cx(control,target)
    for i,_ in reversed(support[:-1]): qc.cx(i,target)
    for i,p in reversed(support):
        if p == "X": qc.h(i)
        elif p == "Y": qc.h(i); qc.s(i)


@dataclass
class Primitive:
    component_id: str
    coefficient: float
    label: str
    basis: QuantumCircuit
    reflected_aux: int | None = None

    def dense(self):
        v = Operator(self.basis).data
        p = v@Pauli(self.label).to_matrix()@v.conj().T
        if self.reflected_aux is not None:
            z = ["I"]*self.basis.num_qubits; z[-1-self.reflected_aux] = "Z"
            s = Pauli("".join(z)).to_matrix(); p = s@p@s
        return p

    def apply(self, qc, ancilla, *, angle=None):
        ids = list(range(self.basis.num_qubits))
        if self.reflected_aux is not None: qc.z(self.reflected_aux)
        qc.compose(self.basis.inverse(),qubits=ids,inplace=True)
        if angle is None and self.coefficient < 0: qc.p(math.pi,ancilla)
        apply_pauli(qc,self.label,angle=None if angle is None else math.copysign(1,self.coefficient)*angle,
                    control=ancilla)
        qc.compose(self.basis,qubits=ids,inplace=True)
        if self.reflected_aux is not None: qc.z(self.reflected_aux)


@dataclass
class DiagonalBlock:
    block_id: str
    terms: dict[str,float]
    basis: QuantumCircuit

    def dense(self):
        v = Operator(self.basis).data
        return v@pdense(self.terms,self.basis.num_qubits)@v.conj().T

    def apply(self,qc,ancilla,time):
        ids = list(range(self.basis.num_qubits))
        qc.compose(self.basis.inverse(),qubits=ids,inplace=True)
        for p,c in self.terms.items(): apply_pauli(qc,p,angle=c*time,control=ancilla)
        qc.compose(self.basis,qubits=ids,inplace=True)


@dataclass
class PRConstruction:
    name: str
    n: int
    blocks: list[DiagonalBlock]
    primitives: list[Primitive]
    identity: float
    outer_basis: QuantumCircuit
    ld: int | None
    metadata: dict = field(default_factory=dict)

    def tail(self):
        if not self.primitives: return None
        return normalize_involutory_tail(self.name,[
            InvolutoryTailTerm(p.component_id,p.coefficient,p.dense()) for p in self.primitives])

    def dense(self):
        h = sum((b.dense() for b in self.blocks),np.zeros((2**self.n,2**self.n),complex))
        h += self.identity*np.eye(2**self.n)
        h += sum((p.coefficient*p.dense() for p in self.primitives),np.zeros_like(h))
        v = Operator(self.outer_basis).data
        return v@h@v.conj().T


def split_identity(terms, n):
    terms = dict(terms)
    return terms.pop("I"*n,0.),terms


def frame_construct(factors, *, anchor=None):
    """All-factor diagonal core in a chosen input-derived frame + exact JW residual."""
    if not factors or any(np.iscomplexobj(g) and np.max(abs(g.imag))>1e-12 for g in factors):
        raise ValueError("Nonempty real Hermitian factor input required")
    n = len(factors[0])
    if anchor is None: v = np.eye(n)
    else:
        _,v = np.linalg.eigh(factors[anchor])
        # Deterministic column signs; no dense target/ground-state information.
        for j in range(n):
            pivot = np.argmax(abs(v[:,j]))
            if v[pivot,j] < 0: v[:,j] *= -1
    transformed = [v.T@g@v for g in factors]
    full,core = {},{}
    for g in transformed:
        f = orbital_paulis(g); d = orbital_paulis(np.diag(np.diag(g)))
        full = padd(full,pmul(f,f)); core = padd(core,pmul(d,d))
    core = real_terms(core); residual = real_terms(padd(full,pscale(core,-1)))
    ident,residual = split_identity(residual,n)
    identity = QuantumCircuit(n)
    name = "frame_identity" if anchor is None else f"frame_factor_{anchor}"
    return PRConstruction(name,n,[DiagonalBlock("shared_core",core,identity)],
                          [Primitive(p,c,p,identity) for p,c in residual.items()],ident,
                          gaussian_circuit(v),None,
                          {"orbital_frame":matrix(v),"core_terms":core,"residual_terms":residual,
                           "constructor":"fixed candidate frames; algebraic exact residual, no label-angle search"})


def native_df_construct(factors, ld):
    n = len(factors[0]); rank = len(factors)
    if not 0 <= ld <= rank: raise ValueError("Invalid DF prefix")
    model = DFModel(np.ones(rank),[np.asarray(g,complex) for g in factors],np.zeros((n,n)),0.,n)
    raw = build_df_blocks_givens(model)
    order = sorted(range(rank),key=lambda j:(-float(np.vdot(factors[j],factors[j]).real),j))
    blocks,primitives,ident = [],[],0.
    for j in order:
        b = raw[j]; basis = QuantumCircuit(n)
        for gate,qubits in b.U_ops:
            if gate.name == "unitary": raise ValueError("Dense Gaussian fallback")
            basis.append(gate,qubits)
        d = orbital_paulis(np.diag(b.eta)); terms = real_terms(pmul(d,d))
        if j in order[:ld]: blocks.append(DiagonalBlock(f"df_{j}",terms,basis))
        else:
            c,nonidentity = split_identity(terms,n); ident += c
            primitives += [Primitive(f"df{j}:{p}",coef,p,basis) for p,coef in nonidentity.items()]
    return PRConstruction(f"native_df_ld{ld}",n,blocks,primitives,ident,QuantumCircuit(n),ld,
                          {"rank":rank,"ranking":order,"basis_control":"diagonal only"})


def schedule(construction, step, q, events=None):
    seq=[]
    for j in range(q):
        for i in range(len(construction.blocks)): seq.append(("d",i,step/2))
        if events is not None: seq.append(("e",j,0.))
        for i in reversed(range(len(construction.blocks))): seq.append(("d",i,step/2))
    # If deterministic-only with two blocks this merges its central halves too.
    merged=[]
    for item in seq:
        if merged and item[0] == merged[-1][0] == "d" and item[1] == merged[-1][1]:
            a=merged.pop();merged.append(("d",item[1],a[2]+item[2]))
        else: merged.append(item)
    return merged


def corrected_pr_mean(construction, time, q, k=2):
    tail=construction.tail(); dt=time/q; eye=np.eye(2**construction.n,dtype=complex)
    d=eye.copy()
    for block in construction.blocks: d=expm(-.5j*dt*block.dense())@d
    if tail:
        poly=finite_taylor_operator(tail.normalized_hamiltonian,tail.lambda_r*dt,k)
        dist=finite_rte_distribution(tail.lambda_r*dt,k)
        norm_total=dist.exact_finite_distribution**q
    else: poly=eye; norm_total=1.
    # Palindrome reverse block order is the same construction, not d dagger.
    right=eye.copy()
    for block in reversed(construction.blocks): right=expm(-.5j*dt*block.dense())@right
    short=right@poly@d
    result=np.exp(-1j*time*construction.identity)*np.linalg.matrix_power(short,q)
    v=Operator(construction.outer_basis).data
    return v@result@v.conj().T,float(norm_total)


def trajectory(construction,time,q,seed,k=2):
    tail=construction.tail(); dt=time/q
    if tail:
        dist=finite_rte_distribution(tail.lambda_r*dt,k)
        events=sample_rte_events(tail.components,dist,sample_count=q,seed=seed)
        operators={c.component_id:p for c,p in zip(tail.components,tail.operators)}
    else: events=None; operators={}
    n=construction.n; anc=n; qc=QuantumCircuit(n+1); ids=list(range(n))
    qc.compose(construction.outer_basis.inverse(),qubits=ids,inplace=True)
    u=np.eye(2**n,dtype=complex); primitives={p.component_id:p for p in construction.primitives}
    for kind,index,duration in schedule(construction,dt,q,events):
        if kind == "d":
            block=construction.blocks[index];block.apply(qc,anc,duration)
            u=expm(-1j*duration*block.dense())@u
        else:
            e=events[index]
            for component in e.product_component_ids: primitives[component].apply(qc,anc)
            primitives[e.rotation_component_id].apply(qc,anc,angle=e.rotation_angle)
            qc.p(float(np.angle(e.phase)),anc)
            u=event_unitary(e,operators)@u
    qc.p(-time*construction.identity,anc)
    qc.compose(construction.outer_basis,qubits=ids,inplace=True)
    v=Operator(construction.outer_basis).data
    u=np.exp(-1j*time*construction.identity)*v@u@v.conj().T
    return qc,u,[] if events is None else [e.to_dict() for e in events]


def wrapper(circuit,u,axis,prepared):
    n=circuit.num_qubits-1; anc=n; out=QuantumCircuit(n+1)
    for qubit,gate in prepared:
        getattr(out,gate)(qubit)
    out.h(anc);out.compose(circuit,inplace=True)
    if axis == "Y": out.sdg(anc)
    out.h(anc)
    prep=QuantumCircuit(n)
    for qubit,gate in prepared: getattr(prep,gate)(qubit)
    identity=np.eye(2**n); ctrl=np.block([[identity,np.zeros_like(identity)],[np.zeros_like(identity),u]])
    had=np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2),identity)
    read=np.kron(np.diag([1,-1j]) if axis == "Y" else np.eye(2),identity)
    reference=had@read@ctrl@had@np.kron(np.eye(2),Operator(prep).data)
    return out,reference


def reconstruct_ir(ir):
    qc=QuantumCircuit(ir["qubits"]);qc.global_phase=ir["global_phase"]
    for op in ir["operations"]: getattr(qc,op["name"])(*op["parameters"],*op["qubits"])
    return qc


@dataclass
class NativeAudit:
    cap: int = 512
    records: list = field(default_factory=list)

    def compile(self,qc,reference,label):
        if len(self.records) >= self.cap or qc.num_qubits > 5: raise ValueError("Compile/qubit cap")
        built=Operator(qc).data;error=norm(built-reference)
        if error > ATOL: raise AssertionError(f"Built circuit differs: {label}, {error}")
        compiled=transpile(qc,basis_gates=["rz","sx","x","cx"],optimization_level=1,
                           seed_transpiler=SEED,num_processes=1)
        if compiled.size()>10000: raise ValueError("Per-circuit native gate cap")
        raw=Operator(compiled).data;overlap=np.trace(built.conj().T@raw)/len(raw)
        scalar=norm(raw-overlap*built);repair=0.
        if norm(raw-reference)>ATOL:
            if scalar>ATOL or abs(abs(overlap)-1)>ATOL: raise AssertionError("Non-scalar compiler error")
            repair=-float(np.angle(overlap));compiled.global_phase+=repair
        ir={"id":len(self.records),"label":label,"qubits":compiled.num_qubits,
            "global_phase":float(compiled.global_phase),"operations":[
                {"name":i.operation.name,"qubits":[compiled.find_bit(q).index for q in i.qubits],
                 "parameters":[float(x) for x in i.operation.params]} for i in compiled.data]}
        reconstructed=Operator(reconstruct_ir(ir)).data
        resid=norm(reconstructed-reference)
        if resid>ATOL: raise AssertionError("Saved IR/absolute operator mismatch")
        ir["sha256"]=hashlib.sha256(json.dumps(ir,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.records.append(ir)
        counts=compiled.count_ops()
        return {"ir_id":ir["id"],"ir_sha256":ir["sha256"],"qubits":qc.num_qubits,
                **{m:int(counts.get(m,0)) for m in ("rz","cx","sx","x")},
                "size":compiled.size(),"depth":compiled.depth(),"measurement_count":1,
                "state_preparation_included":True,"built_residual":error,
                "compiled_reconstructed_residual":resid,"certified_scalar_residual":scalar,
                "global_phase_repair":repair}


def shot_budget(normalization,bias,epsilon):
    if not all(math.isfinite(x) for x in (normalization,bias,epsilon)) or normalization<1 or bias<0 or epsilon<=0:
        raise ValueError("Invalid bias/range budget")
    axis_epsilon=epsilon/math.sqrt(2)
    if bias>=axis_epsilon: return None
    return math.ceil(2*normalization**2/(axis_epsilon-bias)**2*math.log(4/CONFIDENCE_FAILURE))


def summarize_cost(rows,normalization,bias):
    per_axis={}
    raw={}
    for axis in ("X","Y"):
        data=[r["cost"] for r in rows if r["axis"]==axis]
        raw[axis]=data
        per_axis[axis]={m:{"mean":float(np.mean([x[m] for x in data])),
                            "se":float(np.std([x[m] for x in data],ddof=1)/math.sqrt(len(data)))
                            if len(data)>1 else 0.} for m in METRICS}
    return [{"epsilon_complex":eps,"axis_epsilon":eps/math.sqrt(2),
             "shots_per_axis":n,"bias_kind":"exact small-matrix operator diagnostic, not analytic certificate",
             "normalization":normalization,"bias":bias,"per_axis_cost":per_axis,
             "expected_work":None if n is None else {
                 m:n*sum(per_axis[a][m]["mean"] for a in ("X","Y")) for m in METRICS},
             "expected_work_se":None if n is None else {
                 m:n*float(np.std([x[m]+y[m] for x,y in zip(raw["X"],raw["Y"],strict=True)],ddof=1)
                             /math.sqrt(len(raw["X"]))) if len(raw["X"])>1 else 0. for m in METRICS},
             "uncertainty":"paired X/Y cost draws; SE of their sum, not independent-axis quadrature"}
            for eps in EPSILONS for n in [shot_budget(normalization,bias,eps)]]


def run_a(audit):
    factors=[np.array([[.8,.09,0],[.09,-.35,0],[0,0,.15]]),
             np.array([[-.2,0,0],[0,.6,.07],[0,.07,-.45]])]
    n=3; h=sum(second_quantize_one_body(g)@second_quantize_one_body(g) for g in factors)
    constructions=[native_df_construct(factors,k) for k in (0,1,2)]
    constructions += [frame_construct(factors,anchor=k) for k in (None,0,1)]
    time=.4;rows=[];candidates=[];samples=[]
    for con in constructions:
        residual=norm(con.dense()-h)
        if residual>ATOL: raise AssertionError("A input reconstruction")
        tail=con.tail();candidates.append({"name":con.name,"ld":con.ld,"metadata":con.metadata,
                        "identity":con.identity,"tail_lambda":0. if tail is None else tail.lambda_r,
                        "tail_components":len(con.primitives),"reconstruction_residual":residual,
                        "whole_hamiltonian":matrix(con.dense()),
                        "deterministic_blocks":[{"id":b.block_id,"matrix":matrix(b.dense()),
                                                 "terms":b.terms} for b in con.blocks],
                        "tail_matrix":matrix(np.zeros_like(h) if tail is None else tail.dense_hamiltonian)})
        for q in (1,2,4):
            corr,normalization=corrected_pr_mean(con,time,q)
            bias=norm(corr-expm(-1j*time*h));eligible=any(shot_budget(normalization,bias,e) is not None for e in EPSILONS)
            row={"candidate":con.name,"q":q,"delta":time/q,"K":2 if tail else None,
                 "corrected_mean":matrix(corr),"normalization":normalization,"operator_bias":bias,
                 "exclusion_reason":None if eligible else "bias exhausts every fixed per-axis precision budget"}
            local=[]
            if eligible:
                for trial in range(8 if tail else 1):
                    seed=SEED+1000*trial+q
                    qc,u,events=trajectory(con,time,q,seed)
                    for axis in ("X","Y"):
                        circ,ref=wrapper(qc,u,axis,[(0,"x")])
                        cost=audit.compile(circ,ref,f"A/{con.name}/q{q}/trial{trial}/{axis}")
                        item={"candidate":con.name,"q":q,"trial":trial,"seed":seed,"axis":axis,
                              "cost":cost,"events":events};local.append(item);samples.append(item)
                row["precision_resources"]=summarize_cost(local,normalization,bias)
            else: row["precision_resources"]=[]
            rows.append(row)
    return {"input_factors":[matrix(g) for g in factors],"hamiltonian":matrix(h),"modes":n,
            "geometry":None,"chemical_basis":None,"df_rank":2,"rank_policy":"fixed exact synthetic rank2",
            "time":time,"q_grid":[1,2,4],"state":"one particle in mode0, preparation X0 included",
            "diagnostic_sector_dimension":3,"candidates":candidates,"rows":rows,"sampled_wrappers":samples,
            "algorithm_claim":"Finite input-derived frame candidate generator; no dense-oracle scalable selection claim"}


def ising_terms(fields,edges):
    mixed_access_metadata(fields,edges)
    n=len(fields);out={}
    for i,c in enumerate(fields):
        p=["I"]*n;p[n-1-i]="Z";out["".join(p)]=float(c)
    for i,j,c in edges:
        p=["I"]*n;p[n-1-i]=p[n-1-j]="Z";out["".join(p)]=float(c)
    return real_terms(out)


def mixed_access_metadata(fields,edges, *, max_degree=2):
    n=len(fields);neighbors=[set() for _ in range(n)]
    if not n or not all(math.isfinite(x) for x in fields): raise ValueError("Finite nonempty fields required")
    seen=set()
    for i,j,c in edges:
        if i==j or not 0<=i<n or not 0<=j<n: raise ValueError("Invalid Ising edge")
        key=tuple(sorted((i,j)))
        if key in seen or not math.isfinite(c): raise ValueError("Unique finite edges required")
        seen.add(key)
        if c: neighbors[i].add(j);neighbors[j].add(i)
    degrees=[len(x) for x in neighbors]
    return {"degrees":degrees,"branch_counts":[2**d for d in degrees],
            "accepted":max(degrees,default=0)<=max_degree,"degree_cap":max_degree,
            "neighbors":[sorted(x) for x in neighbors],
            "excluded_reason":None if max(degrees,default=0)<=max_degree else "explicit multiplexing degree cap"}


def mixed_controlled(fields,edges,target,alpha,time):
    meta=mixed_access_metadata(fields,edges)
    if not meta["accepted"]: raise ValueError("Mixed oracle degree cap")
    n=len(fields);anc=n;qc=QuantumCircuit(n+1);neighbors=meta["neighbors"][target]
    rest={p:c for p,c in ising_terms(fields,edges).items() if p[n-1-target]=="I"}
    for p,c in rest.items(): apply_pauli(qc,p,angle=time*c,control=anc)
    coupling={j:sum(c for i,k,c in edges if {i,k}=={target,j}) for j in neighbors}
    for bits in itertools.product((0,1),repeat=len(neighbors)):
        detuning=fields[target]+sum(coupling[j]*(-1)**b for j,b in zip(neighbors,bits))
        theta=math.atan2(alpha,detuning);radius=math.hypot(detuning,alpha)
        for j,b in zip(neighbors,bits):
            if b==0: qc.x(j)
        if neighbors: qc.append(RYGate(-theta).control(len(neighbors)),[*neighbors,target])
        else: qc.ry(-theta,target)
        qc.append(RZGate(2*time*radius).control(len(neighbors)+1),[*neighbors,anc,target])
        if neighbors: qc.append(RYGate(theta).control(len(neighbors)),[*neighbors,target])
        else: qc.ry(theta,target)
        for j,b in zip(neighbors,bits):
            if b==0: qc.x(j)
    return qc


def c_evolution(fields,edges,alpha,time,q,method):
    n=len(fields);anc=n;dt=time/q;qc=QuantumCircuit(n+1)
    terms=ising_terms(fields,edges);a=pdense(terms,n)
    xs=[]
    for i in range(n):
        p=["I"]*n;p[n-1-i]="X";xs.append(Pauli("".join(p)).to_matrix())
    def append_a(t):
        for p,c in terms.items(): apply_pauli(qc,p,angle=t*c,control=anc)
    def append_x(t):
        for i in range(n):
            p=["I"]*n;p[n-1-i]="X";apply_pauli(qc,"".join(p),angle=t*alpha,control=anc)
    if method=="thrift":
        for _ in range(q):
            for i in reversed(range(n)):
                qc.compose(mixed_controlled(fields,edges,i,alpha,dt),inplace=True)
                if i: append_a(-dt)
        short=expm(-1j*dt*(a+alpha*xs[0]))
        for x in xs[1:]:short=short@expm(1j*dt*a)@expm(-1j*dt*(a+alpha*x))
    elif method=="symmetric_s2":
        append_a(dt/2)
        for step in range(q): append_x(dt);append_a(dt/2 if step==q-1 else dt)
        short=expm(-.5j*dt*a)@expm(-1j*dt*alpha*sum(xs))@expm(-.5j*dt*a)
    elif method=="ordinary_first":
        for _ in range(q): append_x(dt);append_a(dt)
        short=expm(-1j*dt*a)@expm(-1j*dt*alpha*sum(xs))
    else: raise ValueError("Unknown C comparator")
    return qc,np.linalg.matrix_power(short,q),a+alpha*sum(xs)


def run_c(audit):
    inputs=[("old_two",[1.,.7],[(0,1,.3)],.2),
            ("chain_three",[.9,.6,-.4],[(0,1,.2),(1,2,-.15)],.4)]
    rows=[]
    for name,fields,edges,time in inputs:
        for alpha in (.1,.2):
            for method in ("ordinary_first","symmetric_s2","thrift"):
                for q in (1,2,4,8):
                    qc,u,h=c_evolution(fields,edges,alpha,time,q,method)
                    bias=norm(u-expm(-1j*time*h));costs=[]
                    for axis in ("X","Y"):
                        circ,ref=wrapper(qc,u,axis,[(i,"h") for i in range(len(fields))])
                        cost=audit.compile(circ,ref,f"C/{name}/a{alpha}/{method}/q{q}/{axis}")
                        costs.append({"axis":axis,"cost":cost})
                    rows.append({"input":name,"fields":fields,"edges":edges,"alpha":alpha,"time":time,
                                 "q":q,"delta":time/q,"method":method,"operator_bias":bias,
                                 "access":mixed_access_metadata(fields,edges),"costs":costs,
                                 "precision_resources":summarize_cost(costs,1.,bias)})
    scaling=[]
    for n in range(2,9):
        for graph,edges in (("chain",[(i,i+1,.1) for i in range(n-1)]),
                            ("star",[(0,i,.1) for i in range(1,n)])):
            scaling.append({"n":n,"graph":graph,**mixed_access_metadata([1.]*n,edges)})
    return {"rows":rows,"scaling_metadata_only":scaling,"geometry":None,"df_rank":None,"ld":None,
            "state":"all plus, H on every system qubit included",
            "scope":"Known conditional SU(2) access generated from graph; no novel composition claim"}


def enlarged_connection():
    """A genuine Gaussian-isometry / projected diagonal two-body fixture."""
    n=3;m=4
    v=np.eye(m)
    for i,j,angle in ((0,1,math.pi/8),(2,3,math.pi/6),(1,2,math.pi/10)):
        r=np.eye(m);r[i,i]=r[j,j]=math.cos(angle);r[i,j]=-math.sin(angle);r[j,i]=math.sin(angle)
        v=v@r
    basis=gaussian_circuit(v);f=Operator(basis).data
    terms={};edges=[(0,1,.7),(1,2,.4),(2,3,.25),(0,3,-.2)]
    for i,j,c in edges:
        gi=np.zeros((m,m));gj=gi.copy();gi[i,i]=gj[j,j]=1
        terms=padd(terms,pscale(pmul(orbital_paulis(gi),orbital_paulis(gj)),c))
    terms=real_terms(terms);ht=f@pdense(terms,m)@f.conj().T;hp=ht[:2**n,:2**n]
    s=np.diag([1.]*(2**n)+[-1.]*(2**n));hb=(ht+s@ht@s)/2
    u=v[:n,:]
    if norm(u@u.T-np.eye(n))>ATOL:raise AssertionError("Orbital isometry")
    # Independent normal-ordered physical quartic expression (not compression fitting).
    aa=[]
    for j in range(n):aa.append(pdense(annihilator_paulis(n,j),n))
    target=np.zeros_like(hp)
    for i,j,c in edges:
        ci=sum(u[p,i]*aa[p] for p in range(n));cj=sum(u[p,j]*aa[p] for p in range(n))
        target+=c*ci.conj().T@cj.conj().T@cj@ci
    residual=norm(hp-target)
    if residual>ATOL: raise AssertionError("Projected quartic/isometry mismatch")
    ident,nonidentity=split_identity(terms,m)
    prim=[Primitive(f"{p}:{bit}",c/2,p,basis,m-1 if bit else None)
          for p,c in nonidentity.items() for bit in (0,1)]
    reflected=PRConstruction("enlarged_generator_reflection",m,[],prim,ident,QuantumCircuit(m),0)
    # Exact Pauli expansion here is a small reference comparator, not a scalable construction.
    from qiskit.quantum_info import SparsePauliOp
    direct_terms=real_terms(dict(zip(SparsePauliOp.from_operator(Operator(hp),atol=0,rtol=0).paulis.to_labels(),
                                   SparsePauliOp.from_operator(Operator(hp),atol=0,rtol=0).coeffs)))
    ci,rt=split_identity(direct_terms,n)
    direct=PRConstruction("direct_projected_pauli_rte",n,[],
                          [Primitive(p,c,p,QuantumCircuit(n)) for p,c in rt.items()],ci,QuantumCircuit(n),0)
    return reflected,direct,{"orbital_isometry":matrix(u),"unitary_completion":matrix(v),
                            "diagonal_density_edges":edges,"diagonal_paulis":terms,
                            "physical_hamiltonian":matrix(hp),"enlarged_hamiltonian":matrix(ht),
                            "symmetric_generator":matrix(hb),"encoding_residual":residual,
                            "physical_modes":n,"enlarged_modes":m,"geometry":None,"chemical_basis":None,
                            "df_rank":None,"source":"isometric THC Eq.(4)-(7) structural fixture, not fitted molecular data",
                            "compression_claim":False,"direct_dictionary":"dense small reference, not efficient general oracle"}


def run_b(audit):
    reflected,direct,metadata=enlarged_connection();hp=direct.dense();time=.2;rows=[];samples=[]
    for con in (reflected,direct):
        all_q=[];chosen=None
        for q in (1,2,4):
            corr,normalization=corrected_pr_mean(con,time,q)
            physical=corr[:len(hp),:len(hp)]
            bias=norm(physical-expm(-1j*time*hp))
            all_q.append({"q":q,"delta":time/q,"normalization":normalization,"physical_operator_bias":bias,
                          "shots_per_axis":[shot_budget(normalization,bias,e) for e in EPSILONS],
                          "corrected_mean_physical_block":matrix(physical)})
            if chosen is None and shot_budget(normalization,bias,max(EPSILONS)) is not None:
                chosen=(q,normalization,bias)
        if chosen:
            q,b,bias=chosen;local=[]
            for trial in range(8):
                qc,u,events=trajectory(con,time,q,SEED+2000+trial)
                for axis in ("X","Y"):
                    circ,ref=wrapper(qc,u,axis,[(0,"x"),(1,"x")])
                    cost=audit.compile(circ,ref,f"B/{con.name}/q{q}/trial{trial}/{axis}")
                    item={"candidate":con.name,"q":q,"trial":trial,"axis":axis,"events":events,"cost":cost}
                    local.append(item);samples.append(item)
            resources=summarize_cost(local,b,bias)
        else:resources=[]
        rows.append({"candidate":con.name,"ld":0,"rte_r":1,"K":2,"tail_lambda":con.tail().lambda_r,
                     "identity":con.identity,"all_q_diagnostics":all_q,"cost_q":None if not chosen else chosen[0],
                     "precision_resources":resources,
                     "task":"physical first-moment complex signal; no channel or state protection claim"})
    return {"metadata":metadata,"time":time,"rows":rows,"sampled_wrappers":samples,
            "unresolved":["Molecular compression fitting not performed","No reset/echo first-moment circuit accounting",
                          "No demonstrated asymptotic compression or total resource benefit",
                          "Eight cost draws are exploratory; no general compression theorem"]}


def run_all():
    audit=NativeAudit()
    a=run_a(audit);c=run_c(audit);b=run_b(audit)
    return {"schema_version":1,"status":"LIMITED_CONSTRUCTION_COMPARISON_COMPLETE_AWAITING_GPT_REVIEW",
            "A":a,"C":c,"B":b,"native_ir":audit.records,"compiled_circuits":len(audit.records),
            "epsilon_complex":list(EPSILONS),"confidence_failure_total_xy":CONFIDENCE_FAILURE,
            "shot_budget":"Hoeffding per-axis with epsilon/sqrt(2)-operator_bias; 2-axis union bound",
            "quantum_shots_executed":0,"molecular_loads":0,"gpu_calls":0,"ground_state_solves":0,
            "central_hypothesis_adopted":None,"next_stage_authorized":False,"mandatory_stop":True}
