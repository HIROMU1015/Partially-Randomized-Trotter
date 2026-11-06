"""Saved A1 gate/state connection; no science, RNG or fixture access on import.

Real qualification is an explicit future scientific call, NOT run by tests.
Exact Fraction application is deliberately limited to this small local-gate
alphabet. Existing builders determine all order/phase/basis choices.
"""
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import math
import subprocess

from . import a1_new_target_n2 as n2
from .shared_cpu import module
from .df_correction_comparison.intervals import configure,iv,bounds,upper,outward_upper

N1='744e1a09661ebba6841fe5faf456f88c41c47780'
N2='5e9bd08068738cdf1070bdc43bee8194b113ae33'
P1=Path('artifacts/o2_a1_new_target/2026-11-06/n1')
P2=Path('artifacts/o2_a1_new_target/2026-11-06/n2')
DOC=Path('docs/research/o2_a1_engine_connection_20261106')
ALLOWED={'h','sdg','cz','p','rz','rzz','crz','crzz','xx_plus_yy'}
ZERO=(F(0),F(0))


def digest(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def read(path):return json.loads(Path(path).read_text())
def finite(x):
    x=float(x)
    if not math.isfinite(x):raise ValueError('nonfinite parameter')
    return x

def inspect_fixed_bundle(root):
    """Read-only byte/identity checks; NO real gate/state numerical evaluation."""
    root=Path(root);files={}
    for commit,directory in ((N1,P1),(N2,P2)):
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',str(directory)],cwd=root,text=True).splitlines()
        for path in paths:
            expected=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
            if (root/path).read_bytes()!=expected:raise ValueError('fixed payload byte mismatch: '+path)
            files[path]=digest(expected)
    one=read(root/P1/'RESULT.json');two=read(root/P2/'SOURCE.json');state=read(root/P2/'STATE_RESULT.json')
    if one['identities']!=two['N1_identities'] or state['hamiltonian_hash']!=one['identities']['hamiltonian_hash']:
        raise ValueError('saved identity binding mismatch')
    if files[str(P2/'engine_internal/STATE_VECTOR.json')]!=state['state_vector_sha256']:
        raise ValueError('saved state identity mismatch')
    if two['N2_source_commit']!='a4141795bd3e23670612d0f0d4a43678457a1a37':raise ValueError('N2 scientific SOURCE mismatch')
    return dict(N1_commit=N1,N2_commit=N2,N1_identities=one['identities'],state_identity=state['state_vector_sha256'],
        file_sha256=files,scope='static bytes/schema/identity only',new_scientific_calls=0)


def technical_profile(root):
    n2.n1.dependencies();p=n2.profile(root)
    name='tests/tracks/algorithm_codesign/test_a1_engine_connection.py'
    p['source_sha256'][name]=digest((Path(root)/name).read_bytes())
    return p


def restore_operations(rows):
    """Recreate only the named saved basis gates; exact metadata/hash required."""
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    constructors={'rz':RZGate,'xx_plus_yy':XXPlusYYGate}
    ops=[]
    for row in rows:
        if row['name'] not in constructors:raise ValueError('unsupported saved basis gate')
        gate=constructors[row['name']](*(finite(x) for x in row['parameters']))
        ops.append((gate,tuple(row['qubits'])))
    observed=[asdict(x) for x in module('df_rte_tail').describe_basis_change_operations(ops)]
    if canonical(observed)!=canonical(rows):raise ValueError('saved local matrix/metadata mismatch')
    return tuple(ops)


def gate_key(gate):
    if getattr(gate,'num_ctrl_qubits',0) and (gate.num_ctrl_qubits!=1 or gate.ctrl_state!=1):
        raise ValueError('only ordinary closed control accepted')
    if gate.name not in ALLOWED:raise ValueError('unsupported engine gate: '+gate.name)
    return gate.name,tuple(finite(p).hex() for p in gate.params)


def interval_matrix(gate):
    """Qiskit1.3 gate definitions at EXACT stored binary64 angles, not libm."""
    import mpmath as mp
    configure();name,params=gate_key(gate);p=[iv(F(float.fromhex(x))) for x in params]
    z=lambda r=0,i=0:mp.iv.mpc(r,i)
    phase=lambda t:z(mp.iv.cos(t),mp.iv.sin(t))
    if name in ('crz','crzz'):
        from qiskit.circuit.library import RZGate,RZZGate
        a=interval_matrix((RZGate if name=='crz' else RZZGate)(finite(gate.params[0])))
        n=len(a);out=[[z(int(i==j)) for j in range(2*n)] for i in range(2*n)]
        for i in range(n):
            for j in range(n):out[2*i+1][2*j+1]=a[i][j]
        return out
    if name=='xx_plus_yy':
        t,b=p;c=mp.iv.cos(t/2);s=mp.iv.sin(t/2);out=[[z(int(i==j)) for j in range(4)] for i in range(4)]
        out[1][1]=out[2][2]=z(c);out[1][2]=z(0,-s)*phase(-b);out[2][1]=z(0,-s)*phase(b)
        return out
    if name=='h':
        v=1/mp.iv.sqrt(iv(2));return [[z(v),z(v)],[z(v),z(-v)]]
    if name=='sdg':return [[z(1),z()],[z(),z(0,-1)]]
    if name=='cz':diagonal=[z(1),z(1),z(1),z(-1)]
    elif name=='rz':diagonal=[phase(-p[0]/2),phase(p[0]/2)]
    elif name=='rzz':diagonal=[phase(-p[0]/2),phase(p[0]/2),phase(p[0]/2),phase(-p[0]/2)]
    elif name=='p':diagonal=[z(1),phase(p[0])]
    else:raise ValueError('unsupported interval gate')
    return [[diagonal[i] if i==j else z() for j in range(len(diagonal))] for i in range(len(diagonal))]


def matrix_record(gate):
    import numpy as np
    from qiskit.quantum_info import Operator
    # Same local matrix interface as the saved basis registry. Controlled RZZ
    # has no to_matrix in Qiskit1.3; Operator evaluates its fixed definition.
    a=np.asarray(Operator(gate).data,dtype=np.complex128)
    if a.shape!=(1<<gate.num_qubits,)*2 or not np.isfinite(a).all():raise ValueError('invalid local matrix')
    pairs=[[ (F(float(x.real)),F(float(x.imag))) for x in row] for row in a]
    return a,pairs,digest(a.tobytes(order='C'))


def certify_gate(gate):
    """Frobenius enclosure for actual binary64 matrix vs same ideal gate."""
    import mpmath as mp
    a,pairs,sha=matrix_record(gate);ideal=interval_matrix(gate);s=F(0)
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            for actual,expected in zip(pairs[i][j],(ideal[i][j].real,ideal[i][j].imag)):
                lo,hi=bounds(expected-iv(actual));s+=max(abs(lo),abs(hi))**2
    delta=outward_upper(upper(mp.iv.sqrt(iv(s))))
    return dict(key=gate_key(gate),matrix_sha256=sha,matrix=pairs,error_upper=delta)


def multiply(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def add(a,b):return a[0]+b[0],a[1]+b[1]
def norm_squared(v):return sum((x*x+y*y for x,y in v),F(0))


def apply_exact(v,matrix,qubits):
    """Small local-gate contraction, all real/imag operations exact rational."""
    n=(len(v)-1).bit_length();qs=tuple(qubits);d=1<<len(qs)
    if len(v)!=(1<<n) or len(set(qs))!=len(qs) or any(q<0 or q>=n for q in qs):raise ValueError('invalid gate support')
    if len(matrix)!=d or any(len(row)!=d for row in matrix):raise ValueError('matrix/support size mismatch')
    mask=sum(1<<q for q in qs);out=list(v)
    for base in range(len(v)):
        if base&mask:continue
        indices=[base|sum(((j>>k)&1)<<q for k,q in enumerate(qs)) for j in range(d)]
        for i,at in enumerate(indices):
            value=ZERO
            for j,src in enumerate(indices):value=add(value,multiply(matrix[i][j],v[src]))
            out[at]=value
    return out


def require_probability(p):
    p=F(p)
    if not 0<=p<=1:raise ValueError('Born probability outside [0,1]; no clipping')
    return p


def born_probability(v,ancilla):
    total=norm_squared(v)
    if total<=0:raise ValueError('nonpositive state norm')
    plus=sum((x*x+y*y for i,(x,y) in enumerate(v) if not ((i>>ancilla)&1)),F(0))
    # Exact normalization: initial sqrt(norm) cancels in this ratio. Final
    # normalization of rounded-gate state is explicitly charged in 4E below.
    return require_probability(plus/total)


def threshold_count(p,bits=53):
    p=require_probability(p)
    if type(bits) is not int or not 1<=bits<=53:raise ValueError('invalid bit grid')
    a=p*(1<<bits);return -(-a.numerator//a.denominator)


def compose_error(delta,max_gates,state_bound):
    delta,state_bound=F(delta),F(state_bound)
    if delta<0 or state_bound<0 or type(max_gates) is not int or max_gates<1:raise ValueError('invalid uniform envelope')
    # Bound the product with the existing fixed-precision directed context;
    # do not create enormous exact numerator/denominator powers for long paths.
    configure();E=outward_upper(upper((1+iv(delta))**max_gates-1))
    if E>=1:raise ValueError('positive approximate output norm not guaranteed; STOP')
    return dict(product_vector_error_upper=str(E),gate_and_final_normalization_coordinate_upper=str(4*E),
        state_coordinate_upper=str(state_bound),gate_application_arithmetic='0 (exact rational)',
        Born_sum_division_arithmetic='0 (exact rational)',initial_normalization='exact ratio cancellation',
        conditional_grid_coordinate_upper=str(F(2,2**53)),
        total_coordinate_upper=str(state_bound+4*E+F(2,2**53)),budget_adopted=False)


class SavedEngine:
    """Private probability engine, never a CV feature/fit/allocation interface.

    Construction/qualification on REAL records is a deferred scientific call.
    Neither operation is performed by the static inspection entry point.
    """
    def __init__(self,prep,policy,config,law,state_record,*,step_time):
        from dataclasses import fields
        self.dep=n2.n1.dependencies();self.step_time=finite(step_time);self.n=state_record['num_system_qubits']
        self.prep,self.policy,self.config=prep,policy,config
        cfg=config['config']
        if cfg['tail_hash']!=prep['tail_hash'] or cfg['evolution_time']!=self.step_time or cfg['rte_steps']!=1 or cfg['finite_taylor_order']!=2:
            raise ValueError('saved config/tail/time mismatch')
        self.registry=module('df_rte_tail').DFBasisRegistry();self.definitions={};self.basis_ops={}
        metadata=prep['basis_operations'];ids={c['basis_hash']:c['basis_id'] for c in prep['components']}
        ids.update({b['basis_hash']:b['basis_id'] for b in prep['deterministic_blocks']})
        for bh,rows in metadata.items():
            ops=restore_operations(rows);definition=self.registry.register(ops,num_system_qubits=self.n,basis_id=ids[bh])
            if definition.basis_hash!=bh:raise ValueError('restored full basis identity mismatch')
            self.basis_ops[bh]=ops
        for row in policy['definitions']:
            md=row['definition'];ops=restore_operations(md['operations'])
            d=self.registry.register(ops,num_system_qubits=self.n,basis_id=md['basis_id'])
            if d.basis_hash!=md['basis_hash']:raise ValueError('restored selected basis identity mismatch')
            self.definitions[(row['source_basis_id'],row['source_basis_hash'],tuple(row['support']))]=d
        self.blocks=[]
        for b in prep['deterministic_blocks']:
            values=dict(b);values['basis_change_operations']=tuple(self.dep.rte.BasisChangeOperation(**{**r,'qubits':tuple(r['qubits']),'parameters':tuple(r['parameters'])}) for r in b['basis_change_operations'])
            values['runtime_basis_operations']=self.basis_ops[b['basis_hash']]
            cls=self.dep.partial.DFDeterministicOneBodySpec if b['block_kind']=='one_body' else self.dep.partial.DFDeterministicFragmentSpec
            self.blocks.append(cls(**values))
        allowed={f.name for f in fields(self.dep.rte.RTEComponent)};self.components=[];self.specs=[]
        interface=module('df_rte_circuit')
        for c in prep['components']:
            if c['is_identity'] or c['coefficient_abs']<=0:raise ValueError('extracted identity/nonzero catalog required')
            md=tuple(self.dep.rte.BasisChangeOperation(**{**r,'qubits':tuple(r['qubits']),'parameters':tuple(r['parameters'])}) for r in metadata[c['basis_hash']])
            args={k:v for k,v in c.items() if k in allowed};args['diagonal_pauli_support']=tuple(c['diagonal_pauli_support']);args['basis_change_operations']=md
            self.components.append(self.dep.rte.RTEComponent(**args))
            self.specs.append(interface.DFRTEComponentCircuitSpec(component_id=c['component_id'],coefficient_abs=c['coefficient_abs'],coefficient_sign=c['coefficient_sign'],df_fragment_id=c['df_fragment_id'],basis_id=c['basis_id'],basis_hash=c['basis_hash'],diagonal_pauli_support=tuple(c['diagonal_pauli_support']),basis_change_operations=md,num_system_qubits=self.n))
        self.components=tuple(self.components);self.specs=tuple(self.specs)
        required={(c.basis_id,c.basis_hash,tuple(c.diagonal_pauli_support)) for c in self.components}
        if set(self.definitions)!=required:raise ValueError('incomplete selected basis policy coverage')
        self.index={c.component_id:i for i,c in enumerate(self.components)}
        self.positive={r['label'] for r in law['label_masses'] if F(r['exact'])>0}
        if not self.positive or not self.positive<=set(self.index):raise ValueError('law/catalog support mismatch')
        self.distribution=self.dep.rte.RTEFiniteDistribution(**config['distribution'])
        if tuple(self.distribution.orders)!=(0,2):raise ValueError('fixed K2 orders required')
        self.raw_state=tuple((F(float.fromhex(re)),F(float.fromhex(im))) for re,im in state_record['qiskit_full_complex_hex'])
        if len(self.raw_state)!=(1<<self.n):raise ValueError('saved state dimension mismatch')
        self.certificate=None;self.table={}

    def _request(self):return NS(step_time=self.step_time,controlled=True,ancilla_qubit=self.n)
    def _phases(self,circuit):
        request=NS(step_time=self.step_time,repetition_count=1,controlled=True,ancilla_qubit=self.n,
            preparation=NS(constant_coefficient=float.fromhex(self.prep['constant_coefficient_hex']),extracted_identity_coefficient=float.fromhex(self.prep['identity_coefficient_hex'])))
        self.dep.repeated.QiskitDFPartialS2RepeatedCircuitBuilder()._append_aggregated_phases(circuit,request)

    def selected_circuit(self,order,labels,axis):
        from qiskit import QuantumCircuit
        if axis not in ('X','Y') or order not in (0,2) or len(labels)!=order+1 or any(x not in self.positive for x in labels):raise ValueError('path/axis outside fixed positive Q support')
        event=self.dep.rte._make_event([self.index[labels[-1]],*(self.index[x] for x in labels[:-1])],self.components,self.distribution,(0,2).index(order))
        interface=module('df_rte_circuit');req=interface.DFRTEEventSequenceCircuitRequest((event,),self.specs,
            controlled=True,ancilla_qubit=self.n,cancel_adjacent_equal_bases=True,
            tail_id=self.config['config']['tail_id'],tail_hash=self.prep['tail_hash'],occurrence_rte_steps=1)
        plan=self.dep.policy.make_run_threshold_basis_plan(req,self.definitions,maximum_support_run_length=1,training_fingerprint=self.policy['policy_fingerprint'])
        builder=module('df_rte_qiskit').QiskitDFRTEEventCircuitBuilder(basis_registry=self.registry)
        circuit=QuantumCircuit(self.n+1);self._phases(circuit);step=self.dep.partial.QiskitDFPartialS2CircuitBuilder()
        for block in self.blocks:step._append_block(circuit,block,self._request())
        event_result=builder.build_sequence(req,basis_plan=plan)
        circuit.compose(event_result.circuit,inplace=True)
        for block in reversed(self.blocks):step._append_block(circuit,block,self._request())
        # Same wrapper body as the fixed builder, with no unrelated trajectory
        # builder, RNG, prices, or mutable training metadata.
        path_hash=digest(canonical(dict(preparation=self.prep['preparation_hash'],policy=self.policy['policy_fingerprint'],order=order,labels=labels)).encode())
        evolution=NS(circuit=circuit,repetition_count=1,step_time=self.step_time,trajectory_fingerprint=path_hash,
            provenance_fingerprint=self.policy['policy_fingerprint'],circuit_semantics_fingerprint=path_hash,
            constant_phase=-self.step_time*float.fromhex(self.prep['constant_coefficient_hex']),
            extracted_identity_phase=-self.step_time*float.fromhex(self.prep['identity_coefficient_hex']),
            rte_relative_phase=event_result.relative_ancilla_phase)
        return self.dep.hadamard.QiskitRPEHadamardInterrogationBuilder()._build_validated(NS(evolution=evolution,axis='cosine' if axis=='X' else 'sine',include_measurement=False),round_index=0,ancilla=self.n,total_time=self.step_time).circuit

    def uniform_inventory(self):
        """All possible local gates + symbolic no-cancellation path-length cap.

        No molecular full path is built/enumerated. Only fixed D subcircuits,
        stored basis operations/inverses, and finite RTE/wrapper primitives.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PhaseGate,HGate,SdgGate
        gates=[];fixed_length=0;fixed_records=[];step=self.dep.partial.QiskitDFPartialS2CircuitBuilder()
        for b in self.blocks:
            c=QuantumCircuit(self.n+1);step._append_block(c,b,self._request());fixed_length+=len(c.data)
            gates.extend(i.operation for i in c.data)
            fixed_records.append(dict(block_id=b.block_id,half_step_gates=[
                dict(key=gate_key(i.operation),qubits=[c.find_bit(q).index for q in i.qubits]) for i in c.data]))
        c=QuantumCircuit(self.n+1);self._phases(c);gates.extend(i.operation for i in c.data);scalar_length=len(c.data)
        scalar_records=[dict(key=gate_key(i.operation),qubits=[c.find_bit(q).index for q in i.qubits]) for i in c.data]
        all_ops=list(self.basis_ops.values())+[d.runtime_operations for d in self.definitions.values()]
        if not all_ops:raise ValueError('empty basis inventory')
        max_basis_length=max(map(len,all_ops))
        for ops in all_ops:
            for g,qs in ops:gates.extend((g,g.inverse()))
        helper=module('df_rte_qiskit').QiskitDFRTEEventCircuitBuilder
        for order in (0,2):
            angle=math.atan(self.distribution.dimensionless_step_time/(order+1))
            for sign in (-1,1):
                for support in ((0,),(0,1)) if self.n>=2 else ((0,),):
                    for role in ('rotation','product'):
                        c=QuantumCircuit(self.n+1);helper._append_central_application(c,NS(unsigned_rotation_angle=sign*angle),NS(diagonal_pauli_support=support,role=role,coefficient_sign=sign),controlled=True,ancilla_qubit=self.n)
                        gates.extend(i.operation for i in c.data)
        # Exact same float accumulation order of pi contributions (n=2 phase
        # plus at most two negative product signs); no identities in tail.
        phase=0.
        for _ in range(3):phase+=math.pi;gates.append(PhaseGate(phase))
        gates.extend((HGate(),SdgGate()))
        # at most 3 applications, two CZs each, and one accumulated phase.
        cap=2*fixed_length+scalar_length+3*(2*max_basis_length+2)+1+3
        self.inventory_info=dict(deterministic_block_subcircuits=len(self.blocks),
            stored_basis_catalogs=len(all_ops),candidate_gate_entries=len(gates),
            deterministic_half_step_records=fixed_records,
            scalar_phase_records=scalar_records,
            actual_full_path_builds=0,actual_state_evolutions=0)
        return gates,cap

    def qualify(self,state_bound):
        gates,cap=self.uniform_inventory();table={}
        for gate in gates:
            key=gate_key(gate);_,_,matrix_sha=matrix_record(gate)
            if key in table:
                if table[key]['matrix_sha256']!=matrix_sha:raise ValueError('same gate key has inconsistent numerical matrix')
            else:table[key]=certify_gate(gate)
        delta=max(row['error_upper'] for row in table.values())
        result=compose_error(delta,cap,state_bound)
        result.update(status='UNIFORM_GATE_METHOD_EVALUATED',local_gate_count=len(table),max_gate_count=cap,maximum_local_operator_error=str(delta),inventory=self.inventory_info,full_support_scope='every positive n=0/2 label path, both axes; no representative-path extrapolation',N3_authorized=False)
        self.table=table;self.certificate=result;return result

    def probability(self,order,labels,axis):
        if self.certificate is None:raise ValueError('uniform gate certificate missing')
        circuit=self.selected_circuit(order,labels,axis)
        if len(circuit.data)>self.certificate['max_gate_count'] or float(circuit.global_phase)!=0 or circuit.num_clbits:raise ValueError('gate count/global phase/classical scope mismatch')
        v=list(self.raw_state)+[ZERO]*len(self.raw_state)
        for inst in circuit.data:
            key=gate_key(inst.operation)
            if key not in self.table:raise ValueError('uncertified gate variant; STOP')
            row=self.table[key];_,_,matrix_sha=matrix_record(inst.operation)
            if row['matrix_sha256']!=matrix_sha:raise ValueError('qualified gate matrix changed')
            v=apply_exact(v,row['matrix'],[circuit.find_bit(q).index for q in inst.qubits])
        return born_probability(v,self.n)


def from_fixed_bundle(root):
    """DEFERRED REAL DESERIALIZATION: not invoked during current technical task."""
    root=Path(root);binding=inspect_fixed_bundle(root)
    prep=read(root/P1/'PREPARATION.json');policy=read(root/P1/'POLICY.json');config=read(root/P1/'CONFIG.json');law=read(root/P1/'LAW.json')
    state=read(root/P2/'engine_internal/STATE_VECTOR.json');state['num_system_qubits']=8
    return SavedEngine(prep,policy,config,law,state,step_time=.02),binding


def qualify_fixed_bundle(root,output_dir):
    """FUTURE SCIENTIFIC CALL: separate explicit execution permission required.

    No real path, state evolution, solve/eigh, sampling, compile or costs.
    Never called by static inspection or the artificial tests.
    """
    root=Path(root);out=Path(output_dir)
    dirty=n2.git(root,'status','--porcelain=v1','--untracked-files=all')
    out.mkdir(parents=True,exist_ok=False)
    counts=dict(saved_bundle_decode=0,uniform_gate_qualification=0,real_path_build=0,real_state_evolution=0,preparation=0,CDF=0,solve=0,eigh=0,RNG=0,shots=0,compile=0,variance_N_cost=0)
    try:
        locked=read(root/DOC/'PROFILE_LOCK.json')
        if dirty:raise ValueError('source worktree was not clean before real qualification')
        if canonical(technical_profile(root))!=canonical(locked):raise ValueError('source/environment profile changed')
        counts['saved_bundle_decode']=1;engine,binding=from_fixed_bundle(root)
        counts['uniform_gate_qualification']=1;result=engine.qualify(F('0.000000781'))
        n2.write(out/'UNIFORM_ERROR.json',result)
        n2.write(out/'GATE_CERTIFICATES.json',[dict(key=row['key'],matrix_sha256=row['matrix_sha256'],
            matrix_complex_rationals=[[[str(re),str(im)] for re,im in line] for line in row['matrix']],
            error_upper=str(row['error_upper'])) for row in engine.table.values()])
        n2.write(out/'SOURCE.json',dict(binding=binding,source_commit=n2.git(root,'rev-parse','HEAD'),actual_calls=counts))
        n2.write(out/'STOP.json',dict(status='MANDATORY_STOP',budget_adopted=False,N3_authorized=False))
        return result
    except Exception as exc:
        n2.write(out/'STOP.json',dict(status='STOPPED',reason=str(exc),actual_calls=counts,no_retry=True));raise
