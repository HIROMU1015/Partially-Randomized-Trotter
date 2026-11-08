"""State-free saved-PR connection. Calling it on real N1 is deferred science.

The fixed engine remains unchanged. Only its metadata restoration is separated
from state loading; its selected builder, event policy and phase code are reused.
No files, state, Born, random source or compiler are accessed on import.
"""
from dataclasses import asdict, dataclass, fields
from fractions import Fraction as F
import json
import math

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import PhaseGate, RZGate, RZZGate, XXPlusYYGate, ZGate

from . import a1_engine_connection as old
from . import vacuum_reference_cswap as prototype
from .shared_cpu import module


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record_hashes(records):
    return {name:old.digest(old.canonical(value).encode()) for name,value in records.items()}


def select_paths(records):
    """Cost-blind two-path rule, NOT sampling. Real use is a science call."""
    law=records['LAW.json']
    require(law['uniform_grid_size']==2**53 and tuple(law['orders'])==(0,2)
        and law['application_label_order']=='products_then_rotation'
        and law['Q_definition']=='Q(order)*product(Q(label_position)); independent uniform-word model',
        'actual finite law convention mismatch')
    masses={r['label']:F(r['exact']) for r in law['label_masses']}
    orders=tuple(F(r['exact']) for r in law['order_masses'])
    require(len(masses)==len(law['label_masses']) and len(orders)==2
        and all(x>=0 and (x*2**53).denominator==1 for x in (*masses.values(),*orders))
        and sum(masses.values())==sum(orders)==1 and all(x>0 for x in orders),
        'law support/grid/mass mismatch')
    positive=sorted((x for x,p in masses.items() if p>0),key=lambda x:x.encode('utf-8'))
    require(len(positive)>=3,'three positive labels required; no replacement selection')
    identities={k:records['PREPARATION.json'][k] for k in
                ('hamiltonian_hash','partition_hash','preparation_hash','tail_hash')}
    identities['policy']=records['POLICY.json']['policy_fingerprint']
    return tuple(dict(order=k,labels=labels,
        identity=old.digest(old.canonical(dict(identities=identities,order=k,labels=labels)).encode()),
        mass_exact=str(orders[(0,2).index(k)]*math.prod(masses[x] for x in labels)))
        for k,labels in ((0,(positive[0],)),(2,tuple(positive[:3]))))


class SavedPR(old.SavedEngine):
    """Reuse fixed selected_circuit without constructing/reading any state.

    The original constructor couples metadata restoration to a saved state.
    The small restoration below intentionally duplicates only that constructor's
    metadata portion, retaining its exact constructors and hash/matrix checks.
    """
    def __init__(self,records,*,expected_hashes):
        require(record_hashes(records)==expected_hashes,'input/source record identity mismatch')
        records=json.loads(old.canonical(records)) # owned, immutable-by-contract snapshot
        self.dep=old.n2.n1.dependencies()
        self.prep=prep=records['PREPARATION.json'];self.policy=policy=records['POLICY.json']
        self.config=config=records['CONFIG.json'];law=records['LAW.json']
        source=records['SOURCE.json'];scope=source['scope'];self.n=scope['system_qubits']
        cfg=config['config'];self.step_time=old.finite(cfg['evolution_time'])
        require(type(self.n) is int and 1<=self.n<=8
            and (scope['m'],scope['q'],scope['r'],scope['K'])==(0,1,1,2)
            and scope['delta']==self.step_time and scope['coefficient_atol']==prep['coefficient_atol']==0.
            and cfg['tail_hash']==prep['tail_hash'] and cfg['rte_steps']==1
            and cfg['finite_taylor_order']==2,'fixed context mismatch')
        decl=policy['declaration']
        require(decl['source_commit']==source['source_commit']
            and decl['preparation_hash']==prep['preparation_hash'] and decl['tail_hash']==prep['tail_hash']
            and (decl['schema'],decl['rule'],decl['maximum_support_run_length'],decl['boundary'],decl['controlled'])
            ==('a1_new_target_selected_policy_v1','support_run_le_1',1,'boundary_optimized','ordinary_diag_I_U')
            and old.digest(old.canonical(decl).encode())==policy['policy_fingerprint']
            and old.digest(old.canonical(policy['definitions']).encode())==decl['definition_sha256'],
            'selected SOURCE/policy fingerprint mismatch')
        self.registry=module('df_rte_tail').DFBasisRegistry()
        self.definitions={};self.basis_ops={};self.blocks=[]
        ids={c['basis_hash']:c['basis_id'] for c in prep['components']}
        ids.update({b['basis_hash']:b['basis_id'] for b in prep['deterministic_blocks']})
        for bh,rows in prep['basis_operations'].items():
            ops=old.restore_operations(rows)
            definition=self.registry.register(ops,num_system_qubits=self.n,basis_id=ids[bh])
            require(definition.basis_hash==bh,'restored full basis identity mismatch')
            self.basis_ops[bh]=ops
        for row in policy['definitions']:
            md=row['definition'];ops=old.restore_operations(md['operations'])
            d=self.registry.register(ops,num_system_qubits=self.n,basis_id=md['basis_id'])
            require(d.basis_hash==md['basis_hash'],'selected basis identity mismatch')
            key=(row['source_basis_id'],row['source_basis_hash'],tuple(row['support']))
            require(key not in self.definitions,'duplicate selected definition')
            self.definitions[key]=d
        for i,b in enumerate(prep['deterministic_blocks']):
            require(b['order_index']==i and b['num_system_qubits']==self.n
                and old.canonical(b['basis_change_operations'])==old.canonical(prep['basis_operations'][b['basis_hash']]),
                'block order/basis mismatch')
            values=dict(b)
            values['basis_change_operations']=tuple(self.dep.rte.BasisChangeOperation(
                **{**r,'qubits':tuple(r['qubits']),'parameters':tuple(r['parameters'])}) for r in b['basis_change_operations'])
            values['runtime_basis_operations']=self.basis_ops[b['basis_hash']]
            cls={'one_body':self.dep.partial.DFDeterministicOneBodySpec,
                 'df_fragment':self.dep.partial.DFDeterministicFragmentSpec}.get(b['block_kind'])
            require(cls is not None,'unknown deterministic block');self.blocks.append(cls(**values))
        allowed={f.name for f in fields(self.dep.rte.RTEComponent)}
        interface=module('df_rte_circuit');components=[];specs=[]
        for c in prep['components']:
            require(not c['is_identity'] and c['coefficient_abs']>0,'extracted positive catalog required')
            md=tuple(self.dep.rte.BasisChangeOperation(**{**r,'qubits':tuple(r['qubits']),
                'parameters':tuple(r['parameters'])}) for r in prep['basis_operations'][c['basis_hash']])
            args={k:v for k,v in c.items() if k in allowed}
            args.update(diagonal_pauli_support=tuple(c['diagonal_pauli_support']),basis_change_operations=md)
            components.append(self.dep.rte.RTEComponent(**args))
            specs.append(interface.DFRTEComponentCircuitSpec(component_id=c['component_id'],
                coefficient_abs=c['coefficient_abs'],coefficient_sign=c['coefficient_sign'],
                df_fragment_id=c['df_fragment_id'],basis_id=c['basis_id'],basis_hash=c['basis_hash'],
                diagonal_pauli_support=tuple(c['diagonal_pauli_support']),basis_change_operations=md,
                num_system_qubits=self.n))
        self.components=tuple(components);self.specs=tuple(specs)
        required={(c.basis_id,c.basis_hash,tuple(c.diagonal_pauli_support)) for c in components}
        require(set(self.definitions)==required,'selected coverage mismatch')
        self.index={c.component_id:i for i,c in enumerate(components)}
        require(len(self.index)==len(components),'duplicate component identity')
        masses={r['label']:F(r['exact']) for r in law['label_masses']}
        require(set(masses)==set(self.index) and len(masses)==len(law['label_masses']),
                'law/catalog identity mismatch')
        self.positive={x for x,p in masses.items() if p>0}
        self.distribution=self.dep.rte.RTEFiniteDistribution(**config['distribution'])
        require(tuple(self.distribution.orders)==(0,2),'K2 distribution mismatch')

    def probability(self,*a,**kw):raise ValueError('state/Born disabled in resource adapter')
    def qualify(self,*a,**kw):raise ValueError('old engine qualification disabled')
    def uniform_inventory(self,*a,**kw):raise ValueError('old qualification disabled')

    def body_with_boundaries(self,order,labels):
        """Trace existing block/application ends WITHOUT inserting any gates."""
        require(order in (0,2) and len(labels)==order+1 and all(x in self.positive for x in labels),
                'path outside positive Q support')
        event=self.dep.rte._make_event([self.index[labels[-1]],*(self.index[x] for x in labels[:-1])],
            self.components,self.distribution,(0,2).index(order))
        interface=module('df_rte_circuit')
        req=interface.DFRTEEventSequenceCircuitRequest((event,),self.specs,controlled=True,
            ancilla_qubit=self.n,cancel_adjacent_equal_bases=True,tail_id=self.config['config']['tail_id'],
            tail_hash=self.prep['tail_hash'],occurrence_rte_steps=1)
        plan=self.dep.policy.make_run_threshold_basis_plan(req,self.definitions,
            maximum_support_run_length=1,training_fingerprint=self.policy['policy_fingerprint'])
        base=module('df_rte_qiskit').QiskitDFRTEEventCircuitBuilder;cuts=[]
        class Traced(base):
            @staticmethod
            def _append_central_application(circuit,event,application,**kw):
                phase=base._append_central_application(circuit,event,application,**kw)
                cuts.append(len(circuit.data));return phase
        circuit=QuantumCircuit(self.n+1);self._phases(circuit)
        step=self.dep.partial.QiskitDFPartialS2CircuitBuilder();boundaries=[]
        for block in self.blocks:
            step._append_block(circuit,block,self._request());boundaries.append(len(circuit.data))
        offset=len(circuit.data);rte=Traced(basis_registry=self.registry).build_sequence(req,basis_plan=plan)
        require(len(cuts)==order+1,'application boundary trace mismatch')
        cuts[-1]=len(rte.circuit.data) # trailing shared basis/event phase stays in final application
        circuit.compose(rte.circuit,inplace=True);boundaries.extend(offset+x for x in cuts)
        for block in reversed(self.blocks):
            step._append_block(circuit,block,self._request());boundaries.append(len(circuit.data))
        require(float(circuit.global_phase)==0. and len(boundaries)>=2,'controlled body/global phase mismatch')
        # Midpoint in semantic units, earlier boundary for odd unit count.
        chosen=len(boundaries)//2
        cut=boundaries[chosen-1]
        require(0<cut<len(circuit.data),'no nontrivial natural split')
        R=decontrol(circuit,self.n,0,cut);L=decontrol(circuit,self.n,cut,len(circuit.data))
        return circuit,L,R,dict(rule='semantic_unit_midpoint_earlier_tie',
            unit_ends=boundaries,unit_count=len(boundaries),right_units=chosen,controlled_gate_cut=cut,
            scalar_rule='retain exact ancilla P terms; correction includes their total')


@dataclass(frozen=True)
class Lane:
    n: int
    operations: tuple
    scalar_terms: tuple

    def circuit(self):
        c=QuantumCircuit(self.n)
        constructors={'rz':RZGate,'rzz':RZZGate,'xx_plus_yy':XXPlusYYGate,'z':ZGate}
        for name,params,qs in self.operations:
            require(name in constructors,'unsupported saved lane gate')
            gate=constructors[name](*(float.fromhex(x) for x in params))
            prototype._vacuum_gate_phase(gate)
            require(len(set(qs))==len(qs) and all(type(q) is int and 0<=q<self.n for q in qs),
                    'lane wire mapping')
            c.append(gate,qs)
        return c

    @property
    def scalar(self):return sum((F(float.fromhex(x)) for x in self.scalar_terms),F(0))


def decontrol(c,n,start,stop):
    """Read exact existing control1 branch; basis gates remain uncontrolled."""
    operations=[];phases=[]
    for item in c.data[start:stop]:
        g=item.operation;qs=tuple(c.find_bit(q).index for q in item.qubits)
        if qs==(n,):
            require(type(g) is PhaseGate or isinstance(g,PhaseGate),'unsupported signal phase')
            phases.append(old.finite(g.params[0]).hex());continue
        if n in qs:
            require(isinstance(g,ControlledGate) and g.num_ctrl_qubits==1 and g.ctrl_state==1
                and qs[0]==n,'unsupported control/order')
            g=g.base_gate;qs=qs[1:]
        require(g.name in ('rz','rzz','xx_plus_yy','z') and all(q<n for q in qs),
                'unsupported decontrolled gate')
        prototype._vacuum_gate_phase(g)
        operations.append((g.name,tuple(old.finite(p).hex() for p in g.params),qs))
    lane=Lane(n,tuple(operations),tuple(phases));lane.circuit();return lane


def build_reference(L,R,axis,*,mapping=None):
    require(L.n==R.n and axis in ('X','Y'),'lane/axis mismatch')
    n=L.n;mapping=tuple(range(2*n+1)) if mapping is None else tuple(mapping)
    require(len(mapping)==2*n+1 and set(mapping)==set(range(2*n+1))
            and all(type(q) is int for q in mapping),'reference wire permutation')
    target,reference,a=mapping[:n],mapping[n:2*n],mapping[-1]
    lc,rc=L.circuit(),R.circuit()
    ell=sum((prototype._vacuum_gate_phase(i.operation) for i in lc.data),F(0))
    rr=sum((prototype._vacuum_gate_phase(i.operation) for i in rc.data),F(0))
    scalar=L.scalar+R.scalar;exact=ell+rr+scalar
    angle=float(exact);require(math.isfinite(angle),'nonfinite correction angle')
    c=QuantumCircuit(2*n+1);c.reset(reference);c.h(a)
    for t,v in zip(target,reference):c.cswap(a,t,v)
    c.compose(lc.inverse(),qubits=target,inplace=True);c.compose(rc,qubits=reference,inplace=True)
    for t,v in zip(target,reference):c.cswap(a,t,v)
    c.p(angle,a)
    if axis=='Y':c.sdg(a)
    c.h(a);bit=ClassicalRegister(1,'readout');c.add_register(bit);c.measure(a,bit[0])
    ledger=dict(L_scalar_free_vacuum=str(ell),R_scalar_free_vacuum=str(rr),original_scalar=str(scalar),
        correction_exact=str(exact),correction_hex=angle.hex(),
        correction_rounding_upper=str(abs(F(angle)-exact)),ledger_arithmetic='exact Fraction',
        lane_global_phase='0; original scalar terms transferred only to signal correction',
        reference_initialization='ideal vacuum reset',postselection=False)
    return c,ledger


def controlled_wrapper(context,path,axis):
    c=context.selected_circuit(path['order'],tuple(path['labels']),axis)
    bit=ClassicalRegister(1,'readout');c.add_register(bit);c.measure(context.n,bit[0]);return c
