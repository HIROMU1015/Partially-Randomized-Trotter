"""Definition-based Qiskit 1.3 reference lowering without numeric optimization."""
from importlib.metadata import version
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CXGate,XGate,HGate,ZGate,SGate,SdgGate,SXGate,SXdgGate,TGate,TdgGate
from qiskit.transpiler import PassManager,TransformationPass,TranspilerError
from qiskit.transpiler.passes import BasisTranslator,UnrollCustomDefinitions,InverseCancellation

BASIS=('rz','sx','x','cx')
PROFILE=dict(name='Track C reference A',qiskit='1.3.0',basis=list(BASIS),
    passes=['reject opaque unitary','fixed inverse cancellation','UnrollCustomDefinitions',
            'reject opaque unitary','BasisTranslator','fixed X/CX inverse cancellation'],
    numeric_optimization=False,approximate_commutation=False,angle_cutoff=None,
    topology=None,seed=17,seed_scope='no stochastic pass is used',
    equivalence='already-controlled full joint circuit, up to one common global phase',
    control_after_compilation='not qualified; preserve control before lowering')


class RejectOpaqueUnitary(TransformationPass):
    def run(self,dag):
        for node in dag.op_nodes():
            op=node.op
            while op is not None:
                if op.name in ('unitary','isometry'):
                    raise TranspilerError('Reference A requires definitions, not numeric matrix synthesis')
                op=getattr(op,'base_gate',None)
        return dag


def compile_reference(circuit:QuantumCircuit,*,callback=None)->QuantumCircuit:
    """Lower an already-controlled joint circuit, preserving relative phases.

    One common global phase is physically immaterial and is not promised to
    be strict. Do not add a new quantum control after this lowering.
    """
    if version('qiskit')!='1.3.0':raise RuntimeError('Reference A is fixed to Qiskit 1.3.0')
    fixed=[CXGate(),XGate(),HGate(),ZGate(),(SGate(),SdgGate()),
           (SXGate(),SXdgGate()),(TGate(),TdgGate())]
    pm=PassManager([RejectOpaqueUnitary(),InverseCancellation(fixed),
        UnrollCustomDefinitions(SessionEquivalenceLibrary,list(BASIS)),RejectOpaqueUnitary(),
        BasisTranslator(SessionEquivalenceLibrary,list(BASIS)),
        InverseCancellation([CXGate(),XGate()])])
    native=pm.run(circuit.copy(),num_processes=1,callback=callback)
    extra=set(native.count_ops())-set(BASIS)-{'measure','barrier'}
    if extra:raise TranspilerError('Unlowered operations: '+str(extra))
    return native
