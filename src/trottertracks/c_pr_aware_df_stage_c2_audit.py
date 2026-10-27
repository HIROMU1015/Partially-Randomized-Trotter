"""Static/source audit only: construct a pass manager, never run a circuit."""
import hashlib
import inspect
from pathlib import Path


def compiler_audit():
    import qiskit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.transpiler.passes import (InverseCancellation, Optimize1qGatesDecomposition,
        BasisTranslator, UnitarySynthesis, HighLevelSynthesis, UnrollCustomDefinitions)
    if qiskit.__version__ != '1.3.0': raise ValueError('PINNED_QISKIT')
    profile=dict(optimization_level=1,basis_gates=['rz','sx','x','cx'],
                 seed_transpiler=17,approximation_degree=1.,qubits_initially_zero=False)
    pm=generate_preset_pass_manager(**profile)
    def tasks(controller):
        return [dict(name=type(t).__name__,module=type(t).__module__,
            nested=tasks(t) if hasattr(t,'tasks') else []) for t in controller.tasks]
    stages={name:tasks(getattr(pm,name).to_flow_controller()) if getattr(pm,name) else [] for name in pm.stages}
    files={}
    for cls in (InverseCancellation,Optimize1qGatesDecomposition,BasisTranslator,
                UnitarySynthesis,HighLevelSynthesis,UnrollCustomDefinitions):
        p=Path(inspect.getsourcefile(cls))
        files[cls.__name__]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return dict(status='STATIC_SOURCE_AND_PASS_SCHEDULE_ONLY',profile=profile,
        stages=stages,local_python_sources=files,pass_manager_executed=False,
        preprocessing='UnrollCustomDefinitions(empty EquivalenceLibrary,basis+u)',
        phase_wrapper='task-local symbolic MPFR160 declared phase;not a whole-operator enclosure',
        whole_kernel_cross_event_1q_runs=True,
        synthesis_classification='UNRESOLVED_UNIFORM_P0_ROUNDING_AND_REFERENCE_ENCLOSURE',
        mathematical_route='local analytic unitary proofs plus all-path compiler proof and shell-aware telescoping',
        useful_global_synthesis_bound=None,cost_support=None,
        blockers=['Rust Optimize1qGatesDecomposition multiplies matrices in f64 and resynthesizes runs',
                  'Rust default ANGLE_ZERO_EPSILON=1e-12 is a simplification threshold,not an operator error certificate',
                  'basis translation/global phase/ref numerical errors need their own enclosures',
                  'whole-P0 path and optimization-loop output ranges lack a derived uniform resource proof'],
        actual_H4_analysis=False,scientific_calls=0)
