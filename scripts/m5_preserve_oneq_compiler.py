"""Conservative control: retain 1q runs, keep all other preset opt1 tasks.

No Euler generation, angle simplification, scalar accumulation or shared runtime
patch occurs in the replaced pass. This is a distinct compiler policy, not a
claim that small-angle simplification is the sole source of the H6 residual.
"""
import json
import math
from qiskit.passmanager import ConditionalController, DoWhileController, FlowControllerLinear
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from m5_phase_repair_prototype import require

POLICY = 'M5_OPT1_PRESERVE_ONEQ_RUNS_CONTROL_V1'
BASIS = {'rz','sx','x','cx','id','barrier','measure'}


class PreserveOneQubitRuns(TransformationPass):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def run(self, dag):
        counts = dict(dag.count_ops())
        phase = float(dag.global_phase)
        require(set(counts) <= BASIS and math.isfinite(phase), 'BOUND_BASIS_DAG_REQUIRED')
        # Deliberately return the same unmodified DAG. No gate or angle loop.
        self.recorder(dict(policy=POLICY, gate_counts=counts,
            global_phase_binary64_hex=phase.hex(), retained_same_DAG=True,
            oneq_resynthesis_executed=False, small_angle_simplification_executed=False,
            scalar_phase_generated=False, postcompile_correction=False))
        return dag


def patched_pass_manager(*, recorder, basis_gates, optimization_level, seed_transpiler):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    require(basis_gates == ['rz','sx','x','cx'] and optimization_level == 1 and seed_transpiler == 17,
            'REGISTERED_PRESERVE_ONEQ_COMPILER_PROFILE')
    manager = generate_preset_pass_manager(basis_gates=basis_gates,
        optimization_level=optimization_level, seed_transpiler=seed_transpiler)
    replaced = []

    def replace(task):
        if type(task) is Optimize1qGatesDecomposition:
            replaced.append(task)
            return PreserveOneQubitRuns(recorder)
        if type(task) is DoWhileController:
            return DoWhileController([replace(t) for t in task.tasks], do_while=task.do_while)
        if type(task) is ConditionalController:
            return ConditionalController([replace(t) for t in task.tasks], condition=task.condition)
        if type(task) is FlowControllerLinear:
            return FlowControllerLinear([replace(t) for t in task.tasks])
        require(not hasattr(task, 'tasks'), 'UNRECOGNIZED_FLOW_CONTROLLER')
        return task

    manager.optimization = PassManager([replace(manager.optimization.to_flow_controller())])
    require(len(replaced) == 1, 'EXACTLY_ONE_ONEQ_PASS_SITE_REPLACED')
    return manager


def compile_with_preserve_oneq(circuit, *, phase_journal, callback=None,
                              basis_gates, optimization_level, seed_transpiler, num_processes):
    require(num_processes == 1 and not circuit.parameters and set(circuit.count_ops()) <= BASIS,
            'ONE_BOUND_BASIS_COSINE_INPUT')
    ordinal = 0
    with open(phase_journal, 'x') as stream:
        def record(row):
            nonlocal ordinal
            stream.write(json.dumps(dict(row, pass_ordinal=ordinal), sort_keys=True)+'\n')
            stream.flush()
            ordinal += 1
        manager = patched_pass_manager(recorder=record, basis_gates=basis_gates,
            optimization_level=optimization_level, seed_transpiler=seed_transpiler)
        compiled = manager.run(circuit, callback=callback, num_processes=1)
    require(ordinal > 0 and not compiled.parameters, 'PRESERVATION_PASS_EXECUTED')
    return compiled
