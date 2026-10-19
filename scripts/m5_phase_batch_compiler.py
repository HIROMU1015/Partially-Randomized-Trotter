"""Project-private opt1 phase accumulator; the installed kernel is unchanged."""
from fractions import Fraction
import hashlib
import json
import math
import struct

import symengine
from qiskit.circuit import ParameterExpression
from qiskit.passmanager import ConditionalController, DoWhileController, FlowControllerLinear
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from m5_phase_repair_prototype import _rounded_phase, payload, require

POLICY = 'M5_OPT1Q_EXACT_PHASE_BATCH_V1'


class PhaseLedger:
    def __init__(self):
        self.count = 0
        self.digest = hashlib.sha256()

    def append(self, value):
        require(type(value) is float and math.isfinite(value), 'FINITE_BINARY64_PHASE_INCREMENT')
        self.count += 1
        self.digest.update(struct.pack('>d', value))


class ExactPhase(ParameterExpression):
    """Constant rational expression consumed only inside one Rust pass call.

    The Rust Param conversion preserves ParameterExpression subclasses before
    attempting float coercion. Its addition dispatch calls this exact __add__.
    The inherited symbolic expression also contains the true rational value.
    Nothing is installed or monkeypatched in the shared runtime.
    """
    def __init__(self, value, ledger):
        require(type(value) is Fraction, 'EXACT_PHASE_FRACTION')
        self.exact_value = value
        self.ledger = ledger
        super().__init__({}, symengine.Rational(value.numerator, value.denominator))

    def __add__(self, other):
        self.ledger.append(other)
        return ExactPhase(self.exact_value + Fraction.from_float(other), self.ledger)

    def __radd__(self, other):
        return self.__add__(other)

    def __float__(self):
        raise ValueError('NO_PHASE_ROUNDING_INSIDE_REPLACEMENT_LOOP')


class PhaseBatchOptimize1qGatesDecomposition(Optimize1qGatesDecomposition):
    def __init__(self, original, recorder):
        super().__init__(basis=original._basis_gates, target=original._target)
        require(self._global_decomposers == original._global_decomposers and self._target is original._target,
                'UNCHANGED_ORIGINAL_DECOMPOSER_SETTINGS')
        self.recorder = recorder

    def run(self, dag):
        from qiskit._accelerate import euler_one_qubit_decomposer
        require(set(dag.count_ops()) <= {'rz', 'sx', 'x', 'cx', 'id', 'barrier', 'measure'},
                'PHASE_BATCH_BASIS_ONLY_NO_CONTROL_FLOW')
        initial = float(dag.global_phase)
        require(math.isfinite(initial), 'FINITE_INITIAL_PHASE')
        ledger = PhaseLedger()
        dag.global_phase = ExactPhase(Fraction.from_float(initial), ledger)
        # Same accelerated function and settings as the installed run() body.
        euler_one_qubit_decomposer.optimize_1q_gates_decomposition(
            dag, target=self._target, global_decomposers=self._global_decomposers,
            basis_gates=self._basis_gates)
        accumulated = dag.global_phase
        require(isinstance(accumulated, ExactPhase) and accumulated.ledger is ledger,
                'EXACT_ACCUMULATOR_SURVIVES_ENTIRE_KERNEL')
        final_exact = accumulated.exact_value
        rounded = _rounded_phase(final_exact)
        dag.global_phase = rounded
        require(type(dag.global_phase) is float,
                'ONLY_NUMERIC_PHASE_LEAVES_BATCH_PASS')
        self.recorder(dict(policy=POLICY, initial_phase_binary64_hex=initial.hex(),
            initial_exact=payload(Fraction.from_float(initial)), total_exact=payload(final_exact),
            phase_increment_count=ledger.count, phase_increment_sha256=ledger.digest.hexdigest(),
            assigned_phase_binary64_hex=rounded.hex(), stored_phase_binary64_hex=float(dag.global_phase).hex(),
            accelerated_kernel_unchanged=True, inside_loop_float_coercions=0,
            phase_rounding_assignments_at_boundary=1, postcompile_correction=False))
        return dag


def patched_pass_manager(*, recorder, basis_gates, optimization_level, seed_transpiler):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    require(basis_gates == ['rz', 'sx', 'x', 'cx'] and optimization_level == 1 and seed_transpiler == 17,
            'REGISTERED_PHASE_BATCH_COMPILER_PROFILE')
    manager = generate_preset_pass_manager(basis_gates=basis_gates,
        optimization_level=optimization_level, seed_transpiler=seed_transpiler)
    replacements = []

    def replace(task):
        if type(task) is Optimize1qGatesDecomposition:
            replacements.append(task)
            return PhaseBatchOptimize1qGatesDecomposition(task, recorder)
        if type(task) is DoWhileController:
            return DoWhileController([replace(t) for t in task.tasks], do_while=task.do_while)
        if type(task) is ConditionalController:
            return ConditionalController([replace(t) for t in task.tasks], condition=task.condition)
        if type(task) is FlowControllerLinear:
            return FlowControllerLinear([replace(t) for t in task.tasks])
        require(not hasattr(task, 'tasks'), 'UNRECOGNIZED_FLOW_CONTROLLER')
        return task

    manager.optimization = PassManager([replace(manager.optimization.to_flow_controller())])
    require(len(replacements) == 1, 'EXACTLY_ONE_OPT1_PASS_SITE_REPLACED')
    return manager


def compile_with_phase_batch(circuit, *, phase_journal, callback=None,
                             basis_gates, optimization_level, seed_transpiler, num_processes):
    require(num_processes == 1, 'ONE_REGISTERED_COSINE_COMPILE')
    require(not circuit.parameters and set(circuit.count_ops()) <= {'rz','sx','x','cx','id','barrier','measure'},
            'SAVED_BOUND_BASIS_INPUT_ONLY')
    ordinal = 0
    with open(phase_journal, 'x') as stream:
        def record(row):
            nonlocal ordinal
            stream.write(json.dumps(dict(row, pass_ordinal=ordinal), sort_keys=True) + '\n')
            stream.flush()
            ordinal += 1
        manager = patched_pass_manager(recorder=record, basis_gates=basis_gates,
            optimization_level=optimization_level, seed_transpiler=seed_transpiler)
        compiled = manager.run(circuit, callback=callback, num_processes=1)
    require(ordinal > 0 and not compiled.parameters, 'PHASE_BATCH_PASS_EXECUTED_AND_NUMERIC')
    return compiled
