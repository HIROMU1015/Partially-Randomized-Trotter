"""Fail-fast Stage2 validation orchestration. No work is run on import."""
from fractions import Fraction
from . import c1_stage2_native as native


def call_limits():
    # Logical units; composite checks and their internal work are documented.
    return dict(input_load=1, export_load=1, saved_operator_load=9, car_context=1,
                xdf_generator=11, model_prepare=2, gaussian_build=20,
                local_reference=40, local_build=40, local_controlled_check=40,
                ordered_reference=2, ordered_build=4, ordered_controlled_check=4,
                adjoint_build=4, adjoint_check=4, wrapper_build=8, raw_signal=8,
                raw_signal_pair_check=4, compile=12, compiled_controlled_check=4,
                compiled_signal=8, compiled_signal_pair_check=4, metrics=12)


def validate(backend, proposal, call):
    """All local gates precede any ordered full build; failures propagate.

    backend is numerical only after the approval boundary, or an artificial
    recording double in technical tests. No retries or exception suppression.
    """
    time = proposal['total_time']
    local_times = tuple(proposal['local_times'])
    full = {}
    for arm, count in [('CDF', 9), ('XDF', 11)]:
        for index in range(count):
            for tau in local_times:
                label = f'{arm}:{index}:{tau}'
                target = call('local_reference', label, lambda: backend.local_reference(arm, index, tau))
                qc = call('local_build', label, lambda: backend.local_build(arm, index, tau))
                call('local_controlled_check', label, lambda: backend.controlled_check(qc, target, label))
    for arm in ['CDF', 'XDF']:
        target = call('ordered_reference', arm, lambda: backend.ordered_reference(arm, time))
        for optimized in [False, True]:
            label = arm+('::shared' if optimized else '::literal')
            qc = call('ordered_build', label, lambda: backend.ordered_build(arm, time, optimized))
            call('ordered_controlled_check', label, lambda: backend.controlled_check(qc, target, label))
            adj = call('adjoint_build', label, lambda: backend.adjoint(qc))
            call('adjoint_check', label, lambda: backend.adjoint_check(adj, target, label))
            full[label] = (arm, optimized, qc, target)
    wrappers = {}
    for label, (arm, optimized, qc, target) in full.items():
        values = []
        for axis in ['X', 'Y']:
            key = label+'::'+axis
            w = call('wrapper_build', key, lambda: backend.wrapper(arm, qc, time, axis, optimized))
            values.append(call('raw_signal', key, lambda: backend.signal(w, axis)))
            wrappers[key] = (w, axis)
        call('raw_signal_pair_check', label, lambda: backend.signal_check(values, target, label))
    compiled_full, compiled_wrappers, metrics = {}, {}, {}
    # Compiler checks run only after all raw full/adjoint/X/Y gates pass.
    for label, (_, _, qc, target) in full.items():
        compiled = call('compile', label, lambda: backend.compile(qc))
        call('compiled_controlled_check', label, lambda: backend.controlled_check(compiled, target, 'compiled:'+label))
        compiled_full[label] = compiled
        metrics[label] = call('metrics', label, lambda: backend.metrics(compiled))
    for label, (_, _, _, target) in full.items():
        values = []
        for axis in ['X', 'Y']:
            key = label+'::'+axis
            qc, _ = wrappers[key]
            compiled = call('compile', key, lambda: backend.compile(qc))
            values.append(call('compiled_signal', key, lambda: backend.signal(compiled, axis)))
            compiled_wrappers[key] = compiled
            metrics[key] = call('metrics', key, lambda: backend.metrics(compiled))
        call('compiled_signal_pair_check', label, lambda: backend.signal_check(values, target, 'compiled:'+label))
    return dict(status='STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS', metrics=metrics,
                total_time=time, PF=proposal['product_formula'], new_fit=0,
                equal_accuracy_total_resource_claim=False, Stage3_authorized=False,
                mandatory_stop=True)


class QiskitBackend:
    def __init__(self, models, generators, proposal, *, mark, keep):
        from .pr_dictionary_validation.fixture import CompilerProfile
        self.models, self.generators, self.proposal = models, generators, proposal
        self.builder = native.NativeBuilder(call=mark)
        self.references = {}
        self.profile = CompilerProfile(**{**proposal['compiler'],
                                         'basis_gates': tuple(proposal['compiler']['basis_gates'])})
        self.profile.validate()
        self.keep = keep
        self.gates = []
        self.adjoint_count = 0
        self.compile_count = 0

    def local_reference(self, arm, index, tau):
        from scipy.linalg import expm
        value = expm(-1j*float(Fraction(tau))*self.generators[arm][index])
        self.references[arm, index, Fraction(tau)] = value
        return value

    def local_build(self, arm, index, tau):
        qc = self.builder.local(self.models[arm], index, tau)
        self.keep(f'local_{arm}_{index}_{str(tau).replace("/", "_")}', qc)
        return qc

    def controlled_check(self, qc, target, label):
        from .pr_dictionary_validation.gates import controlled_gate
        report = controlled_gate(qc, target, self.proposal['operator_atol'])
        self.gates.append(dict(id=label, **report))
        return report

    def ordered_reference(self, arm, time):
        import numpy as np
        result = np.eye(len(self.generators[arm][0]), dtype=complex)
        for index, tau in native.ordered_schedule(len(self.models[arm].blocks), time):
            result = self.references[arm, index, tau] @ result
        return result

    def ordered_build(self, arm, time, optimized):
        qc = self.builder.ordered(self.models[arm], time, optimized=optimized)
        self.keep(f'full_{arm}_{optimized}', qc)
        return qc

    def adjoint(self, qc):
        result = native.actual_adjoint(qc)
        self.keep(f'adjoint_{self.adjoint_count}', result)
        self.adjoint_count += 1
        return result

    def adjoint_check(self, qc, target, label):
        return self.controlled_check(qc, target.conj().T, 'adjoint:'+label)

    def wrapper(self, arm, qc, time, axis, optimized):
        result = native.wrapper(qc, self.models[arm], time, axis, optimized=optimized,
                                occupied=self.proposal['occupied'])
        self.keep(f'wrapper_{arm}_{optimized}_{axis}', result)
        return result

    def signal(self, qc, axis):
        from .pr_dictionary_validation.gates import wrapper_signal
        return wrapper_signal(qc, axis, measured=True)

    def signal_check(self, values, target, label):
        from .pr_dictionary_validation.gates import signal_gate
        index = sum(1 << p for p in self.proposal['occupied'])
        report = signal_gate(*values, target[index, index], self.proposal['signal_atol'])
        self.gates.append(dict(id='XY_pair:'+label, **report))
        return report

    def compile(self, qc):
        from .pr_dictionary_validation.gates import compiled_wrapper
        result = compiled_wrapper(qc, self.profile)
        self.keep(f'compiled_{self.compile_count}', result)
        self.compile_count += 1
        return result

    def metrics(self, qc):
        from .pr_dictionary_validation.f2_execution import circuit_metrics
        return circuit_metrics(qc)
