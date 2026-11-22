"""Share the canonical precompile circuit walk in the fixed PF price worker."""
from contextlib import contextmanager
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


def shared_fingerprints(evolution, axes):
    """Hash the native evolution and measured wrappers in one gate walk.

    The byte order follows H6's verified shared writer. Use the fixed worker's
    own native/canonical modules and fast writer after its provider install.
    """
    from h46_b3_fast_fingerprint import FastWriter
    from qiskit import ClassicalRegister
    from qiskit.circuit import Gate
    from trotterlib.rpe_hadamard_interrogation import QiskitRPEHadamardInterrogationBuilder
    from trotterlib.rte_compiled_cost import _numeric_parameter_payload
    from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as canonical

    if (evolution.num_clbits or evolution.parameters or evolution.layout is not None
            or evolution.calibrations or not axes or not set(axes) <= {'cosine', 'sine'}):
        raise ValueError('Expected fixed numeric evolution and measured axes')

    def shell(axis):
        ancilla = evolution.num_qubits - 1
        result = QiskitRPEHadamardInterrogationBuilder._new_wrapper_circuit(evolution)
        result.h(ancilla)
        result.global_phase += evolution.global_phase
        if axis == 'sine':
            result.sdg(ancilla)
        result.h(ancilla)
        result.add_register(ClassicalRegister(1, 'rpe_measure'))
        result.measure(ancilla, 0)
        return result

    def encode_positions(bits):
        order = {bit: index for index, bit in enumerate(bits)}

        @cache
        def encode(operands):
            return canonical._ENCODER.encode([order[bit] for bit in operands]).encode()
        return encode

    def header(circuit):
        writer.sink.write(b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":')
        writer.scalar(_numeric_parameter_payload(circuit.global_phase))
        writer.sink.write(b',"instructions":[')

    def instruction(circuit, item, clbits, *, comma):
        writer.sink.write((b',' if comma else b'') + b'{"clbits":' + clbits(item.clbits) + b',"operation":')
        writer.operation(item.operation, circuit)
        writer.sink.write(b',"qubits":' + qubits(item.qubits) + b'}')

    def trailer(circuit):
        writer.sink.write(b'],"num_clbits":')
        writer.scalar(circuit.num_clbits)
        writer.sink.write(b',"num_qubits":')
        writer.scalar(circuit.num_qubits)
        writer.sink.write(b'}')

    shells = {axis: shell(axis) for axis in axes}
    prototype = next(iter(shells.values()))
    if any(value.qubits != evolution.qubits or value.global_phase.hex() != prototype.global_phase.hex()
           for value in shells.values()):
        raise ValueError('Inconsistent measured wrapper headers')
    native, wrapped = canonical._HashSink(), canonical._HashSink()
    writer = FastWriter(native)
    qubits = encode_positions(evolution.qubits)
    clbits = encode_positions(prototype.clbits)
    header(evolution)
    writer.sink = wrapped
    header(prototype)
    instruction(prototype, prototype.data[0], clbits, comma=False)
    native.flush()
    wrapped.flush()

    class SharedBody:
        def update(self, chunk):
            native.hasher.update(chunk)
            wrapped.hasher.update(chunk)

    writer.sink = canonical._HashSink()
    writer.sink.hasher = SharedBody()
    for index, item in enumerate(evolution.data):
        if not isinstance(item.operation, Gate):
            raise ValueError('Expected a unitary native evolution')
        if index == 0:
            wrapped.hasher.update(b',')
        instruction(evolution, item, clbits, comma=index != 0)
    writer.sink.flush()
    writer.sink = native
    trailer(evolution)
    native.flush()

    result = {}
    for axis, circuit in shells.items():
        sink = canonical._HashSink()
        sink.hasher = wrapped.hasher.copy()
        writer.sink = sink
        clbits = encode_positions(circuit.clbits)
        for item in circuit.data[1:]:
            instruction(circuit, item, clbits, comma=True)
        trailer(circuit)
        sink.flush()
        result[axis] = sink.hasher.hexdigest()
    return native.hasher.hexdigest(), result


@contextmanager
def use_shared_fingerprints(native, axes):
    """Only substitute a wrapper's first hash, keeping postcompile checks live."""
    from h46_b3_fast_fingerprint import FingerprintSession
    from trottertracks.resource_applicability import ax2b_native_df_v5 as api

    compose = native.compose
    build = api.build_native_hadamard_wrapper
    fingerprint = FingerprintSession.fingerprint
    evolution = None
    digests = {}
    first_hashes = {}

    def compose_candidate(row, preparation, requests, scalar, hash_circuit):
        nonlocal evolution, digests
        shared = None

        def share(circuit):
            nonlocal shared
            if shared is None:
                try:
                    actual, wrappers = shared_fingerprints(circuit, axes)
                except ValueError:
                    shared = (None, {})
                else:
                    shared = circuit, wrappers
                    return actual
            return hash_circuit(circuit)

        candidate = compose(row, preparation, requests, scalar, share)
        evolution = candidate.circuit
        digests = shared[1] if shared and shared[0] is evolution else {}
        return candidate

    def build_wrapper(candidate, *, axis, include_measurement=True, max_instructions):
        # The original builder still checks the native fingerprint, bounds,
        # operation types, phase and measurement convention.
        circuit = build(candidate, axis=axis, include_measurement=include_measurement,
                        max_instructions=max_instructions)
        if include_measurement and candidate.circuit is evolution and axis in digests:
            first_hashes[id(circuit)] = circuit, digests[axis]
        return circuit

    def hash_once(session, circuit):
        saved = first_hashes.pop(id(circuit), None)
        if saved is not None and saved[0] is circuit:
            return saved[1]
        return fingerprint(session, circuit)

    with (patch.object(native, 'compose', compose_candidate),
          patch.object(api, 'build_native_hadamard_wrapper', build_wrapper),
          patch.object(FingerprintSession, 'fingerprint', hash_once)):
        yield


@contextmanager
def pf_price_worker(caller):
    """Install inside the fixed caller's forked price worker, after native setup."""
    worker = caller.worker

    def accelerated_worker(cell, output, progress):
        import pf_native_price_fast

        accelerated_native = pf_native_price_fast.accelerated_native
        axes = tuple({'X': 'cosine', 'Y': 'sine'}[axis] for axis in cell.get('axes', ('X', 'Y')))

        @contextmanager
        def accelerated(native):
            with accelerated_native(native), use_shared_fingerprints(native, axes):
                yield

        with patch.object(pf_native_price_fast, 'accelerated_native', accelerated):
            return worker(cell, output, progress)

    with patch.object(caller, 'worker', accelerated_worker):
        yield


def main():
    entry = Path(__file__).resolve()
    spec = spec_from_file_location('_rpe_fixed_pf_price_entry', entry.with_name('run_pf_preparation_stage.py'))
    stage = module_from_spec(spec)
    spec.loader.exec_module(stage)
    load_provider = stage.load_provider

    def load_accelerated_provider(root):
        provider = load_provider(root)
        native_child = provider.native_child

        def child(path):
            import run_pf_m0_prices as caller

            with pf_price_worker(caller):
                return native_child(path)

        provider.native_child = child
        return provider

    # The existing stage forks a new Python interpreter for the native child.
    # Keep that entry on the same request-bound acceleration file.
    stage.__file__ = str(entry)
    stage.load_provider = load_accelerated_provider
    return stage.main()


if __name__ == '__main__':
    raise SystemExit(main())
