"""Reuse native DF circuit pieces and the canonical bytes shared by X and Y."""

from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import resource

from qiskit import ClassicalRegister, QuantumCircuit, qpy
from qiskit.circuit import Gate
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.rpe_hadamard_interrogation import QiskitRPEHadamardInterrogationBuilder
from trotterlib.rte import CompilerSettings
from trotterlib.rte_compiled_cost import _numeric_parameter_payload, transpile_and_measure_cost
from trottertracks.resource_applicability import ax2b_native_df_v5 as native
from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as canonical


@cache
def fast_fingerprint_module():
    source = Path(__file__).resolve().parent / "optimizer-source/scripts/h46_b3_fast_fingerprint.py"
    spec = spec_from_file_location("h46_b3_fast_fingerprint", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint_context():
    return fast_fingerprint_module().use_fast_fingerprint


def hadamard_wrapper(evolution, axis, *, shell=False):
    if axis not in ("cosine", "sine"):
        raise ValueError("Expected a cosine or sine wrapper")
    ancilla = evolution.num_qubits - 1
    wrapper = QiskitRPEHadamardInterrogationBuilder._new_wrapper_circuit(evolution)
    wrapper.h(ancilla)
    if shell:
        wrapper.global_phase += evolution.global_phase
    else:
        wrapper.compose(evolution, inplace=True)
    if axis == "sine":
        wrapper.sdg(ancilla)
    wrapper.h(ancilla)
    wrapper.add_register(ClassicalRegister(1, "rpe_measure"))
    wrapper.measure(ancilla, 0)
    return wrapper


class SharedWriter(fast_fingerprint_module().FastWriter):
    def header(self, circuit):
        self.sink.write(b'{"fingerprint_policy":"qiskit_recursive_numeric_circuit_v2","global_phase":')
        self.scalar(_numeric_parameter_payload(circuit.global_phase))
        self.sink.write(b',"instructions":[')

    def item(self, circuit, item, qubits, clbits, *, comma):
        self.sink.write((b',' if comma else b'') + b'{"clbits":')
        self.sink.write(clbits(item.clbits))
        self.sink.write(b',"operation":')
        self.operation(item.operation, circuit)
        self.sink.write(b',"qubits":')
        self.sink.write(qubits(item.qubits))
        self.sink.write(b'}')

    def trailer(self, circuit):
        self.sink.write(b'],"num_clbits":')
        self.scalar(circuit.num_clbits)
        self.sink.write(b',"num_qubits":')
        self.scalar(circuit.num_qubits)
        self.sink.write(b'}')


class SharedBody:
    """Buffer one canonical instruction stream before feeding both SHA states."""

    def __init__(self, evolution, wrapper):
        self.sink = canonical._HashSink()
        self.sink.hasher = self
        self.evolution = evolution
        self.wrapper = wrapper

    def update(self, chunk):
        self.evolution.hasher.update(chunk)
        self.wrapper.hasher.update(chunk)


def encoded_indices(bits):
    positions = {bit: index for index, bit in enumerate(bits)}

    @cache
    def encode(operands):
        return canonical._ENCODER.encode([positions[bit] for bit in operands]).encode()

    return encode


def shared_fingerprints(evolution, axes=("cosine", "sine")):
    """Fingerprint the native circuit and measured wrappers from one ordered walk."""
    if (evolution.num_clbits or evolution.parameters or evolution.layout is not None
            or evolution.calibrations):
        raise ValueError("Expected an unlaid-out numeric evolution with no classical bits")
    shells = {axis: hadamard_wrapper(evolution, axis, shell=True) for axis in axes}
    prototype = next(iter(shells.values()))
    if any(shell.qubits != evolution.qubits or shell.global_phase.hex() != prototype.global_phase.hex()
           for shell in shells.values()):
        raise ValueError("Inconsistent Hadamard wrapper qubit order or phase")

    ev_sink, wrapper_sink = canonical._HashSink(), canonical._HashSink()
    writer = SharedWriter(ev_sink)
    writer.header(evolution)
    writer.sink = wrapper_sink
    writer.header(prototype)
    qubits = encoded_indices(evolution.qubits)
    clbits = encoded_indices(prototype.clbits)
    writer.item(prototype, prototype.data[0], qubits, clbits, comma=False)
    ev_sink.flush()
    wrapper_sink.flush()
    writer.sink = SharedBody(ev_sink, wrapper_sink).sink
    for index, item in enumerate(evolution.data):
        if not isinstance(item.operation, Gate):
            raise ValueError("Native evolution must contain unitary gates only")
        if index == 0:
            wrapper_sink.hasher.update(b',')
        writer.item(evolution, item, qubits, clbits, comma=index != 0)
    writer.sink.flush()
    writer.sink = ev_sink
    writer.trailer(evolution)
    ev_sink.flush()
    digests = {}
    for axis, shell in shells.items():
        sink = canonical._HashSink()
        sink.hasher = wrapper_sink.hasher.copy()
        writer.sink = sink
        clbits = encoded_indices(shell.clbits)
        for item in shell.data[1:]:
            writer.item(shell, item, qubits, clbits, comma=True)
        writer.trailer(shell)
        sink.flush()
        digests[axis] = sink.hasher.hexdigest()
    return ev_sink.hasher.hexdigest(), digests


class PriceFingerprints:
    """Share precompile hashes only within one evolution and its forked axes."""

    def __init__(self, session, axes=("cosine", "sine")):
        self.session = session
        self.requested_axes = axes

    def prepare(self, circuit):
        self.evolution, self.axes = shared_fingerprints(circuit, self.requested_axes)
        self.circuit = circuit
        return self.evolution


def acquire_axis(axis, output, evolution, fingerprints):
    from run_full_m7 import COMPILER_HASH, digest, require, write

    require(evolution.circuit is fingerprints.circuit and evolution.circuit_fingerprint == fingerprints.evolution,
            "Price worker received a different native evolution")
    wrapper = hadamard_wrapper(evolution.circuit, axis)
    require(len(wrapper.data) == len(evolution.circuit.data) + 3 + (axis == "sine"),
            "Hadamard wrapper instruction count")
    fingerprint = fingerprints.axes[axis]
    compiler = CompilerSettings(("rz", "sx", "x", "cx"), None, None, 1, None, None, 17, "1.3.0")
    write(output / f"{axis}_compile_started.json", {"fingerprint": fingerprint, "compiler": COMPILER_HASH})
    print(f"COMPILE_START {axis} {fingerprint}", flush=True)
    cost = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fingerprint)
    require(cost.compiler_settings_hash == COMPILER_HASH, "Compiled compiler hash")
    require(fingerprints.session.fingerprint(wrapper) == fingerprint,
            "Hadamard input differs from its precompile fingerprint")
    metrics = {name: getattr(cost, name) for name in (
        "rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size",
    )}
    receipt = {"metrics": metrics, "fingerprint": fingerprint, "compiler": COMPILER_HASH}
    write(output / f"{axis}_compiled_metrics.json", receipt)
    circuit_file = output / f"{axis}_compiled.qpy"
    temporary = circuit_file.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        qpy.dump(cost.transpiled_circuit, stream)
    temporary.replace(circuit_file)
    receipt.update(cache_hit=False, qpy=circuit_file.name, qpy_sha256=digest(circuit_file),
                   peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write(output / f"{axis}_price.json", receipt)
    print(f"COMPILE_DONE {axis} RZ={metrics['rz_count']}", flush=True)
    return receipt


class CircuitPieces:
    """Share pieces for one fixed preparation and the symmetric directional control."""

    def __init__(self, preparation):
        self.preparation = preparation
        self.n = native.validate_blocks(preparation.deterministic_blocks, preparation.num_system_qubits)
        self.blocks = preparation.deterministic_blocks
        self.bound = 1 + sum(
            native.block_instruction_bound(block, mode)
            for block in self.blocks for mode in ("UNCONTROLLED", "DIRECTIONAL")
        )
        self.deterministic = {}
        self.flanks = {}

    def schedule(self, candidate, core, *, random_tail):
        return core.schedule(
            candidate["formula"], candidate["q"], "ABS_TIME_PER_OUTER" if random_tail else "PER_PIECE",
            candidate["R_outer"] if random_tail else 1, T=candidate["T"],
        )

    def deterministic_piece(self, time):
        key = time.hex()
        if key not in self.deterministic:
            piece = native.build_deterministic_native(
                self.blocks, num_system_qubits=self.n, T=time, q=1, scalar=0.0, formula="2nd",
                control_policy="symmetric_directional", max_instructions=self.bound,
            )
            self.deterministic[key] = piece.circuit
        return self.deterministic[key]

    def pr_flanks(self, time):
        key = time.hex()
        if key not in self.flanks:
            forward = QuantumCircuit(self.n + 1)
            reverse = QuantumCircuit(self.n + 1)
            for block in self.blocks:
                native._append_block(forward, block, time / 2, "UNCONTROLLED", self.n)
            for block in reversed(self.blocks):
                native._append_block(reverse, block, time / 2, "DIRECTIONAL", self.n)
            self.flanks[key] = forward, reverse
        return self.flanks[key]

    def prepare_pr_flanks(self, candidate, core):
        for step in self.schedule(candidate, core, random_tail=True):
            self.pr_flanks(step["signed_time"])

    def assemble(self, candidate, core, requests, scalar, fingerprint):
        steps = self.schedule(candidate, core, random_tail=bool(requests))
        circuit = QuantumCircuit(self.n + 1)
        bound = 1 + len(steps) * self.bound
        if requests:
            if len(steps) != len(requests):
                raise ValueError("SIGNED_REQUEST_COUNT")
            builder = QiskitDFRTEEventCircuitBuilder(
                basis_registry=self.preparation.rte_preparation.basis_registry,
            )
            for step, request in zip(steps, requests, strict=True):
                time = step["signed_time"]
                if (request.step_time != time or
                        request.preparation.preparation_hash != self.preparation.preparation_hash):
                    raise ValueError("SIGNED_NATIVE_IDENTITY")
                tail = builder.build_sequence(request.rte_occurrence, basis_plan=request.rte_basis_plan).circuit
                native._gate_only(tail, self.n + 1)
                forward, reverse = self.pr_flanks(time)
                piece = forward.copy()
                piece.compose(tail, inplace=True)
                piece.compose(reverse, inplace=True)
                bound += len(tail.data)
                circuit.compose(piece, inplace=True)
        else:
            # Flatten one period for gate emission. Replay global phase in
            # the original piece order to keep Qiskit's binary64 accumulation.
            count = len(steps) // candidate["q"]
            pieces = [self.deterministic_piece(step["signed_time"]) for step in steps[:count]]
            period = QuantumCircuit(self.n + 1)
            for piece in pieces:
                period.compose(piece, inplace=True)
            period.global_phase = 0.0
            if candidate["q"] == 1:
                circuit = period
            else:
                for _ in range(candidate["q"]):
                    circuit.compose(period, inplace=True)
            phase = QuantumCircuit(1)
            for _ in range(candidate["q"]):
                for piece in pieces:
                    phase.global_phase += piece.global_phase
            circuit.global_phase = phase.global_phase
        if scalar * candidate["T"] != 0:
            circuit.p(-scalar * candidate["T"], self.n)
        if len(circuit.data) > bound:
            raise RuntimeError("POST_BUILD_INSTRUCTION_BOUND")
        return native.NativeEvolution(
            circuit, self.n, candidate["T"], candidate["q"], candidate["formula"],
            "symmetric_directional", bound, fingerprint(circuit),
        )
