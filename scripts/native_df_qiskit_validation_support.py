"""Local artificial-only connection support; never load molecular artifacts.

The reference provider is loaded from exact Git bytes in a temporary directory.
No alternate SOURCE is copied into src or substituted for the base library.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib
import itertools
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types

BASE = "1ea42fbf1bd1c8341ffd410b8067467b4ba4e170"
REFERENCE = "87b6b9f46b7e8ba9abe132ee21478c114b6f46f7"
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATHS = (
    "src/trottertracks/resource_applicability/ax2b_native_df_v5.py",
    "src/trottertracks/resource_applicability/ax2a_control_plan.py",
    "src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py",
    "src/trottertracks/resource_applicability/ax2b_diagnostics_v4.py",
)
_TEMP = None
PROVENANCE = []
OBSERVATIONS = []


def git_bytes(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + path])


def package(name, directory):
    """A namespace shell bypasses unrelated __init__ exports, not actual modules."""
    if name in sys.modules:
        raise RuntimeError("Validation requires a clean interpreter namespace: " + name)
    mod = types.ModuleType(name)
    mod.__path__ = [str(directory)]
    sys.modules[name] = mod


def bootstrap():
    global _TEMP
    package("trotterlib", ROOT / "src/trotterlib")
    package("trotterlib.df_trotter", ROOT / "src/trotterlib/df_trotter")
    _TEMP = tempfile.TemporaryDirectory(prefix="native-df-qiskit-reference-")
    snapshot = Path(_TEMP.name)
    package("native_validation_reference", snapshot)
    for path in REFERENCE_PATHS:
        data = git_bytes(REFERENCE, path)
        destination = snapshot / Path(path).name
        destination.write_bytes(data)
        PROVENANCE.append({"role": "READ_ONLY_REFERENCE", "commit": REFERENCE, "path": path,
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    native = importlib.import_module("native_validation_reference.ax2b_native_df_v5")
    # Every common library .py except the new counter is byte-identical at both
    # fixed commits; assert this for the actually imported closure in receipt().
    return native


def receipt_sources():
    sources = list(PROVENANCE)
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("trotterlib.") or not getattr(module, "__file__", None):
            continue
        p = Path(module.__file__).resolve()
        relative = p.relative_to(ROOT).as_posix()
        actual = p.read_bytes()
        if relative == "src/trotterlib/native_df_gate_model.py":
            role = "CURRENT_COUNTER_WITH_DECLARED_BUGFIX"
        else:
            if actual != git_bytes(BASE, relative) or actual != git_bytes(REFERENCE, relative):
                raise RuntimeError("Common source identity mismatch: " + relative)
            role = "COMMON_LIBRARY_IDENTICAL_AT_BASE_AND_REFERENCE"
        sources.append({"role": role, "path": relative, "bytes": len(actual),
                        "sha256": hashlib.sha256(actual).hexdigest(),
                        "base_commit": BASE, "reference_commit": REFERENCE})
    return sources


def qiskit_gate(gate):
    from qiskit.circuit import library as lib
    classes = {"rz": lib.RZGate, "p": lib.PhaseGate, "z": lib.ZGate, "h": lib.HGate,
               "sx": lib.SXGate, "sxdg": lib.SXdgGate, "x": lib.XGate, "sdg": lib.SdgGate,
               "cx": lib.CXGate, "cz": lib.CZGate, "rzz": lib.RZZGate,
               "crz": lib.CRZGate, "rxx": lib.RXXGate, "ryy": lib.RYYGate,
               "xx_plus_yy": lib.XXPlusYYGate}
    if gate.name == "crzz":
        return lib.RZZGate(gate.angle).control(1)
    constructor = classes[gate.name]
    if gate.name == "xx_plus_yy":
        return constructor(gate.angle, gate.beta)
    return constructor(gate.angle) if gate.name in ("rz", "p", "rzz", "crz", "rxx", "ryy") else constructor()


def circuit_from_ir(ir, n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for gate in ir.operations:
        qc.append(qiskit_gate(gate), list(gate.qubits))
    qc.global_phase = ir.global_phase
    return qc


def fixed_decompose(qc):
    """Independent Qiskit realization of profile explicit_rz_sx_x_cx_v1.

    Does not call model.lower/lower_gate/count and never calls transpile.
    """
    from qiskit import QuantumCircuit
    output = QuantumCircuit(qc.num_qubits)
    output.global_phase = qc.global_phase

    def emit(name, q, a=0.0, beta=0.0):
        if name in ("rz", "sx", "x", "cx"):
            getattr(output, name)(a, *q) if name == "rz" else getattr(output, name)(*q)
        elif name in ("p", "z", "sdg"):
            a = a if name == "p" else math.pi if name == "z" else -math.pi / 2
            output.rz(a, q[0]); output.global_phase += a / 2
        elif name == "h":
            output.rz(math.pi / 2, q[0]); output.sx(q[0]); output.rz(math.pi / 2, q[0])
            output.global_phase += math.pi / 4
        elif name == "sxdg":
            output.sx(q[0]); output.x(q[0])
        elif name == "rzz":
            output.cx(*q); output.rz(a, q[1]); output.cx(*q)
        elif name == "crz":
            output.rz(a / 2, q[1]); output.cx(*q); output.rz(-a / 2, q[1]); output.cx(*q)
        elif name == "crzz":
            output.cx(q[1], q[2]); emit("crz", (q[0], q[2]), a); output.cx(q[1], q[2])
        elif name == "cz":
            emit("h", (q[1],)); output.cx(*q); emit("h", (q[1],))
        elif name == "rxx":
            for j in q: emit("h", (j,))
            emit("rzz", q, a)
            for j in q: emit("h", (j,))
        elif name == "ryy":
            for j in q: output.sx(j)
            emit("rzz", q, a)
            for j in q: emit("sxdg", (j,))
        elif name == "xx_plus_yy":
            output.rz(beta, q[0]); emit("rxx", q, a / 2); emit("ryy", q, a / 2); output.rz(-beta, q[0])
        else:
            raise ValueError("Undeclared gate in fixed profile: " + name)
    for item in qc.data:
        gate = item.operation
        q = tuple(qc.find_bit(bit).index for bit in item.qubits)
        args = tuple(float(p) for p in gate.params)
        emit(gate.name, q, *(args or (0.0,)))
    return output


def qiskit_cost(qc):
    from trotterlib.native_df_gate_model import Cost
    fixed = fixed_decompose(qc)
    counts = fixed.count_ops()
    return Cost(len(qc.data), *(int(counts.get(k, 0)) for k in ("rz", "cx", "sx", "x")))


def word(qc):
    return [(item.operation.name, tuple(qc.find_bit(b).index for b in item.qubits),
             tuple(float(p) for p in item.operation.params)) for item in qc.data]


def artificial_inputs():
    """Two system modes; analytic Gaussian matrix, no molecule/SCF/DF solver."""
    import numpy as np
    from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw_givens
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, DFDeterministicFragmentSpec
    from trotterlib.df_rte_tail import DFBasisRegistry
    from trotterlib.native_df_gate_model import basis_from_runtime
    t = .37
    rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], complex)
    u = np.diag(np.exp(1j * np.array([.23, -.41]))) @ rotation
    v = np.diag(np.exp(1j * np.array([-.17, .31]))) @ rotation.T
    registry = DFBasisRegistry()
    definitions = tuple(registry.register(U_to_qiskit_ops_jw_givens(matrix),
                          num_system_qubits=2, basis_id=key) for matrix, key in ((u, "artificial-a"), (v, "artificial-b")))
    a, b = definitions
    blocks = (
        DFDeterministicOneBodySpec("artificial-onebody", "one_body", None, None, a.basis_id,
          a.basis_hash, a.metadata.operations, (.7, -.2), 2, 0, a.runtime_operations),
        DFDeterministicFragmentSpec("artificial-squared", "df_fragment", 0, 1, b.basis_id,
          b.basis_hash, b.metadata.operations, (.4, -.9), -.6, 2, 1, b.runtime_operations),
    )
    bases = tuple(basis_from_runtime(d.basis_id, d.basis_hash, d.runtime_operations) for d in definitions)
    return registry, definitions, blocks, bases, (u, v)


def rte_inputs(same_basis=False, identity=False):
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEComponentCircuitSpec, DFRTEIdentityCircuitSpec
    from trotterlib.native_df_gate_model import Component
    registry, definitions, _, bases, _ = artificial_inputs()
    a, b = definitions
    if same_basis:
        b = a
    components, specs, theory = [], [], []
    entries = [("Zplus", a, (0,), 1), ("ZZminus", b, (0, 1), -1)]
    if identity:
        entries.append(("Iminus", None, (), -1))
    probability = Fraction(1, len(entries))
    for key, d, support, sign in entries:
        components.append(rte.RTEComponent(key, float(probability), 1.0, sign,
            df_fragment_id=None if d is None else key, basis_id=None if d is None else d.basis_id,
            basis_hash=None if d is None else d.basis_hash, is_identity=d is None,
            diagonal_pauli_support=support, basis_change_operations=() if d is None else d.metadata.operations))
        if d is None:
            specs.append(DFRTEIdentityCircuitSpec(key, 1.0, sign, 2))
        else:
            specs.append(DFRTEComponentCircuitSpec(key, 1.0, sign, key, d.basis_id, support,
                d.metadata.operations, 2, d.basis_hash))
        basis = None if d is None else bases[0 if d == a else 1]
        theory.append(Component(key, probability, basis, support, sign))
    return registry, tuple(components), tuple(specs), tuple(theory)


def build_events(components, tau=.4, maximum_order=4):
    from trotterlib import rte
    distribution = rte.finite_rte_distribution(tau, maximum_order)
    # Full artificial finite enumeration, no PRNG or sampling entry point.
    return distribution, tuple(rte.iter_rte_events(components, distribution, max_events=1024))


def build_rte(registry, specs, events, reuse):
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    request = DFRTEEventSequenceCircuitRequest(events=tuple(events), component_specs=specs,
             controlled=True, ancilla_qubit=2, cancel_adjacent_equal_bases=reuse)
    return QiskitDFRTEEventCircuitBuilder(basis_registry=registry).build_sequence(request)


def normalized_fractions(values):
    values = tuple(Fraction(float(v)) for v in values)
    denominator = sum(values)
    return tuple(v / denominator for v in values)


def exterior_fock_lift(u):
    """Independent minors oracle, Qiskit integer basis: occupation bit p = mode p."""
    import numpy as np
    n = len(u)
    out = np.zeros((1 << n, 1 << n), complex)
    for i, j in itertools.product(range(1 << n), repeat=2):
        rows = [p for p in range(n) if i & (1 << p)]
        cols = [p for p in range(n) if j & (1 << p)]
        if len(rows) == len(cols):
            out[i, j] = np.linalg.det(u[np.ix_(rows, cols)]) if rows else 1
    return out
