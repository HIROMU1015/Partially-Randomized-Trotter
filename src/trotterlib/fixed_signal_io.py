"""JSON input and provenance for the artificial fixed-signal technical runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .pauli_signal_adapter import PauliHamiltonian


def load_artificial_input(path, representation):
    """No loaders capable of running SCF/DF or fetching molecular data."""
    data = json.loads(Path(path).read_text())
    if data.get("input_kind") != "artificial":
        raise ValueError("This technical runner requires input_kind=artificial; molecular execution is not connected")
    if data.get("state_order") != "qiskit_little_endian":
        raise ValueError("Explicit state_order=qiskit_little_endian required")
    state = np.asarray([complex(*pair) for pair in data["state"]], dtype=complex)
    model = data["hamiltonian"]
    metadata = dict(data.get("generation", {}), input_kind="artificial")
    if representation == "pauli_jw":
        if model["format"] == "pauli_terms":
            h = PauliHamiltonian.from_terms(model["terms"], num_qubits=model["num_qubits"],
                                             model_id=data["model_id"], metadata=metadata)
        elif model["format"] == "spatial_mo_integrals":
            h = PauliHamiltonian.from_spatial_integrals(model["one_body"], model["eri_chemist"],
                nuclear_repulsion=model["nuclear_repulsion"], model_id=data["model_id"], metadata=metadata)
        else:
            raise ValueError("Unsupported Pauli input format")
    elif representation == "df_native":
        from .df_hamiltonian import DFHamiltonian
        if model["format"] != "df_native":
            raise ValueError("df_native requires supplied DF arrays")
        h = DFHamiltonian(float(model["constant"]), np.asarray(model["one_body"], dtype=complex),
                          np.asarray(model["lambdas"], dtype=float),
                          tuple(np.asarray(g, dtype=complex) for g in model["g_matrices"]),
                          dict(metadata, model_id=data["model_id"]))
    else:
        raise ValueError("Unknown Hamiltonian representation")
    return h, state, dict(input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          model_id=data["model_id"], state_generation=data.get("state_generation", {}))


def source_provenance(root):
    root = Path(root)
    paths = ["src/trotterlib/fixed_signal_accounting.py", "src/trotterlib/pauli_signal_adapter.py",
             "src/trotterlib/fixed_signal_evaluation.py", "src/trotterlib/fixed_signal_io.py",
             "scripts/run_fixed_signal_evaluation.py", "src/trotterlib/rte.py",
             "src/trotterlib/df_partial_s2.py", "src/trotterlib/df_partial_s2_repeated.py",
             "src/trotterlib/df_rte_qiskit.py", "src/trotterlib/rte_compiled_cost.py"]
    paths = sorted(set(paths) | {str(p.relative_to(root)) for p in (root/"src/trotterlib").rglob("*.py")})
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                base_commit="b81d4096e806efb6523304764d3a0680b7663320",
                dirty=bool(git("status", "--porcelain")),
                files={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
