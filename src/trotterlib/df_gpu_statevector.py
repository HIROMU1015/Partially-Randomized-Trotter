from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


def _collect_cuda_lib_dirs() -> list[str]:
    cuda_dirs: list[str] = []
    site_packages_roots = [
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
        os.path.join(sys.prefix, "lib64", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
    ]
    rel_paths = [
        os.path.join("nvidia", "nvjitlink", "lib"),
        os.path.join("nvidia", "cusparse", "lib"),
        os.path.join("nvidia", "cusolver", "lib"),
        os.path.join("nvidia", "cublas", "lib"),
        os.path.join("nvidia", "cuda_nvrtc", "lib"),
        os.path.join("nvidia", "cuda_runtime", "lib"),
        os.path.join("cuquantum", "lib"),
        os.path.join("cutensor", "lib"),
    ]
    for site_packages in site_packages_roots:
        for rel_path in rel_paths:
            lib_dir = os.path.join(site_packages, rel_path)
            if os.path.isdir(lib_dir):
                cuda_dirs.append(lib_dir)
    return cuda_dirs


def _prepend_library_path(lib_dirs: Sequence[str]) -> None:
    current_dirs = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged: list[str] = []
    for path in [*lib_dirs, *current_dirs]:
        if path and path not in merged:
            merged.append(path)
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


_prepend_library_path(_collect_cuda_lib_dirs())

try:
    from qiskit_aer import AerSimulator
except Exception as exc:  # pragma: no cover - depends on GPU server env
    AerSimulator = None
    AER_IMPORT_ERROR = exc
else:
    AER_IMPORT_ERROR = None


_AER_SIMULATOR_CACHE: dict[tuple[str, ...], object] = {}
_AER_GPU_SIMULATOR_KWARGS = {
    "method": "statevector",
    "device": "GPU",
    # Keep the same numerical policy as the reference DF project.
    "fusion_enable": False,
}


@dataclass(frozen=True)
class DFGPUParameterizedTemplate:
    """Pre-transpiled Aer GPU circuit with one symbolic time parameter."""

    circuit: QuantumCircuit
    time_parameter_name: str
    num_qubits: int
    optimization_level: int
    input_num_instructions: int
    full_num_instructions: int
    transpiled_num_instructions: int
    prepare_profile: dict[str, object]


def _contains_instruction_name(circuit: QuantumCircuit, name: str) -> bool:
    return any(getattr(inst.operation, "name", "") == name for inst in circuit.data)


def _get_cached_aer_simulator(gpu_ids: Sequence[str]) -> object:
    visible_devices = tuple(str(g) for g in gpu_ids if str(g) != "")
    simulator = _AER_SIMULATOR_CACHE.get(visible_devices)
    if simulator is None:
        simulator = AerSimulator(**_AER_GPU_SIMULATOR_KWARGS)
        _AER_SIMULATOR_CACHE[visible_devices] = simulator
    return simulator


def _transpile_for_aer_gpu(
    circuit: QuantumCircuit,
    simulator: object,
    *,
    optimization_level: int,
) -> QuantumCircuit:
    prepared = circuit
    for _ in range(3):
        if not (
            _contains_instruction_name(prepared, "rzz")
            or _contains_instruction_name(prepared, "xx_plus_yy")
        ):
            break
        prepared = prepared.decompose(reps=1)

    transpiled = transpile(
        prepared,
        simulator,
        optimization_level=int(optimization_level),
    )
    if _contains_instruction_name(transpiled, "rzz") or _contains_instruction_name(
        transpiled, "xx_plus_yy"
    ):
        raise ValueError(
            "GPU transpilation left unsupported instructions in the circuit; "
            "'rzz' and 'xx_plus_yy' must be decomposed before simulator.run()."
        )
    return transpiled


def _find_template_parameter(
    circuit: QuantumCircuit,
    parameter_name: str,
) -> Parameter:
    matches = [
        parameter
        for parameter in circuit.parameters
        if getattr(parameter, "name", None) == parameter_name
    ]
    if len(matches) != 1:
        raise ValueError(
            "Parameterized GPU template must contain exactly one matching "
            f"'{parameter_name}' parameter; found {len(matches)}."
        )
    return matches[0]


def _bind_global_phase(
    circuit: QuantumCircuit,
    parameter: Parameter,
    value: float,
) -> float:
    raw_phase = getattr(circuit, "global_phase", 0.0) or 0.0
    if hasattr(raw_phase, "bind"):
        raw_phase = raw_phase.bind({parameter: float(value)})
    elif hasattr(raw_phase, "assign"):
        raw_phase = raw_phase.assign(parameter, float(value))
    phase = float(raw_phase or 0.0)
    circuit.global_phase = phase
    return phase


def build_parameterized_gpu_template(
    qc: QuantumCircuit,
    psi0: np.ndarray,
    *,
    time_parameter_name: str,
    gpu_ids: Sequence[str] = ("0",),
    optimization_level: int = 0,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
    debug_label: str = "",
) -> DFGPUParameterizedTemplate:
    """Build and transpile a one-parameter statevector circuit for repeated GPU runs."""
    started = perf_counter()
    profile: dict[str, object] = {
        "backend": "aer_statevector_gpu",
        "execution_strategy": "parameterized_pretranspiled_template",
        "num_qubits": int(qc.num_qubits),
        "input_num_instructions": int(len(qc.data)),
        "optimization_level": int(optimization_level),
        "gpu_ids": [str(g) for g in gpu_ids if str(g) != ""],
        "aer_simulator_kwargs": dict(_AER_GPU_SIMULATOR_KWARGS),
        "time_parameter_name": str(time_parameter_name),
    }
    if AerSimulator is None:
        message = "GPU execution requested, but qiskit-aer GPU backend is unavailable."
        if AER_IMPORT_ERROR is not None:
            message = f"{message} import_error={AER_IMPORT_ERROR}"
        raise RuntimeError(message)

    _find_template_parameter(qc, time_parameter_name)

    t0 = perf_counter()
    visible_devices = [str(g) for g in gpu_ids if str(g) != ""]
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    profile["set_cuda_visible_devices_s"] = perf_counter() - t0
    _emit_debug(
        enabled=debug,
        debug_print=debug_print,
        label=debug_label,
        stage="template_set_cuda_visible_devices",
        elapsed_s=float(profile["set_cuda_visible_devices_s"]),
    )

    t0 = perf_counter()
    init_state = Statevector(np.asarray(psi0, dtype=np.complex128).reshape(-1))
    full_qc = QuantumCircuit(qc.num_qubits)
    full_qc.set_statevector(init_state)
    full_qc.compose(qc, inplace=True, copy=False)
    full_qc.save_statevector()
    profile["compose_with_initial_state_s"] = perf_counter() - t0
    profile["full_num_instructions"] = int(len(full_qc.data))

    t0 = perf_counter()
    simulator_key = tuple(visible_devices)
    profile["simulator_cache_hit"] = simulator_key in _AER_SIMULATOR_CACHE
    simulator = _get_cached_aer_simulator(visible_devices)
    profile["create_simulator_s"] = perf_counter() - t0

    t0 = perf_counter()
    transpiled = _transpile_for_aer_gpu(
        full_qc,
        simulator,
        optimization_level=int(optimization_level),
    )
    profile["transpile_s"] = perf_counter() - t0
    profile["transpiled_num_instructions"] = int(len(transpiled.data))
    _find_template_parameter(transpiled, time_parameter_name)
    profile["total_s"] = perf_counter() - started
    _emit_debug(
        enabled=debug,
        debug_print=debug_print,
        label=debug_label,
        stage="template_transpile",
        elapsed_s=float(profile["transpile_s"]),
    )

    return DFGPUParameterizedTemplate(
        circuit=transpiled,
        time_parameter_name=str(time_parameter_name),
        num_qubits=int(qc.num_qubits),
        optimization_level=int(optimization_level),
        input_num_instructions=int(len(qc.data)),
        full_num_instructions=int(len(full_qc.data)),
        transpiled_num_instructions=int(len(transpiled.data)),
        prepare_profile=dict(profile),
    )


def _split_circuit(circuit: QuantumCircuit, num_splits: int) -> list[QuantumCircuit]:
    if num_splits <= 1:
        return [circuit]
    instructions = list(circuit.data)
    subcircuits: list[QuantumCircuit] = []
    for idx in range(num_splits):
        start = (len(instructions) * idx) // num_splits
        stop = (len(instructions) * (idx + 1)) // num_splits
        subcircuit = QuantumCircuit(
            *circuit.qregs,
            *circuit.cregs,
            name=f"{circuit.name or 'df'}_part{idx}",
        )
        for inst, qargs, cargs in instructions[start:stop]:
            subcircuit.append(inst, qargs, cargs)
        subcircuits.append(subcircuit)
    return subcircuits


def _emit_debug(
    *,
    enabled: bool,
    debug_print: Callable[[str], None],
    label: str,
    stage: str,
    elapsed_s: float,
) -> None:
    if enabled:
        prefix = f"{label} " if label else ""
        debug_print(f"{prefix}{stage}: {elapsed_s:.3f}s")


def _run_statevector_backend(
    qc: QuantumCircuit,
    psi0: np.ndarray,
    *,
    gpu_ids: Sequence[str],
    optimization_level: int,
    debug: bool,
    debug_print: Callable[[str], None],
    debug_label: str,
) -> tuple[np.ndarray, dict[str, object]]:
    started = perf_counter()
    profile: dict[str, object] = {
        "backend": "aer_statevector_gpu",
        "mode": "single",
        "num_qubits": int(qc.num_qubits),
        "input_num_instructions": int(len(qc.data)),
        "optimization_level": int(optimization_level),
        "gpu_ids": [str(g) for g in gpu_ids if str(g) != ""],
        "aer_simulator_kwargs": dict(_AER_GPU_SIMULATOR_KWARGS),
    }
    if AerSimulator is None:
        message = "GPU execution requested, but qiskit-aer GPU backend is unavailable."
        if AER_IMPORT_ERROR is not None:
            message = f"{message} import_error={AER_IMPORT_ERROR}"
        raise RuntimeError(message)

    t0 = perf_counter()
    visible_devices = [str(g) for g in gpu_ids if str(g) != ""]
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    profile["set_cuda_visible_devices_s"] = perf_counter() - t0
    _emit_debug(
        enabled=debug,
        debug_print=debug_print,
        label=debug_label,
        stage="set_cuda_visible_devices",
        elapsed_s=float(profile["set_cuda_visible_devices_s"]),
    )

    t0 = perf_counter()
    init_state = Statevector(np.asarray(psi0, dtype=np.complex128).reshape(-1))
    full_qc = QuantumCircuit(qc.num_qubits)
    full_qc.set_statevector(init_state)
    full_qc.compose(qc, inplace=True, copy=False)
    full_qc.save_statevector()
    profile["compose_with_initial_state_s"] = perf_counter() - t0
    profile["full_num_instructions"] = int(len(full_qc.data))

    t0 = perf_counter()
    simulator_key = tuple(visible_devices)
    profile["simulator_cache_hit"] = simulator_key in _AER_SIMULATOR_CACHE
    simulator = _get_cached_aer_simulator(visible_devices)
    profile["create_simulator_s"] = perf_counter() - t0

    t0 = perf_counter()
    circuit_for_run = _transpile_for_aer_gpu(
        full_qc,
        simulator,
        optimization_level=int(optimization_level),
    )
    profile["transpile_s"] = perf_counter() - t0
    profile["transpiled_num_instructions"] = int(len(circuit_for_run.data))
    global_phase = float(getattr(circuit_for_run, "global_phase", 0.0) or 0.0)
    profile["global_phase"] = global_phase

    t0 = perf_counter()
    result = simulator.run(circuit_for_run).result()
    profile["simulator_run_result_s"] = perf_counter() - t0

    t0 = perf_counter()
    state = np.asarray(result.get_statevector(), dtype=np.complex128)
    profile["extract_statevector_s"] = perf_counter() - t0
    if global_phase != 0.0:
        state = np.exp(1j * global_phase) * state
    profile["total_s"] = perf_counter() - started
    return state, profile


def run_parameterized_gpu_template(
    template: DFGPUParameterizedTemplate,
    *,
    time_value: float,
    gpu_ids: Sequence[str] = ("0",),
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
    debug_label: str = "",
) -> tuple[np.ndarray, dict[str, object]]:
    """Bind a pre-transpiled template's time parameter and run it on Aer GPU."""
    started = perf_counter()
    profile: dict[str, object] = {
        "backend": "aer_statevector_gpu",
        "mode": "single",
        "execution_strategy": "parameterized_pretranspiled_template",
        "num_qubits": int(template.num_qubits),
        "input_num_instructions": int(template.input_num_instructions),
        "full_num_instructions": int(template.full_num_instructions),
        "transpiled_num_instructions": int(template.transpiled_num_instructions),
        "optimization_level": int(template.optimization_level),
        "gpu_ids": [str(g) for g in gpu_ids if str(g) != ""],
        "aer_simulator_kwargs": dict(_AER_GPU_SIMULATOR_KWARGS),
        "time_parameter_name": template.time_parameter_name,
        "template_prepare_total_s": float(template.prepare_profile.get("total_s", 0.0)),
        "template_prepare_transpile_s": float(
            template.prepare_profile.get("transpile_s", 0.0)
        ),
    }
    if AerSimulator is None:
        message = "GPU execution requested, but qiskit-aer GPU backend is unavailable."
        if AER_IMPORT_ERROR is not None:
            message = f"{message} import_error={AER_IMPORT_ERROR}"
        raise RuntimeError(message)

    t0 = perf_counter()
    visible_devices = [str(g) for g in gpu_ids if str(g) != ""]
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    profile["set_cuda_visible_devices_s"] = perf_counter() - t0
    _emit_debug(
        enabled=debug,
        debug_print=debug_print,
        label=debug_label,
        stage="set_cuda_visible_devices",
        elapsed_s=float(profile["set_cuda_visible_devices_s"]),
    )

    t0 = perf_counter()
    simulator_key = tuple(visible_devices)
    profile["simulator_cache_hit"] = simulator_key in _AER_SIMULATOR_CACHE
    simulator = _get_cached_aer_simulator(visible_devices)
    profile["create_simulator_s"] = perf_counter() - t0

    t0 = perf_counter()
    time_parameter = _find_template_parameter(
        template.circuit,
        template.time_parameter_name,
    )
    circuit_for_run = template.circuit.assign_parameters(
        {time_parameter: float(time_value)},
        inplace=False,
    )
    profile["assign_parameters_s"] = perf_counter() - t0
    global_phase = _bind_global_phase(
        circuit_for_run,
        time_parameter,
        float(time_value),
    )
    profile["global_phase"] = global_phase

    t0 = perf_counter()
    result = simulator.run(circuit_for_run).result()
    profile["simulator_run_result_s"] = perf_counter() - t0

    t0 = perf_counter()
    state = np.asarray(result.get_statevector(), dtype=np.complex128)
    profile["extract_statevector_s"] = perf_counter() - t0
    if global_phase != 0.0:
        state = np.exp(1j * global_phase) * state
    profile["total_s"] = perf_counter() - started
    return state, profile


def simulate_statevector_gpu(
    qc: QuantumCircuit,
    psi0: np.ndarray,
    *,
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    debug: bool = False,
    debug_print: Callable[[str], None] = print,
    debug_label: str = "",
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a Qiskit statevector circuit on the Aer GPU backend."""
    if chunk_splits <= 1:
        return _run_statevector_backend(
            qc,
            psi0,
            gpu_ids=gpu_ids,
            optimization_level=optimization_level,
            debug=debug,
            debug_print=debug_print,
            debug_label=debug_label,
        )

    state = np.asarray(psi0, dtype=np.complex128).reshape(-1)
    profiles: list[dict[str, object]] = []
    for idx, subcircuit in enumerate(_split_circuit(qc, int(chunk_splits))):
        state, profile = _run_statevector_backend(
            subcircuit,
            state,
            gpu_ids=gpu_ids,
            optimization_level=optimization_level,
            debug=debug,
            debug_print=debug_print,
            debug_label=f"{debug_label} chunk={idx + 1}/{chunk_splits}".strip(),
        )
        profile = dict(profile)
        profile["chunk_index"] = int(idx)
        profiles.append(profile)

    total_profile: dict[str, object] = {
        "backend": "aer_statevector_gpu",
        "mode": "chunked",
        "chunk_count": len(profiles),
        "chunks": profiles,
        "input_num_instructions": int(len(qc.data)),
    }
    for profile in profiles:
        for key, value in profile.items():
            if key.endswith("_s") and isinstance(value, (int, float)):
                total_profile[key] = float(total_profile.get(key, 0.0)) + float(value)
    return state, total_profile
