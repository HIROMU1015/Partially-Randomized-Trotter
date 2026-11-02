"""Fail before grant consumption if the frozen Gaussian lowering cannot import.

Metadata and imports only: no molecule, array input, circuit build or transpile.
The native implementation remains the one in the immutable science closure.
"""
import importlib.metadata as md
import inspect
from pathlib import Path
from .h6_matched_contract_v1 import file_hash

REQUIRED = {'qiskit-nature': '0.7.2', 'qiskit-algorithms': '0.3.1',
            'qiskit-aer': '0.15.1', 'psutil': '5.9.8'}


def versions():
    actual = {}
    for name, expected in REQUIRED.items():
        try:
            actual[name] = md.version(name)
        except md.PackageNotFoundError as exc:
            raise ValueError('H6_NATIVE_DEPENDENCY_MISSING:' + name) from exc
        if actual[name] != expected:
            raise ValueError('H6_NATIVE_DEPENDENCY_VERSION:' + name)
    return actual


def native_dependencies():
    actual = versions()
    # Use the exact import function used by U_to_qiskit_ops_jw; do not implement
    # an alternative decomposition or instantiate a scientific circuit here.
    from trotterlib.df_trotter.ops import _import_bogoliubov_transform
    try:
        native = _import_bogoliubov_transform()
    except ImportError as exc:
        raise ValueError('H6_NATIVE_LOWERING_IMPORT_FAILED') from exc
    if native.__module__ != 'qiskit_nature.second_q.circuit.library.bogoliubov_transform':
        raise ValueError('H6_NATIVE_LOWERING_INTERFACE_CHANGED')
    path = Path(inspect.getfile(native)).resolve()
    return dict(schema='h6_native_lowering_dependencies_v2', versions=actual,
                native_class=native.__module__ + '.' + native.__name__,
                source_path=str(path), source_sha256=file_hash(path),
                scientific_circuit_constructed=False, transpile_performed=False)
