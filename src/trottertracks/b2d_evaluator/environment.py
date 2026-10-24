"""Metadata-only dependency checks and provenance; no bitwise environment lock."""
import importlib.metadata as metadata
import importlib.util
import os
import sys

ENVIRONMENT_PATH = "docs/research/b2d_stage1_execution_ready_20261025/environment.json"
MODULES = ("numpy", "scipy", "qiskit", "qiskit_nature", "openfermion", "openfermionpyscf",
           "pyscf", "numba", "matplotlib")
CONTROLLED_ENV = ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def inspect_environment():
    """Record versions and locations without importing scientific packages."""
    versions = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in metadata.distributions()}
    origins = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise ValueError("Required scientific dependency is not installed: " + name)
        origins[name] = spec.origin
    return dict(python_runtime=dict(implementation=sys.implementation.name,
                version=list(sys.version_info)), dependency_versions=dict(sorted(versions.items())),
        executable=sys.executable, prefix=sys.prefix, package_origins=origins,
        environment_variables={name: os.environ.get(name) for name in CONTROLLED_ENV})


def verify_environment(expected):
    from .scope import REQUIRED_DEPENDENCIES
    actual = inspect_environment()
    if actual["python_runtime"] != expected["python_runtime"]:
        raise ValueError("Scientific Python runtime version differs.")
    for name in REQUIRED_DEPENDENCIES:
        if actual["dependency_versions"].get(name) != expected["dependency_versions"][name]:
            raise ValueError("Scientific dependency version differs: " + name)
    return actual
