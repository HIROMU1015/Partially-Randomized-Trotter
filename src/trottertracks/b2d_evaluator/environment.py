"""Metadata-only runtime binding. Never import a scientific dependency here."""
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import sys

from trottertracks.pr_dictionary_validation.authorization import file_sha

ENVIRONMENT_PATH = "docs/research/b2d_stage1_execution_ready_20261025/environment.json"
MODULES = ("numpy", "scipy", "qiskit", "qiskit_nature", "openfermion", "openfermionpyscf",
           "pyscf", "numba", "matplotlib")
CONTROLLED_ENV = ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path.read_bytes()))


def inspect_environment():
    """Run with the selected existing venv's -I -B; no arrays or circuits."""
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("Isolated venv interpreter with -I -B required.")
    versions, records = {}, {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in versions:
            raise PermissionError("Duplicate runtime distribution.")
        versions[name] = dist.version
        record = dist.read_text("RECORD")
        if record is None:
            raise PermissionError("Unbound installed distribution RECORD.")
        records[name] = file_sha(record.encode())
    origins = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise PermissionError("Required scientific package is not installed: "+name)
        path = Path(spec.origin).resolve()
        if not path.is_relative_to(prefix):
            raise PermissionError("Scientific import would escape pinned venv.")
        origins[name] = identity(path)
    from packaging.requirements import Requirement
    conflicts=[]
    for dist in metadata.distributions():
        for text in dist.requires or ():
            required=Requirement(text)
            if required.marker is not None and not required.marker.evaluate({"extra":""}):
                continue
            key=required.name.lower().replace("_","-")
            if key not in versions or not required.specifier.contains(versions[key],prereleases=True):
                conflicts.append(dict(package=dist.metadata["Name"],requirement=text))
    if conflicts:
        raise PermissionError("Runtime dependency requirements are inconsistent: "+str(conflicts))
    return dict(schema="b2d_environment_v1", python_runtime=dict(
        implementation=sys.implementation.name, version=list(sys.version_info)),
        interpreter=identity(sys.executable), prefix=str(prefix), base_prefix=sys.base_prefix,
        pyvenv_cfg=identity(prefix/"pyvenv.cfg"), dependency_versions=dict(sorted(versions.items())),
        distribution_records=dict(sorted(records.items())), package_origins=origins,
        dependency_requirements_consistent=True,
        isolated=True, dont_write_bytecode=True,
        controlled_environment={name:os.environ.get(name) for name in CONTROLLED_ENV},
        validation="METADATA_AND_IMPORT_ORIGINS_ONLY_NO_SCIENTIFIC_EXECUTION")


def verify_environment(expected):
    if inspect_environment() != expected:
        raise PermissionError("Bound Stage-1 environment changed.")
