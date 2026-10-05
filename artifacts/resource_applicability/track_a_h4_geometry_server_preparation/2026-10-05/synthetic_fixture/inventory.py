"""Read-only CPU environment inventory; no repository/science imports."""
import contextlib
import datetime
import hashlib
import importlib.metadata as md
import inspect
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError) as error:
        return {"unavailable": str(error)}


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def main():
    import numpy as np
    import qiskit
    from qiskit.circuit import library
    output = Path(sys.argv[1])
    packages = {}
    for name in ["numpy", "scipy", "qiskit", "rustworkx", "openfermion", "openfermionpyscf", "pyscf"]:
        try:
            dist = md.distribution(name)
            record = dist.read_text("RECORD")
            packages[name] = {
                "version": dist.version, "location": str(dist.locate_file("")),
                "installer": dist.read_text("INSTALLER"), "direct_url": dist.read_text("direct_url.json"),
                "wheel_metadata": dist.read_text("WHEEL"),
                "installed_record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
                "wheel_archive_identity": None,
                "provenance_scope": "installed metadata; original wheel archive not available/verified",
            }
        except md.PackageNotFoundError:
            packages[name] = {"missing": True}
    blas = io.StringIO()
    with contextlib.redirect_stdout(blas):
        np.show_config()
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = value.strip()
    process_summary = command(["ps", "-eo", "uid,pcpu,rss", "--no-headers"])
    aggregates = {"process_count": 0, "summed_lifetime_pcpu": 0.0, "summed_rss_kib": 0, "other_uid_process_count": 0}
    for line in process_summary.get("stdout", "").splitlines():
        try:
            uid, pcpu, rss = line.split()
            aggregates["process_count"] += 1
            aggregates["summed_lifetime_pcpu"] += float(pcpu)
            aggregates["summed_rss_kib"] += int(rss)
            aggregates["other_uid_process_count"] += int(int(uid) != os.getuid())
        except ValueError:
            pass
    cpu_rows = []
    cpu_table = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    for line in cpu_table.get("stdout", "").splitlines():
        if line and not line.startswith("#"):
            cpu, core, socket, node = line.split(",")
            cpu_rows.append({"cpu": int(cpu), "core": core, "socket": socket, "node": node})
    affinity = sorted(os.sched_getaffinity(0))
    allowed_rows = [row for row in cpu_rows if row["cpu"] in affinity]
    cgroup_text = read("/proc/self/cgroup")
    cgroup_records = []
    if isinstance(cgroup_text, str):
        for line in cgroup_text.splitlines():
            if line.startswith("0::"):
                current = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
                while str(current).startswith("/sys/fs/cgroup"):
                    record = {"path": str(current)}
                    for filename in ["cpu.max", "cpu.weight", "cpuset.cpus.effective", "cpuset.mems.effective", "memory.max", "memory.high", "memory.current", "memory.swap.max", "pids.max"]:
                        value = read(current / filename)
                        if isinstance(value, str): record[filename] = value
                    cgroup_records.append(record)
                    if current == Path("/sys/fs/cgroup"): break
                    current = current.parent
    signature = inspect.signature(qiskit.transpile)
    defaults = {}
    for key, value in signature.parameters.items():
        defaults[key] = repr(value.default)
    env_keys = ["PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "QISKIT_PARALLEL", "QISKIT_NUM_PROCS", "RAYON_NUM_THREADS", "QISKIT_SETTINGS"]
    scheduler_keys = ["SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "PBS_JOBID", "LSB_JOBID"]
    filesystems = {}
    for name, path in [("worktree", output.parent), ("temporary", "/tmp")]:
        stat = os.statvfs(path)
        filesystems[name] = {"path": str(path), "total_bytes": stat.f_blocks * stat.f_frsize, "available_bytes": stat.f_bavail * stat.f_frsize}
    config = getattr(qiskit, "user_config", None)
    config_values = config.get_config() if config is not None else {}
    result = {
        "schema_version": "track_a_server_environment_inventory_v0",
        "observed_jst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        "os_release": read("/etc/os-release"), "kernel": platform.uname()._asdict(),
        "cpu_lscpu": command(["lscpu", "-J"]), "cpu_topology": cpu_rows,
        "numa": {p.parent.name: read(p) for p in Path("/sys/devices/system/node").glob("node*/cpulist")},
        "cpu_affinity": affinity, "available_logical_cpu_count": len(affinity),
        "available_physical_core_count": len({(row["socket"], row["core"]) for row in allowed_rows}),
        "loadavg": os.getloadavg(), "proc_loadavg": read("/proc/loadavg"),
        "cpu_pressure": read("/proc/pressure/cpu"), "memory_pressure": read("/proc/pressure/memory"),
        "memory": meminfo, "filesystem": filesystems, "cgroup": cgroup_records,
        "resource_limits": {name: resource.getrlimit(getattr(resource, name)) for name in ["RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NPROC", "RLIMIT_NOFILE"]},
        "shared_process_summary": aggregates, "scheduler_environment": {key: os.environ.get(key) for key in scheduler_keys},
        "python": {"executable": sys.executable, "version": sys.version, "prefix": sys.prefix, "base_prefix": sys.base_prefix},
        "dependencies": packages, "numpy_blas_configuration": blas.getvalue(),
        "process_environment": {key: os.environ.get(key) for key in env_keys},
        "qiskit_user_config_read_only": config_values,
        "qiskit_transpile_signature": str(signature), "qiskit_transpile_defaults": defaults,
        "api_presence": {name: hasattr(library, name) for name in ["XXPlusYYGate", "RXXGate", "RYYGate", "PhaseGate", "UnitaryGate"]},
        "qiskit_plugin_metadata": [{"group": ep.group, "name": ep.name, "value": ep.value} for ep in md.entry_points() if ep.group.startswith("qiskit.")],
        "actions": {"shared_environment_changes": 0, "science_module_imports": 0, "molecular_access": 0, "gpu_access": 0},
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary = {key: result[key] for key in ["available_logical_cpu_count", "available_physical_core_count", "loadavg", "cpu_pressure", "memory_pressure", "cgroup", "python", "api_presence", "qiskit_user_config_read_only"]}
    summary["memory"] = {key: meminfo.get(key) for key in ["MemTotal", "MemAvailable", "SwapTotal", "SwapFree"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
