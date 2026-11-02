"""SP-1 launch binding and one-process caps; no science is run by this module."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_binding(auth, contract_hash, head, parents, changed, dirty, allowed):
    source = auth.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise PermissionError("new SP-1 source commit is not bound")
    if (auth.get("status") != "APPROVED_FOR_ONE_SP1_RUN"
            or auth.get("science_execution_authorized") is not True
            or type(auth.get("runs")) is not int or auth["runs"] != 1
            or type(auth.get("retries")) is not int or auth["retries"] != 0
            or auth.get("mandatory_STOP") is not True):
        raise PermissionError("separate SP-1 one-shot authorization required")
    instruction = auth.get("explicit_execution_instruction")
    if not isinstance(instruction, str) or len(instruction.strip()) < 10:
        raise PermissionError("explicit SP-1 execution instruction missing")
    if auth.get("contract_sha256") != contract_hash:
        raise PermissionError("SP-1 contract identity mismatch")
    if head == source or parents != [source] or dirty:
        raise PermissionError("clean direct authorization-only child required")
    if not changed or not set(changed).issubset(allowed):
        raise PermissionError("authorization child changed a forbidden path")


def verify_source(root, contract):
    manifest = json.loads((root/contract["source_manifest_path"]).read_text())
    if manifest.get("focused_tests_passed") is not True or manifest.get("fusion_audit_passed") is not True:
        raise PermissionError("reviewable source preparation incomplete")
    for relative, digest in manifest["critical_sha256"].items():
        if sha(root/relative) != digest:
            raise PermissionError(f"critical source/input identity mismatch: {relative}")
    return manifest


def verify_launch(root, contract_path):
    root = Path(root)
    contract = json.loads(Path(contract_path).read_text())
    auth = json.loads((root/contract["authorization_path"]).read_text())
    source = auth.get("source_commit")
    # Pending preparation refuses BEFORE data access or source revision queries.
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise PermissionError("final source review / separate SP-1 authorization pending")
    head = git(root, "rev-parse", "HEAD")
    changed = git(root, "diff", "--name-only", source, head).splitlines()
    parents = git(root, "show", "-s", "--format=%P", head).split()
    allowed = {contract["authorization_path"], contract["optional_receipt_path"]}
    validate_binding(auth, sha(contract_path), head, parents, changed,
                     bool(git(root, "status", "--porcelain", "--untracked-files=all")), allowed)
    if contract["authorization_path"] not in changed:
        raise PermissionError("authorization JSON must be committed in direct child A")
    verify_source(root, contract)
    return contract, auth, head


def package_tree_sha(name):
    dist = importlib.metadata.distribution(name)
    files = sorted(str(p) for p in dist.files if str(p).endswith(".py"))
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode()+b"\0")
        digest.update(Path(dist.locate_file(relative)).read_bytes()+b"\0")
    return digest.hexdigest()


def verify_runtime(root, contract):
    ref = contract["tool_identity"]
    path = root/ref["path"]
    if sha(path) != ref["sha256"]:
        raise PermissionError("stored tool identity changed")
    identity = json.loads(path.read_text())
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if platform.python_version() != identity["python"] or versions != identity["packages"]:
        raise RuntimeError("fixed B runtime package lock mismatch")
    for name, digest in identity["source_py_tree_sha256"].items():
        if package_tree_sha(name) != digest:
            raise RuntimeError(f"runtime source tree changed: {name}")
    return {"python": platform.python_version(), "packages_match": True,
            "source_py_tree_sha256": identity["source_py_tree_sha256"],
            "synthesizer_calls": 0}


def consume_marker(directory, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("SP-1 output directory already contains evidence; no retry")
    marker = directory/"one_shot_consumed.json"
    with marker.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return marker


def check_limits(caps, elapsed, cpu, peak_rss_kib):
    if elapsed >= caps["wall_seconds"]:
        raise TimeoutError("SP-1 wall cap hit; no retry")
    if cpu >= caps["cpu_seconds"]:
        raise TimeoutError("SP-1 CPU cap hit; no retry")
    if peak_rss_kib > caps["RSS_MiB"]*1024:
        raise MemoryError("SP-1 RSS cap hit; no retry")


class BudgetGuard:
    """POSIX periodic wall/CPU/RSS guard, one process, no workers or GPU."""
    def __init__(self, caps):
        self.caps = caps
        self.start = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_start = usage.ru_utime+usage.ru_stime

    def usage(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"wall_seconds": time.monotonic()-self.start,
                "cpu_seconds": usage.ru_utime+usage.ru_stime-self.cpu_start,
                "peak_RSS_KiB": usage.ru_maxrss, "processes": 1}

    def check(self, *_):
        u = self.usage()
        check_limits(self.caps, u["wall_seconds"], u["cpu_seconds"], u["peak_RSS_KiB"])

    def cpu_signal(self, *_):
        raise TimeoutError("SP-1 OS CPU cap hit; no retry")

    def __enter__(self):
        self.old_alarm = signal.getsignal(signal.SIGALRM)
        self.old_cpu_signal = signal.getsignal(signal.SIGXCPU)
        self.old_timer = signal.getitimer(signal.ITIMER_REAL)
        self.old_cpu = resource.getrlimit(resource.RLIMIT_CPU)
        self.old_as = resource.getrlimit(resource.RLIMIT_AS)
        try:
            signal.signal(signal.SIGALRM, self.check)
            signal.signal(signal.SIGXCPU, self.cpu_signal)
            cpu_soft = math.ceil(self.cpu_start+self.caps["cpu_seconds"])
            if self.old_cpu[1] != resource.RLIM_INFINITY:
                cpu_soft = min(cpu_soft, self.old_cpu[1])
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, self.old_cpu[1]))
            as_soft = self.caps["virtual_address_MiB"]*1024**2
            if self.old_as[1] != resource.RLIM_INFINITY:
                as_soft = min(as_soft, self.old_as[1])
            resource.setrlimit(resource.RLIMIT_AS, (as_soft, self.old_as[1]))
            period = self.caps["watchdog_period_seconds"]
            signal.setitimer(signal.ITIMER_REAL, period, period)
            self.check()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_alarm)
        signal.signal(signal.SIGXCPU, self.old_cpu_signal)
        resource.setrlimit(resource.RLIMIT_CPU, self.old_cpu)
        resource.setrlimit(resource.RLIMIT_AS, self.old_as)
        if any(self.old_timer):
            signal.setitimer(signal.ITIMER_REAL, *self.old_timer)


def serialize_result(result, output_cap):
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if len(payload.encode()) > output_cap:
        raise RuntimeError("SP-1 output cap hit; no retry")
    return payload
