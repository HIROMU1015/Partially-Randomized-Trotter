"""Single-process resource/call/output guards for an authorized future run."""
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import time

CAPS = {"processes": 1, "threads": 1, "retries": 0, "main_calls": 55275,
        "recipe_total_calls": 110550, "hard_total_calls": 111000,
        "per_LP_wall_seconds": 2, "wall_seconds": 3600, "CPU_seconds": 3300,
        "RSS_MiB": 1536, "virtual_MiB": 4096, "output_MiB": 128}


class TechnicalFailure(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class ResourceCap(TechnicalFailure):
    pass


def single_thread_environment():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "HIGHS_THREADS"):
        os.environ[key] = "1"


class BudgetGuard:
    def __init__(self, domain="REGISTERED_SAVED_TABLE", synthetic_caps=None):
        if synthetic_caps and domain != "SYNTHETIC":
            raise PermissionError("registered cap changes prohibited")
        self.caps = CAPS | (synthetic_caps or {})
        self.domain = domain
        self.wall_start, self.cpu_start = time.monotonic(), time.process_time()
        self.main_calls = self.aux_calls = self.output_bytes = 0
        self.seen_main, self.seen_aux = set(), set()
        self.active_start = None
        self.current_phase = None

    def check(self):
        if time.monotonic()-self.wall_start >= self.caps["wall_seconds"]:
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: wall")
        if time.process_time()-self.cpu_start >= self.caps["CPU_seconds"]:
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: CPU")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= self.caps["RSS_MiB"]*1024:
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: RSS")
        if self.active_start is not None and time.monotonic()-self.active_start >= self.caps["per_LP_wall_seconds"]:
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: per-LP wall")

    @contextmanager
    def lp_call(self, call_id, baseline, auxiliary=False):
        self.check()
        if self.current_phase == "BUDGET_FREEZE" and baseline == "B3":
            raise TechnicalFailure("B3 solver call during BUDGET_FREEZE")
        if auxiliary:
            if call_id not in self.seen_main or call_id in self.seen_aux:
                raise TechnicalFailure("auxiliary without main or auxiliary retry")
        else:
            if call_id in self.seen_main:
                raise TechnicalFailure("retry forbidden")
        next_main = self.main_calls + (not auxiliary)
        next_total = self.main_calls+self.aux_calls+1
        if (next_main > self.caps["main_calls"]
                or next_total > self.caps["recipe_total_calls"]
                or next_total > self.caps["hard_total_calls"]):
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: LP calls")
        if auxiliary:
            self.seen_aux.add(call_id)
            self.aux_calls += 1
        else:
            self.seen_main.add(call_id)
            self.main_calls += 1
        self.active_start = time.monotonic()
        try:
            yield
            self.check()
        finally:
            self.active_start = None

    @contextmanager
    def enforce_OS_limits(self):
        single_thread_environment()
        old_handler = signal.getsignal(signal.SIGALRM)
        old_cpu_handler = signal.getsignal(signal.SIGXCPU)
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        old_as = resource.getrlimit(resource.RLIMIT_AS)
        old_cpu = resource.getrlimit(resource.RLIMIT_CPU)
        ceiling = self.caps["virtual_MiB"]*1024**2
        soft_cpu = int(time.process_time()+self.caps["CPU_seconds"])+1
        def alarm_handler(_signum, _frame):
            self.check()
        def cpu_handler(_signum, _frame):
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: OS CPU")
        try:
            resource.setrlimit(resource.RLIMIT_AS, (min(ceiling, old_as[1]) if old_as[1] >= 0 else ceiling, old_as[1]))
            resource.setrlimit(resource.RLIMIT_CPU, (min(soft_cpu, old_cpu[1]) if old_cpu[1] >= 0 else soft_cpu, old_cpu[1]))
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.signal(signal.SIGXCPU, cpu_handler)
            signal.setitimer(signal.ITIMER_REAL, .05, .05)
            yield
            self.check()
        finally:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
            signal.signal(signal.SIGALRM, old_handler)
            signal.signal(signal.SIGXCPU, old_cpu_handler)
            resource.setrlimit(resource.RLIMIT_AS, old_as)
            resource.setrlimit(resource.RLIMIT_CPU, old_cpu)

    def write(self, path, value, compressed_append=False, terminal=False):
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)+"\n").encode()
        if compressed_append:
            payload = gzip.compress(payload, mtime=0)
        # Reserve space for a small terminal failure receipt. The failure
        # receipt must not re-trigger the wall/CPU cap while recording STOP.
        limit = self.caps["output_MiB"]*1024**2-(0 if terminal else 65536)
        if self.output_bytes+len(payload) > limit:
            raise ResourceCap("D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP: output")
        if not terminal:
            self.check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab" if compressed_append else "xb") as stream:
            stream.write(payload)
        self.output_bytes += len(payload)
        return payload

    def usage(self):
        return {"main_LP_calls": self.main_calls, "auxiliary_LP_calls": self.aux_calls,
                "total_LP_calls": self.main_calls+self.aux_calls,
                "wall_seconds": time.monotonic()-self.wall_start,
                "CPU_seconds": time.process_time()-self.cpu_start,
                "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "output_bytes": self.output_bytes, "processes": 1, "retries": 0}
