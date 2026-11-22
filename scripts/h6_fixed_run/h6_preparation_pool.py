"""Use PBS CPU slots for signals and memory-weighted slots for pricing."""

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import resource
import time


GIB = 1 << 30
CONTEXT = mp.get_context("fork")
AXIS_OPEN, AXIS_RESERVED, AXIS_SERIAL, AXIS_PARALLEL = range(4)
_price_phases = None
_axis_slot = None


@dataclass
class ActivePrice:
    process: object
    connection: object
    profile: tuple
    memory: int
    path: Path
    axis_slot: object
    pair_memory: int
    extra_memory: int = 0
    extra_cpu: int = 0


def price_size(cell, block_bound, stages):
    head = cell["q"] * stages * block_bound
    tail = cell["q"] * (cell["R_outer"] or 0) * ((cell["K"] or 0) + 1)
    return head, tail


def price_profile(size):
    # Round to powers of two so observations apply to similarly sized circuits.
    return tuple(0 if count == 0 else 1 << (count - 1).bit_length() for count in size)


def estimated_bytes(profile):
    head, tail = profile
    return GIB + head * 1024 + tail * (128 << 10)


def reserved_bytes(forecast, measured, successful, running):
    if successful >= 2:
        return max(GIB, math.ceil(measured * 1.5))
    return max(forecast * (2 if running and not successful else 1), math.ceil(measured * 1.5))


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def record_price_phase(name, seconds, rss):
    if _price_phases is not None:
        _price_phases[name] = (seconds, rss)


def use_spare_axis():
    if _axis_slot is None:
        return False
    with _axis_slot.get_lock():
        parallel = _axis_slot.value == AXIS_RESERVED
        _axis_slot.value = AXIS_PARALLEL if parallel else AXIS_SERIAL
        return parallel


@contextmanager
def price_phase(name):
    if _price_phases is None:
        yield
        return
    began = time.monotonic()
    yield
    record_price_phase(name, time.monotonic() - began, peak_rss())


def measured_price(connection, execute, task, axis_slot):
    global _price_phases, _axis_slot
    _price_phases = {}
    _axis_slot = axis_slot
    with connection:
        cpu_start = time.process_time()
        error = None
        try:
            execute(task)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"[:1024]
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = time.process_time() - cpu_start + child.ru_utime + child.ru_stime
        connection.send((error, peak_rss(), cpu, _price_phases))


def result(process, connection):
    process.join()
    try:
        if process.exitcode != 0:
            return f"Price worker exited with status {process.exitcode}", 0, 0, {}
        return connection.recv()
    except EOFError:
        return "Price worker completed without accounting", 0, 0, {}
    finally:
        connection.close()
        process.close()


def current_rss(pid, include_children=False):
    rss = 0
    directory = Path(f"/proc/{pid}")
    try:
        value = next(line for line in (directory / "status").read_text().splitlines() if line.startswith("VmRSS:"))
        rss = int(value.split()[1]) * 1024
        if include_children:
            children = (directory / "task" / str(pid) / "children").read_text().split()
            rss += sum(current_rss(child, True) for child in children)
    except (OSError, StopIteration):
        pass
    return rss


def prices(jobs, execute, slots, memory, size_for, checkpoint, pair_memory_for=None):
    grouped = defaultdict(list)
    for job in jobs:
        size = size_for(job[1])
        grouped[price_profile(size)].append((estimated_bytes(size), job))
    pending = {
        profile: deque(job for _, job in sorted(group, key=lambda entry: entry[0], reverse=True))
        for profile, group in grouped.items()
    }
    profiles = sorted(pending, key=estimated_bytes, reverse=True)
    forecasts = {profile: estimated_bytes(profile) for profile in profiles}
    short = {profile for profile in profiles if forecasts[profile] <= memory // slots}
    short_target = slots // 2
    observed = defaultdict(int)
    successes = defaultdict(int)
    by_phase = {}
    active = {}
    in_profile = defaultdict(int)
    reserved = 0
    errors = {}
    completed = 0
    peak_workers = peak_reserved = peak_worker_rss = 0
    extra_cpu = parallel_completed = 0
    cpu_seconds = 0.
    begin = time.monotonic()
    next_checkpoint = begin

    try:
        while profiles or active:
            overrun = sum(max(0, current_rss(job.process.pid, bool(job.extra_cpu))
                              - job.memory - job.extra_memory) for job in active.values())
            while profiles and len(active) + extra_cpu < slots:
                free = memory - reserved - overrun
                selected = None
                active_short = sum(in_profile[profile] for profile in short)
                order = profiles
                if active_short < min(len(active), short_target):
                    # The first long task gets a head start. While it runs,
                    # use part of the CPU allocation for shorter tasks that fit.
                    order = [p for p in reversed(profiles) if p in short] + [p for p in profiles if p not in short]
                for profile in order:
                    need = reserved_bytes(forecasts[profile], observed[profile], successes[profile], in_profile[profile])
                    if need <= free or not active:
                        selected = profile, min(need, memory)
                        break
                if selected is None:
                    break
                profile, need = selected
                task = pending[profile].popleft()
                if not pending[profile]:
                    profiles.remove(profile)
                pair_memory = pair_memory_for(task[1]) if pair_memory_for else 0
                axis_slot = CONTEXT.Value("b", AXIS_OPEN) if pair_memory else None
                read_end, write_end = CONTEXT.Pipe(duplex=False)
                process = CONTEXT.Process(target=measured_price, args=(write_end, execute, task, axis_slot))
                try:
                    process.start()
                except BaseException:
                    read_end.close()
                    raise
                finally:
                    write_end.close()
                active[process.sentinel] = ActivePrice(process, read_end, profile, need, task[3], axis_slot, pair_memory)
                in_profile[profile] += 1
                reserved += need
                peak_workers = max(peak_workers, len(active))
                peak_reserved = max(peak_reserved, reserved)

            # All ordinary work is in flight before reserving a second axis.
            # That keeps the extra CPU and its memory out of the path queue.
            if not profiles and len(active) + extra_cpu < slots:
                for job in active.values():
                    if job.axis_slot is None or len(active) + extra_cpu >= slots:
                        continue
                    extra = max(0, job.pair_memory - job.memory)
                    if extra + reserved + overrun > memory:
                        continue
                    with job.axis_slot.get_lock():
                        if job.axis_slot.value == AXIS_OPEN:
                            job.extra_memory, job.extra_cpu = extra, 1
                            reserved += extra
                            extra_cpu += 1
                            peak_reserved = max(peak_reserved, reserved)
                            job.axis_slot.value = AXIS_RESERVED

            for sentinel in wait(active, timeout=10):
                job = active.pop(sentinel)
                error, rss, cpu, phases = result(job.process, job.connection)
                reserved -= job.memory + job.extra_memory
                extra_cpu -= job.extra_cpu
                in_profile[job.profile] -= 1
                cpu_seconds += cpu
                peak_worker_rss = max(peak_worker_rss, rss, max((p for _, p in phases.values()), default=0))
                parallel = job.axis_slot is not None and job.axis_slot.value == AXIS_PARALLEL
                if not parallel:
                    observed[job.profile] = max(observed[job.profile], rss)
                if error is None:
                    if parallel:
                        parallel_completed += 1
                    else:
                        successes[job.profile] += 1
                    for name, (seconds, peak) in phases.items():
                        stats = by_phase.setdefault(name, dict(tasks=0, wall_seconds=0., peak_rss_bytes=0))
                        stats["tasks"] += 1
                        stats["wall_seconds"] += seconds
                        stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], peak)
                    completed += 1
                else:
                    errors[job.path.name] = error

            now = time.monotonic()
            if now >= next_checkpoint or not active and not profiles:
                checkpoint(dict(phase="price", done=completed, errors=len(errors), active=len(active),
                    worker_limit=slots, peak_workers=peak_workers, cpu_seconds=cpu_seconds,
                    wall_seconds=now-begin, memory_budget=memory, reserved_bytes=reserved,
                    peak_reserved_bytes=peak_reserved, peak_worker_rss_bytes=peak_worker_rss,
                    extra_axis_cpus=extra_cpu, completed_parallel_prices=parallel_completed, phase_metrics=by_phase))
                next_checkpoint = now + 30
    finally:
        for job in active.values():
            result(job.process, job.connection)
    return errors


def signals(jobs, execute, slots, checkpoint):
    if not jobs:
        return {}
    errors = {}
    completed = 0
    begin = time.monotonic()
    baseline = resource.getrusage(resource.RUSAGE_CHILDREN)
    with ProcessPoolExecutor(max_workers=min(slots, len(jobs)), mp_context=CONTEXT) as pool:
        tasks = {pool.submit(execute, job): job[3] for job in jobs}
        for future in as_completed(tasks):
            path = tasks[future]
            try:
                future.result()
            except Exception as error:
                errors[path.name] = f"{type(error).__name__}: {error}"
            else:
                completed += 1
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    checkpoint(dict(phase="signal", done=completed, errors=len(errors), worker_limit=slots,
        cpu_seconds=usage.ru_utime + usage.ru_stime - baseline.ru_utime - baseline.ru_stime,
        wall_seconds=time.monotonic()-begin, peak_worker_rss_bytes=usage.ru_maxrss * 1024))
    return errors
