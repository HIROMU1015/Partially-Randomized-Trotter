"""Use PBS CPU slots for signals and memory-weighted slots for pricing."""

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import resource
import time


GIB = 1 << 30
CONTEXT = mp.get_context("fork")


def price_profile(cell, block_bound, stages):
    head = cell["q"] * stages * block_bound
    tail = cell["q"] * (cell["R_outer"] or 0) * ((cell["K"] or 0) + 1)
    # Round to powers of two so observations apply to similarly sized circuits.
    return tuple(0 if count == 0 else 1 << (count - 1).bit_length() for count in (head, tail))


def estimated_bytes(profile):
    head, tail = profile
    return GIB + head * 1024 + tail * (128 << 10)


def measured_price(connection, execute, task):
    with connection:
        cpu_start = time.process_time()
        error = None
        try:
            execute(task)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"[:1024]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        connection.send((error, rss, time.process_time() - cpu_start))


def result(process, connection):
    process.join()
    try:
        if process.exitcode != 0:
            return f"Price worker exited with status {process.exitcode}", 0, 0
        return connection.recv()
    except EOFError:
        return "Price worker completed without accounting", 0, 0
    finally:
        connection.close()
        process.close()


def current_rss(process):
    try:
        value = next(line for line in Path(f"/proc/{process.pid}/status").read_text().splitlines()
                     if line.startswith("VmRSS:"))
        return int(value.split()[1]) * 1024
    except (OSError, StopIteration):
        return 0


def prices(jobs, execute, slots, memory, profile_for, checkpoint):
    pending = defaultdict(deque)
    for job in jobs:
        pending[profile_for(job[1])].append(job)
    profiles = sorted(pending, key=estimated_bytes, reverse=True)
    observed = defaultdict(int)
    active = {}
    in_profile = defaultdict(int)
    reserved = 0
    errors = {}
    completed = 0
    peak_workers = peak_reserved = 0
    cpu_seconds = 0.
    begin = time.monotonic()
    next_checkpoint = begin

    try:
        while profiles or active:
            overrun = sum(max(0, current_rss(process) - need)
                          for process, _, _, need, _ in active.values())
            while profiles and len(active) < slots:
                free = memory - reserved - overrun
                selected = None
                for profile in profiles:
                    measured = observed[profile]
                    # Initially admit one per profile. Later results can raise
                    # the budget based on the observed peak, with 25% headroom.
                    need = max(estimated_bytes(profile), math.ceil(measured * 1.25))
                    if (need <= free and (measured or not in_profile[profile])) or not active:
                        selected = profile, min(need, memory)
                        break
                if selected is None:
                    break
                profile, need = selected
                task = pending[profile].popleft()
                if not pending[profile]:
                    profiles.remove(profile)
                read_end, write_end = CONTEXT.Pipe(duplex=False)
                process = CONTEXT.Process(target=measured_price, args=(write_end, execute, task))
                try:
                    process.start()
                except BaseException:
                    read_end.close()
                    raise
                finally:
                    write_end.close()
                active[process.sentinel] = process, read_end, profile, need, task[3]
                in_profile[profile] += 1
                reserved += need
                peak_workers = max(peak_workers, len(active))
                peak_reserved = max(peak_reserved, reserved)

            for sentinel in wait(active, timeout=10):
                process, connection, profile, need, path = active.pop(sentinel)
                error, rss, cpu = result(process, connection)
                reserved -= need
                in_profile[profile] -= 1
                cpu_seconds += cpu
                if error is None:
                    observed[profile] = max(observed[profile], rss)
                    completed += 1
                else:
                    errors[path.name] = error

            now = time.monotonic()
            if now >= next_checkpoint or not active and not profiles:
                checkpoint(dict(phase="price", done=completed, errors=len(errors), active=len(active),
                    worker_limit=slots, peak_workers=peak_workers, cpu_seconds=cpu_seconds,
                    wall_seconds=now-begin, memory_budget=memory, reserved_bytes=reserved,
                    peak_reserved_bytes=peak_reserved, peak_worker_rss_bytes=max(observed.values(), default=0)))
                next_checkpoint = now + 30
    finally:
        for process, connection, _, _, _ in active.values():
            result(process, connection)
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
