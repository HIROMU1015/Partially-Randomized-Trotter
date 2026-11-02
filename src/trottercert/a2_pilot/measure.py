"""Owned-process timing and bounded logs, never scientific bound arithmetic."""

import io
import resource
import time

from ..a2_spec01_03.common import integer


class Timings:
    def __init__(self):
        self.rows, self.current = [], None

    def transition(self, label):
        wall, cpu = time.perf_counter(), time.process_time()
        if self.current is not None:
            name, w, c = self.current
            self.rows.append({"stage": name, "wall_seconds": format(wall - w, ".9f"),
                              "CPU_seconds": format(cpu - c, ".9f"),
                              "process_peak_RSS_KiB_so_far": str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
        self.current = None if label is None else (label, wall, cpu)


class BoundedLog(io.TextIOBase):
    def __init__(self, limit):
        self.limit, self.data, self.dropped_bytes = limit, bytearray(), 0

    def write(self, text):
        value = text.encode("utf-8")
        room = max(0, self.limit - len(self.data))
        self.data.extend(value[:room])
        self.dropped_bytes += max(0, len(value) - room)
        return len(text)

    def flush(self):
        pass


def maximum_rat_bit_lengths(record):
    maximum = {"numerator_bits": 0, "denominator_bits": 0}

    def walk(value):
        if type(value) is dict:
            if set(value) == {"num", "den"}:
                maximum["numerator_bits"] = max(maximum["numerator_bits"], abs(integer(value["num"])).bit_length())
                maximum["denominator_bits"] = max(maximum["denominator_bits"], integer(value["den"], natural=True).bit_length())
            else:
                for sub in value.values():
                    walk(sub)
        elif type(value) is list:
            for sub in value:
                walk(sub)
    walk(record)
    return {k: str(v) for k, v in maximum.items()}
