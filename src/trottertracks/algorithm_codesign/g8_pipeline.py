"""On-demand conditional native description. No reference/table/global normalizer."""
from collections import OrderedDict
from fractions import Fraction as F
import json
import time
from .g7_provider import bind_synthesized_provider_ir


class AcquisitionCache:
    """Shared bounded memo, populated only after a live symbolic request.

    No eviction/resynthesis in the acquisition memo: a new miss beyond its fixed
    capacity is a technical STOP. The injected backend is called once per key.
    """
    def __init__(self, backend, capacity=32, byte_cap=1024**2):
        self.backend, self.capacity, self.byte_cap = backend, capacity, byte_cap
        self.rows, self.requests, self.hits = {}, 0, 0
        self.calls, self.bytes, self.peak_bytes = 0, 0, 0
        self.failed = False

    def get(self, key):
        if self.failed: raise RuntimeError('G8 acquisition already failed; no retry')
        self.requests += 1
        if key in self.rows:
            self.hits += 1
            return self.rows[key]
        if self.calls >= self.capacity:
            raise RuntimeError('G8 acquisition key cap; no retry/eviction/resynthesis')
        self.calls += 1
        try:
            row = self.backend(F(key))
        except Exception:
            self.failed = True
            raise
        size = len(json.dumps(row, sort_keys=True).encode())
        if self.bytes + size > self.byte_cap:
            self.failed = True
            raise MemoryError('G8 acquisition cache byte cap; no retry')
        self.rows[key] = row
        self.bytes += size; self.peak_bytes = max(self.peak_bytes, self.bytes)
        return row


class NativePipeline:
    def __init__(self, generator, acquisition, budget, provider_delta, capacity=8, byte_cap=128*1024):
        self.generator, self.acquisition, self.budget = generator, acquisition, budget
        self.delta, self.capacity, self.byte_cap = F(provider_delta), capacity, byte_cap
        self.cache, self.sizes, self.bytes, self.peak_bytes = OrderedDict(), {}, 0, 0
        self.attempts = self.accepted = self.zeros = self.hits = self.misses = self.evictions = 0
        self.observed_keys = set()
        self.timing = {'local_query_and_symbolic_angle_wall': 0.0, 'local_query_and_symbolic_angle_CPU': 0.0,
                       'tangent_key_construction_wall': 0.0, 'tangent_key_construction_CPU': 0.0,
                       'cache_lookup_wall': 0.0, 'provider_description_wall': 0.0,
                       'provider_description_CPU': 0.0, 'miss_acquisition_wall': 0.0}

    def _lookup(self, key):
        wall = time.monotonic()
        if key in self.cache:
            self.hits += 1; row = self.cache.pop(key); self.cache[key] = row
        else:
            self.misses += 1
            acquire_wall = time.monotonic()
            row = self.acquisition.get(key)
            self.timing['miss_acquisition_wall'] += time.monotonic() - acquire_wall
            size = len(json.dumps(row, sort_keys=True).encode())
            if size > self.byte_cap: raise MemoryError('single sequence exceeds row cache cap')
            while self.cache and (len(self.cache) >= self.capacity or self.bytes + size > self.byte_cap):
                old, _ = self.cache.popitem(last=False)
                self.bytes -= self.sizes.pop(old); self.evictions += 1
            self.cache[key], self.sizes[key] = row, size
            self.bytes += size; self.peak_bytes = max(self.bytes, self.peak_bytes)
        self.timing['cache_lookup_wall'] += time.monotonic() - wall
        return row

    def step(self, bits):
        if self.attempts >= self.budget['hard_attempt_cap_two_axes']:
            raise RuntimeError('G8 attempt budget exhausted')
        self.attempts += 1
        wall, cpu = time.monotonic(), time.process_time()
        event = self.generator.sample(bits)
        self.timing['local_query_and_symbolic_angle_wall'] += time.monotonic() - wall
        self.timing['local_query_and_symbolic_angle_CPU'] += time.process_time() - cpu
        if event is None:
            self.zeros += 1
            return None  # no cache request, synthesis, provider or readout
        if self.accepted >= self.budget['accepted_call_cap_two_axes']:
            raise RuntimeError('G8 high-probability resource cap exceeded; abort without scientific prefix')
        wall, cpu = time.monotonic(), time.process_time()
        key = str(F(event['ratio'])); self.observed_keys.add(key)
        self.timing['tangent_key_construction_wall'] += time.monotonic() - wall
        self.timing['tangent_key_construction_CPU'] += time.process_time() - cpu
        row = self._lookup(key)
        self.accepted += 1
        wall, cpu = time.monotonic(), time.process_time()
        native = bind_synthesized_provider_ir(event, {key: row})
        n_provider = sum(event['provider_calls'].values())
        error = 2 * F(row['strict_operator_error_upper']) + n_provider * self.delta
        self.timing['provider_description_wall'] += time.monotonic() - wall
        self.timing['provider_description_CPU'] += time.process_time() - cpu
        return {'event': event, 'conditional_native_description': native,
                'conditional_event_error_upper': error,
                'provider_error_is_assumption_not_measurement': True,
                'T_Rz': 2 * row['T_count'], 'provider_calls': event['provider_calls']}

    def counters(self):
        return {'attempts': self.attempts, 'accepted': self.accepted, 'zeros': self.zeros,
                'row_hits': self.hits, 'row_misses': self.misses, 'row_evictions': self.evictions,
                'row_cache_entries': len(self.cache), 'row_cache_bytes': self.bytes,
                'row_peak_cache_bytes': self.peak_bytes, 'observed_tangents': sorted(self.observed_keys, key=F),
                'timing': dict(self.timing), 'quantum_measurements': 0}
