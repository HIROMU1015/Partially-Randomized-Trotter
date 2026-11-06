"""Opt-in memory/copy reduction for the existing Aer state-action primitive.

No installation, monkey patch or change to the sealed C6b process. A caller
must explicitly select this class in a separately fixed execution revision.
Cost circuits and the existing simulation_circuit conversion remain unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .c1_h6_state_action import StateActions, simulation_circuit


def stage_memory_estimates(*, raw, reference, direct):
    """Estimate each stage from COMPLETED process peaks, never current low RSS.

    Operational planning only, not a rigorous maximum or a scientific statistic.
    A scheduler must still check actual available memory at dispatch.
    """
    result = {}
    for name, rows in [('raw', raw), ('reference', reference), ('direct', direct)]:
        if not rows or any(r.get('status') != 'PASS' or type(r.get('peak_RSS_KiB')) is not int
                           or r['peak_RSS_KiB'] <= 0 for r in rows):
            raise ValueError('COMPLETED_STAGE_MEMORY_EVIDENCE_REQUIRED:'+name)
        result[name+'_task_memory_estimate_bytes'] = max(r['peak_RSS_KiB'] for r in rows)*1024
    return result


class _NativeExecutor(ThreadPoolExecutor):
    """Apply the thread budget on the thread that actually calls Aer native code."""

    def __init__(self, threads):
        super().__init__(max_workers=1)
        self.native_threads = threads

    def submit(self, fn, /, *args, **kwargs):
        def execute():
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=self.native_threads, user_api='openmp'):
                return fn(*args, **kwargs)
        return super().submit(execute)


class BatchedStateActions(StateActions):
    """Reuse immutable operations; batch independent inputs at native parallelism.

Every input column and output amplitude is retained. This is the same Aer
double-precision statevector backend, not a new numerical simulator. Batching
does not change a circuit, its phase, native compiler, probes or tolerances.
"""

    def apply(self, circuit, states):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit_aer.jobs import AerJob
        from threadpoolctl import threadpool_limits

        states = np.asarray(states, dtype=np.complex128)
        if (states.ndim != 2 or states.shape[1] != (1 << circuit.num_qubits)
                or len(states) == 0 or not np.isfinite(states).all()):
            raise ValueError('STATE_BATCH_SCHEMA')
        if id(circuit) not in self.cache:
            body = simulation_circuit(circuit)
            phase = float(body.global_phase)
            body.global_phase = 0.
            self.cache[id(circuit)] = (circuit, body, phase)
        original, body, phase = self.cache[id(circuit)]
        if original is not circuit:
            raise ValueError('CIRCUIT_CACHE_IDENTITY')

        # One experiment per allocated native parallel slot. This bounds circuit
        # metadata/native result retention by simultaneous work, rather than by
        # all256 inputs. The caller still owns the complete input/output arrays.
        count = min(self.threads, len(states))
        output = np.empty_like(states)
        self.last_batch_sizes = []
        self.last_parallel_metadata = []
        # Aer submits its job to a background executor. An OpenMP limit on the
        # Python caller alone does not set that background thread's task-local
        # OpenMP budget. The pinned Aer's _run_circuits omits the executor when
        # constructing AerJob, despite exposing an executor option. Reuse that
        # exact AerJob/native callback with our explicit executor; no monkey patch.
        with _NativeExecutor(self.threads) as executor, threadpool_limits(limits=1, user_api='blas'):
            for begin in range(0, len(states), count):
                chunk = states[begin:begin+count]
                batch = []
                for psi in chunk:
                    qc = QuantumCircuit(circuit.num_qubits)
                    qc.set_statevector(psi)
                    # No mutation of body or local UnitaryGate arrays after
                    # conversion; avoid repeating operation/matrix deep copies.
                    qc.compose(body, inplace=True, copy=False)
                    qc.save_statevector()
                    batch.append(qc)
                job = AerJob(self.backend, str(uuid4()), self.backend._execute_circuits_job,
                             circuits=batch, parameter_binds=None, run_options={'shots':1}, executor=executor)
                job.submit()
                result = job.result()
                if not result.success:
                    raise ValueError('STATE_ACTION_BACKEND_FAIL: '+str(result.status))
                for i in range(len(chunk)):
                    output[begin+i] = np.asarray(result.get_statevector(i))
                self.last_batch_sizes.append(len(batch))
                self.last_parallel_metadata.append({k:v for k,v in result.metadata.items()
                    if k.startswith('parallel_') or k.startswith('omp_')})
                del result, batch
        # Restore only the declared phase; no reference comparison or phase fit.
        output *= np.exp(1j*phase)
        if not np.isfinite(output).all():
            raise ValueError('NONFINITE_STATE_ACTION')
        return output
