"""Acquisition-only PF adapter primitives, with lazy dependency injection.

This module has no molecular loader/production dispatcher. NativeKernel uses the
existing provider's build/wrapper/fingerprint/transpile functions only after a
future scoped grant AND a sealed live gate. Preparation is not execution approval.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from pf_direct_compile_contract import (
    AXES, SYSTEM_LAYOUTS, authorize_metadata, digest, require, verify_bound_files,
)


class Journal:
    """Exclusive local output and before-call reservations; no refund/retry."""
    def __init__(self, root, caps):
        self.root = Path(root)
        require(not self.root.exists() and not self.root.is_symlink(), 'OUTPUT_ALREADY_USED')
        self.root.mkdir(exist_ok=False)
        self.caps = dict(caps); self.used = {key: 0 for key in caps}
        self.ledger = self.root/'ledger.jsonl'
        self.ledger.touch(exist_ok=False)

    def write(self, name, value):
        require(Path(name).name == name, 'OUTPUT_NAME')
        with (self.root/name).open('x') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False); handle.write('\n')

    def take(self, name, count=1):
        require(name in self.caps and type(count) is int and count > 0
                and self.used[name]+count <= self.caps[name], 'CALL_NOT_AUTHORIZED:'+name)
        with self.ledger.open('a') as handle:
            handle.write(json.dumps(dict(call=name, count=count, before_call=True, nonrefundable=True))+'\n')
            handle.flush(); os.fsync(handle.fileno())
        self.used[name] += count


def acquire_pair(contract, approval, proposals, candidate_key, *, live_gate, backend_factory):
    """Future coordinator. No loader/import/output occurs on a rejected grant.

    live_gate is required to bind SOURCE Git blobs, inputs/provider/runtime/profile
    and a fresh resource receipt to this exact contract. No real live_gate or loader
    is supplied in this technical-preparation packet; tests use explicit fakes.
    """
    selected = authorize_metadata(contract, approval, proposals)
    pair = [r for r in selected if (r['system'], r['candidate_id']) == tuple(candidate_key)]
    require(len(pair) == 2 and {r['axis'] for r in pair} == set(AXES), 'CANDIDATE_NOT_SELECTED')
    verify_bound_files(contract)
    receipt = live_gate(contract)
    require(receipt.get('status') == 'PASS' and receipt.get('contract_digest') == digest(contract)
            and receipt.get('SOURCE') == contract['SOURCE']
            and receipt.get('run_id') == contract['run_id'], 'FRESH_GATE_NOT_BOUND')
    # Each pair owns its own output and exact calls. Independent tasks share no STOP.
    from pf_direct_compile_contract import call_caps
    output = Path(contract['output_root'])/(pair[0]['system']+'__'+pair[0]['candidate_id'])
    journal = Journal(output, call_caps(pair))
    try:
        journal.take('saved_input_load')
        backend = backend_factory(contract, pair, journal)
        journal.take('saved_preparation_restore')
        backend.restore(pair[0])
        journal.take('native_evolution')
        evolution = backend.build(pair[0])
        journal.take('control_probe', 4)
        control = backend.control(evolution)
        require(control.get('status') == 'PASS', 'CONTROL_FAILED')
        journal.write('control.json', control)
        rows = []
        for axis in AXES:
            row = backend.compile_axis(evolution, axis, journal)
            validate_measurement(row, contract['compiler_hash'], axis)
            row.update(system=pair[0]['system'], candidate_id=pair[0]['candidate_id'],
                       SOURCE=contract['SOURCE'], observation_kind='DIRECT_COMPILED_FULL_WRAPPER',
                       model_evaluated=False)
            journal.write('measured_'+axis+'.json', row)
            rows.append(row)
        require(journal.used == journal.caps, 'INCOMPLETE_CALL_LEDGER')
        result = dict(status='COMPLETE_MANDATORY_STOP', rows=rows, calls=journal.used,
                      new_sampling=False, new_fit=False, cost_reaccounting=False,
                      next_stage_authorized=False)
        journal.write('result.json', result)
        return result
    except BaseException as error:
        journal.write('STOP.json', dict(status='MANDATORY_STOP', reason=type(error).__name__+':'+str(error),
            calls=journal.used, reservations_refunded=False, automatic_retry=False))
        raise


def validate_measurement(row, compiler_hash, axis):
    require(row['axis'] == axis and row['compiler_hash'] == compiler_hash, 'COMPILE_AXIS_OR_COMPILER_CHANGED')
    require(isinstance(row.get('fingerprint_before'), str) and bool(row['fingerprint_before'])
            and row['fingerprint_before'] == row['fingerprint_after'] == row['compiled_input_fingerprint'],
            'COMPILE_FINGERPRINT_CHANGED')
    for name in ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size'):
        require(type(row[name]) is int and row[name] >= 0, 'NONINTEGER_COMPILED_METRIC:'+name)
    counts = row['posttranspile_gate_counts']
    require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()), 'GATE_COUNTS')
    require(counts.get('rz', 0) == row['RZ'] and counts.get('cx', 0) == row['CX'], 'METRIC_COUNT_MISMATCH')
    require(type(row.get('native_exit_code')) is int and row['native_exit_code'] == 0 and row.get('reaped') is True, 'NATIVE_EXIT_OR_REAP')
    require(type(row['compile_seconds']) in (int, float) and math.isfinite(row['compile_seconds']) and row['compile_seconds'] >= 0, 'COMPILE_TIME')


def accept_owned_measurement(row, witness, task_id):
    """Attach exit/reap facts only from a bound owned-worker receipt.

    The production spawner/wait4 witness is not bound by this preparation. A
    low-level NativeKernel result alone is intentionally insufficient for PASS.
    """
    require(witness.get('task_id') == task_id and witness.get('result_digest') == digest(row),
            'OWNED_RESULT_IDENTITY')
    require(type(witness.get('exit_code')) is int and witness['exit_code'] == 0 and witness.get('reaped') is True
            and witness.get('cleanup_errors') == [], 'OWNED_WORKER_NOT_REAPED')
    return dict(row, native_exit_code=witness['exit_code'], reaped=witness['reaped'])


class NativeKernel:
    """Provider-API adapter; context must come from a separately sealed loader.

    No SCF/DF/signal/shot/model calls. Context supplies exact saved scalar and
    deterministic blocks, with an independently bound control oracle. This class
    never constructs those inputs itself or silently transfers another profile.
    """
    def __init__(self, context):
        self.context = context

    def restore(self, task):
        c = self.context; n, rank = SYSTEM_LAYOUTS[task['system']]
        require((c['system'], c['candidate_id'], c['q'], c['PF'], c['prefix']) ==
                (task['system'], task['candidate_id'], int(task['q']), task['PF'], int(task['prefix'])),
                'PREPARATION_TASK_IDENTITY')
        require(c['system_qubits'] == n and c['DF_rank'] == rank, 'INPUT_SERIES_MISMATCH')
        require(c['identity_gate_passed'] is True and c['scalar_policy'] == 'original_plus_own_tail_once'
                and c['control_policy'] == 'symmetric_directional', 'PREPARATION_OR_SCALAR_NOT_BOUND')
        require(c['independent_control_oracle_bound'] is True, 'INDEPENDENT_CONTROL_ORACLE_MISSING')
        require(c['native_instruction_limit'] is None, 'ARBITRARY_INSTRUCTION_CEILING')
        self.m = c['provider']; self.native = c['native']; self.compiler = c['compiler']

    def build(self, task):
        c = self.context
        return self.native.build_deterministic_native(c['deterministic_blocks'],
            num_system_qubits=c['system_qubits'], T=0.8, q=int(task['q']), formula=task['PF'],
            scalar=c['saved_scalar'], control_policy='symmetric_directional', max_instructions=math.inf)

    def control(self, evolution):
        # Existing independent control callback must retain branch/X/Y semantics.
        return self.context['control_oracle'](evolution)

    def compile_axis(self, evolution, axis, journal):
        journal.take('wrapper_build')
        wrapper = self.m.build_native_hadamard_wrapper(evolution, axis=axis,
            include_measurement=True, max_instructions=math.inf)
        before = self.m.canonical_qiskit_circuit_fingerprint(wrapper)
        journal.take('wrapper_compile')
        start = time.monotonic()
        result = self.m.transpile_and_measure_cost(wrapper, self.compiler, actual_circuit_fingerprint=before)
        after = self.m.canonical_qiskit_circuit_fingerprint(wrapper)
        return dict(axis=axis, RZ=result.rz_count, CX=result.cx_count, RZ_depth=result.rz_depth,
            CX_depth=result.cx_depth, depth=result.total_depth, size=result.circuit_size,
            pretranspile_gate_counts=dict(result.pretranspile_gate_counts),
            posttranspile_gate_counts=dict(result.posttranspile_gate_counts),
            compiler_hash=result.compiler_settings_hash,
            fingerprint_before=before, fingerprint_after=after,
            compiled_input_fingerprint=result.actual_circuit_fingerprint,
            compile_seconds=time.monotonic()-start,
            # Coordinator must run inside an owned worker; real exit/reap is the
            # scheduler's responsibility, never synthesized by this low-level API.
            native_exit_code=None, reaped=False)
