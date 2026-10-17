"""Scoped H8 adapter: original native worker, conditioned steps only."""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import time
import h8_conditional_s2_contract as c
import h8_conditional_s2_sampling as sampling


class Writer:
    def __init__(self, output):
        self.output = Path(output)
        self.sequence = 0

    def write(self, name, value, **kwargs):
        c.require(Path(name).name == name, 'OWN_OUTPUT_NAME')
        # Stage telemetry may revisit a stage; retain every observation.
        path = self.output/name
        if path.exists():
            self.sequence += 1
            path = self.output/(str(self.sequence)+'_'+name)
        c.save(path, value)


@contextmanager
def trace_stage(writer, name, **context):
    started = time.monotonic()
    writer.write('stage_'+name+'_start.json', dict(stage=name, phase='start', monotonic=started, **context))
    try:
        yield
    finally:
        writer.write('stage_'+name+'_end.json', dict(stage=name, phase='end', elapsed=time.monotonic()-started, **context))


def evaluate(task, directory, config, grant):
    own = os.sched_getaffinity(0)
    c.require(len(own) == 1, 'SINGLE_OWN_WORKER_CPU')
    c.authorize(config, grant, worker_cpu=next(iter(own)))
    c.verify(config, git=False)
    sys.path.insert(0, str(Path(config['provider_root'])/'src'))
    frozen = importlib.import_module('trottertracks.resource_applicability.h8_fast_cost_port_v1')
    optimized = importlib.import_module('trottertracks.resource_applicability.h8_resource_map_optimization_v1')
    native = importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    molecular = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    bind = importlib.import_module('trottertracks.resource_applicability.h4_common_panel_port_v1').bound_function
    for module in (frozen, optimized, native, molecular):
        c.require(Path(module.__file__).resolve().is_relative_to(Path(config['provider_root'])/'src'), 'PINNED_IMPORT_ORIGIN')
    writer = Writer(directory)
    sample = {}
    port_instance = []
    class ConditionalPort(frozen.MatchedPort):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            port_instance.append(self)

        def sampled_steps(self, cell, master):
            c.require(cell['id'] == task['cell_id'] and master == task['seed'], 'EXACT_SAMPLING_TASK')
            self.calls.take('trajectory')
            self.calls.take('occurrence', cell['q'])
            steps, receipt = sampling.make_steps(self.preps[cell['id']], q=cell['q'], master=master,
                expected=c.evidence()['laws'][cell['id']])
            sample.update(receipt)
            writer.write('conditional_sampling.json', receipt)
            return steps

    def checked_prepare(ham, prefix):
        c.require(prefix == 7 and ham.n_qubits == 16 and len(ham.lambdas) == 28, 'SAVED_H8_PREPARATION_SCOPE')
        # This is the original native view of the SAVED DF, not SCF/DF/state
        # generation. Exact saved hashes reject any basis/normalization drift.
        p = frozen._prepare(ham, prefix)
        ref = c.evidence()['preparation']
        binding = c.verify_saved_preparation(p, ref)
        writer.write('preparation_identity.json', dict(saved_preparation=ref, partition_binding=binding, snapshot_sha256=
            next(r['sha256'] for r in config['files'] if r['logical_name'] == 'h8_input_snapshot.npz'),
            SCF=0, DF=0, state_generation=0, new_signal=0))
        return p

    retained = []
    def evolution(*args, **kwargs):
        if retained:
            return retained.pop()
        value = frozen.partial_native_from_step_requests(*args, **kwargs)
        # An inconclusive fast structure proof is not a failed native circuit.
        # The pinned validator uses its original two-branch fallback for the
        # first trajectory; do not silently drop those four logical probes.
        writer.write('native_constructor.json', dict(
            pinned_native_constructor=True, ancilla_structure_fast_path=optimized.ancilla_diagonal(value.circuit, 16)))
        retained.append(replace(value, circuit=value.circuit.copy()))
        return value

    original = c.read(next(r['path'] for r in config['files'] if r['logical_name'] == 'original_manifest'))
    cell = dict(id=task['cell_id'], method='B2', prefix=7, q=task['q'], r=4, K=4, R=task['q']*4, formula='2nd')
    caps = dict(original['plan']['caps_proposed'], compile=2, trajectory=1, occurrence=task['q'],
                control_probe=4 if task['validate'] else 0, primitive=0, reference_matvec=0)
    manifest = dict(kind='H8_CONDITIONAL_S2_NATIVE_COST', input_identity=dict(original['input_identity']),
        plan=dict(original['plan'], cells=[cell], caps_proposed=caps))
    paths = {r['logical_name']: r['path'] for r in config['files']}
    for field, name in (('snapshot_path', 'h8_input_snapshot.npz'), ('snapshot_receipt_path', 'snapshot_receipt.json'),
                        ('df_receipt_path', 'df_receipt.json')):
        manifest['input_identity'][field] = paths[name]
    native_task = dict(kind='COST', cell=cell, cell_id=cell['id'], seed=task['seed'], phase='conditional_s2',
                       replica=task['replica'], ordinary=False, validate=task['validate'], expected_signal=None)
    class Trace:
        def stage(self, name, **context):
            return trace_stage(writer, name, **context)
    class Progress:
        def update(self, **values):
            writer.write('progress.json', values)
        def phase(self, name):
            self.update(phase=name)
    worker = bind(frozen.cost_task, dict(MatchedPort=ConditionalPort, _prepare=checked_prepare,
        partial_native_from_step_requests=evolution, validate_control=optimized.validate_control))
    try:
        with optimized.hashing() as cache:
            worker(c.ROOT, manifest, native_task, writer, Progress(), Trace())
            writer.write('hash_reuse_receipt.json', cache.snapshot())
    finally:
        retained.clear()
    result = c.read(Path(directory)/'cost_result.json')
    trajectory = c.read(Path(directory)/'trajectory.json')
    c.require(result['calls_attempted'] == port_instance[0].calls.used, 'WORKER_CALL_RECEIPT')
    if task['validate']:
        proof = c.read(Path(directory)/'validation_symmetric_directional.json')
        c.require(proof['max_error'] < 1e-8, 'CONTROL_VALIDATION')
    else:
        proof = dict(inherited_pinned_native_constructor=True, new_control_probes=0,
                     correctness_validation_policy='first trajectory per candidate; original registered policy')
    out = dict(task=task, sampling=sample, rows=result['rows'], event_digest=trajectory['event_digest'],
        actual_calls=result['calls_attempted'], SOURCE=config['SOURCE'], compiler=c.COMPILER, profile=c.PROFILE,
        control=proof, state_preparation_excluded=True, original_n32_mixed=False,
        trajectory_sha256=c.sha(Path(directory)/'trajectory.json'),
        sampling_receipt_sha256=c.sha(Path(directory)/'conditional_sampling.json'))
    c.paired_result(task, out)
    return out
