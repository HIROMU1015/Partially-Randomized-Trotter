"""Artificial fixtures produced by the unchanged pinned Stream/run_pair/Meter.

Imports only old mock tests and safe driver; no scientific module or saved result.
"""
from hashlib import sha256
from itertools import cycle
import json
from pathlib import Path
import sys
from unittest.mock import patch

import test_g6_w2c_b3_driver as T
import test_g6_w2c_b3_f01_provenance as P
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
D = T.M


def metadata(stream):
    return dict(output_bytes=stream.bytes, output_rows=stream.rows, output_chain_sha256=stream.previous,
                file_sha256=stream.file_hash.hexdigest(), peak_line_bytes=stream.peak_line_bytes)


def emit_file(path, documents):
    meter = D.Meter(T.FakeClock())
    stream = D.Stream(path, meter, byte_limit=1 << 36, flush_rows=1)
    for doc in documents:
        stream.emit(doc)
    stream.close()
    return metadata(stream)


def bundle(root, n=18):
    root = Path(root)
    root.mkdir(parents=True)
    pairs, summaries = [], []
    clock = T.FakeClock()
    contract = dict(artificial_only=True, id='ARTIFICIAL_B3_OFFLINE_MIXED', profile_N=n,
                    scientific_execution=False)
    digest = D.identity(contract)
    for phase, pass_name in (('correctness', 'baseline'), ('profile', 'baseline'), ('profile', 'instrumented')):
        for arm in D.ARMS:
            for layer in D.LAYERS:
                spec = dict(phase=phase, pass_name=pass_name, arm=arm, layer=layer, N=n)
                spec['path'] = '_'.join(V.key(spec)) + '.jsonl'
                meter = D.Meter(clock)
                before = clock()
                stream = D.Stream(root/spec['path'], meter, byte_limit=1 << 30, flush_rows=4096)
                with meter.span('setup.constructor_and_preflight'):
                    endpoint = D.LegacyEndpoint(T.modules(), arm, layer,
                                               T.accepted_candidate()['decisions']['angle'], lambda: 'artificial-source')
                cells = [dict(arm=arm, layer=layer, axis=a, N=n) for a in D.AXES]
                summary = D.run_pair(endpoint, lambda axis: T.Bits(cycle((0, 1, 2))), cells, stream, meter,
                                     phase=phase, pass_name=pass_name, correctness_trials=16,
                                     memory_read=lambda: {'VmRSS_bytes': None, 'VmHWM_bytes': None})
                stream.close()
                summary.update({k: spec[k] for k in ('phase', 'pass_name', 'arm', 'layer')})
                summary.update(metadata(stream), coarse=meter.rows, reconciliation=meter.reconcile(D.delta(before, clock())),
                               functions={} if pass_name == 'instrumented' else None,
                               memory={'VmRSS_bytes': 10000, 'VmHWM_bytes': 11000},
                               memory_baseline={'VmRSS_bytes': 9000, 'VmHWM_bytes': 10000},
                               unknown_bit_operations=None,
                               cache={'angle': 'DISABLED', 'law': 'FROZEN_HELD_STATE_ONLY',
                                      'runtime_cache_bytes': 0, 'OS': 'uncontrolled'})
                if pass_name == 'instrumented':
                    total = summary['reconciliation']['outer_total_ns']
                    summary['fine_reconciliation'] = D.Meter(clock).reconcile(total)
                else:
                    summary['fine_reconciliation'] = None
                pairs.append(spec)
                summaries.append(summary)
    shared = D.Meter(clock)
    sb = clock()
    with shared.span('shared.source_check'):
        pass
    with shared.span('shared.worker_dispatch_and_receive'):
        pass
    final = dict(schema='g6_w2c_b3_driver_result_v1', contract_digest=digest,
                 admission=dict(contract_digest=digest, approved=True, fresh_claim=True, previously_consumed=False,
                                claim_id='B3_NEW_ARTIFICIAL_OFFLINE_ONLY', retry=0, resume=False,
                                scope='B3_CORRECTNESS_AND_FIXED_FULL_N_TWO_PASS'),
                 pairs=summaries, mandatory_STOP=True, population_cost_certificate=False, physical_RZ_certified=False,
                 shared_setup=dict(stages=shared.rows, reconciliation=shared.reconcile(D.delta(sb, clock()))),
                 final_receipt_write_cost_ns=None,
                 final_receipt_cost_status='SELF_MEASUREMENT_TRAILER_NOT_INCLUDED_IN_ARM_TOTALS')
    sm = D.Meter(clock)
    before = clock()
    stream = D.Stream(root/'summary.jsonl', sm, byte_limit=1 << 30, flush_rows=1)
    stream.emit(final)
    stream.close()
    receipt = dict(summary_chain_sha256=stream.previous, shared_finalization_stages=sm.rows,
                   reconciliation=sm.reconcile(D.delta(before, clock())), receipt_self_write_cost_ns=None, mandatory_STOP=True)
    (root/'completion_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    manifest = dict(schema='b3_artificial_serial_inputs_v1', artificial_only=True, producer_commit=V.BASE_COMMIT,
                    artificial_contract=contract, contract_digest=digest, profile_N={a: n for a in D.ARMS},
                    pairs=pairs, summary='summary.jsonl', receipt='completion_receipt.json')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest, final


def failure(root, mode='sample', n=18):
    root = Path(root)
    root.mkdir(parents=True)
    spec = dict(phase='profile', pass_name='baseline', arm='full_return', layer='A', N=n)
    clock = T.FakeClock()
    meter = D.Meter(clock)
    stream = D.Stream(root/'pair.jsonl', meter, byte_limit=1 << 30, flush_rows=1)
    endpoint = P.ToyEndpoint(fail_sample=lambda t: mode == 'sample' and t.endswith('/X/1'),
                             fail_angle=mode == 'angle')
    if mode in ('partial', 'flush'):
        stream.file = P.FailingFile(stream, mode)
    if mode == 'file_hash':
        actual_hash = stream.file_hash
        class BadHash:
            calls = 0
            def update(self, b):
                self.calls += 1
                if self.calls == 2:
                    raise OSError('artificial file hash failure')
                actual_hash.update(b)
            def hexdigest(self):
                return actual_hash.hexdigest()
        stream.file_hash = BadHash()
    cells = [dict(arm=spec['arm'], layer='A', axis=a, N=n) for a in D.AXES]
    try:
        D.run_pair(endpoint, lambda axis: T.Bits(cycle((2,))), cells, stream, meter,
                   phase='profile', pass_name='baseline', correctness_trials=16)
    except Exception as exc:
        f = exc.b3_failure_provenance
        control = dict(kind='TYPED_FAILURE', trial_failure_provenance=T.A.canonical(D.exact(f)), mandatory_STOP=True)
        (root/'pair.failure.json').write_text(json.dumps(control, indent=2)+'\n')
    else:
        raise AssertionError('Fault not reached')
    finally:
        try:
            stream.close()
        except OSError:
            pass
    return spec, f
