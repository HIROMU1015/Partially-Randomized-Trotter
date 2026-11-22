from contextlib import nullcontext
import multiprocessing as mp
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate

from trotterlib.df_partial_s2 import DFDeterministicFragmentSpec, DFDeterministicOneBodySpec
from trottertracks import pr_candidate_completion as completion
from trottertracks import rpe_df_price_flanks as df, rpe_q3_round_tables as q3
from trottertracks.resource_applicability import ax2b_native_df_v5 as native


def test_df_price_flanks_are_shared_before_fork_and_keep_each_tail():
    common = dict(basis_id='fixture', basis_hash='fixture', basis_change_operations=(),
                  num_system_qubits=2, runtime_basis_operations=((XXPlusYYGate(.19, -.23), (0, 1)),))
    blocks = (
        DFDeterministicOneBodySpec(block_id='one', block_kind='one_body', original_fragment_index=None,
                                  rank=None, diagonal_eigenvalues=(.5, -.22), order_index=0, **common),
        DFDeterministicFragmentSpec(block_id='df', block_kind='df_fragment', original_fragment_index=0,
                                    rank=2, diagonal_eta=(.13, -.79), lam=.51, order_index=1, **common),
    )
    times = (.17, -.19, .17)
    context = SimpleNamespace(native=native, prep=SimpleNamespace(deterministic_blocks=blocks, num_system_qubits=2),
                              h={'core': SimpleNamespace(schedule=lambda *a, **kw: [
                                  {'signed_time': time} for time in times])})

    class NativeRuntime:
        def context(self, request):
            return context

    instance = NativeRuntime()
    runner = SimpleNamespace(runtime=lambda: SimpleNamespace(D=SimpleNamespace(NativeRuntime=NativeRuntime)))
    request = dict(physical_context=dict(physical_candidate=dict(formula='4th', q=1, R_outer=3, T=.15)))

    def circuits(path):
        result = []
        for index, signed_time in enumerate(times):
            tail = QuantumCircuit(3)
            tail.cp(.2 * (1 + index + path), 2, 0)
            tail.rz(.8 * (1 + index + path), 1)
            tail.global_phase = .11 * (1 + index + path)
            evolution = native.build_partial_native(blocks, num_system_qubits=2, T=signed_time, q=1,
                scalar=-.31, controlled_tails=(tail,), control_policy='symmetric_directional', max_instructions=500)
            fingerprints = [evolution.circuit_fingerprint]
            for axis in ('cosine', 'sine'):
                wrapper = native.build_native_hadamard_wrapper(evolution, axis=axis, max_instructions=510)
                fingerprints.append(native.canonical_qiskit_circuit_fingerprint(wrapper))
            result.append((fingerprints, evolution.circuit.global_phase.hex(), evolution.instruction_upper_bound))
        return result

    expected = [circuits(0), circuits(1)]
    assert expected[0] != expected[1]
    fork = mp.get_context('fork')
    read, write = fork.Pipe(duplex=False)

    def child():
        instance.context(request)
        write.send([circuits(0), circuits(1), append.call_count])

    with patch.object(native, '_append_block', wraps=native._append_block) as append:
        with df.reuse_df_price_flanks(runner):
            runner.runtime()
            instance.context(request)
            assert append.call_count == 8
            process = fork.Process(target=child)
            process.start()
            write.close()
            assert read.poll(10)
            first, second, calls = read.recv()
            process.join(10)
            assert process.exitcode == 0
            assert [first, second] == expected
            assert calls == 8
            with pytest.raises(RuntimeError, match='PRE_BUILD_INSTRUCTION_BUDGET'):
                native.build_partial_native(blocks, num_system_qubits=2, T=.17, q=1, scalar=0.,
                    controlled_tails=(QuantumCircuit(3),), control_policy='symmetric_directional', max_instructions=1)
        assert circuits(0) == expected[0]
        assert append.call_count == 20
    read.close()


def test_new_df_command_preserves_fixed_runner_arguments_and_leaves_audits_alone(tmp_path):
    (tmp_path/'src').mkdir()
    runner = tmp_path/'runner.py'
    runner.write_text('import json, sys\ndef runtime():\n    raise AssertionError("unexpected provider load")\n'
                      'def main():\n    print(json.dumps(sys.argv[1:]))\n    return 0\n')
    base = [sys.executable, '-B', '-c', completion._DF_PRICE_BOOTSTRAP, str(runner), str(tmp_path), 'calibrate']
    assert completion.accelerate_df_price_command(base, 'DF_MODEL_FREEZE', df.__file__) == base
    command = completion.accelerate_df_price_command(base, 'DF_CALIBRATION', df.__file__)
    assert subprocess.check_output(command, text=True).strip() == '["calibrate"]'


def test_q3_tables_reuse_by_exact_row_and_update_the_m13_globals():
    def build(row, **options):
        return dict(row=row, options=options)

    risk = SimpleNamespace(round_distribution=build)
    opts = dict(h=.1, target_energy=0., radius=1., probability_rule='EXACT_AFFINE', maximum_outcomes=9)
    calls = []

    def watched(row, **options):
        calls.append(row['m'])
        return build(row, **options)

    risk.round_distribution = watched
    namespace = dict(round_distribution=watched)
    exec('def extended(row, options):\n    return round_distribution(row, **options)', namespace)
    for enabled in (False, True):
        calls.clear()
        installation = q3.reuse_round_tables(risk, namespace['extended']) if enabled else nullcontext()
        with installation:
            for m in range(14):
                row = dict(m=m, raw=dict(real=.5, imag=.25), N_X=2, N_Y=2)
                assert risk.round_distribution(row, **opts) == namespace['extended'](row, opts)
            assert risk.round_distribution(row, **dict(opts, radius=.5))['options']['radius'] == .5
            risk.round_distribution(row, **dict(opts, target_energy=-0.))
        assert len(calls) == (16 if enabled else 30)
    assert risk.round_distribution is watched
    assert namespace['round_distribution'] is watched
