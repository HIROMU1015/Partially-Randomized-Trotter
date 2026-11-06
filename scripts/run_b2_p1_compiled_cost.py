"""Fixed m=0 B2 native full-wrapper compiler connection; no MC or signals."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trotterlib import b2_rpe_connection as connection
from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory
from trotterlib.b2_p1_signal_validation import DF_COMMIT, DF_PATH, DF_SHA
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider, configure_law
from trotterlib.pr2_s0_s1_validation import load_snapshot
from trotterlib.rpe_resource_accounting import RPERoundSpecification, RPE_COST_METRICS
from trotterlib.rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from trotterlib.rte import CompilerSettings, _make_event
from trotterlib.rte_compiled_cost import (
    canonical_backend_fingerprint_or_none, canonical_qiskit_circuit_fingerprint,
    compiler_settings_hash, transpile_and_measure_cost,
)

PRICE_COMMIT = '0daf01d51c022977a1c0a7d29dde34a1c4c82d7c'
PRICE_PATH = 'artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/result.json'
CORRECT_COMMIT = 'ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5'
CORRECT_PATH = 'artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json'
SPEC_COMMIT = '89165cf70958e7e4676d0580e3acb7faf588772c'
SPEC_PATH = 'artifacts/b2_hphys_ground_state/2026-11-06/run01/agent_b_input_spec.json'
CORE = (
    'src/trotterlib/b2_rpe_connection.py',
    'src/trotterlib/o1_m1_resource_bridge.py',
    'src/trotterlib/rte.py',
    'src/trotterlib/rte_compiled_cost.py',
    'src/trotterlib/df_partial_s2.py',
    'src/trotterlib/df_partial_s2_repeated.py',
    'src/trotterlib/df_rte_tail.py',
    'src/trotterlib/df_rte_circuit.py',
    'src/trotterlib/df_rte_qiskit.py',
    'src/trotterlib/df_trotter/ops.py',
    'src/trotterlib/rpe_hadamard_interrogation.py',
    'src/trottertracks/resource_applicability/ax2a_native_df.py',
)
OWN = ('src/trotterlib/b2_p1_compiled_cost.py', 'scripts/run_b2_p1_compiled_cost.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n')


def saved(commit, path):
    raw = git('show', commit+':'+path)
    return json.loads(raw), dict(commit=commit, path=path,
        blob=git('rev-parse', commit+':'+path).decode().strip(),
        sha256=hashlib.sha256(raw).hexdigest())


def compiler_from_saved(record, runtime_version):
    """Read the actual prior price profile, never the signal context profile."""
    fields = dict(record['compiler'])
    fields['basis_gates'] = tuple(fields['basis_gates'])
    require(fields['coupling_map'] is None and fields['backend_name'] is None,
            'Prior logical all-to-all profile required; no backend fallback')
    compiler = CompilerSettings(**fields)
    require(compiler.qiskit_version == runtime_version, 'Actual Qiskit version differs')
    return compiler


def fixed_events(round_, prior):
    """Replay prior controlled_checks outer_step_cases[1], without sampling."""
    representatives = prior['controlled_correctness']['representative_events']
    negative = next(i for i, e in enumerate(representatives)
                    if e['taylor_order'] == 2 and e['phase_real'] == -1
                    and e['phase_imag'] == 0)
    selection = (0, negative, len(representatives)-1, 0)
    components = round_.tail.components
    lookup = {c.component_id: i for i, c in enumerate(components)}
    events = []
    for i in selection:
        row = representatives[i]
        event = _make_event(tuple(lookup[c] for c in row['component_ids']), components,
            round_.distribution, round_.distribution.orders.index(row['taylor_order']))
        require(complex(event.phase) == complex(row['phase_real'], row['phase_imag']),
                'Saved representative phase differs')
        events.append(event)
    return tuple(events), dict(prior_case=1, representative_indices=selection,
        rule='existing controlled_checks case1: first, first negative order2, last, first',
        random_sampling=False, price_based_selection=False)


def run(private_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(schema_version='b2_p1_explicit_compiled_cost_v1', status='PREFLIGHT',
        started_utc=datetime.now(timezone.utc).isoformat(), mandatory_stop=True,
        fixed_conditions=dict(target='H_phys', model='linear H4; 1.00 Angstrom; STO-3G',
            DF_rank=12, L_D=3, r=4, K=2, delta=.02, m=0, Q=1,
            state_role='original_H_ground_state', state_preparation_included=False),
        axes={}, expected_trajectory_cost=None, price_MC=None, total_rpe_cost=None,
        guarantee_status='not_certified', compile_calls_started=0)
    try:
        source = git('rev-parse', 'HEAD').decode().strip()
        source_hashes = {}
        for name in (*CORE, *OWN):
            raw = (ROOT/name).read_bytes()
            require(raw == git('show', source+':'+name), 'Uncommitted SOURCE: '+name)
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
        result['source'] = dict(commit=source,
            tree=git('rev-parse', source+'^{tree}').decode().strip(), sha256=source_hashes)
        prior, correctness_identity = saved(CORRECT_COMMIT, CORRECT_PATH)
        spec, spec_identity = saved(SPEC_COMMIT, SPEC_PATH)
        old_price, price_identity = saved(PRICE_COMMIT, PRICE_PATH)
        require(old_price['status'] == 'FIXED_R8_PRICE_PILOT_COMPLETE',
                'Prior compiler price evidence is incomplete')
        checks = prior['controlled_correctness']['checks']
        require(prior['controlled_correctness']['status'] == 'PASS'
                and len(checks) == 16 and all(c['passed'] for c in checks),
                'Representative controlled correctness is ineligible')
        for name in CORE:
            require((ROOT/name).read_bytes() == git('show', prior['execution_SOURCE']+':'+name),
                    'Native correctness SOURCE differs: '+name)
        require((ROOT/'src/trotterlib/rte_compiled_cost.py').read_bytes() ==
                git('show', old_price['source']['head']+':src/trotterlib/rte_compiled_cost.py'),
                'Prior price compiler wrapper SOURCE differs')
        versions = {p: importlib.metadata.version(p) for p in
                    ('qiskit', 'numpy', 'scipy', 'openfermion', 'qiskit-nature')}
        compiler = compiler_from_saved(old_price, versions['qiskit'])
        provider = MatchedNoPrepCostProvider(compiler)
        require(provider.backend_hash == 'no_backend', 'Backend context differs')
        prior_cost = old_price['arms']['DF']['price']
        # The saved price's metadata establishes actual full-wrapper scope/backend.
        require(prior_cost['circuit_cost_scope'] == RPE_HADAMARD_INTERROGATION_SCOPE
                and dict(prior_cost['metadata'])['backend_hash'] == provider.backend_hash,
                'Prior measured price scope/backend differs')
        for axis in ('cosine', 'sine'):
            require(prior_cost[axis+'_expected_cost']['compiler'] == json.loads(json.dumps(asdict(compiler))),
                    'Actual saved axis compiler differs')
        files = {name: Path(private_dir)/name for name in prior['input_identity']['file_SHA256']}
        private_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in files.items()}
        require(private_hashes == prior['input_identity']['file_SHA256'], 'Private input identity differs')
        for name in ('original_hamiltonian', 'original_state', 'df_hamiltonian',
                     'sector', 'spin_orbital_order', 'state_role'):
            require(spec[name] == prior['input_identity'][name], 'Saved identity differs: '+name)
        df_raw = git('show', DF_COMMIT+':'+DF_PATH)
        require(hashlib.sha256(df_raw).hexdigest() == DF_SHA, 'DF snapshot identity differs')
        result.update(input=dict(df_commit=DF_COMMIT, df_path=DF_PATH, df_sha256=DF_SHA,
            private_file_sha256=private_hashes,
            scientific_identity={k: spec[k] for k in ('original_hamiltonian', 'original_state',
                'df_hamiltonian', 'sector', 'spin_orbital_order', 'state_role')},
            state_vectors_used=False, physical_H_rebuilt=False),
            reused_evidence=dict(correctness=correctness_identity, input_spec=spec_identity,
                                 measured_compiler=price_identity),
            compiler=dict(settings=asdict(compiler), settings_hash=compiler_settings_hash(compiler),
                backend_fingerprint=provider.backend_hash, target=None,
                connectivity='all_to_all_logical', signal_seed_17_adopted=False),
            runtime=dict(python=sys.version, packages=versions, host=platform.node(),
                available_cpus=len(os.sched_getaffinity(0)), load_average=os.getloadavg()),
            preflight_status='PASS')
        require(platform.node() == 'GPUA100-gleap02', 'Expected SERVER 2')
        dump(output/'preflight.json', result)
        # No molecular build has run above. Load the existing DF; discard its state.
        with tempfile.TemporaryDirectory(prefix='b2-cost-df-') as temp:
            path = Path(temp)/'snapshot.npz'; path.write_bytes(df_raw)
            ham, sector, unused_df_state, metadata = load_snapshot(path)
            del unused_df_state
        require(ham.n_blocks == 12 and metadata['hamiltonian_hash'] == spec['df_hamiltonian']
                and metadata['fragment_order'] == list(range(12)), 'DF scientific input differs')
        preparation = connection.prepare_b2(ham, L_D=3)
        config, distribution = configure_law(SimpleNamespace(
            tail=preparation.rte_preparation.symbolic_tail), delta_time=.02, r=4, K=2)
        law_prior = prior['normalization_law']
        require(config.lambda_r == law_prior['lambda_R'] and
                config.tail_hash == law_prior['tail_hash'] and
                preparation.preparation_hash == law_prior['preparation_hash'] and
                preparation.extracted_identity_coefficient == law_prior['tail_identity'],
                'Canonical preparation/law/identity differs')
        context = connection.RPEConnectionContext(spec['original_hamiltonian'], spec['df_hamiltonian'],
            spec['original_state'], spec['sector'], spec['spin_orbital_order'], source,
            compiler, provider.backend_hash, 'rz_count', None, None, None, None)
        round_ = connection.B2RPERound(context, preparation, RPERoundSpecification(0,.02),
            3,4,2,.02,1,config,distribution)
        events, selection = fixed_events(round_, prior)
        request = round_.explicit_request(events)
        del request  # Validation only; the adapter reuses the same explicit replay.
        result['trajectory_selection'] = selection
        result['law'] = dict(config=asdict(config), distribution=asdict(distribution),
            preparation_hash=preparation.preparation_hash,
            scalar=preparation.constant_coefficient,
            tail_identity=preparation.extracted_identity_coefficient)
        dump(output/'trajectory.json', dict(selection=selection,
            canonical_events=[e.to_dict() for e in events], round_binding=round_.binding))
        def record_compile(circuit, settings, **kwargs):
            require(result['compile_calls_started'] < 2, 'Only two fixed compiler calls')
            axis = ('cosine', 'sine')[result['compile_calls_started']]
            require(circuit.num_qubits == 9 and circuit.num_clbits == 1
                and circuit.count_ops().get('measure') == 1
                and circuit.data[-1].operation.name == 'measure', 'Full measured wrapper scope')
            result['compile_calls_started'] += 1
            tick = time.perf_counter()
            cost = transpile_and_measure_cost(circuit, settings, **kwargs)
            compiled = cost.transpiled_circuit
            require(cost.backend_fingerprint == provider.backend_hash and
                    cost.compiler_settings_hash == compiler_settings_hash(compiler) and
                    cost.actual_circuit_fingerprint == canonical_qiskit_circuit_fingerprint(circuit),
                    'Compiled context/identity differs')
            require(compiled.count_ops().get('measure') == 1 and compiled.num_clbits == 1,
                    'Compiled measurement scope differs')
            record = dict(axis=axis, SOURCE=source, round_binding=round_.binding,
                scope=context.cost_scope, state_preparation_included=False,
                compiler_hash=cost.compiler_settings_hash, backend_fingerprint=cost.backend_fingerprint,
                actual_circuit_fingerprint=cost.actual_circuit_fingerprint,
                compiled_circuit_fingerprint=canonical_qiskit_circuit_fingerprint(compiled),
                metrics={k: getattr(cost,k) for k in RPE_COST_METRICS},
                pretranspile_gate_counts=dict(cost.pretranspile_gate_counts),
                posttranspile_gate_counts=dict(cost.posttranspile_gate_counts),
                qubits=cost.qubit_count, compiled_global_phase=cost.global_phase,
                elapsed_seconds=time.perf_counter()-tick,
                process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            dump(output/(axis+'.json'), record)
            result['axes'][axis] = record
            print(axis+': '+str(record['metrics']), flush=True)
            return cost
        provider.compile_function = record_compile
        result['price'] = price_explicit_trajectory(round_, provider, events)
        for name, expected in source_hashes.items():
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,
                    'Consumed SOURCE changed: '+name)
        require({name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
                == private_hashes, 'Consumed private inputs changed')
        result['status'] = 'PASS_SPECIFIED_TRAJECTORY_M0_ONLY'
    except Exception as exc:
        result.update(status='STOP_INELIGIBLE_OR_FAILED', error=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds'] = time.perf_counter()-started
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    dump(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.private_dir, args.output)
    print(result['status'], result.get('error', ''), flush=True)
    sys.exit(0 if result['status'] == 'PASS_SPECIFIED_TRAJECTORY_M0_ONLY' else 1)
