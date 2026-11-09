"""One finite Q3-N audit; no trajectory, circuit, phase, RPE or price calls.

Fixed scope: ba616902 NEXT_VALIDATION_PLAN sections 1--4. Numeric entry is
main() only. Imports and artificial helper tests do not load the science cache.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import lzma
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'ba6169028d6d71105aa1f474dfcd285461339070'
BASE = 'f2638a7845685a119619382b850441f5d8d33713'
PROVIDER = 'eaf70d886d42364a01b7e949ebc21f3b026bbe72'
SELF = 'scripts/run_pr_reference_v6q3_n.py'
CHECKER = 'scripts/check_pr_reference_v6q3_n.py'
CACHE = 'artifacts/pr_reference_interference_pilot/2026-11-06/common.pkl.xz'
A1 = 'artifacts/pr_reference_interference_pilot/2026-11-06/A1_REFERENCE_PHASE.json'
V2 = 'artifacts/pr_reference_bprime/2026-11-07/V2_INPUTS_CONSTANTS.json'
PACKED_SHA = '85d7e21a41e770596b92327d909f38f0ae091c875e2dad3651abeb660f26ae16'
PICKLE_SHA = '68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2'
SOURCE_PATHS = ('src/trottertracks/track_c_h4.py',
    'src/trotterlib/df_trotter/ops.py', 'src/trotterlib/df_partial_s2.py',
    'src/trotterlib/df_partial_s2_repeated.py', 'src/trotterlib/df_rte_qiskit.py',
    'src/trotterlib/df_rte_tail.py', 'src/trotterlib/rte.py')
TARGET = dict(m=17, q=131072, r=32, bits=88)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git_bytes(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stored_literal(value):
    """Type and stored value only; no basis exponent audit."""
    kind = type(value).__module__+'.'+type(value).__qualname__
    if isinstance(value, float):
        return dict(type=kind, hex=float.__float__(value).hex())
    if type(value) is int:
        return dict(type=kind, integer=str(value))
    raise TypeError('Unsupported binary input type: '+kind)


def generator(key, role, value, binary_input, **metadata):
    f = binary_input(value)
    denominator = f.denominator
    require(denominator > 0 and denominator & (denominator-1) == 0, 'Non-dyadic generator')
    k = denominator.bit_length()-1
    return dict(key=key, role=role, literal=stored_literal(value),
        numerator=str(f.numerator), denominator_power=k,
        required_ell=k+(role == 'angle'), **metadata)


def audit_basis(label, operations, described, binary_input, classes):
    require(len(operations) == len(described), 'Basis metadata length: '+label)
    rows = []
    for index, ((gate, wires), meta) in enumerate(zip(operations, described, strict=True)):
        name = gate.name
        require(name in classes and type(gate) is classes[name], 'Non-genuine basis gate: '+label)
        width, arity = (1, 1) if name == 'rz' else (2, 2)
        require(gate.num_qubits == len(wires) == width and len(gate.params) == arity,
                'Basis gate shape: '+label)
        require(gate.num_clbits == 0 and getattr(gate, 'condition', None) is None,
                'Classical basis operation: '+label)
        require(len(set(wires)) == width and all(type(w) is int and 0 <= w < 8 for w in wires),
                'Basis wire map: '+label)
        require(meta.name == name and tuple(meta.qubits) == tuple(wires)
                and tuple(meta.parameters) == tuple(str(p) for p in gate.params),
                'Saved basis metadata mismatch: '+label)
        for value in gate.params:
            binary_input(value)  # Reuse the unchanged provider acceptance rule.
        rows.append(dict(index=index, name=name, gate_class=type(gate).__module__+'.'+type(gate).__qualname__,
            wires=list(wires), width=width, num_clbits=0, condition=None,
            parameters=[stored_literal(p) for p in gate.params], saved_metadata_matches=True))
    return dict(label=label, operations=rows, parameter_count=sum(len(x['parameters']) for x in rows))


def lattice_result(records):
    require(bool(records), 'Empty generator pool')
    ell = max(x['required_ell'] for x in records)
    return dict(ell=ell, normal_exponent_limit=1022, sufficient=ell <= 1022,
        maximum_keys=[x['key'] for x in records if x['required_ell'] == ell],
        exact_nonzero_lower_bound='2^-'+str(ell), exact_upper_bound='2*M_saved',
        basis_angles_cancel_by_SOURCE_pairing=True,
        normalization_model='inherited correctly-rounded binary64 add/sub/remainder; finite updates',
        runtime_wheel_formally_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (SELF, CHECKER):
        require((ROOT/path).read_bytes() == git_bytes(args.source_commit, path), 'Execution SOURCE changed: '+path)
    bindings = {}
    for path in (*SOURCE_PATHS, 'src/trottertracks/pr_reference_v6q.py'):
        ref = PROVIDER if path.endswith('/pr_reference_v6q.py') else BASE
        raw = git_bytes(ref, path)
        require((ROOT/path).read_bytes() == raw, 'Frozen SOURCE changed: '+path)
        bindings[path] = dict(ref=ref, sha256=sha(raw))
    require(not args.output.exists(), 'Fresh output required; do not repeat a campaign')
    args.output.mkdir(parents=True)
    (args.output/'.gitignore').write_text('/common.pkl\n')
    calls = dict(cache_loads=0, diagonal_primitive_calls=0, scalar_evaluations=0,
        saved_central_angle_reads=0, provider_phase_calls=0, new_atan_calls=0,
        trajectory_circuit_compile_simulation_sampling_RPE_RZ_CT_price_calls=0)
    progress = dict(stage='inputs', calls=calls, basis_catalog=[], generator_pool=[])
    start = time.monotonic()
    try:
        source = dict(execution_commit=args.source_commit, scope_commit=PLAN,
            scientific_input_commit=BASE, provider_commit=PROVIDER, target=TARGET,
            driver_sha256=sha(Path(__file__).read_bytes()), checker_sha256=sha((ROOT/CHECKER).read_bytes()),
            source_bindings=bindings, cache_archive_path=CACHE, cache_archive_sha256=PACKED_SHA,
            cache_pickle_sha256=PICKLE_SHA, arithmetic_model='inherited correctly-rounded binary64 add/sub/remainder',
            python=sys.version, executable=sys.executable, host=os.uname().nodename,
            affinity_cpus=len(os.sched_getaffinity(0)), memory_observation=Path('/proc/meminfo').read_text(),
            cpu_pressure=Path('/proc/pressure/cpu').read_text(), memory_pressure=Path('/proc/pressure/memory').read_text())
        packed = git_bytes(BASE, CACHE)
        require(sha(packed) == PACKED_SHA, 'Cache archive hash')
        raw = lzma.decompress(packed)
        require(sha(raw) == PICKLE_SHA, 'Cache pickle hash')
        cache = args.output/'common.pkl'
        cache.write_bytes(raw)
        inputs = {name:git_bytes(BASE, path) for name, path in [('A1', A1), ('V2', V2)]}
        source['saved_input_hashes'] = {name:sha(value) for name, value in inputs.items()}
        save(args.output/'SOURCE.json', source)
        sys.path.insert(0, str(ROOT/'src'))
        os.environ['TRACK_C_H4_COMMON_CACHE'] = str(cache.resolve())
        from trottertracks.track_c_h4 import common, diagonal_primitives
        from trottertracks.pr_reference_v6q import _binary_input
        from qiskit.circuit.library import RZGate, XXPlusYYGate
        require(version('qiskit') == '1.3.0', 'Fixed Qiskit runtime')
        require(common.cache_info().currsize == 0, 'Loader used before campaign')
        calls['cache_loads'] += 1
        prep = common()['preps'][3]
        require(prep.num_system_qubits == 8 and prep.ld == 3 and len(prep.deterministic_blocks) == 4,
                'Fixed H4/LD3 preparation')
        a1, witness = (json.loads(inputs[name]) for name in ('A1', 'V2'))
        require(a1['all_passed'] and a1['basis_families'] == ['rz', 'xx_plus_yy'], 'Saved A1 qualification')
        expected_d = {x['index'] for x in a1['checks'] if x['kind'] == 'D_basis'}
        expected_r = {x['basis_id'] for x in a1['checks'] if x['kind'] == 'R_basis'}
        registry = prep.rte_preparation.basis_registry._definitions
        require(expected_d == set(range(4)) and expected_r == set(registry), 'Complete A1 pool coverage')
        progress['stage'] = 'N1_basis_metadata'
        classes = dict(rz=RZGate, xx_plus_yy=XXPlusYYGate)
        for index, block in enumerate(prep.deterministic_blocks):
            progress['basis_catalog'].append(dict(kind='D_basis', index=index, basis_id=block.basis_id,
                basis_hash=block.basis_hash, **audit_basis('D'+str(index), block.runtime_basis_operations,
                block.basis_change_operations, _binary_input, classes)))
        for bid, definition in registry.items():
            require(bid == definition.basis_id and definition.num_system_qubits == 8, 'Registry metadata')
            progress['basis_catalog'].append(dict(kind='R_basis', basis_id=bid, basis_hash=definition.basis_hash,
                **audit_basis(bid, definition.runtime_operations, definition.metadata.operations, _binary_input, classes)))
        components = prep.rte_preparation.symbolic_tail.components
        require(all(not c.is_identity and c.basis_id in registry and c.diagonal_pauli_support is not None
            and len(c.diagonal_pauli_support) in (1, 2) for c in components), 'Fixed tail support/no identity')
        component_bindings = []
        for component in components:
            definition = registry[component.basis_id]
            require(component.basis_hash == definition.basis_hash and
                component.basis_change_operations == definition.metadata.operations, 'Tail component basis binding')
            component_bindings.append(dict(component_id=component.component_id, basis_id=component.basis_id,
                basis_hash=component.basis_hash, support=list(component.diagonal_pauli_support)))
        save(args.output/'N1_CATALOG.json', dict(bases=progress['basis_catalog'], components=component_bindings,
            A1_D_indices=sorted(expected_d), A1_R_basis_ids=sorted(expected_r),
            all_supported_finite=True, all_saved_metadata_matches=True, basis_exponents_evaluated=False))
        progress['stage'] = 'N2_generators'
        contexts = [(i, .01) for i in range(4)]+[(0, .02)]
        for index, step in contexts:
            key = 'D'+str(index)+'@'+step.hex()
            calls['diagonal_primitive_calls'] += 1
            primitive = diagonal_primitives(prep.deterministic_blocks[index], step)
            progress['generator_pool'].append(generator(key+'/global', 'phase', primitive.global_phase, _binary_input))
            for qubit, angle in primitive.rz:
                progress['generator_pool'].append(generator(key+'/rz/'+str(qubit), 'angle', angle, _binary_input))
            for left, right, angle in primitive.rzz:
                progress['generator_pool'].append(generator(key+'/rzz/'+str(left)+'/'+str(right), 'angle', angle, _binary_input))
        calls['scalar_evaluations'] += 1
        n = TARGET['q']//2
        scalar = -.02*n*(prep.constant_coefficient+prep.extracted_identity_coefficient)
        progress['generator_pool'].append(generator('scalar/n='+str(n), 'phase', scalar, _binary_input,
            expression='-.02*n*(constant_coefficient+extracted_identity_coefficient)', n=n))
        require(witness['all_angles_equal'] and witness['lambda_prep_hex'] == float(prep.exact_rte_lambda_r).hex()
            == witness['lambda_tail_hex'], 'Saved central angle SOURCE binding')
        for order in (0, 2):
            selected = [x for x in witness['angles'] if x['r'] == 32 and x['order'] == order]
            require(len(selected) == 1, 'Unique saved central angle')
            row = selected[0]
            require(row['exact_equal'] and row['actual_hex'] == row['old_interval_hex'], 'Saved angle witness mismatch')
            calls['saved_central_angle_reads'] += 1
            progress['generator_pool'].append(generator('RTE/r32/order'+str(order), 'angle',
                float.fromhex(row['actual_hex']), _binary_input, saved_witness=row, signs_covered=[-1, 1]))
        require(math.tau.hex() == (2.*math.pi).hex(), 'Fixed binary normalizer period')
        for key, value in [('zero', 0.), ('pi', math.pi), ('tau', math.tau)]:
            progress['generator_pool'].append(generator(key, 'phase', value, _binary_input))
        grid = lattice_result(progress['generator_pool'])
        save(args.output/'N2_GENERATORS.json', dict(contexts=[dict(block=i, time_hex=t.hex()) for i,t in contexts],
            records=progress['generator_pool'], lattice=grid, scalar_constants=dict(
                constant=stored_literal(prep.constant_coefficient), extracted_identity=stored_literal(prep.extracted_identity_coefficient))))
        status = 'PASS_IN_FIXED_ARITHMETIC_MODEL' if grid['sufficient'] else 'INCONCLUSIVE_LATTICE'
        result = dict(status=status, target=TARGET, N1_basis_count=len(progress['basis_catalog']),
            N1_operation_count=sum(len(b['operations']) for b in progress['basis_catalog']),
            N1_parameter_types=dict(Counter(p['type'] for b in progress['basis_catalog'] for op in b['operations'] for p in op['parameters'])),
            N1_component_count=len(components), N2_generator_count=len(progress['generator_pool']),
            lattice=grid, calls=calls, old_M_and_Bnorm_preserved=True, postrun_N3_required=True,
            actual_m17_caller_connected=False, V6P_cost_promoted=False,
            wall_seconds=time.monotonic()-start, next_action='MANDATORY_STOP')
        save(args.output/'RESULTS.json', result)
        save(args.output/'STOP.json', dict(status=status, scientific_campaigns=1, scientific_retries=0,
            N1_N2_complete=True, calls=calls, next_action='MANDATORY_STOP'))
        print(json.dumps(result, indent=2))
    except Exception as exc:
        save(args.output/'PARTIAL.json', progress)
        save(args.output/'STOP.json', dict(status='FAIL_INPUT_OR_SOURCE', scientific_campaigns=1,
            scientific_retries=0, stage=progress['stage'], calls=calls,
            error_type=type(exc).__name__, error=str(exc), next_action='MANDATORY_STOP'))
        raise


if __name__ == '__main__':
    main()
