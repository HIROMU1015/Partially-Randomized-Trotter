"""H4-P contract/controller. Numerical imports live only in the granted port.

Preparing orbital basis operations is permitted only by a separate H4-P grant.
This entry point never evaluates a signal, builds a wrapper or seals science.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .ax2a_preparation import digest
from .ax2b_bound_launch_v2 import (
    environment, safe_path, scope_plan, source_paths as bound_source_paths,
    verify_input, verify_sources as verify_bound_sources,
)
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_h6_controller import BoundedWriter
from .ax2b_limits import CallBudget

RUNNER = 'scripts/resource_applicability/run_track_a_ax2b_h4_native_receipt_v1.py'
STATIC = 'artifacts/resource_applicability/track_a_ax2b_h4_prelaunch_preparation/2026-10-10/metadata_preflight_v3.json'
STATIC_SHA = '3edf7cc598bcdad148b258bae8cba210e3706a430198c50672c6d9bd2c923753'
STATIC_COMMIT = 'f0a5021b8f97dd9116578c7c3e455b7bc2559792'
OUTPUT = 'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1'
KIND = 'H4_NATIVE_RECEIPT'
FORBIDDEN = ('signal', 'reference', 'matvec_probe', 'sampling', 'wrapper_build',
             'compile', 'solver', 'input_generation', 'H6', 'H8', 'gpu')


def bounded_json(path, maximum=4*2**20):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('JSON_INPUT_SIZE')
    return json.loads(path.read_text(encoding='utf-8'))


def plan():
    return {'kind': KIND, 'T': .8, 'cells': scope_plan('H4_LIMITED')['cells'],
            'caps': {'total_wall_seconds': 900, 'address_space_bytes': 8*2**30,
                     'output_bytes': 16*2**20, 'log_bytes': 65536, 'diagnostics': 128,
                     'snapshot_loads': 1, 'native_preparation_calls': 8,
                     'primitive': 2000, 'untranspiled_instructions': 1000000},
            'forbidden_calls': dict.fromkeys(FORBIDDEN, 0), 'output_path': OUTPUT,
            'retry': False, 'resume': False, 'science_manifest_sealed': False,
            'N': None, 'G': None, 'numerical_allowance_certified': False,
            'accuracy_eligibility': 'UNDETERMINED', 'mandatory_stop': True,
            'next_stage_authorized': False, 'H6_status': 'H6_NOT_AUTHORIZED',
            'contract_status': 'DRAFT_NOT_AUTHORIZATION'}


def draft():
    return {'schema': 'track_a_ax2b_h4_native_preparation_v1', 'kind': KIND,
            'status': 'H4_NATIVE_RECEIPT_NOT_AUTHORIZED', 'plan': plan(),
            'source_commit': None, 'source_hashes': None, 'environment': None,
            'input_binding': None, 'static_binding': None, 'assigned_resources': None,
            'execution_plan_sealed': False, 'science_authorized': False,
            'launch_allowed': False, 'mandatory_stop': True,
            'contract_status': 'DRAFT_NOT_AUTHORIZATION'}


def source_paths(root):
    return sorted(set(bound_source_paths(root)) | {RUNNER})


def verify_sources(root, commit, hashes):
    if not isinstance(hashes, dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('H4P_SOURCE_CLOSURE')
    verify_bound_sources(root, commit, {k: v for k, v in hashes.items() if k != RUNNER})
    if file_hash(safe_path(root, RUNNER)) != hashes[RUNNER]:
        raise ValueError('LOCAL_H4P_RUNNER_CHANGED')
    blob = subprocess.check_output(['git', 'show', commit+':'+RUNNER], cwd=root)
    if hashlib.sha256(blob).hexdigest() != hashes[RUNNER]:
        raise ValueError('H4P_RUNNER_COMMIT_BYTES')


def static_receipt(root, binding):
    expected = {'path': STATIC, 'sha256': STATIC_SHA, 'commit': STATIC_COMMIT}
    if binding != expected or file_hash(safe_path(root, STATIC)) != STATIC_SHA:
        raise ValueError('STATIC_RECEIPT_BINDING')
    blob = subprocess.check_output(['git', 'show', STATIC_COMMIT+':'+STATIC], cwd=root)
    if hashlib.sha256(blob).hexdigest() != STATIC_SHA:
        raise ValueError('STATIC_RECEIPT_COMMIT_BYTES')
    receipt = bounded_json(safe_path(root, STATIC))
    if receipt['manifest']['plan']['cells'] != plan()['cells']:
        raise ValueError('STATIC_CELLS_CHANGED')
    return receipt


def bind_metadata(root, commit, cpu):
    """Metadata-only preparation: no payload decode or native basis work."""
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
        raise ValueError('ASSIGNED_CPU_UNAVAILABLE')
    result = draft()
    result['source_commit'] = commit
    result['source_hashes'] = {p: file_hash(safe_path(root, p)) for p in source_paths(root)}
    verify_sources(root, commit, result['source_hashes'])
    result['static_binding'] = {'path': STATIC, 'sha256': STATIC_SHA, 'commit': STATIC_COMMIT}
    result['input_binding'] = static_receipt(root, result['static_binding'])['manifest']['input_binding']
    verify_input(root, result['input_binding'], 'H4_LIMITED')
    result['environment'] = environment()
    result['assigned_resources'] = {'assigned_cpu': cpu, 'science_workers': 1, 'blas_threads': 1}
    if safe_path(root, OUTPUT).exists():
        raise ValueError('EXCLUSIVE_OUTPUT_REQUIRED')
    # Only the receipt acquisition plan is sealed; H4 science remains unsealed.
    result['execution_plan_sealed'] = True
    return result


def validate_launch(root, manifest, authorization, *, requested, output, worker=False):
    # Reject before source/input/environment I/O and before numerical imports.
    if requested is not True or not isinstance(authorization, dict) or authorization.get('approved_by_user') is not True:
        raise ValueError('SEPARATE_EXPLICIT_H4P_USER_GRANT_REQUIRED')
    if authorization.get('schema') != 'track_a_ax2b_h4_native_authorization_v1' or authorization.get('kind') != KIND:
        raise ValueError('H4P_AUTHORIZATION_SCHEMA')
    if authorization.get('manifest_digest') != digest(manifest):
        raise ValueError('H4P_AUTHORIZATION_MANIFEST_BINDING')
    if (manifest.get('schema') != 'track_a_ax2b_h4_native_preparation_v1'
            or manifest.get('kind') != KIND or manifest.get('plan') != plan()
            or manifest.get('status') != 'H4_NATIVE_RECEIPT_NOT_AUTHORIZED'
            or manifest.get('execution_plan_sealed') is not True
            or manifest.get('science_authorized') is not False
            or manifest.get('launch_allowed') is not False
            or manifest.get('mandatory_stop') is not True
            or manifest.get('contract_status') != 'DRAFT_NOT_AUTHORIZATION'):
        raise ValueError('SEALED_H4P_PLAN_REQUIRED_NOT_AUTHORIZATION')
    cpu = authorization.get('assigned_cpu')
    if (type(cpu) is not int or cpu not in os.sched_getaffinity(0)
            or manifest.get('assigned_resources') != {'assigned_cpu': cpu, 'science_workers': 1, 'blas_threads': 1}):
        raise ValueError('ASSIGNED_RESOURCES_REQUIRED')
    fixed = safe_path(root, OUTPUT)
    if Path(output).resolve() != fixed or authorization.get('exclusive_output') != str(fixed):
        raise ValueError('FIXED_EXCLUSIVE_OUTPUT_REQUIRED')
    if authorization.get('retry') is not False or authorization.get('resume') is not False:
        raise ValueError('NO_RETRY_OR_RESUME')
    if not worker and fixed.exists():
        raise ValueError('EXCLUSIVE_OUTPUT_REQUIRED')
    if worker:
        if not fixed.is_dir() or bounded_json(fixed/'launch_binding.json', 8192) != {
                'manifest_digest': digest(manifest), 'authorization_digest': digest(authorization)}:
            raise ValueError('WORKER_LAUNCH_BINDING')
        if (fixed/'worker_claim.json').exists():
            raise ValueError('ONE_SHOT_WORKER_ALREADY_CLAIMED')
    verify_sources(root, manifest.get('source_commit'), manifest.get('source_hashes'))
    verify_input(root, manifest.get('input_binding'), 'H4_LIMITED')
    if manifest.get('environment') != environment():
        raise ValueError('ENVIRONMENT_CHANGED')
    static_receipt(root, manifest.get('static_binding'))
    return cpu


class ProductionPort:
    """Instantiated only inside the limited, granted worker. No science actions."""
    synthetic_only = False

    def __init__(self, root, manifest):
        # Imports (including Qiskit) happen after AS/affinity/thread gates.
        from trotterlib.pr2_new_series_validation import _load_snapshot_once
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
        from .ax2b_molecular_ports_v2 import actual_bounds, check_bounds
        self.load_snapshot = _load_snapshot_once
        self.prepare_native, self.prepare_discard = _prepare, _prepare_discard
        self.actual_bounds, self.check_bounds = actual_bounds, check_bounds
        self.root, self.manifest = Path(root), manifest

    def load(self):
        ham, sector, full, state, metadata, layout = self.load_snapshot(
            safe_path(self.root, self.manifest['input_binding']['path']))
        if (ham.n_qubits, ham.n_blocks, sector.dimension) != (8, 12, 36):
            raise ValueError('H4_SAVED_INPUT_SCOPE')
        # Preserve the saved input; no Hamiltonian/state regeneration or solver.
        return ham

    def prepare(self, ham, cell):
        from dataclasses import asdict
        from .ax2b_native_df_v5 import block_instruction_bound
        prep = (self.prepare_discard if cell['method'] == 'B0' else self.prepare_native)(ham, cell['prefix'])
        blocks = []
        for block in prep.deterministic_blocks:
            one = block.original_fragment_index is None
            blocks.append({'primitive_id': 'one' if one else str(block.original_fragment_index),
                'block_id': block.block_id, 'basis_id': block.basis_id, 'basis_hash': block.basis_hash,
                'n_qubits': block.num_system_qubits, 'order_index': block.order_index,
                'basis_operations': [asdict(op) for op in block.basis_change_operations],
                'runtime_basis_operation_count': len(block.runtime_basis_operations),
                'diagonal_hex': [float(x).hex() for x in (block.diagonal_eigenvalues if one else block.diagonal_eta)],
                'lambda_hex': None if one else float(block.lam).hex(),
                'instruction_bounds': {mode: block_instruction_bound(block, mode)
                    for mode in ('UNCONTROLLED', 'ORDINARY', 'DIRECTIONAL')}})
        specs = [asdict(spec) for spec in prep.rte_preparation.component_specs]
        bounds = self.actual_bounds([cell], {cell['id']: prep}, T=.8)
        self.check_bounds(bounds, plan()['caps'])
        return {'cell_id': cell['id'], 'preparation_hash': prep.preparation_hash,
            'hamiltonian_hash': prep.hamiltonian_hash, 'partition_hash': prep.partition_hash,
            'ld': prep.ld, 'constant_hex': float(prep.constant_coefficient).hex(),
            'extracted_identity_hex': float(prep.extracted_identity_coefficient).hex(),
            'lambda_r_hex': float(prep.exact_rte_lambda_r).hex(),
            'identity_policy': prep.identity_policy, 'coefficient_atol': prep.coefficient_atol,
            'deterministic_blocks': blocks, 'component_spec_count': len(specs),
            'component_specs_digest': digest(specs), 'component_specs': specs,
            'bounds_row': bounds['cells'][0]}


def assemble_bounds(records, static):
    """Combine frozen per-cell formulas without retaining all prepared Gates."""
    expected = static['static_coverage']
    rows, probes, identities = [], set(), {}
    if len(records) != 8 or [r['cell_id'] for r in records] != [c['id'] for c in plan()['cells']]:
        raise ValueError('EXACT_EIGHT_CELL_RECEIPTS_REQUIRED')
    for cell, record, old in zip(plan()['cells'], records, expected['cells']):
        row = record['bounds_row']
        schedule = dict(row['schedule'])
        times = schedule.pop('registered_validation_times_v2')
        if (row['cell_id'] != cell['id'] or old['cell_id'] != cell['id']
                or digest(schedule) != digest(old['schedule'])
                or digest(times) != digest(old['registered_validation_times_v2'])):
            raise ValueError('STATIC_NATIVE_COVERAGE_MISMATCH:'+cell['id'])
        blocks = record['deterministic_blocks']
        if len(blocks) != cell['prefix']+1:
            raise ValueError('NATIVE_BLOCK_COUNT')
        for block in blocks:
            if block['n_qubits'] != 8:
                raise ValueError('NATIVE_REGISTER')
            key, value = block['primitive_id'], digest(block)
            # B0 uses the truncated Hamiltonian, but deterministic blocks must agree.
            if key in identities and identities[key] != value:
                raise ValueError('NATIVE_BLOCK_IDENTITY_CHANGED:'+key)
            identities[key] = value
        for i, t in times:
            probes.add((blocks[i]['primitive_id'], t))
        b = row['wrapper_instruction_upper_bounds']
        if (set(b) != {'ordinary', 'symmetric_directional'}
                or any(type(v) is not int or not 0 < v <= plan()['caps']['untranspiled_instructions'] for v in b.values())):
            raise ValueError('STRUCTURAL_INSTRUCTION_BOUND_CAP')
        rows.append(row)
    if digest(sorted(probes)) != digest(expected['unique_primitive_time_pairs']):
        raise ValueError('EXACT_PRIMITIVE_TIME_SET_MISMATCH')
    if 3*len(probes) != 537 or 3*len(probes) > plan()['caps']['primitive']:
        raise ValueError('PRIMITIVE_COVERAGE_CAP')
    return {'primitive_actions': 3*len(probes), 'primitive_probe_count': 3,
            'cells': rows, 'policy': 'all actual unique times x saved state/first/last sector columns',
            'structural_bounds_are_not_compiled_costs': True}


def run_receipt(port, writer, manifest, static):
    """One saved-input load, exactly eight preparations; retain failures/partials."""
    calls = CallBudget(snapshot_loads=1, native_preparation_calls=8, **dict.fromkeys(FORBIDDEN, 0))
    records, reason, receipt = [], None, None
    try:
        calls.take('snapshot_loads')
        ham = port.load()
        for cell in plan()['cells']:
            calls.take('native_preparation_calls')
            record = port.prepare(ham, cell)
            if record.get('cell_id') != cell['id']:
                raise ValueError('PREPARATION_CELL_BINDING')
            writer.write(cell['id']+'_native.json', record, diagnostic=True)
            records.append(record)
        bounds = assemble_bounds(records, static)
        receipt = {'schema': 'track_a_ax2b_h4_native_receipt_v1', 'kind': KIND,
            'manifest_digest': digest(manifest), 'source_commit': manifest['source_commit'],
            'input_binding': manifest['input_binding'], 'environment': manifest['environment'],
            'cell_receipt_digests': {r['cell_id']: digest(r) for r in records},
            'coverage_binding': {'sealed': True, 'actual_rank': 12,
                'schedule_digest': digest(plan()['cells']), 'expected_bounds': bounds},
            'structural_bounds_are_not_compiled_costs': True,
            'synthetic_only': port.synthetic_only, 'science_manifest_sealed': False,
            'N': None, 'G': None, 'numerical_allowance_certified': False,
            'accuracy_eligibility': 'UNDETERMINED', 'mandatory_stop': True,
            'next_stage_authorized': False, 'H6_status': 'H6_NOT_AUTHORIZED',
            'contract_status': 'DRAFT_NOT_AUTHORIZATION'}
        writer.write('native_receipt.json', receipt)
    except Exception as error:
        reason = type(error).__name__+':'+str(error)[:512]
    terminal = {'status': KIND+('_COMPLETE' if reason is None else '_STOP'), 'reason': reason,
        'manifest_digest': digest(manifest), 'calls': calls.used,
        'completed_preparations': len(records),
        'receipt_sha256': file_hash(writer.output/'native_receipt.json') if reason is None else None,
        'synthetic_only': port.synthetic_only, 'science_manifest_sealed': False,
        'N': None, 'G': None, 'numerical_allowance_certified': False,
        'accuracy_eligibility': 'UNDETERMINED', 'mandatory_stop': True,
        'next_stage_authorized': False, 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION'}
    return terminal
