"""Byte bindings, saved-input adapter and one-shot authorization for the diagnostic.

Drafts are rejected before any saved H8 input is opened or output is created.
The adapter normalizes saved JSON/CSV only; it calls no shot rule, cost model,
selection, signal, compiler, sampler or scientific provider.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import h8_noncancellation as numeric

SCOPE = numeric.SCOPE
require = numeric.require
d0 = numeric.d0
PLAN_SCHEMA = 'H8_NONCANCELLATION_DIAGNOSTIC_EXECUTION_PLAN_V1'
GRANT_SCHEMA = 'H8_NONCANCELLATION_DIAGNOSTIC_GRANT_V1'
MANIFEST = 'references/h8_noncancellation_inputs_v1.json'
BASE = dict(signed_result='8e54303bfe46bbebbf4d37148cbe26ef4013ae38',
            evidence_audit='b75039c1d5ead475d8c0ff8b62dba04a9978a6fa',
            B2_readiness='3a450af0fb431269ac94152807992883c1490b26')
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
SEED_SOURCE = 'cb8a56c6b2a772827275f9200b02638030f33f72'
SEED_NAMESPACE = 'H8_CORE_RESOURCE_MAP_V1_20261013'
D0 = dict(SOURCE='20f89bc32b161ee22e4a1ed709f9e3414c19633a',
          analysis_source='fa75c7997e135da7b7a0c4e23949e47252bd7a2e',
          result='681ec23c93fdd1090d325908f4b155a79115afdf')
B1 = dict(result='83a3c4a6688e66576f85471bf41a45e29cbfb705', SOURCE='6fb8a0a71986fda571d82b5658686f5e74a32f8b')
SOURCE_PATHS = ('src/trotterlib/h8_noncancellation.py', 'scripts/h8_noncancellation_contract.py',
                'scripts/run_h8_noncancellation_diagnostic.py', 'scripts/audit_h8_noncancellation_evidence.py',
                'tests/test_h8_noncancellation_diagnostic.py', MANIFEST, 'references/registration_v1.json',
                *numeric.FROZEN_D0)
B2_ROLES = ('driver_terminal', 'saved_B0_acceptance', 'launcher_terminal', 'final_gate',
            'launcher_log_terminal', 'witness_native_exit')
# Shared with the signed run01 input receipts; the same bytes must be bound here.
SIGNED_ROLES = ('registry', 'old_signals', 'old_B0I', 'A_signals', 'A_export', 'full_reference',
                'old_D0_contract', 'old_completion')
SCIENCE = dict(system='LINEAR_H8', separation_angstrom=1., basis='STO-3G', DF_rank=28, modes=16,
               sector_dimension=4900, T=.8, qs=[1, 2, 4, 8], deterministic_candidates=184,
               old_deterministic=32, additional_deterministic=152, deterministic_axes=368,
               PR='PREEXISTING_CONFIRMATION_UNION6_N32', PR_axes=12)
ACCOUNTING = dict(epsilons=list(numeric.EPSILONS), allowance_factors=list(numeric.FACTORS),
                  allowance_factor_meaning='MULTIPLIERS_1_AND_10', alpha_axis=numeric.ALPHA_AXIS,
                  epsilon_axis='epsilon/sqrt(2)', maximum_u_over_headroom=numeric.MAXIMUM_U_OVER_HEADROOM,
                  frozen_rule=dict(numeric.FROZEN_D0), only_change='abs(total_signed_axis)->declared_bias_axis',
                  combined_empirical_u='SAVED_COMBINED_U_TIMES_FACTOR_ONCE_PER_AXIS_NO_REPETITION_NO_RSS',
                  normalization='SAVED_log_B_PLUS_FACTOR_TIMES_SAVED_log_B_margin',
                  metric='SHOT_WEIGHTED_FULL_WRAPPER_RZ_STATE_PREPARATION_EXCLUDED', P=0,
                  closure=dict(numeric.TOLERANCES))
BIAS_KINDS = dict(usual='abs(total_signed_axis)', B0_B0I='abs(discard_axis)+abs(PF_axis)',
                  full_PF='abs(PF_axis)', PR='abs(outer_PF_axis)+abs(finite_RTE_axis)',
                  rigorous_bias_guarantee=None)
PROHIBITED = dict(new_signal=0, compile=0, sampling=0, PR_sampling=0, SCF_DF_state=0, fit=0, refit=0,
                  new_shot_rule=0, unregistered_candidates=0)
_AUTHORIZED = object()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def hexdigest(value, length):
    return type(value) is str and len(value) == length and all(c in '0123456789abcdef' for c in value)


def strict(data):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'DUPLICATE_JSON_KEY')
            out[k] = v
        return out

    def number(value):
        result = float(value)
        require(math.isfinite(result), 'NONFINITE_JSON')
        return result

    def bad(value):
        raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=bad)


def home_path(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'ABSOLUTE_HOME_PATH')
    require(path.resolve().is_relative_to(Path('/home/AbeHiromu')), 'HOME_ONLY')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'NO_SYMLINK_PATH')
    return path


def read_verified(path, receipt):
    path = Path(path)
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), 'REGULAR_INPUT_REQUIRED')
    h, blocks, size = hashlib.sha256(), [], 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
            blocks.append(block)
            size += len(block)
    require(size == receipt['bytes'] and h.hexdigest() == receipt['sha256'], 'FIXED_INPUT_BYTES_SHA')
    return b''.join(blocks)


def environment():
    """The frozen D0 H8 interpreter profile, so paired SE floats reproduce exactly."""
    exe = Path(sys.executable).resolve()
    import statistics
    stats = Path(statistics.__file__).resolve()
    return dict(executable=str(exe), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                version=sys.version, statistics=str(stats),
                statistics_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
                dependencies='PYTHON_STDLIB_ONLY_NO_COMPILER_OR_PROVIDER_IMPORTS')


class Permit:
    def __init__(self, *, synthetic, scope, _authorization=None):
        require(scope == SCOPE, 'NUMERICAL_SCOPE')
        require(synthetic is True or _authorization is _AUTHORIZED, 'APPROVED_GRANT_PERMIT_REQUIRED')
        self.synthetic = synthetic

    def require(self, kind):
        require(kind == ('ARTIFICIAL' if self.synthetic else 'SAVED_H8'), 'DATA_KIND_AUTHORIZATION')


def b2_binding_complete(dependency):
    roles = dependency.get('receipts', [])
    return (dependency.get('binding_complete') is True and dependency.get('required_SOURCE') == numeric.B2['SOURCE']
            and dependency.get('scope') == numeric.B2['scope'] and [r['role'] for r in roles] == list(B2_ROLES)
            and all(type(r['bytes']) is int and r['bytes'] > 0 and hexdigest(r['sha256'], 64) and
                    type(r['relative_path']) is str and not Path(r['relative_path']).is_absolute() and
                    '..' not in Path(r['relative_path']).parts for r in roles))


def authorize(plan, grant, *, source_root=None):
    """Reject drafts before opening any saved H8 input or creating any output."""
    require(grant.get('approved') is True and grant.get('runtime_authorization') is True,
            'EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED')
    source_root = ROOT if source_root is None else source_root
    require(plan['schema'] == PLAN_SCHEMA and plan['scope'] == grant['scope'] == SCOPE and
            grant['schema'] == GRANT_SCHEMA, 'PLAN_GRANT_SCOPE')
    require(grant['plan_sha256'] == digest(plan) and grant['SOURCE'] == plan['SOURCE'] and
            hexdigest(plan['SOURCE'], 40), 'PLAN_SOURCE_BINDING')
    require(plan['science'] == SCIENCE and plan['accounting'] == ACCOUNTING and
            plan['bias_kinds'] == BIAS_KINDS and plan['prohibited'] == PROHIBITED and
            plan['base_commits'] == BASE, 'FROZEN_SCIENCE_ACCOUNTING_SCOPE')
    stages = grant['authorized_stages']
    require(stages in ([numeric.STAGES[0]], list(numeric.STAGES)), 'REGISTERED_STAGE_SET')
    if 'COST_COMPARISON' in stages:
        require(b2_binding_complete(plan['B2_dependency']), 'B2_FULL_PASS_RECEIPT_BINDING_REQUIRED_FOR_COST_STAGE')
    require(grant['one_shot'] is True and grant['automatic_retry'] is False and
            grant['next_stage_authorized'] is False and type(grant['authorization_evidence']) is str and
            bool(grant['authorization_evidence'].strip()), 'ONE_SHOT_USER_EVIDENCE')
    require(plan['environment'] == environment(), 'PREPARATION_ENVIRONMENT_BINDING')
    require(len(plan['source_files']) == len(SOURCE_PATHS) and
            {r['path'] for r in plan['source_files']} == set(SOURCE_PATHS), 'SOURCE_CLOSURE')
    for f in plan['source_files']:
        rel = Path(f['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'SOURCE_PATH')
        require(git_blob(read_verified(home_path(source_root/rel), f)) == f['git_blob'], 'SOURCE_GIT_BLOB')
    require(digest(plan['input_manifest']) == plan['input_manifest_sha256'] and
            plan['input_manifest'] == strict((source_root/MANIFEST).read_bytes()), 'FROZEN_INPUT_MANIFEST')
    input_root, output = home_path(grant['input_root']), home_path(grant['output_directory'])
    require(output.parent.is_dir() and not output.exists() and
            not output.with_name(output.name + '.one_shot_lock').exists() and
            not output.is_relative_to(input_root) and not input_root.is_relative_to(output) and
            not output.is_relative_to(source_root), 'EXCLUSIVE_SEPARATE_OUTPUT')
    b2_root = None
    if 'COST_COMPARISON' in stages:
        b2_root = home_path(grant['B2_receipt_root'])
        require(not output.is_relative_to(b2_root) and not b2_root.is_relative_to(output), 'B2_RECEIPTS_SEPARATE')
    gate = grant['fresh_gate']
    require(all(gate[k] is True for k in ('source_checked', 'environment_checked',
                                          'resources_noninterference_checked', 'B2_noninterference_checked')),
            'FRESH_RESOURCE_AND_BINDING_GATE')
    checked = datetime.fromisoformat(gate['checked_UTC'])
    require(checked.tzinfo is not None and
            0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 300, 'FRESH_GATE_STALENESS')
    # Admission estimates for small JSON outputs, not CPU/RAM/worker/wall caps.
    fs = shutil.disk_usage(output.parent)
    require(fs.free >= plan['resource_estimates']['new_output_bytes'] and
            os.statvfs(output.parent).f_favail >= plan['resource_estimates']['new_output_inodes'],
            'OUTPUT_FREE_BLOCKS_INODES')
    return Permit(synthetic=False, scope=SCOPE, _authorization=_AUTHORIZED), input_root, output, b2_root


def task_seed(binding, cell_id, phase, replica):
    """Frozen H8 resource-map seed text (pre_review_h8_d0_adapter_v1.task_seed)."""
    key = f"{SEED_NAMESPACE}|{binding['seed_source']}|{binding['input']}|{cell_id}|{phase}|{replica}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def receipt_of(receipts, name, key='logical_name'):
    rows = [r for r in receipts if r[key] == name]
    require(len(rows) == 1, 'UNIQUE_RECEIPT:' + name)
    return rows[0]


def assemble(objects, raw_receipts, permit):
    """Lineage joins and normalization only; numbers are passed through untouched."""
    permit.require('SAVED_H8')
    o, r = objects, raw_receipts
    registry, selection, completion = o['registry'], o['confirmation_selection'], o['old_completion']
    contract, binding = o['old_D0_contract'], o['old_D0_contract']['binding']
    signed, signed_binding = o['signed_axis_results'], o['signed_binding']
    require(signed_binding['raw_axis_result_sha256'] == r['signed_axis_results']['sha256'] and
            signed_binding['raw_axis_result_bytes'] == r['signed_axis_results']['bytes'] and
            signed_binding['SOURCE'] == signed['SOURCE'] == numeric.SIGNED['SOURCE'] and
            signed_binding['plan_digest'] == signed['plan_sha256'] and
            signed_binding['grant_digest'] == signed['grant_sha256'] and
            signed['status'] == 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY' and
            signed['post_execution_binding_verified'] is True, 'SIGNED_RUN01_BINDING')
    for role in SIGNED_ROLES:
        old = receipt_of(signed['input_receipts'], role, 'role')
        require(old['bytes'] == r[role]['bytes'] and old['sha256'] == r[role]['sha256'], 'SIGNED_INPUT_ROLE:' + role)
    for name, role in (('signal_summary.json', 'old_signals'), ('wrapper_metrics.csv', 'wrapper_metrics'),
                       ('selection_before_confirmation.json', 'confirmation_selection'),
                       ('completion_audit.json', 'old_completion')):
        row = receipt_of(contract['input_receipts'], name)
        require(row['bytes'] == r[role]['bytes'] and row['sha256'] == r[role]['sha256'], 'D0_INPUT_RECEIPT:' + name)
    require(binding['compiler'] == numeric.COMPILER and binding['source'] == PROVIDER and
            binding['seed_source'] == SEED_SOURCE and binding['result'] == D0['result'] and
            binding['analysis_source'] == D0['analysis_source'] and
            binding['selection_digest'] == d0.digest(selection) and
            selection['signal_digest'] == d0.digest(o['old_signals']['signals']) and
            selection['fixed_before_confirmation'] is True and selection['exploration_confirmation_pooled'] is False,
            'D0_SELECTION_SIGNAL_BINDING')
    require(completion['SOURCE'] == PROVIDER and completion['seed_source'] == SEED_SOURCE and
            completion['input_SHA'] == binding['input'] and completion['compiler_hash'] == numeric.COMPILER and
            completion['selected_ids'] == selection['selected_ids'], 'RESOURCE_MAP_COMPLETION_BINDING')
    terminal, provenance = o['D0_terminal'], o['D0_provenance']
    accounting = receipt_of(terminal['output_file_receipts'], 'candidate_accounting_v1.json', 'name')
    require(terminal['status'] == 'COMPLETE_MANDATORY_STOP' and terminal['first_stop'] is None and
            accounting['bytes'] == r['D0_accounting']['bytes'] and accounting['sha256'] == r['D0_accounting']['sha256'] and
            provenance['SOURCE'] == D0['SOURCE'] and provenance['fixed_D0_SOURCE'] == D0['analysis_source'] and
            provenance['contract_sha256'] == r['old_D0_contract']['sha256'], 'D0_RESULT_LINEAGE')
    b0i = receipt_of(o['B0I_file_receipts']['files'], 'case_accounting_v1.json', 'name')
    signals_receipt = receipt_of(o['B0I_file_receipts']['files'], 'case_scalar_signal_receipts_v1.json', 'name')
    require(b0i['bytes'] == r['B0I_accounting']['bytes'] and b0i['sha256'] == r['B0I_accounting']['sha256'] and
            signals_receipt['sha256'] == r['old_B0I']['sha256'], 'OLD_B0I_ACCOUNTING_LINEAGE')
    exit_ = o['B1_launcher_native_exit']
    require(o['B1_terminal']['SOURCE'] == B1['SOURCE'] and exit_['native_exit_code'] == 0 and
            exit_['wait_completed'] is True, 'B1_RESULT_LINEAGE')
    union = selection['selected_ids']
    cells = {c['id']: c for c in registry['old'] + registry['additional']}
    registrations = selection['registrations']
    for item in registrations:
        task = item['task']
        require(task['seed'] == task_seed(binding, task['cell_id'], 'confirmation', task['replica']),
                'CONFIRMATION_SEED_DERIVATION')
    old = o['old_signals']['signals']
    records = [x for x in old if x['cell']['id'] in cells or x['cell']['id'] in union]
    records += [w['signal'] for w in o['old_B0I']] + o['A_signals']['signals']
    saved, means, metrics = [], {}, []
    for row in o['D0_accounting']['H8_EXPLORATION92']:
        if row['cell_id'] in cells:
            saved.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                              accounting=row['accounting']))
            if row['method'] == 'B1' and row['status'] == 'DERIVED_SAVED_SCALAR':
                means.setdefault(row['cell_id'], []).append(row['metrics']['RZ']['axis_means'])
    for row in o['D0_accounting']['H8_CONFIRMATION_UNION6']:
        if row['cell_id'] in union:
            saved.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                              accounting=row['accounting']))
            metrics.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                                status=row['status'], metrics=row['metrics'], sample_phase=row['sample_phase'],
                                sample_count=row['sample_count']))
    for row in o['B0I_accounting']:
        require(row['cell_id'] in cells and cells[row['cell_id']]['method'] == 'B0I', 'B0I_SAVED_DOMAIN')
        saved.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                          accounting=row['accounting']))
    for signal in o['A_signals']['signals']:
        for e, variants in signal['precision_rows'].items():
            for f, a in variants.items():
                saved.append(dict(cell_id=signal['cell']['id'], epsilon=e, factor=f, accounting=a))
    full = o['full_reference']
    require(full['numerical_allowance_certified'] is False and full['reference_agreement']['certified'] is False,
            'REFERENCE_CERTIFICATION_NOT_CLAIMED')
    return dict(kind='SAVED_H8', records=records, signed=signed,
        saved_usual=saved, confirmation=dict(selected_ids=union, registrations=registrations),
        full_reference=full['reference'], wrappers=o['wrapper_metrics'],
        wrapper_binding=dict(compiler=binding['compiler'], allowed_sources=binding['allowed_wrapper_sources']),
        full_PF_saved_axis_means=means, PR_saved_metrics=metrics, Taylor_orders=completion['Taylor_orders'],
        B1_model=o['B1_models'], B1_terminal=o['B1_terminal'], B1_anchors=o['B1_anchors'],
        B2_saved_B0_registry=o['B2_saved_B0_registry'], B2=o.get('B2'),
        lineage_limitations=dict(prefix19_runtime_SHA='ORIGINAL_FIELD_MISSING_PUBLICATION_HASH_SEPARATE',
            prefix19_candidates=16, old_native_projection_cases=8, old_overlay_reconstructed_candidates=12,
            shared_reference_and_scalar=True, scalar_extraction_recomputed=False,
            exact_truncated_is_numerical_not_certified=True, legacy_full_PF_runtime_profile='ORIGINAL_OR_UNKNOWN',
            old_B0_wrapper_costs_reused=False, B0_anchor_original_acquisition_status='STOP',
            S4_model_validated_on='FOUR_REGISTERED_B2_AXES_ONLY', model_extrapolation_flagged=True,
            PR_population_mean_guaranteed=False, PR_rare_layers='SAVED_HISTOGRAM_UNOBSERVED_ORDERS_NOT_BOUNDED',
            empirical_u_certified=False, rigorous_bias_guarantee=None)), registry


def load_saved(manifest, input_root, permit, *, b2_dependency=None, b2_root=None):
    permit.require('SAVED_H8')
    require(manifest['schema'] == 'H8_NONCANCELLATION_INPUTS_V1' and
            manifest['real_numerical_analysis_authorized'] is False, 'MANIFEST_DOES_NOT_AUTHORIZE_EXECUTION')
    raw, receipts = {}, {}
    for row in manifest['files']:
        role, rel = row['role'], Path(row['snapshot_path'])
        require(role not in raw and not rel.is_absolute() and '..' not in rel.parts, 'INPUT_ROLE_PATH')
        raw[role] = read_verified(home_path(input_root/rel), row)
        receipts[role] = dict(role=role, path=str(input_root/rel), bytes=row['bytes'], sha256=row['sha256'])
    if b2_dependency is not None:
        for row in b2_dependency['receipts']:
            role = 'B2_' + row['role']
            raw[role] = read_verified(home_path(b2_root/row['relative_path']), row)
            receipts[role] = dict(role=role, path=str(b2_root/row['relative_path']),
                                  bytes=row['bytes'], sha256=row['sha256'])
    objects = {k: strict(v) for k, v in raw.items() if k not in ('A_signals', 'wrapper_metrics')}
    export = objects['A_export']
    with gzip.GzipFile(fileobj=io.BytesIO(raw['A_signals'])) as stream:
        decoded = stream.read(export['original_json_bytes']+1)
    require(len(decoded) == export['original_json_bytes'] and
            hashlib.sha256(decoded).hexdigest() == export['original_json_sha256'] and
            len(raw['A_signals']) == export['gzip_bytes'] and
            hashlib.sha256(raw['A_signals']).hexdigest() == export['gzip_sha256'], 'A_EXPORT_BYTES')
    objects['A_signals'] = strict(decoded)
    require(objects['A_signals']['status'] == 'COMPLETE_MANDATORY_STOP', 'PHASE_A_COMPLETE')
    objects['wrapper_metrics'] = d0.parse_wrappers(raw['wrapper_metrics'])
    if b2_dependency is not None:
        objects['B2'] = {k[3:]: objects.pop(k) for k in list(objects) if k.startswith('B2_') and
                         k != 'B2_saved_B0_registry'}
    dataset, registry = assemble(objects, receipts, permit)
    return dataset, registry, list(receipts.values())
