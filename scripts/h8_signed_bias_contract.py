"""Byte bindings and one-shot authorization for saved-value arithmetic."""
import gzip
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import h8_signed_bias as numeric

SCOPE = numeric.SCOPE
AUDIT = 'b75039c1d5ead475d8c0ff8b62dba04a9978a6fa'
PHASE_A = '01309059636f2432e29b663bcd2eb0e400f5ae0a'
A_SOURCE = 'fc463d9e07825f34e496acac6b40db7aec1aed6d'
OLD_I_SOURCE = '54ed98c5fa5b7190f77255271f4e3bec8db15630'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
SOURCE_PATHS = ('src/trotterlib/h8_signed_bias.py', 'scripts/h8_signed_bias_contract.py',
                'scripts/run_h8_signed_bias_validation.py', 'tests/test_h8_signed_bias_validation.py',
                'references/registration_v1.json', 'references/evidence_inputs_v1.json')
require = numeric.require
_AUTHORIZED = object()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


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
            h.update(block); blocks.append(block); size += len(block)
    require(size == receipt['bytes'] and h.hexdigest() == receipt['sha256'], 'FIXED_INPUT_BYTES_SHA')
    return b''.join(blocks)


def environment():
    exe = Path(sys.executable).resolve()
    return dict(executable=str(exe), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                version=sys.version, dependencies='PYTHON_STDLIB_ONLY_NO_COMPILER_CALLS')


class Permit:
    def __init__(self, *, synthetic, scope, _authorization=None):
        require(scope == SCOPE, 'NUMERICAL_SCOPE')
        require(synthetic is True or _authorization is _AUTHORIZED, 'APPROVED_GRANT_PERMIT_REQUIRED')
        self.synthetic = synthetic

    def require(self, kind):
        require(kind == ('ARTIFICIAL' if self.synthetic else 'SAVED_H8'), 'DATA_KIND_AUTHORIZATION')


def authorize(plan, grant, *, source_root=ROOT):
    """Reject the draft before opening any saved H8 input or creating output."""
    require(grant.get('approved') is True and grant.get('runtime_authorization') is True,
            'EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED')
    require(plan['schema'] == 'H8_SIGNED_COMPONENT_EXECUTION_PLAN_V1' and
            plan['scope'] == grant['scope'] == SCOPE and
            grant['schema'] == 'H8_SIGNED_COMPONENT_GRANT_V1', 'PLAN_GRANT_SCOPE')
    require(grant['plan_sha256'] == digest(plan) and grant['SOURCE'] == plan['SOURCE'] and
            type(plan['SOURCE']) is str and len(plan['SOURCE']) == 40, 'PLAN_SOURCE_BINDING')
    require(plan['tolerances'] == numeric.TOLERANCES and
            plan['comparison_rule'] == 'MATH_ISCLOSE_PER_REAL_IMAG_AXIS' and
            plan['science'] == dict(system='LINEAR_H8', separation_angstrom=1., basis='STO-3G',
                DF_rank=28, modes=16, sector_dimension=4900, T=.8, candidates=184, axes=368),
            'FROZEN_SCIENCE_AND_TOLERANCES')
    require(plan['prohibited'] == dict(new_signal=0, compile=0, sampling=0, SCF_DF_state=0,
                noncancellation_shots=0, PR_cost_comparison=0, ranking=0, fit=0), 'FROZEN_SCOPE')
    require(grant['one_shot'] is True and grant['automatic_retry'] is False and
            grant['next_stage_authorized'] is False and
            type(grant['authorization_evidence']) is str and bool(grant['authorization_evidence'].strip()),
            'ONE_SHOT_USER_EVIDENCE')
    require(plan['environment'] == environment(), 'PREPARATION_ENVIRONMENT_BINDING')
    require(plan['audit_commit'] == AUDIT and plan['Phase_A_result'] == PHASE_A and
            plan['saved_compiler_profile'] == PROFILE, 'FIXED_AUDIT_SOURCE_PROFILE')
    require(len(plan['source_files']) == len(SOURCE_PATHS) and
            {r['path'] for r in plan['source_files']} == set(SOURCE_PATHS), 'SOURCE_CLOSURE')
    for f in plan['source_files']:
        rel = Path(f['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'SOURCE_PATH')
        read_verified(home_path(source_root/rel), f)
    require(digest(plan['input_manifest']) == plan['input_manifest_sha256'], 'INPUT_MANIFEST_BINDING')
    require(plan['input_manifest'] == strict((source_root/'references/evidence_inputs_v1.json').read_bytes()),
            'FROZEN_AUDIT_INPUT_MANIFEST')
    input_root = home_path(grant['input_root'])
    output = home_path(grant['output_directory'])
    require(output.parent.is_dir() and not output.exists() and
            not output.is_relative_to(input_root) and not input_root.is_relative_to(output) and
            not output.is_relative_to(source_root), 'EXCLUSIVE_SEPARATE_OUTPUT')
    require(grant['fresh_gate']['source_checked'] is True and
            grant['fresh_gate']['environment_checked'] is True and
            grant['fresh_gate']['resources_noninterference_checked'] is True,
            'FRESH_RESOURCE_AND_BINDING_GATE')
    checked = datetime.fromisoformat(grant['fresh_gate']['checked_UTC'])
    require(checked.tzinfo is not None and
            0 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 300, 'FRESH_GATE_STALENESS')
    # These are admission estimates for a few small scalar JSON files, not
    # CPU/RAM/worker or scientific wall-time caps.
    fs = shutil.disk_usage(output.parent)
    import os
    require(fs.free >= plan['resource_estimates']['new_output_bytes'] and
            os.statvfs(output.parent).f_favail >= plan['resource_estimates']['new_output_inodes'],
            'OUTPUT_FREE_BLOCKS_INODES')
    return Permit(synthetic=False, scope=SCOPE, _authorization=_AUTHORIZED), input_root, output


def load_saved(manifest, input_root, permit):
    permit.require('SAVED_H8')
    require(manifest['schema'] == 'H8_BIAS_METADATA_AUDIT_INPUTS_V1' and
            manifest['real_numerical_analysis_authorized'] is False,
            'ORIGINAL_METADATA_MANIFEST_RETAINED_NUMERICAL_AUTHORIZATION_IS_SEPARATE')
    raw, receipts = {}, []
    for row in manifest['files']:
        role, rel = row['role'], Path(row['snapshot_path'])
        require(role not in raw and not rel.is_absolute() and '..' not in rel.parts, 'INPUT_ROLE_PATH')
        path = home_path(input_root/rel)
        raw[role] = read_verified(path, row)
        receipts.append(dict(role=role, path=str(path), bytes=row['bytes'], sha256=row['sha256']))
    objects = {k: strict(v) for k, v in raw.items() if not k.startswith('producer_') and
               k not in ('A_signals', 'H6_policy', 'H6_result')}
    export = objects['A_export']
    with gzip.GzipFile(fileobj=io.BytesIO(raw['A_signals'])) as stream:
        decoded = stream.read(export['original_json_bytes']+1)
    require(len(decoded) == export['original_json_bytes'] and
            hashlib.sha256(decoded).hexdigest() == export['original_json_sha256'] and
            len(raw['A_signals']) == export['gzip_bytes'] and
            hashlib.sha256(raw['A_signals']).hexdigest() == export['gzip_sha256'], 'A_EXPORT_BYTES')
    additional = strict(decoded)
    require(additional['status'] == objects['A_terminal']['status'] == 'COMPLETE_MANDATORY_STOP' and
            objects['A_binding']['SOURCE'] == objects['A_terminal']['SOURCE'] == A_SOURCE and
            objects['A_binding']['provider_SOURCE'] == PROVIDER and
            objects['A_binding']['profile_digest'] == PROFILE, 'A_SOURCE_PROFILE_COMPLETION')
    combined = [x for x in objects['A_file_receipts']['files']
                if x['path'] == 'signal_output/H8_signal_complete.json']
    require(len(combined) == 1 and combined[0]['bytes'] == len(decoded) and
            combined[0]['sha256'] == export['original_json_sha256'], 'A_ORIGINAL_FILE_RECEIPT')
    workers = objects['A_workers']
    require(len(workers['workers']) == 35 and workers['all_reaped'] is True and
            all(w['exit_code'] == 0 and w['reaped'] is True for w in workers['workers']) and
            workers['original_result_hash_missing'] == ['prefix_19'] and
            workers['original_receipts_modified'] is False and
            [w['task_id'] for w in workers['workers'] if not w['original_result_sha256_present']] == ['prefix_19'] and
            all(type(w.get('publication_result_sha256')) is str and len(w['publication_result_sha256']) == 64
                for w in workers['workers']),
            'A_WORKER_RECEIPTS_AND_ORIGINAL_HASH_LIMITATION')
    old_contract, old_completion = objects['old_D0_contract'], objects['old_completion']
    old_receipt = [r for r in old_contract['input_receipts'] if r['logical_name'] == 'signal_summary.json']
    require(len(old_receipt) == 1 and old_receipt[0]['bytes'] == len(raw['old_signals']) and
            old_receipt[0]['sha256'] == hashlib.sha256(raw['old_signals']).hexdigest() and
            old_contract['binding']['source'] == old_completion['SOURCE'] == PROVIDER and
            old_completion['status'] == 'COMPLETE_SAVED_RESULTS_VERIFIED', 'OLD_SAVED_SIGNAL_LINEAGE')
    provenance = objects['old_B0I_provenance']
    require(provenance['execution_SOURCE'] == objects['old_B0I_completion']['SOURCE'] == OLD_I_SOURCE and
            provenance['profile_digest'] == PROFILE and
            objects['old_B0I_completion']['status'] == 'COMPLETE_MANDATORY_STOP', 'OLD_OVERLAY_LINEAGE')
    ref_receipts = [r for r in objects['A_binding']['input_receipts'] if r['logical_name'] == 'full_H_reference']
    ref_hash = hashlib.sha256(raw['full_reference']).hexdigest()
    require(len(ref_receipts) == 1 and ref_receipts[0]['sha256'] == ref_hash ==
            objects['A_reference']['reference_receipt_sha256'] and
            objects['full_reference']['numerical_allowance_certified'] is False and
            objects['full_reference']['reference_agreement']['certified'] is False, 'REFERENCE_LINEAGE')
    old = objects['old_signals']['signals']
    require(len(old) == 92 and len(additional['signals']) == 152 and
            len(objects['old_B0I']) == 12, 'FIXED_INPUT_COUNTS')
    projections = objects['old_B0I_correctness']
    projection_cases = [(r['prefix'], r['q']) for r in projections]
    require(len(projection_cases) == len(set(projection_cases)) == 8 and
            all(r['status'] == 'PASS' and r['certified'] is False for r in projections),
            'OLD_NATIVE_PROJECTION_LIMITATION')
    dataset = dict(kind='SAVED_H8', records=[r for r in old if r['cell']['method'] in ('B0', 'B1')] +
        [w['signal'] for w in objects['old_B0I']] + additional['signals'],
        old_B0I_wrappers=objects['old_B0I'], full_reference=objects['full_reference']['reference'],
        old_native_projection_cases=projection_cases,
        lineage_limitations=dict(prefix19_runtime_SHA='ORIGINAL_FIELD_MISSING_PUBLICATION_HASH_SEPARATE',
            prefix19_candidates=16, original_records_modified=False, old_native_projection_cases=8,
            old_overlay_endpoint_cases=12, shared_reference_and_scalar=True,
            scalar_extraction_recomputed=False, historical_unknown_runtime_profile_backfilled=False,
            source_commits=dict(Phase_A=A_SOURCE, old_overlay=OLD_I_SOURCE, provider=PROVIDER),
            profile_digest=PROFILE))
    origins = {}
    for rows, role, source, profile in [([r for r in old if r['cell']['method'] in ('B0', 'B1')],
            'old_signals', PROVIDER, 'ORIGINAL_OR_UNKNOWN_NOT_BACKFILLED'),
            ([w['signal'] for w in objects['old_B0I']], 'old_B0I', OLD_I_SOURCE, PROFILE),
            (additional['signals'], 'A_signals', A_SOURCE, PROFILE)]:
        for row in rows:
            origins[row['cell']['id']] = dict(SOURCE=source, input_role=role,
                input_sha256=hashlib.sha256(raw[role]).hexdigest(), compiler_profile=profile,
                provider_SOURCE=PROVIDER, prefix19_runtime_hash_missing=role == 'A_signals' and
                row['cell']['prefix'] == 19)
    dataset['origins'] = origins
    return dataset, objects['registry'], receipts
