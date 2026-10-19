"""PF direct-acquisition metadata contracts; no scientific imports or execution."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import re
from pathlib import Path

PLAN_COMMIT = 'db06128a4b302faea0bc42d45399508b241a276d'
AXES = ('cosine', 'sine')
SYSTEM_LAYOUTS = {'H2_DF3': (4, 3), 'H4_DF10': (8, 10), 'H6_DF19': (12, 19),
                  'H8_DF28': (16, 28), 'H4_LEGACY_DF12': (8, 12)}
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
FORBIDDEN_CALLS = ('sampling', 'fit', 'bootstrap', 'shot_recalculation', 'cost_reaccounting',
                   'SCF', 'DF', 'state_solver', 'new_signal_grid', 'new_PF_selection')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'NONFINITE_JSON'))


def parse_proposals(raw, expected_sha256):
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'PROPOSAL_BYTES_CHANGED')
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    keys = set()
    for row in rows:
        key = row['task_id']
        require(row['system'] in SYSTEM_LAYOUTS and row['axis'] in AXES, 'UNKNOWN_SYSTEM_OR_AXIS')
        require(key == ':'.join((row['system'], row['candidate_id'], row['axis']))
                and key not in keys, 'TASK_IDENTITY_OR_DUPLICATION')
        require(row['execution_authorized'] == 'False', 'PROPOSAL_IS_NOT_A_GRANT')
        match = re.fullmatch(r'(B0I|B0|B1)_p(\d+)_q(\d+)_(2nd|4th)', row['candidate_id'])
        require(match is not None and match.group(2) == row['prefix']
                and match.group(3) == row['q'] and match.group(4) == row['PF']
                and int(row['q']) > 0 and 0 <= int(row['prefix']) <= SYSTEM_LAYOUTS[row['system']][1], 'PF_PARAMETERS')
        keys.add(key)
    require(bool(rows), 'EMPTY_PROPOSAL')
    return rows


def select_pairs(proposals, task_ids):
    """Metadata selection only, not an execution authorization."""
    require(isinstance(task_ids, list) and task_ids, 'EXACT_SCOPE_REQUIRED')
    require(len(task_ids) == len(set(task_ids)), 'DUPLICATE_SELECTION')
    by_id = {row['task_id']: row for row in proposals}
    require(set(task_ids) <= set(by_id), 'TASK_NOT_IN_PROPOSAL')
    selected = [deepcopy(by_id[key]) for key in sorted(task_ids)]
    groups = defaultdict(set)
    for row in selected:
        groups[(row['system'], row['candidate_id'])].add(row['axis'])
    require(all(axes == set(AXES) for axes in groups.values()), 'PAIRED_SCOPE_REQUIRED')
    return selected


def call_caps(selected):
    pairs = len({(r['system'], r['candidate_id']) for r in selected})
    return dict(saved_input_load=pairs, saved_preparation_restore=pairs,
                native_evolution=pairs, control_probe=4*pairs,
                wrapper_build=len(selected), wrapper_compile=len(selected),
                **{key: 0 for key in FORBIDDEN_CALLS})


def draft(proposals, proposal_sha256, input_registry_sha256):
    return dict(schema='PF_DIRECT_TECHNICAL_PREPARATION_V1', plan_commit=PLAN_COMMIT,
        plan_suitable_for_research=True, technical_preparation_authorized=True,
        proposal_sha256=proposal_sha256, proposal_rows_digest=digest(proposals), input_registry_sha256=input_registry_sha256,
        proposed_axes=len(proposals), selected_task_ids=None, acquisition_scope_approved=False,
        execution_authorized=False, grant_issued=False, SOURCE=None,
        source_closure=None, input_files=None, provider_files=None, runtime_files=None,
        profile_files=None, loader_binding=None, fresh_gate=None, approval_receipt=None,
        output_root=None, run_id=None, call_caps=call_caps([]),
        compiler_hash=COMPILER, control='symmetric_directional', P=0, T='0.8',
        resources=dict(CPU_limit=None, RAM_limit=None, worker_limit=None,
                       admission='actual permitted resources and measured task footprints'),
        next_stage_authorized=False, retry=False, mandatory_STOP=True)


def authorize_metadata(contract, approval, proposals):
    """Reject incomplete/unapproved scopes before loaders, imports, output or locks.

    Actual byte/fresh-resource verification is deliberately a separate sealed live
    gate, not a boolean inferred from this metadata check. No grant is issued here.
    """
    require(contract.get('acquisition_scope_approved') is True
            and contract.get('execution_authorized') is True
            and contract.get('grant_issued') is True, 'SCIENTIFIC_EXECUTION_NOT_AUTHORIZED')
    require(approval.get('action') == 'PF_DIRECT_COMPILE_SCIENTIFIC_EXECUTION'
            and approval.get('approved_by_user') is True, 'EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED')
    require(approval.get('contract_digest') == digest(contract), 'APPROVAL_CONTRACT_MISMATCH')
    require(contract.get('proposal_rows_digest') == digest(proposals), 'PROPOSAL_RECORDS_CHANGED')
    selected = select_pairs(proposals, contract.get('selected_task_ids'))
    require(contract.get('call_caps') == call_caps(selected), 'EXACT_CALL_SCOPE_REQUIRED')
    require(isinstance(contract.get('SOURCE'), str) and re.fullmatch('[0-9a-f]{40}', contract['SOURCE']) is not None,
            'SEALED_SOURCE_REQUIRED')
    for key in ('source_closure', 'input_files', 'provider_files', 'runtime_files', 'profile_files'):
        require(isinstance(contract.get(key), list) and contract[key], 'UNSEALED_'+key)
    require(isinstance(contract.get('loader_binding'), dict) and contract['loader_binding'].get('sealed') is True,
            'PRODUCTION_LOADER_NOT_BOUND')
    require(isinstance(contract.get('output_root'), str) and Path(contract['output_root']).is_absolute()
            and isinstance(contract.get('run_id'), str) and contract['run_id'], 'OUTPUT_AND_RUN_ID_REQUIRED')
    require(contract.get('compiler_hash') == COMPILER and contract.get('control') == 'symmetric_directional'
            and contract.get('P') == 0 and contract.get('T') == '0.8', 'CHANGED_SCIENTIFIC_CONDITIONS')
    require(contract.get('next_stage_authorized') is False and contract.get('retry') is False,
            'NO_AUTOMATIC_NEXT_STAGE_OR_RETRY')
    return selected


def verify_bound_files(contract):
    """Only use after authorize_metadata; hashes do not decode scientific inputs."""
    for key in ('source_closure', 'input_files', 'provider_files', 'runtime_files', 'profile_files'):
        for item in contract[key]:
            require(Path(item['path']).is_file() and sha(item['path']) == item['sha256'],
                    'BOUND_FILE_CHANGED:'+key)


def admissible_pairs(pending, *, permitted_cpus, available_memory_bytes, available_disk_bytes):
    """Admission from supplied live observations; no fixed worker or q ceiling.

    Estimates are contract-bound planning inputs, not measured by this function.
    Unknown footprints wait for an explicit bound, rather than a fabricated value.
    """
    require(type(available_memory_bytes) is int and available_memory_bytes >= 0
            and type(available_disk_bytes) is int and available_disk_bytes >= 0, 'RESOURCE_OBSERVATION')
    slots = len(set(permitted_cpus)); memory = available_memory_bytes; disk = available_disk_bytes
    admitted = []
    for task in pending:
        need = task.get('memory_admission_bytes'); space = task.get('disk_admission_bytes')
        if type(need) is not int or type(space) is not int or need <= 0 or space <= 0:
            continue
        if slots and need <= memory and space <= disk:
            admitted.append(task); slots -= 1; memory -= need; disk -= space
    return admitted
