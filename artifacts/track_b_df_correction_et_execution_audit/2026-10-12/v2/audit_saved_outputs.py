"""Stdlib saved-byte/JSON/IR audit only. Never evaluates an operator or circuit."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_storage_v5.storage import (
    MAGIC, load_json, artifact_identity,
)

AUDIT = ROOT/'artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v2'
RESULT = ROOT/'artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v2'
PREP = ROOT/'artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5'
SOURCE = 'd0a12bae817ce75c5876a06c2db00397914ca9b0'
AUTHORIZATION = '0e147e2746544657072053b7d648b41eee299ffc'
INDEXES = {'PROJECT_MAP.md', 'docs/research/研究概要・現状.md', 'scripts/README.md',
           'src/trotterlib/README.md', 'docs/README.md',
           'docs/tracks/algorithm_codesign/README.md'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).relative_to(ROOT)), 'bytes': len(raw), 'sha256': sha(raw)}


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def main():
    checks = []
    def check(name, value):
        checks.append({'name': name, 'PASS': bool(value)})
        if not value:
            raise AssertionError(name)
    original = json.loads((AUDIT/'original_run_inventory.json').read_text())
    for row in original['files']:
        check('original saved file '+row['path'], file_identity(ROOT/row['path']) == row)
    check('no result file added or removed', sorted(str(p.relative_to(ROOT)) for p in RESULT.rglob('*') if p.is_file()) == sorted(r['path'] for r in original['files']))
    check('all result files review bounded', max(r['bytes'] for r in original['files']) <= 512*1024)
    check('active output and log caps', original['bytes'] <= 64*1024*1024-256*1024 and sum(r['bytes'] for r in original['files'] if r['path'].endswith('.log')) <= 4*1024*1024)
    manifest = json.loads((PREP/'source_manifest_v5.json').read_text())
    appended = []
    for row in manifest['files']:
        raw = (ROOT/row['path']).read_bytes()
        if row['path'] in INDEXES:
            check('source index unchanged prefix '+row['path'], len(raw) >= row['bytes'] and sha(raw[:row['bytes']]) == row['sha256'])
            appended.append({'path': row['path'], 'post_run_appended_bytes': len(raw)-row['bytes']})
        else:
            check('protected source/input/old evidence '+row['path'], len(raw) == row['bytes'] and sha(raw) == row['sha256'])
    check('authorization sole parent', subprocess.check_output(['git', 'rev-list', '--parents', '-n', '1', AUTHORIZATION], cwd=ROOT, text=True).split()[1:] == [SOURCE])
    auth_paths = subprocess.check_output(['git', 'diff', '--name-only', SOURCE, AUTHORIZATION], cwd=ROOT, text=True).splitlines()
    check('authorization-only paths', set(auth_paths) == {str((PREP/'authorization.json').relative_to(ROOT)), 'docs/tracks/algorithm_codesign/df_correction_et_execution_authorization_v5.md'})
    auth = json.loads((PREP/'authorization.json').read_text())
    contract = json.loads((PREP/'contract_v5.json').read_text())
    check('explicit approval, one run, STOP', auth['source_commit'] == SOURCE and auth['status'] == 'APPROVED_FOR_ONE_DF_CORRECTION_E_T_V5_RUN' and auth['science_execution_authorized'] is True and auth['runs'] == 1 and auth['retries'] == 0 and auth['mandatory_STOP'] is True)
    check('contract binding', auth['contract_sha256'] == sha((PREP/'contract_v5.json').read_bytes()))
    check('manifest binding', auth['source_manifest_sha256'] == sha((PREP/'source_manifest_v5.json').read_bytes()))
    marker = json.loads((RESULT/'one_shot_marker.json').read_text())
    terminal = load_json(RESULT/'terminal_status.json')
    marker_hash = sha((RESULT/'one_shot_marker.json').read_bytes())
    check('marker source/authorization hashes', marker['source'] == SOURCE and marker['authorization'] == AUTHORIZATION and marker['authorization_sha256'] == sha((PREP/'authorization.json').read_bytes()) and marker['contract_sha256'] == auth['contract_sha256'] and marker['source_manifest_sha256'] == auth['source_manifest_sha256'])
    check('terminal marker/provenance', terminal['marker_SHA256'] == marker_hash and all(terminal['provenance'][k] == marker[k] for k in terminal['provenance']))
    check('terminal incomplete classification and no winner', terminal['classification'] == 'TECHNICAL_INCONCLUSIVE' and terminal['scientific_winner'] is None and terminal['mandatory_STOP'] is True and terminal['retries'] == 0)
    check('stages stop before E_NATIVE complete', terminal['completed_stages'] == ['CD_SAVED_REFERENCE_ONLY'])
    check('no T inventory, jobs, or sequences', not (RESULT/'synthesis_inventory.json').exists() and not (RESULT/'synthesis_inventory.freeze.json').exists() and not list(RESULT.glob('T-*')))
    outer = json.loads((AUDIT/'outer_process_receipt.json').read_text())
    stdout = (AUDIT/'outer_runner_stdout.log').read_bytes()
    check('outer stdout identity', len(stdout) == outer['stdout_bytes'] and sha(stdout) == outer['stdout_sha256'])
    outward = json.loads(stdout.decode().splitlines()[-1])
    check('outer exit and terminal agree', outer['exit_status'] == 2 and outward['classification'] == terminal['classification'] and outward['reason'] == terminal['technical_reason'] and outward['mandatory_STOP'] is True)
    check('outer run/provenance', outer['run_invocations'] == 1 and outer['retries'] == 0 and outer['source'] == SOURCE and outer['authorization'] == AUTHORIZATION)
    check('child exits', {r['label']: r['exit_status'] for r in terminal['resource_receipt']['child_receipts']} == {'E-r3': 0, 'E-r6': 1})
    log = (RESULT/'E-r6.log').read_text()
    check('direct child traceback', "native_evaluation.py\", line 98" in log and "ArithmeticError: Whole native numerical allowance not certified" in log)
    cd = load_json(RESULT/'CD_saved_substage_receipt.json')
    check('CD saved only and not reclassified', cd['saved_rows'] == 6 and cd['sha256'] == contract['resumption']['saved_CD_sha256'] and cd['bytes'] == contract['resumption']['saved_CD_bytes'] and cd['CD_recomputed'] is False and cd['old_run_reclassified'] is False and cd['E_cost_multiplication_allowed'] is False and cd['PR_total_winner'] is None)
    check('saved E inputs equal jobs', all(json.loads((RESULT/('E-r'+str(r)+'.job.json')).read_text())['frames'] == load_json(RESULT/'E-inputs.json')[str(r)] for r in (3, 6)))
    logical = []
    for path in sorted(RESULT.glob('*.json')):
        root_record = json.loads(path.read_text())
        if root_record.get('kind') == MAGIC:
            item = artifact_identity(path)
            item['logical_path'] = str(path.relative_to(ROOT))
            logical.append(item)
    observations = []
    for path in sorted(RESULT.glob('*.raw-provenance-*.json')):
        row = load_json(path)
        expected_provenance = dict(marker, marker_sha256=marker_hash)
        check('raw observation source '+path.name, all(row['provenance'][k] == expected_provenance[k] for k in row['provenance']))
        check('single captured payload '+path.name, row['fingerprint_attempts'] == 1 and row['fingerprint_exception'] is None and row['recursive_definition_status'] == 'COMPLETE_SAME_SINGLE_PAYLOAD' and digest(row['recursive_payload']) == row['actual_fingerprint'])
        check('compile returned '+path.name, row['phase'] == 'COST_RETURNED_BEFORE_IDENTITY_GUARD' and row['transpile_attempts'] == 1 and row['historical_mismatch_rejects'] is False)
        for name, array in row['input_arrays'].items():
            raw = bytes.fromhex(array['raw_bytes_hex'])
            check('saved array bytes '+path.name+'/'+name, len(raw) == array['bytes'] and sha(raw) == array['sha256'] and array['dtype_str'] == '<c16' and len(raw) == 16*__import__('math').prod(array['shape']))
            if name.startswith('frame_'):
                expected = contract['resumption']['frame_identities'][str(row['context']['active_modes'])][int(name.split('_')[1])]
                check('saved frame contract identity '+path.name+'/'+name, all(array[k] == expected[k] for k in ('dtype_str', 'shape', 'bytes', 'sha256')))
        observations.append({'path': str(path.relative_to(ROOT)), 'context': row['context'], 'phase': row['phase'], 'fingerprint_attempts': row['fingerprint_attempts'], 'actual_fingerprint': row['actual_fingerprint'], 'expected_fingerprint': row['expected_fingerprint'], 'matches_expected': row['matches_expected'], 'fingerprint_exception': row['fingerprint_exception'], 'transpile_attempts': row['transpile_attempts'], 'posttranspile_instruction_count': row['posttranspile_instruction_count'], 'numerical_acceptance_implied_by_phase': False})
        del row
    accepted = []
    for r in (3, 6):
        path = RESULT/('E-r'+str(r)+'.json')
        complete = path.exists()
        data = load_json(path if complete else RESULT/(path.name+'.prefix.json'))
        if not complete:
            check('r6 prefix not scientific result', data['partial_not_scientific_outcome'] is True)
            data = data['prefix']
        for row in data['native']:
            inv = row['inventory'];ir = inv['ordered_native_ir'];counts = Counter();rz = []
            check('IR digest', digest(ir) == inv['ordered_native_ir_sha256'])
            for i, op in enumerate(ir['operations']):
                check('IR position/name/wires', op['position'] == i and op['name'] in ir['basis'] and len(op['qubits']) == (2 if op['name'] == 'cx' else 1) and len(set(op['qubits'])) == len(op['qubits']) and all(type(q) is int and 0 <= q < ir['num_qubits'] for q in op['qubits']))
                counts[op['name']] += 1
                if op['name'] == 'rz':
                    rz.append((i, op['params_binary64_exact'][0]))
            check('IR counts/phase/angle inventory', dict(counts) == inv['gate_counts'] and counts['rz'] == inv['total_RZ'] and counts['cx'] == inv['CX'] and sum(v for k, v in counts.items() if k != 'cx') == inv['1Q'] and ir['global_phase_binary64_exact'] == inv['global_phase_raw'] and rz == [(x['position'], x['raw_binary64_radians']) for x in inv['occurrences']])
            check('stored accepted numerical upper', F(0) <= F(row['numeric_operator_error_upper']) <= F(contract['E']['local_numeric_allowance']))
            accepted.append({'r': r, 'arm': row['arm'], 'controlled': row['controlled'], 'label': row.get('label'), 'scope': 'Saved native substage in incomplete run, no winner', 'gate_counts': inv['gate_counts'], 'CX': inv['CX'], '1Q': inv['1Q'], 'RZ': inv['total_RZ'], 'depth': row['depth'], 'numeric_operator_error_upper': row['numeric_operator_error_upper'], 'numeric_ledger': row['numeric_ledger'], 'ordered_native_ir_sha256': inv['ordered_native_ir_sha256'], 'global_phase_raw': inv['global_phase_raw']})
        if complete:
            check('B executed distributions', all(sum(F(e['executed_conditional_probability']) for e in data['all_B_event_costs'] if e['controlled'] == c) == 1 for c in (False, True)))
    check('observed and accepted row counts', len(observations) == 10 and len(accepted) == 9)
    check('failed controlled r6 reached compile', any(o['context']['active_modes'] == 6 and o['context']['controlled'] is True and o['posttranspile_instruction_count'] == 50554 for o in observations))
    output = {'kind': 'DF_ET_V5_READONLY_SAVED_OUTPUT_AUDIT', 'PASS': len(checks), 'FAIL': 0,
              'checks': checks, 'classification': terminal['classification'], 'technical_reason': terminal['technical_reason'],
              'direct_worker_reason': 'ArithmeticError: Whole native numerical allowance not certified',
              'failed_context': {'active_modes': 6, 'arm': 'A', 'controlled': True},
              'failed_combined_guard_and_ledger_saved': False, 'actual_operator_error_above_cap_claim': False,
              'source_manifest_files': len(manifest['files']), 'index_post_run_append_only': appended,
              'original_result_files': original['file_count'], 'original_result_bytes': original['bytes'],
              'marker': file_identity(RESULT/'one_shot_marker.json'), 'authorization': file_identity(PREP/'authorization.json'),
              'logical_records': logical, 'raw_observations': observations, 'stored_accepted_native_rows': accepted,
              'native_compile_attempts': 10, 'stored_numerically_accepted_native_rows': 9,
              'synthesis_inventory_created': False, 'actual_T_calls': 0, 'T_scientific_comparison_available': False,
              'read_only_auditor_matrix_circuit_synthesis_LP_calls': 0, 'retry': 0, 'mandatory_STOP': True,
              'scientific_winner': None, 'source': SOURCE, 'authorization_commit': AUTHORIZATION}
    # Detailed per-gate booleans would themselves be unnecessary large review output.
    output['checks'] = [item for item in checks if item['name'] != 'IR position/name/wires']
    output['per_gate_IR_checks_passed'] = sum(item['name'] == 'IR position/name/wires' for item in checks)
    path = AUDIT/'saved_output_audit.json'
    with path.open('x') as stream:
        json.dump(output, stream, indent=2);stream.write('\n')
    print(json.dumps({'PASS': output['PASS'], 'FAIL': 0, 'native_attempts': 10, 'accepted_rows': 9, 'saved_output_audit_bytes': path.stat().st_size, 'mandatory_STOP': True}))


if __name__ == '__main__':
    main()
