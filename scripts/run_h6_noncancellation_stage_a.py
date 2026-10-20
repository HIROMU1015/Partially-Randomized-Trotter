"""Seal and launch one saved-only H6 noncancellation stage A child; never stage B."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/trotterlib'))
import h6_noncancellation_stage_a as core

BASE = '6cfc4579223b0ab4e54adbdacbc3b16a88ecf382'
B = 'artifacts/resource_applicability/'
PRIOR = B + 'track_a_h6_noncancellation_investigation/2026-10-20/'
OUT = ROOT / (B + 'track_a_h6_noncancellation_stage_a/2026-10-20')
PRIVATE = Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261015')
ORIGINAL_PYTHON = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
PUBLIC_INPUTS = (PRIOR + 'candidate12_evidence.json', PRIOR + 'literal_decomposition_recovery.json',
    PRIOR + 'future_saved_analysis_DRAFT.json', PRIOR + 'saved_unresolved120.json',
    B + 'track_a_v1_reaccount_readiness/2026-10-16/input_binding.json',
    B + 'track_a_pf_selection_comparison_preparation/2026-10-20/input_bindings.json',
    B + 'track_a_h6_fixed_delta_result_v1/2026-10-15/signal_summary_v1.json')
SOURCE_FILES = ('src/trotterlib/h6_noncancellation_stage_a.py',
    'scripts/run_h6_noncancellation_stage_a.py', 'scripts/check_h6_noncancellation_stage_a.py',
    'tests/test_h6_noncancellation_stage_a.py', 'scripts/d0_saved_value_reanalysis_v1.py',
    'scripts/d0_frozen_shot_rule_v1.py', 'src/trotterlib/track_a_v1_contract.py')
LEGACY_SOURCE = '2261a3ad1d6c23521474b31204572a6c3a51258b'
APPROVAL = '''User explicitly approved stage A from contract seal through one-shot saved-value
analysis and native termination audit at base 6cfc4579223b0ab4e54adbdacbc3b16a88ecf382:
H6 B0I prefix5/10/15 x q1/2/4/8; 11 saved-signal component reconstructions plus
the existing p15/q4 components; closure, original noncancellation envelope,
120 endpoint eligibility decisions and eligible analytic shots. Original
allowance/log_B/alpha/order and all old evidence are preserved. No new signal,
SCF/DF/state/sampling/compile, retry, additional analysis, stage B, detailed
scientific review or final evidence freeze. New publication is not authorized.
This receipt records the current explicit user instruction, not an old grant.
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return core.accounting_source.strict_json(Path(path).read_bytes())


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def write(name, value, *, exclusive=False):
    path = OUT / name
    raw = value.encode() if isinstance(value, str) else encoded(value)
    if exclusive:
        with path.open('xb') as f:
            f.write(raw)
    else:
        temporary = OUT / ('.' + name + '.' + str(os.getpid()) + '.tmp')
        with temporary.open('xb') as f:
            f.write(raw)
        os.replace(temporary, path)


def raw_path(item):
    # Closed role-to-file mapping; no discovery or arbitrary private path reads.
    p, q = item['prefix'], item['q']
    pilot = (p, q) in ((5, 1), (10, 2), (15, 8))
    return PRIVATE / ('h6-native-b0i-pilot6-run04-v3/run04' if pilot else
                      'h6-native-b0i-full-family-run04/run01') / (
        f'case_{p}_{q}_{"scalar_signal" if pilot else "screen"}.json')


def verify(contract):
    for name, value in contract['source_hashes'].items():
        core.require(sha(ROOT / name) == value and hashlib.sha256(
            git('show', contract['SOURCE'] + ':' + name)).hexdigest() == value,
            'COMMITTED_SOURCE_CHANGED:' + name)
    for name, value in contract['public_input_hashes'].items():
        core.require(sha(ROOT / name) == value and hashlib.sha256(
            git('show', BASE + ':' + name)).hexdigest() == value, 'INPUT_CHANGED:' + name)
    for item in contract['private_input_receipts']:
        path = raw_path(item)
        core.require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
                     and sha(path) == item['sha256'], 'PRIVATE_INPUT_CHANGED:' + item['candidate_id'])
    for name, value in contract['protected_evidence_hashes'].items():
        core.require(sha(ROOT / name) == value, 'OLD_EVIDENCE_CHANGED:' + name)
    for name, value in contract['runtime']['file_hashes'].items():
        core.require(sha(name) == value, 'RUNTIME_CHANGED:' + name)
    core.require(sys.version == contract['runtime']['python_version']
                 and sys.executable == ORIGINAL_PYTHON, 'ORIGINAL_PYTHON_REQUIRED')


def input_records():
    metadata = read(ROOT / PRIOR / 'candidate12_evidence.json')['rows']
    recovery = read(ROOT / PRIOR / 'literal_decomposition_recovery.json')['rows']
    fixed = read(ROOT / PUBLIC_INPUTS[-1])
    binding = read(ROOT / PUBLIC_INPUTS[-2])['H6_DF19']['binding']
    original_receipts = read(ROOT / PUBLIC_INPUTS[-3])['input_receipts']
    original_receipts = {r['logical_name']:r for r in original_receipts}
    core.require([r['candidate_id'] for r in metadata] == list(core.IDS), 'EXACT_ORDERED12_INPUTS')
    records = []
    for item in metadata:
        original = original_receipts[item['raw_role']]
        core.require(original['sha256'] == item['raw_sha256'] and original['bytes'] == item['raw_bytes'],
                     'ORIGINAL_PUBLIC_RECEIPT')
        raw = read(raw_path(item))
        s = raw['signal']
        core.require(s['cell']['id'] == item['candidate_id'], 'SIGNAL_CANDIDATE_BINDING')
        for key in ('hamiltonian', 'df', 'state', 'input', 'compiler'):
            core.require(raw['blueprint']['binding'][key] == binding[key], 'SAVED_INPUT_IDENTITY:' + key)
        if item['candidate_id'] == core.REUSED:
            core.require(len(recovery) == 1, 'EXACT_ONE_PUBLIC_COMPONENT_REUSE')
            r = recovery[0]
            public = fixed[r['source_index']]
            core.require(public['cell']['id'] == core.REUSED and public['T'] == .8, 'PUBLIC_REUSE_SCOPE')
            core.require(public['decomposition']['parts_signed'] == r['signed_parts_verbatim'], 'PUBLIC_REUSE_VALUES')
            core.require(all(public[k] == s[k] for k in r['literal_identity_fields']), 'PUBLIC_REUSE_EXACT_SAVED_SIGNAL')
        records.append(s)
    return records, recovery[0]


def resources():
    mem = {line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    location = OUT
    while not location.exists():
        location = location.parent
    disk = os.statvfs(location)
    return dict(UTC=datetime.now(timezone.utc).isoformat(),
        process_affinity=sorted(os.sched_getaffinity(0)), host_load_average=list(os.getloadavg()),
        MemAvailable=mem.get('MemAvailable'), output_available_bytes=disk.f_bavail*disk.f_frsize,
        resource_limits_modified=False, worker_limits_imposed=False,
        scheduling='One stdlib scalar child, parse12 once, cache12 envelopes across120 endpoints; no useful process-parallel workload',
        shared_environment_or_other_processes_modified=False)


def seal(test_receipt):
    core.require(not OUT.exists(), 'FRESH_STAGE_A_OUTPUT_REQUIRED')
    core.require(sys.executable == ORIGINAL_PYTHON, 'USE_ORIGINAL_ARITHMETIC_PYTHON')
    head = git('rev-parse', 'HEAD').decode().strip()
    for name in SOURCE_FILES:
        core.require(git('show', head + ':' + name) == (ROOT / name).read_bytes(), 'COMMIT_SOURCE_FIRST:' + name)
    for name in SOURCE_FILES[-3:]:
        core.require(git('show', LEGACY_SOURCE + ':' + name) == (ROOT / name).read_bytes(), 'LEGACY_SOURCE_CHANGED')
    tests = read(test_receipt)
    core.require(tests['status'] == 'PASS_ARTIFICIAL_AND_SHOT_REGRESSION'
                 and tests['real_saved_numeric_evaluations'] == 0, 'PRE_EXECUTION_TESTS_REQUIRED')
    for name, value in tests['source_hashes'].items():
        core.require(sha(ROOT / name) == value, 'TESTED_SOURCE_CHANGED:' + name)
    meta = read(ROOT / PRIOR / 'candidate12_evidence.json')['rows']
    receipts = [dict(candidate_id=r['candidate_id'], prefix=r['prefix'], q=r['q'],
                     role=r['raw_role'], bytes=r['raw_bytes'], sha256=r['raw_sha256']) for r in meta]
    protected = []
    for prefix in (PRIOR.rstrip('/'), B + 'track_a_pf_selection_cost_comparison/2026-10-20'):
        protected.extend(git('ls-tree', '-r', '--name-only', BASE, prefix).decode().splitlines())
    runtime_paths = {str(Path(sys.executable).resolve())}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file() and not Path(path).resolve().is_relative_to(ROOT):
            runtime_paths.add(str(Path(path).resolve()))
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and Path(path).is_file():
            runtime_paths.add(path)
    value = dict(status='SEALED_APPROVED_STAGE_A_ONLY', scope='H6_NONCANCELLATION_STAGE_A_12x10',
        SOURCE=head, base_commit=BASE, legacy_numeric_SOURCE=LEGACY_SOURCE,
        candidate_ids=list(core.IDS), epsilons=list(core.EPSILONS), allowance_factors=list(core.FACTORS),
        candidate_endpoints=120, reused_component_candidate=core.REUSED, call_caps=core.CAPS,
        prohibited_calls={k:0 for k in core.ZERO_CALLS},
        validation_budget_separate_from_primary=True, one_shot=True, retry=False,
        source_hashes={n:sha(ROOT/n) for n in SOURCE_FILES},
        public_input_hashes={n:sha(ROOT/n) for n in PUBLIC_INPUTS}, private_input_receipts=receipts,
        protected_evidence_hashes={n:sha(ROOT/n) for n in protected},
        runtime=dict(executable=sys.executable, python_version=sys.version, platform=platform.platform(),
                     file_hashes={n:sha(n) for n in sorted(runtime_paths)},
                     closure_scope='Loaded stdlib/code and native-library bytes only; no provider/venv reinventory claimed'),
        arithmetic=dict(components='11 saved complex differences plus1 literal public copy; full-H reference unrotated',
            closure_and_envelope='unchanged track_a_v1_contract.signal_envelopes; original rel/abs1e-12',
            shots='unchanged d0_saved_value_reanalysis_v1.accounting and frozen rule',
            allowance_log_B_alpha='retain original values; alpha_axis0.025, factors1/10, original binary64 order',
            unknown_policy='Save evidence and STOP on unresolved, mismatch, or numerical FAIL; no fallback'),
        physical=dict(system='linear_H6',geometry_angstrom=1.,basis='STO-3G',DF_rank=19,T=.8,
                      P=0,formula='S2',control='symmetric_directional'),
        resources_policy=resources(), approval_receipt_sha256=hashlib.sha256(APPROVAL.encode()).hexdigest(),
        old_records_unchanged=True, stage_B_authorized=False, new_publication_authorized=False,
        detailed_scientific_review=False, final_evidence_freeze=False,
        sealed_UTC=datetime.now(timezone.utc).isoformat())
    verify(value)
    input_records()  # Identity/field joins only; no differences/envelopes/accounting.
    OUT.mkdir(parents=True, exist_ok=False)
    write('sealed_contract.json', value, exclusive=True)
    write('user_approval_receipt.txt', APPROVAL, exclusive=True)
    write('artificial_tests.json', tests, exclusive=True)
    write('grant.json',dict(status='APPROVED_CURRENT_USER_STAGE_A_ONE_SHOT',SOURCE=head,
        contract_sha256=sha(OUT/'sealed_contract.json'),scientific_execution_authorized=True,
        allowed_scope=value['scope'],retry=False,stage_B_authorized=False,
        new_publication_authorized=False,call_caps=core.CAPS),exclusive=True)
    print(json.dumps(dict(status='SEALED_NOT_RUN',SOURCE=head,endpoints=120)))


def run():
    contract = read(OUT/'sealed_contract.json')
    grant = read(OUT/'grant.json')
    verify(contract)
    core.require(grant['contract_sha256'] == sha(OUT/'sealed_contract.json'), 'GRANT_CONTRACT_BINDING')
    core.require(sha(OUT/'user_approval_receipt.txt') == contract['approval_receipt_sha256'], 'APPROVAL_RECEIPT_BINDING')
    with (OUT/'one_shot.marker').open('x') as f:
        f.write(contract['SOURCE']+'\n')
    started = time.monotonic()
    ledger = dict(SOURCE=contract['SOURCE'],status='STARTED_STAGE_A',contract_sha256=sha(OUT/'sealed_contract.json'))
    def journal(state):
        ledger.update(state)
        write('ledger.json', ledger)
    permit = core.Permit(contract, grant, journal)
    journal(permit.state())
    components, endpoints = [], []
    try:
        records, recovery = input_records()
        by_id = {r['candidate_id']:r for r in contract['private_input_receipts']}
        with (OUT/'progress.jsonl').open('xb') as log:
            def emit(kind, value):
                (components if kind == 'component' else endpoints).append(value)
                log.write((json.dumps(dict(kind=kind,record=value),allow_nan=False)+'\n').encode())
                log.flush()
            result = core.analyse(records, recovery, permit, emit)
        for row in result['endpoints']+result['components']:
            row['raw_input_sha256'] = by_id[row['candidate_id']]['sha256']
        write('components.json',dict(SOURCE=contract['SOURCE'],rows=result['components']))
        write('endpoints.json',dict(SOURCE=contract['SOURCE'],rows=result['endpoints']))
        csv_rows = [dict(candidate_id=r['candidate_id'],prefix=r['prefix'],q=r['q'],
            epsilon=r['epsilon'],allowance_factor=r['allowance_factor'],status=r['status'],eligible=r['eligible'],
            N_cosine=r['shots']['cosine'] if r['shots'] else None,
            N_sine=r['shots']['sine'] if r['shots'] else None,N_total=r['N_total'],
            real_headroom=r['accounting']['axes']['real']['headroom'],
            imag_headroom=r['accounting']['axes']['imag']['headroom'],raw_input_sha256=r['raw_input_sha256'])
            for r in result['endpoints']]
        stream = io.StringIO(newline='')
        csv_writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        csv_writer.writeheader()
        csv_writer.writerows({k:json.dumps(v) if type(v) is bool or v is None else v
                             for k,v in row.items()} for row in csv_rows)
        write('endpoints.csv',stream.getvalue())
        write('numeric_validation.json',dict(SOURCE=contract['SOURCE'],status='PASS_AUTHOR_NUMERIC_CHECKS',
            checks=result['validation'],validation_budget=permit.completed,
            no_second_primary_accounting_pass=True,independent_reviewer=False))
        write('summary.json',dict(SOURCE=contract['SOURCE'],**result['summary']))
        verify(contract)
        ledger.update(status='COMPLETE_APPROVED_STAGE_A',wall_seconds=time.monotonic()-started,
                      final_input_SOURCE_runtime_protected_evidence_gate='PASS')
        write('ledger.json',ledger)
        names=('components.json','endpoints.json','endpoints.csv','numeric_validation.json','summary.json','progress.jsonl')
        write('numeric_completion.json',dict(status='COMPLETE_STAGE_A_PENDING_NATIVE_AUDIT',SOURCE=contract['SOURCE'],
            result_sha256={n:sha(OUT/n) for n in names},input_SOURCE_runtime_gate='PASS',
            stage_B_executed=False,additional_numeric_execution_authorized=False))
        write('grant_consumption_receipt.json',dict(status='CONSUMED_COMPLETED',SOURCE=contract['SOURCE'],
            grant_sha256=sha(OUT/'grant.json'),one_shot_consumed=True,further_execution_authorized=False))
        print(json.dumps(dict(status=ledger['status'],**result['summary'])))
    except BaseException as error:
        write('partial_results.json',dict(components=components,endpoints=endpoints))
        ledger.update(status='STOP_NO_RETRY',error_type=type(error).__name__,error=str(error))
        write('ledger.json',ledger)
        write('STOP.json',dict(status='STAGE_A_NUMERIC_OR_TECHNICAL_STOP',error_type=type(error).__name__,
              error=str(error),retry=False,stage_B_authorized=False,mandatory_STOP=True))
        write('grant_consumption_receipt.json',dict(status='CONSUMED_STOP_NO_RETRY',SOURCE=contract['SOURCE'],
            one_shot_consumed=True,further_execution_authorized=False))
        raise


def launch():
    contract=read(OUT/'sealed_contract.json')
    verify(contract)
    core.require(not (OUT/'one_shot.marker').exists() and not (OUT/'STOP.json').exists(), 'FRESH_UNCONSUMED_RUN_ONLY')
    write('prelaunch_gate.json',dict(status='PASS_CURRENT_INPUT_SOURCE_RUNTIME',resources=resources()),exclusive=True)
    write('launch.marker',contract['SOURCE']+'\n',exclusive=True)
    started=time.monotonic()
    with (OUT/'native_stdout.txt').open('xb') as stdout,(OUT/'native_stderr.txt').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdout=stdout,stderr=stderr)
        write('native_started.json',dict(pid=child.pid,parent_pid=os.getpid(),SOURCE=contract['SOURCE']),exclusive=True)
        pid,status,usage=os.wait4(child.pid,0)
        code=os.waitstatus_to_exitcode(status)
        child.returncode=code
    write('native_exit_evidence.json',dict(status='PASS' if code==0 else 'STOP_NATIVE_ERROR',
        SOURCE=contract['SOURCE'],native_exit_code=code,worker_pid=pid,parent_pid=os.getpid(),
        same_parent_wait4=True,reaped=True,live_owned_workers=0,wall_seconds=time.monotonic()-started,
        child_CPU_seconds=usage.ru_utime+usage.ru_stime,child_peak_RSS_bytes=usage.ru_maxrss*1024,
        stdout_sha256=sha(OUT/'native_stdout.txt'),stderr_sha256=sha(OUT/'native_stderr.txt'),
        automatic_retry=False),exclusive=True)
    if code:
        if not (OUT/'STOP.json').exists():
            write('STOP.json',dict(status='STAGE_A_NATIVE_STOP',native_exit_code=code,
                                  retry=False,stage_B_authorized=False,mandatory_STOP=True))
        raise SystemExit(code)
    print(json.dumps(dict(status='PASS_NATIVE_EXIT0_SAME_PARENT_REAP',SOURCE=contract['SOURCE'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal','run','launch'))
    parser.add_argument('--tests-receipt',type=Path)
    args=parser.parse_args()
    if args.stage=='seal':
        core.require(args.tests_receipt is not None,'TEST_RECEIPT_REQUIRED')
        seal(args.tests_receipt)
    elif args.stage=='run':run()
    else:launch()


if __name__=='__main__':main()
