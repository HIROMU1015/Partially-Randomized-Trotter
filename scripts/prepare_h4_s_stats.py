"""Seal an approved S_STATS scope without performing its numerical analysis."""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h4_dp_contract as c
import h4_s_stats as stats
import h4_s_stats_witness

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ['scripts/h4_dp_contract.py', 'scripts/h4_s_stats.py',
                'scripts/h4_s_stats_witness.py', 'scripts/prepare_h4_s_stats.py']
PHASE = 'artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/'
D = 'artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/'
INPUTS = dict(original_S_STATS_contract=PHASE+'S_STATS_contract_DRAFT.json',
    P_registration=PHASE+'P_registration.json', D_result=D+'deterministic_result.json',
    D_completion_audit=D+'completion_audit.json', P_after_D_contract=D+'P_after_D_contract_DRAFT.json',
    design_document='docs/research/track_a_h4_dp_phase0_20261019/h4_pr_confirmation_design_DRAFT.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--approval-receipt', type=Path, required=True)
    p.add_argument('--previous-worktree', type=Path, required=True)
    args = p.parse_args()
    approval = c.read(args.approval_receipt)
    c.require(approval['stage'] == 'S_STATS' and approval['approved_by_user'] is True
              and approval['P_authorized'] is False and approval['one_shot'] is True, 'S_STATS_USER_APPROVAL')
    private = ROOT/'.h4_s_stats_private'
    run = private/'run01_20261019'
    run.mkdir(exist_ok=False)
    copied_receipt = run/'user_approval_PRIVATE.json'
    copied_receipt.write_bytes(args.approval_receipt.read_bytes())
    preserved = []
    previous_private = args.previous_worktree.resolve()/'.h4_dp_phase0_private'
    for name in ['stop_preservation_manifest.json', 'D_run02_20261019/successful_run_preservation_PRIVATE.json']:
        for row in c.read(previous_private/name):
            c.require(c.sha(row['path']) == row['sha256'] and c.sha(row['copy']) == row['sha256'],
                      'OLD_D_RECORD_CHANGED')
            preserved.append(row)
    c.save(run/'previous_preservation_manifest_PRIVATE.json', preserved)
    inputs = []
    for role, name in INPUTS.items():
        raw = subprocess.check_output(['git', 'show', stats.FIXED_RESULT+':'+name], cwd=ROOT)
        h = hashlib.sha256(raw).hexdigest()
        c.require(c.sha(ROOT/name) == h, 'FIXED_INPUT_CHANGED')
        path = run/'inputs'/(role+('.md' if role == 'design_document' else '.json'))
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        inputs.append(dict(role=role, repo_path=name, path=str(path), sha256=h))
    source_hashes = {name: c.sha(ROOT/name) for name in SOURCE_FILES}
    for name, h in source_hashes.items():
        raw = subprocess.check_output(['git', 'show', args.source+':'+name], cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest() == h, 'FIXED_STATS_CODE_SOURCE')
    c.require(not any(name.split('.')[0] in {'numpy', 'scipy', 'qiskit', 'trotterlib'}
                      for name in sys.modules), 'STATS_STDLIB_ONLY')
    runtime_files = {Path(sys.executable).resolve()}
    for module in list(sys.modules.values()):
        for key in ['__file__', '__cached__']:
            name = getattr(module, key, None)
            if name:
                path = Path(name).resolve()
                if path.is_file() and ROOT not in path.parents:
                    runtime_files.add(path)
    # Include mapped CPython/math/libc binaries used by this scalar calculation.
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith('/'):
            path = Path(fields[-1]).resolve()
            if path.is_file() and ROOT not in path.parents:
                runtime_files.add(path)
    runtime_hashes = {str(path): c.sha(path) for path in sorted(runtime_files)}
    mem = {x.split(':', 1)[0]: x.split(':', 1)[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()}
    resources = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        allowed_logical_cpus=len(os.sched_getaffinity(0)), memory_available=mem['MemAvailable'],
        disk_free_bytes=shutil.disk_usage(run).free, processes=1, task_is_small_scalar_summary_analysis=True,
        arbitrary_CPU_memory_wall_caps=None, affinity_changed=False, shared_settings_changed=False,
        transient_resource_pressure_is_diagnostic_only=True)
    c.save(run/'resources_PRIVATE.json', resources)
    spec = dict(stage='S_STATS', status='SEALED_FOR_EXECUTION', technical_sealed=True,
        SOURCE=args.source, fixed_result_commit=stats.FIXED_RESULT, source_hashes=source_hashes,
        inputs=inputs, runtime_hashes=runtime_hashes, python=sys.executable, python_sha256=c.sha(sys.executable),
        output=str(run/'output'), lock=str(run/'ONE_SHOT.json'), numeric_design_pass=1,
        maximum_normal_scenarios=84, fixed_reference_sign_comparisons=14, retry=False,
        next_stage_authorized=False, P_authorized=False, resources_receipt_sha256=c.sha(run/'resources_PRIVATE.json'))
    grant = dict(stage='S_STATS', SOURCE=args.source, spec_digest=c.digest(spec),
        approved_by_user=True, runtime_authorization=True, one_shot=True, retry=False,
        next_stage_authorized=False, approval_receipt=str(copied_receipt), approval_receipt_sha256=c.sha(copied_receipt))
    c.save(run/'spec_SEALED_PRIVATE.json', spec)
    c.save(run/'grant_PRIVATE.json', grant)
    stats.authorize(spec, grant)
    c.save(run/'preflight_PRIVATE.json', dict(status='PASS_WITHOUT_NUMERIC_DESIGN',
        source_files=len(source_hashes), input_files=len(inputs), runtime_files=len(runtime_hashes),
        previous_records_preserved=len(preserved), no_numeric_design_executed=True))
    print('S_STATS_SEALED; NO_NUMERIC_ANALYSIS_YET; SOURCE='+args.source)


if __name__ == '__main__':main()
