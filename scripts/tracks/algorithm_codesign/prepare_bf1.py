#!/usr/bin/env python3
"""Formula/synthetic-only preparation; never opens a science input."""
import os
from pathlib import Path
import subprocess
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT/'src'))

from trottertracks.algorithm_codesign.domain import Domain, fixed_references
from trottertracks.algorithm_codesign.formal_check import order_residuals
from trottertracks.algorithm_codesign.freeze import (EXECUTION_ID, PREPARATION_REL, REVIEWED_COMMIT,
    SCIENCE_OUTPUT_REL, canonical, environment, git, sha, source_inventory, write_new)
from trottertracks.algorithm_codesign.input_contract import INPUT


def main():
    output = ROOT/PREPARATION_REL
    if output.exists():
        raise FileExistsError('Preparation output already exists; do not overwrite')
    domain = Domain()
    manifest = domain.manifest()
    manifest['fixed_references'] = []
    for p, degree in zip(fixed_references(domain), (2, 4, 4, 8)):
        row = p.record()
        row['formal_AB_word_residuals'] = order_residuals(p, degree)
        manifest['fixed_references'].append(row)
    command = [sys.executable, '-B', '-m', 'pytest', '-q',
               'tests/tracks/algorithm_codesign', '-p', 'no:cacheprovider']
    proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
                          text=True, capture_output=True, timeout=60)
    report = dict(schema='bf1_synthetic_semantic_report_v1', scope='synthetic_small_matrix_and_formula_only',
                  all_passed=proc.returncode == 0, returncode=proc.returncode,
                  command=command, stdout=proc.stdout, stderr=proc.stderr,
                  molecular_npz_resolve_stat_hash_load=0, science_signals=0, trajectories=0,
                  circuits_built=0, compilations=0, gpu_queries=0, full_test_suite=False,
                  evidence_scope='local uncommitted source-content checks; not immutable CI')
    print(proc.stdout, end='')
    if proc.returncode:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    plan = dict(schema='bf1_preregistration_source_plan_v1', execution_id=EXECUTION_ID,
                status='CONTENT_SEALED_UNCOMMITTED_EXECUTION_NOT_AUTHORIZED', reviewed_BF0_commit=REVIEWED_COMMIT,
                preparation_head=git(ROOT, 'rev-parse', 'HEAD'), branch=git(ROOT, 'branch', '--show-current'),
                input=INPUT, environment=environment(), sources=source_inventory(ROOT),
                domain_fingerprint=sha(canonical(manifest)),
                output_relative=SCIENCE_OUTPUT_REL,
                primary=dict(epsilon=.01, alpha=.05, materiality_ratio=.95, comparator='all O/L evaluations plus four fixed references'),
                bridge=dict(epsilon=.05, new_coefficient_search=False),
                arms=['O', 'L', 'F'], evaluations_per_arm=32,
                q=[1, 2, 4, 8], R_bud=[5, 10, 20, 40, 80], K=[2, '4 only via frozen trigger'],
                resource_caps=dict(cpu_seconds=28800, wall_seconds=14400, workers=1, blas_threads=1,
                    rss_bytes=4*1024**3, aggregate_rss_bytes=8*1024**3, output_bytes=128*1024**2,
                    ideal_cells=400, finite_cells=4000, science_retries=0),
                scientific_results_present=False, BF1_executed=False, science_execution_authorized=False,
                mandatory_stop=True, automatic_next_stage=None, BF2_authorized=False)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'domain_manifest.json', manifest)
    write_new(output/'synthetic_semantic_report.json', report)
    write_new(output/'source_plan.json', plan)
    domain_bytes = (output/'domain_manifest.json').read_bytes()
    tests_bytes = (output/'synthetic_semantic_report.json').read_bytes()
    auth = dict(schema='bf1_authorization_draft_v1', execution_id=EXECUTION_ID,
                status='DENIED_DRAFT_AWAITING_SOURCE_COMMIT_AND_EXECUTION_REVIEW',
                science_execution_authorized=False, source_commit=None, source_content_sealed=True,
                plan_fingerprint=sha(canonical(plan)), domain_sha256=sha(domain_bytes),
                semantic_report_sha256=sha(tests_bytes), review_verdict=None,
                explicit_user_execution_instruction=None, input=INPUT, run_count=1, mandatory_stop=True,
                science_retry_authorized=False, automatic_next_stage=None, BF2_authorized=False)
    write_new(output/'authorization_draft.json', auth)
    write_new(output/'preparation_audit.json', dict(status='BF1_PREPARATION_COMPLETE_REVIEW_REQUIRED',
               plan_fingerprint=sha(canonical(plan)), domain_sha256=sha(domain_bytes),
               semantic_report_sha256=sha(tests_bytes), source_file_count=len(plan['sources']),
               science_input_operations=0, science_execution_authorized=False,
               source_commit_bound=False, mandatory_stop=True, automatic_next_stage=None))
    print('Prepared source-content packet; science execution remains NOT authorized.')


if __name__ == '__main__':
    main()
