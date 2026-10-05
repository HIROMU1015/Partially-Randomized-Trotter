#!/usr/bin/env python3
"""Formula/synthetic-only preparation; never opens a science input."""
import os
import json
from pathlib import Path
import subprocess
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT/'src'))

from trottertracks.algorithm_codesign.freeze import (AUTHORIZATION_REL, DOMAIN_REL, DOMAIN_SHA256,
    EXECUTION_ID, PREPARATION_REL, PUBLICATION_SCHEME, REVIEWED_COMMIT, REVISION_REVIEWED_COMMIT,
    SCIENCE_OUTPUT_REL, canonical, environment, git, sha, source_inventory, write_new)
from trottertracks.algorithm_codesign.input_contract import INPUT


def main():
    output = ROOT/PREPARATION_REL
    if output.exists():
        raise FileExistsError('Preparation output already exists; do not overwrite')
    domain_bytes = (ROOT/DOMAIN_REL).read_bytes()
    if sha(domain_bytes) != DOMAIN_SHA256:
        raise ValueError('Frozen v1 domain changed; do not regenerate it')
    manifest = json.loads(domain_bytes)
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
    plan = dict(schema='bf1_preregistration_source_plan_v2', execution_id=EXECUTION_ID,
                status='CONTENT_SEALED_UNCOMMITTED_EXECUTION_NOT_AUTHORIZED', reviewed_BF0_commit=REVIEWED_COMMIT,
                reviewed_execution_gate_commit=REVISION_REVIEWED_COMMIT,
                review_verdict='PROCEED_BF1_AFTER_MINIMAL_EXECUTION_GATE_REVISION',
                publication_scheme=PUBLICATION_SCHEME, authorization_relative=AUTHORIZATION_REL,
                preparation_head=git(ROOT, 'rev-parse', 'HEAD'), branch=git(ROOT, 'branch', '--show-current'),
                input=INPUT, environment=environment(), sources=source_inventory(ROOT),
                domain_fingerprint=sha(canonical(manifest)),
                domain_relative=DOMAIN_REL, domain_sha256=DOMAIN_SHA256,
                output_relative=SCIENCE_OUTPUT_REL,
                primary=dict(epsilon=.01, alpha=.05, materiality_ratio=.95, comparator='all O/L evaluations plus four fixed references'),
                bridge=dict(epsilon=.05, new_coefficient_search=False),
                arms=['O', 'L', 'F'], evaluations_per_arm=32,
                cross_objectives=dict(phase='after all searches and primary finite rescoring, before bridge',
                    union='all O/L/F evaluated coefficients plus four fixed references', arms=['O', 'L', 'F'],
                    epsilon=.01, cache_only=True, affects_search=False, affects_primary_classification=False,
                    added_ideal_or_finite_cells=0, maximum_logical_scores=300),
                q=[1, 2, 4, 8], R_bud=[5, 10, 20, 40, 80], K=[2, '4 only via frozen trigger'],
                resource_caps=dict(cpu_seconds=28800, wall_seconds=14400, workers=1, blas_threads=1,
                    rss_bytes=4*1024**3, aggregate_rss_bytes=8*1024**3, output_bytes=128*1024**2,
                    ideal_cells=400, finite_cells=4000, science_retries=0),
                scientific_results_present=False, BF1_executed=False, science_execution_authorized=False,
                mandatory_stop=True, automatic_next_stage=None, BF2_authorized=False)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'synthetic_semantic_report.json', report)
    write_new(output/'source_plan.json', plan)
    tests_bytes = (output/'synthetic_semantic_report.json').read_bytes()
    auth = dict(schema='bf1_authorization_draft_v2', execution_id=EXECUTION_ID,
                status='DENIED_DRAFT_AWAITING_SOURCE_COMMIT_AND_EXECUTION_REVIEW',
                science_execution_authorized=False, source_commit=None, source_content_sealed=True,
                publication_scheme=PUBLICATION_SCHEME, formal_authorization_relative=AUTHORIZATION_REL,
                plan_fingerprint=sha(canonical(plan)), domain_sha256=sha(domain_bytes),
                semantic_report_sha256=sha(tests_bytes), review_verdict=None,
                explicit_user_execution_instruction=None, input=INPUT, run_count=1, mandatory_stop=True,
                science_retry_authorized=False, automatic_next_stage=None, BF2_authorized=False)
    write_new(output/'authorization_draft.json', auth)
    write_new(output/'preparation_audit.json', dict(status='BF1_PREPARATION_COMPLETE_REVIEW_REQUIRED',
               plan_fingerprint=sha(canonical(plan)), domain_sha256=sha(domain_bytes),
               semantic_report_sha256=sha(tests_bytes), source_file_count=len(plan['sources']),
               publication_scheme=PUBLICATION_SCHEME, original_domain_reused=True,
               domain_relative=DOMAIN_REL, previous_preparation_preserved=True,
               science_input_operations=0, science_execution_authorized=False,
               source_commit_bound=False, mandatory_stop=True, automatic_next_stage=None))
    print('Prepared source-content packet; science execution remains NOT authorized.')


if __name__ == '__main__':
    main()
