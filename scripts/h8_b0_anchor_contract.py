"""Executable contract derived from the unchanged, hash-pinned acquisition plan."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h8_b0_anchor_plan as original

PLAN_PATH='artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/acquisition_plan_UNAPPROVED_v1.json'
PLAN_SHA='c7755c88d9a6e465411292bc0d6196c8d8f05f8c5642994fe3d1291d93a27273'
TARGET_PATH='artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/saved_control_targets_v1.json'
SCOPE=original.SCOPE
PROFILE=original.PROFILE
COMPILER=original.COMPILER
PROVIDER=original.PROVIDER
RUNTIME_MAP_SHA='850568c8018943b91dc9dbb031198b684112f88f8870f5a217253c61ee5fb958'
PROVIDER_MAP_SHA='9382176df9a1353c9f02b842edec5943dfbeb06b223387f6a99542e53c878bc0'
digest=original.digest


def plan():
    raw=(ROOT/PLAN_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PLAN_SHA:raise ValueError('ORIGINAL_PLAN_BYTES_CHANGED')
    p=json.loads(raw);original.validate(p);r=p['resources']
    limits=dict(r,max_workers=4,AS_bytes=r['driver_AS_bytes'],RSS_bytes=r['driver_RSS_bytes'],
        CPU_seconds=r['cumulative_CPU_seconds'],aggregate_CPU_seconds=r['cumulative_CPU_seconds'],
        worker_CPU_seconds=r['cumulative_CPU_seconds'],wall_seconds=r['phase_wall_seconds'],
        minimum_free_inodes=r['fresh_free_inodes'],log_bytes=r['logs_bytes'],
        output_files_max=r['output_files'])
    return dict(schema='H8_B0_ANCHOR_EXECUTION_CONTRACT_V1',scope=SCOPE,
        original_plan_sha256=PLAN_SHA,cases=p['cases'],axes=p['axes'],waves=p['waves'],
        call_caps=p['call_caps'],resources=limits,binding=p['binding'],
        control=p['control'],scalar_policy=p['scalar_policy'],
        compiler_options=p['binding']['compiler_options'],control_tolerance=1e-8,
        instruction_cap=r['native_instructions'],transpiled_cap=r['transpiled_instructions'],
        one_shot=True,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,
        fit_authorized=False,holdout_in_fit=False)


SOURCES=(
    'scripts/h8_b0_anchor_contract.py','scripts/h8_b0_anchor_runtime.py',
    'scripts/h8_b0_anchor_adapter.py','scripts/run_h8_b0_anchors.py',
    'scripts/launch_h8_b0_anchors.py','scripts/prepare_h8_b0_anchors.py',
    'scripts/h8_strengthened_export.py','scripts/h8_strengthened_contract.py',
    'scripts/h8_strengthened_runtime.py','scripts/launch_h8_strengthened_phase_a.py',
    'scripts/track_a_phase_a_parallel.py','src/trotterlib/h8_b0_anchor_plan.py',
    PLAN_PATH,TARGET_PATH,
)
