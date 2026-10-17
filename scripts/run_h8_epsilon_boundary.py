#!/usr/bin/env python3
"""One-shot approved C_EPS_H8 only. No other campaign stage can be launched."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/trotterlib'),str(ROOT/'scripts')]
import h8_epsilon_boundary as numeric
import h8_noncancellation_contract as old_adapter
from targeted_validation_plan import digest

MANIFEST='references/h8_epsilon_boundary_inputs_v1.json'
SCOPE=dict(stage='C_EPS_H8',epsilon=list(numeric.EPS),factor=1,bias=list(numeric.MODES),
           deterministic_candidates=184,PR_candidates=6,n_PR=32,P=0,
           new_sampling=0,new_compile=0,new_signal=0,new_model_prediction=0,
           new_bootstrap=0,automatic_retry=False,next_stage_authorized=False)
FILES=('src/trotterlib/h8_epsilon_boundary.py','scripts/run_h8_epsilon_boundary.py',
       'tests/test_h8_epsilon_boundary.py',MANIFEST,
       'src/trotterlib/h8_noncancellation.py','scripts/h8_noncancellation_contract.py',
       'scripts/d0_saved_value_reanalysis_v1.py','scripts/d0_frozen_shot_rule_v1.py',
       'src/trotterlib/targeted_validation_plan.py','references/h8_noncancellation_inputs_v1.json')


def sha(data):return hashlib.sha256(data).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def runtime():
    import math, fractions
    paths=[Path(sys.executable).resolve(),Path(statistics.__file__),Path(fractions.__file__),Path(math.__file__)]
    return dict(python=sys.version,files={str(p):sha(p.read_bytes()) for p in paths})


def authorize(config,grant,root=ROOT):
    req=numeric.require
    req(grant.get('authorized') is True and config.get('authorized') is True,
        'EXPLICIT_EXECUTION_APPROVAL_REQUIRED')
    req(config['scope']==SCOPE and grant['scope_sha256']==digest(SCOPE) and
        grant['config_sha256']==digest(config) and grant['SOURCE']==config['SOURCE'],
        'SCOPE_SOURCE_CONFIG_BINDING')
    req(grant['user_instruction']=='実行準備ができたらそのまま実行してよい' and
        grant['one_shot'] is True and grant['automatic_retry'] is False and
        grant['next_stage_authorized'] is False, 'ONE_SHOT_USER_APPROVAL')
    req(config['runtime']==runtime(),'RUNTIME_DRIFT')
    req(set(config['source_files'])==set(FILES),'SOURCE_CLOSURE')
    for path,expected in config['source_files'].items():
        data=(root/path).read_bytes()
        req(sha(data)==expected and subprocess.check_output(
            ['git','--no-optional-locks','-C',str(root),'show',config['SOURCE']+':'+path])==data,
            'SOURCE_DRIFT:'+path)
    req(sha((root/MANIFEST).read_bytes())==config['manifest_sha256'],'MANIFEST_DRIFT')


class ApprovedInputPermit:
    """Projection protocol only; instantiated after the new C grant passes."""
    def require(self,kind):numeric.require(kind=='SAVED_H8','SAVED_DATA_ONLY')


def load_public(manifest,root=ROOT):
    raw={}
    for row in manifest['public_files']:
        data=subprocess.check_output(['git','--no-optional-locks','-C',str(root),'show',
                                      row['commit']+':'+row['path']])
        numeric.require(len(data)==row['bytes'] and sha(data)==row['sha256'],'PUBLIC_INPUT_DRIFT')
        raw[row['role']]=data
    return raw


def execute(config,grant,*,root=ROOT):
    authorize(config,grant,root)
    out=Path(config['output']).resolve()
    numeric.require(not out.is_relative_to(root) and out.parent.is_dir(),'PRIVATE_NEW_OUTPUT_REQUIRED')
    out.mkdir(exist_ok=False)
    write(out/'one_shot_consumed.json',dict(SOURCE=config['SOURCE'],config_sha256=digest(config),
          grant_sha256=digest(grant),consumed=True,at=now(),automatic_retry=False))
    start=time.monotonic()
    watched={str(root/p):(root/p).read_bytes() for p in FILES}
    original_runtime=runtime()
    def guard():
        numeric.require(runtime()==original_runtime,'RUNTIME_CHANGED')
        for path,data in watched.items():
            numeric.require(Path(path).read_bytes()==data,'BOUND_FILE_CHANGED:'+path)
    try:
        manifest=json.loads((root/MANIFEST).read_text())
        public=load_public(manifest,root)
        original=json.loads((root/old_adapter.MANIFEST).read_text())
        snapshot=Path(config['input_root'])
        for row in original['files']:
            path=snapshot/row['snapshot_path']
            data=old_adapter.read_verified(path,row)
            watched[str(path)]=data
        fresh=dict(at=now(),SOURCE=config['SOURCE'],scope=SCOPE,runtime=original_runtime,
                   source_and_snapshot_hashes='PASS',public_git_inputs='PASS',
                   permitted_CPU_count=len(os.sched_getaffinity(0)),
                   resource_policy='small_stdlib_scalar_job_single_process_no_legacy_caps',
                   memory_observation=next(x.strip() for x in Path('/proc/meminfo').read_text().splitlines()
                                            if x.startswith('MemAvailable:')),
                   compiler='NOT_EXECUTED_SAVED_COMPILER_RECEIPTS_ONLY')
        write(out/'fresh_gate.json',fresh)
        data,registry,receipts=old_adapter.load_saved(original,snapshot,ApprovedInputPermit())
        six=[json.loads(x) for x in gzip.decompress(public['six_rows']).splitlines()]
        nc=[json.loads(x) for x in gzip.decompress(public['noncancel_rows']).splitlines()]
        gate=json.loads(public['six_final_gate'])
        numeric.require(gate['status']=='PASS_MANDATORY_STOP','SAVED_SIX_FINAL_GATE')
        guard()
        # The data are a few thousand scalar rows. Hash checks at endpoint boundaries
        # avoid repeatedly re-reading MB-sized inputs for every scalar operation.
        result=numeric.evaluate(data,registry,six,nc)
        guard()
        write(out/'result.json',result)
        write(out/'input_receipts.json',dict(snapshot=receipts,public=manifest['public_files']))
        write(out/'terminal.json',dict(status=result['status'],at=now(),SOURCE=config['SOURCE'],
            wall_seconds=time.monotonic()-start,rows=result['new_endpoint_rows'],
            anchor_rows_reproduced=result['anchor_rows_reproduced'],
            owned_scientific_children=0,next_stage_authorized=False,source_input_runtime_final_gate='PASS'))
        return 0
    except BaseException as exc:
        write(out/'STOP.json',dict(status='STOP_MANDATORY_STOP',error=type(exc).__name__+':'+str(exc),
                                 at=now(),automatic_retry=False,next_stage_authorized=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--grant',required=True,type=Path)
    args=p.parse_args()
    return execute(json.loads(args.config.read_text()),json.loads(args.grant.read_text()))


if __name__=='__main__':raise SystemExit(main())
