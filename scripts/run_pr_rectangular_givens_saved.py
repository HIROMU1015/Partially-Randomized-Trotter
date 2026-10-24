#!/usr/bin/env python3
"""PS48 candidate A: explicit metadata-only preflight OR separately granted science."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--saved-input-root',required=True)
    parser.add_argument('--baseline-root',required=True)
    parser.add_argument('--plan')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--packet');parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.source_root).resolve()
    sys.path.insert(0,str(root/'src'))
    from trottertracks.pr_dictionary_validation.rectangular_preflight import (
        ARTIFACT,inspect_preparation,verify_execution,
    )
    plan=Path(args.plan) if args.plan else root/ARTIFACT
    if args.check_only:
        v=inspect_preparation(plan,root,args.saved_input_root,args.baseline_root)
        print(json.dumps({k:v[k] for k in ('technical_status','angle_status',
            'scientific_execution_count','plan_sha256')},indent=2))
        return 0
    if not args.packet or not args.output: parser.error('--execute needs --packet and --output')
    v=verify_execution(args.packet,plan,root,args.saved_input_root,args.baseline_root,args.output)
    from trottertracks.pr_dictionary_validation.authorization import claim_packet
    v['output'].parent.mkdir(parents=True,exist_ok=True)
    claim_packet(v);out=v['output'];out.chmod(0o700)
    result=dict(status='FAILED_MANDATORY_STOP_NO_RETRY')
    try:
        os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            NUMBA_NUM_THREADS='1',QISKIT_NUM_PROCS='1',RAYON_NUM_THREADS='1')
        with (out/'binding_snapshot.json').open('x') as f:
            json.dump(dict(plan=v['plan'],plan_sha256=v['plan_sha256'],
                authorization_sha256=v['authorization_digest']),f)
        for name in v['plan']['saved_B_inputs']:
            target=out/'input_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write((v['saved_input_root']/name).read_bytes())
        from trottertracks.pr_dictionary_validation.rectangular_execution import execute
        result=execute(v)
    except BaseException as error:
        result=dict(status='FAILED_MANDATORY_STOP_NO_RETRY',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        with (out/'STOP.json').open('x') as f:
            json.dump(dict(result,next_owner='USER_GPT',followup_authorized=False),f,allow_nan=False)
        records=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()]
        with (out/'output_hashes.json').open('x') as f:
            json.dump(dict(records=records,self_excluded=True),f)
    return 0


if __name__=='__main__':raise SystemExit(main())
