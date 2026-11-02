"""F2 saved-input entry. --check-only is stdlib-only and never consumes a grant."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',required=True);p.add_argument('--source-root',required=True)
    p.add_argument('--output');p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=Path(args.source_root).resolve();sys.path.insert(0,str(root/'src'))
    from trottertracks.pr_dictionary_validation.f2_preflight import inspect_packet,verify_packet
    if args.check_only:
        v=inspect_packet(args.packet,root)
        print(json.dumps(dict(technical_status=v['technical_status'],blockers=v['blockers'],
            counts=v['contract']['counts'],science_executed=False,grant_consumed=False,
            execution_authorization_checked=False),ensure_ascii=False,indent=2))
        return 0
    if not args.output:p.error('--output is required for execution')
    v=verify_packet(args.packet,root,args.output)
    from trottertracks.pr_dictionary_validation.authorization import claim_packet
    claim_packet(v);out=v['output'];out.chmod(0o700)
    result=dict(status='FAILED_MANDATORY_STOP_NO_RETRY')
    try:
        # Coordinate only this job's environment, before importing NumPy/BLAS.
        os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            NUMBA_NUM_THREADS='1',QISKIT_NUM_PROCS='1',RAYON_NUM_THREADS='1')
        with (out/'binding_snapshot.json').open('x') as f:
            json.dump(dict(contract=v['contract'],authorization_sha256=v['authorization_digest']),f)
        for name,data in v['inputs'].items():
            target=out/'input_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
        from trottertracks.pr_dictionary_validation.f2_execution import execute_f2
        result=execute_f2(v)
    except BaseException as error:
        result=dict(status='FAILED_MANDATORY_STOP_NO_RETRY',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        with (out/'STOP.json').open('x') as f:
            json.dump(dict(result,F2_followup_authorized=False,next_owner='USER_GPT'),f,allow_nan=False)
        records=[]
        for path in sorted(out.rglob('*')):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
                records.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=h.hexdigest()))
        with (out/'output_hashes.json').open('x') as f:json.dump(dict(records=records,self_excluded=True),f)
    return 0


if __name__=='__main__':raise SystemExit(main())
