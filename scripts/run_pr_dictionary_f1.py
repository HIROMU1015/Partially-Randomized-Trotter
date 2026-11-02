"""F1-only runner definition. Stage I does not authorize invoking this script."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.source_root).resolve()
    # Only the stdlib gate is imported before permission and identity checks.
    sys.path.insert(0,str(root/'src'))
    from trottertracks.pr_dictionary_validation.authorization import verify_packet,claim_packet
    verified=verify_packet(args.packet,root,args.output)
    claim_packet(verified)
    try:
        from trottertracks.pr_dictionary_validation.execution import execute_registered_f1
        result=execute_registered_f1(verified)
    except BaseException as error:
        (verified['output']/'STOP.json').write_text(json.dumps(dict(
            status='FAILED_MANDATORY_STOP_NO_RETRY',error_type=type(error).__name__,
            error=str(error),authorization=verified['authorization_digest']),ensure_ascii=False)+'\n')
        raise
    else:
        (verified['output']/'STOP.json').write_text(json.dumps(dict(
            status='COMPLETED_MANDATORY_STOP',result=result,
            authorization=verified['authorization_digest']),ensure_ascii=False)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
