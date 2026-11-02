"""B-only future runner; B-PRE does not authorize invoking this script."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True);parser.add_argument('--source-root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    sys.dont_write_bytecode=True
    root=Path(args.source_root).resolve();sys.path.insert(0,str(root/'src'))
    from trottertracks.pr_dictionary_binding.contract import verify,consume
    checked=verify(args.packet,root,args.output)
    consume(checked)
    try:
        from trottertracks.pr_dictionary_binding.engine import execute
        summary=execute(checked)
    except BaseException as error:
        # Keep consumed marker and partial files. No rollback/overwrite/retry.
        name=checked['output']/'STOP.json'
        with name.open('x') as stream:
            stream.write(json.dumps(dict(status='B_FAILED_MANDATORY_STOP_NO_RETRY',
                error_type=type(error).__name__,error=str(error)[:8192]),ensure_ascii=False)+'\n')
        raise
    else:
        with (checked['output']/'STOP.json').open('x') as stream:
            stream.write(json.dumps(dict(status='B_COMPLETE_MANDATORY_STOP',summary=summary),
                                   ensure_ascii=False,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
