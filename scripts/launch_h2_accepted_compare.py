"""Isolated new COMPARE CLI. Default is stdlib-only blocked/preflight inspection."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval');parser.add_argument('--grant')
    args=parser.parse_args()
    if args.execute and not (args.approval and args.grant):parser.error('NEW_EXPLICIT_COMPARE_APPROVAL_AND_GRANT_REQUIRED')
    cfg=json.loads(Path(args.contract).read_text())
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):parser.error('ISOLATED_I_S_B_REQUIRED')
    if str(Path(sys.executable).resolve())!=cfg['runtime']['python_target']:parser.error('SELECTED_INTERPRETER_REQUIRED')
    sys.path[:]=[str(Path(__file__).resolve().parent)]+cfg['runtime']['sys_path']
    import h2_accepted_compare_contract as c
    result=c.execute(cfg,c.read(args.approval),c.read(args.grant)) if args.execute else c.preflight(cfg)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
