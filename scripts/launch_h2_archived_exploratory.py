"""Default: isolated metadata preflight. Execution needs separate approval/grant."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval');parser.add_argument('--grant')
    args=parser.parse_args()
    if args.execute and not (args.approval and args.grant):parser.error('SEPARATE_EXPLICIT_EXPLORATORY_APPROVAL_AND_GRANT_REQUIRED')
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):parser.error('ISOLATED_I_S_B_REQUIRED')
    cfg=json.loads(Path(args.contract).read_text())
    if str(Path(sys.executable).resolve())!=cfg['runtime']['python_target']:parser.error('SELECTED_ANALYSIS_PYTHON_REQUIRED')
    sys.path[:]=[str(Path(__file__).resolve().parent)]+cfg['runtime']['sys_path']
    import h2_archived_exploratory as c
    result=c.execute(cfg,c.read(args.approval),c.read(args.grant)) if args.execute else c.preflight(cfg)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
