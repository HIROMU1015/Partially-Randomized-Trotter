"""Metadata-only by default; one explicit fixed-package approval before real arithmetic."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--approval');p.add_argument('--grant');a=p.parse_args()
    if a.execute and not(a.approval and a.grant):p.error('EXPLICIT_FIXED_BATCH_APPROVAL_REQUIRED')
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):p.error('ISOLATED_I_S_B_REQUIRED')
    cfg=json.loads(Path(a.contract).read_text());root=Path(__file__).resolve().parents[1]
    sys.path[:]=[str(root/'scripts'),str(root/'src/trotterlib'),str(root.parent/'h246-reversal-batch-analysis-assets-20261024/provider/src')]+cfg['runtime']['sys_path']
    import h246_reversal_batch as c
    result=c.execute(cfg,c.read(a.approval),c.read(a.grant)) if a.execute else c.preflight(cfg)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
