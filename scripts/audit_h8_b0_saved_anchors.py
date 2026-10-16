"""Read-only audit of saved run02; no compile, numerical fit or grant creation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
from h8_b0_saved_anchor_audit import audit_saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path)
    parser.add_argument('--execution-worktree',type=Path)
    parser.add_argument('--published-worktree',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    values=(args.evidence_root,args.execution_worktree,args.published_worktree,args.output)
    if not any(values):
        print(json.dumps(dict(mode='READ_ONLY_SAVED_EVIDENCE',new_compile=0,new_fit=0,
            runtime_authorization=False,next_stage_authorized=False)));return
    if not all(values):parser.error('all evidence and output paths are required')
    destination=args.output.resolve()
    if destination.exists() or any(destination.is_relative_to(p.resolve()) for p in values[:3]):
        parser.error('output must be new and outside original evidence/worktrees')
    result=audit_saved(args.evidence_root,args.execution_worktree,args.published_worktree)
    with destination.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
