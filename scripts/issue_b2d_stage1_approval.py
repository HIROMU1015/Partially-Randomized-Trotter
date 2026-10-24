#!/usr/bin/env python3
"""USER ONLY: ordinary-user publication AFTER actual chat approval.

This operation issues a fresh grant but does not claim or execute it. Codex must
not run this command during preparation or manufacture the user's approval.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft",required=True)
    p.add_argument("--packet",required=True)
    p.add_argument("--approval-file",required=True)
    p.add_argument("--approval-reference",required=True)
    p.add_argument("--confirm-contract",required=True)
    p.add_argument("--acknowledge-manual-user-review",action="store_true",required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"src"))
    from trottertracks.b2d_evaluator.manual_authority import issue_manual_packet
    print(json.dumps(issue_manual_packet(a.draft,a.packet,a.approval_file,
                                         a.approval_reference,a.confirm_contract),indent=2))


if __name__=="__main__": main()
