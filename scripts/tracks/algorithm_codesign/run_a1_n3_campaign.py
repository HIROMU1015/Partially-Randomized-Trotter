#!/usr/bin/env python3
"""One authorized N3 campaign; fixed conditions, no retry/resume/parameter sweep."""
from pathlib import Path
from trottertracks.algorithm_codesign.a1_n3_campaign import run,OUTPUT

if __name__=='__main__':
    root=Path(__file__).resolve().parents[3]
    print(run(root,root/OUTPUT),flush=True)
