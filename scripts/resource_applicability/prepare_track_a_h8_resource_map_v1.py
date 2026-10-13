#!/usr/bin/env python3
"""Metadata preparation; --approved records the user's explicit core launch."""
import argparse,json
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_resource_map_contract_v1 import make_manifest,grant,bind_resume_parent
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-directory',type=Path,required=True)
    p.add_argument('--auth-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--approved',action='store_true')
    p.add_argument('--source',help='frozen SOURCE; use same source for manual complete-only resume')
    p.add_argument('--resume-parent',type=Path);p.add_argument('--native-exit',type=Path)
    a=p.parse_args()
    source=a.source or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    m=make_manifest(ROOT,source,a.input_directory,a.output)
    if a.resume_parent:
        if a.native_exit is None:raise ValueError('RESUME_NATIVE_EXIT_REQUIRED')
        bind_resume_parent(m,a.resume_parent,a.native_exit)
    g=grant(m,a.approved)
    a.auth_directory.mkdir(exist_ok=False)
    for n,v in [('manifest.json',m),('grant.json',g)]:
        (a.auth_directory/n).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(SOURCE=source,manifest_sha256=file_hash(a.auth_directory/'manifest.json'),
        grant_sha256=file_hash(a.auth_directory/'grant.json'),runtime_authorization=a.approved)))

if __name__=='__main__':main()
