#!/usr/bin/env python3
"""Freeze conditional metadata now; do not run science or retire heavy."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_handoff_contract_v1 import (
    AUTHORIZATION,pending_template,verify_template)
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args()
    m=pending_template(ROOT,a.parent.resolve(),a.source,a.output.resolve());verify_template(ROOT,m)
    auth=dict(schema='h8_conditional_handoff_user_authorization_v1',approved_by_user=True,
        authorization_basis=AUTHORIZATION,template_sha256=digest(m),source_commit=a.source,
        output=m['output'],remaining_cases=7,remaining_compile=18,total_compile=20,
        one_shot=True,retry=False,H8_full_campaign_authorized=False,
        conditional_start=m['conditional_start'],old_heavy_must_not_be_interrupted=True)
    a.bundle.mkdir(parents=True,exist_ok=False)
    for name,value in [('template.json',m),('conditional_authorization.json',auth)]:
        with (a.bundle/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.bundle/'SHA256SUMS').write_text('\n'.join(file_hash(a.bundle/n)+'  '+n for n in ('template.json','conditional_authorization.json'))+'\n')
    print(json.dumps({'SOURCE':a.source,'bundle':str(a.bundle),'output':m['output'],
        'remaining_cases':7,'science_executed':False,'heavy_interrupted':False}))


if __name__=='__main__':main()
