#!/usr/bin/env python3
"""Pin the newly approved one-shot retry or successful-retry continuation."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_retry_contract_v1 import make_manifest,verify_binding,SCOPES
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=SCOPES,required=True);p.add_argument('--source',required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--completed-heavy',type=Path)
    a=p.parse_args();m,g=make_manifest(ROOT,a.parent.resolve(),a.source,a.output.resolve(),a.scope,a.completed_heavy)
    verify_binding(ROOT,m,g);a.bundle.mkdir(parents=True,exist_ok=False)
    for name,value in [('manifest.json',m),('grant.json',g)]:
        with (a.bundle/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.bundle/'SHA256SUMS').write_text('\n'.join(file_hash(a.bundle/n)+'  '+n for n in ('manifest.json','grant.json'))+'\n')
    print(json.dumps({'SOURCE':a.source,'scope':a.scope,'registrations':len(m['registrations']),
        'approved_caps':m['approved_caps'],'output':m['output'],'science_executed':False}))


if __name__=='__main__':main()
