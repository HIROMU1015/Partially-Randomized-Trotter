"""Update an H4 saved-result comparison by appending fixed sources to INPUTS.json."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from trottertracks.saved_rpe_cost_catalog import Catalog, Reader, ref, report

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--source',action='append',default=[],help='Additional fixed COMMIT:repository/path (repeatable)')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.perf_counter()
    inputs=json.loads(args.inputs.read_text());reader=Reader(ROOT);catalog=Catalog(reader,inputs['comparison'])
    sources=inputs['sources']+[dict(zip(('commit','path'),s.split(':',1)))for s in args.source]
    sources=[dict(s,commit=reader.commit(s['commit']))for s in sources]
    for source in sources:
        print('READ',source['commit'],source['path'],flush=True);catalog.add(source)
    result=catalog.result()
    result['execution_source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['input_manifest']=str(args.inputs);result['wall_seconds']=time.perf_counter()-start
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'CATALOG.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'REPORT.md').write_text(report(result))
    (args.output/'INPUTS.json').write_text(json.dumps(dict(inputs,sources=sources),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(candidates=len(result['candidates']),duplicates=result['duplicates_removed'],
        minima={k:v['conditional_cost']for k,v in result['method_minima'].items()})),flush=True)


if __name__=='__main__':main()
