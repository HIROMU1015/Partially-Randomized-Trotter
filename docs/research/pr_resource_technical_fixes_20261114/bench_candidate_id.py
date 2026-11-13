"""Synthetic parse benchmark against the exact old function, never a kernel."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve()
sys.path.insert(0,str(root/'src'));sys.dont_write_bytecode=True
from trottertracks import pf_pr_rpe_pipeline as p

baseline='ce0517de6e7b58cf4bf4c69f99f998d86099c406'
path='src/trottertracks/pf_pr_rpe_pipeline.py'
source=subprocess.check_output(['git','show',baseline+':'+path],cwd=root)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='candidate_id')
scope=dict(ast=ast,json=json,hashlib=hashlib,Path=Path,__file__=p.__file__)
exec(compile(ast.Module(body=[node],type_ignores=[]),baseline+':'+path,'exec'),scope)
old=scope['candidate_id']
inputs=[dict(system='TOY',method=method,formula=formula,prefix=2,T=.1,q=q,
             R_outer=7 if method=='PR' else None,K=2 if method=='PR' else None)
        for method in ('FULL','B0','B0I','PR') for formula in ('2nd','4th','4th(m5_best)') for q in (1,3,4096)]
assert [old(c) for c in inputs]==[p.candidate_id(c) for c in inputs]
def measure(function):
    samples=[]
    for repeat in range(5):
        start=time.perf_counter_ns()
        for i in range(200):function(inputs[i%len(inputs)])
        samples.append((time.perf_counter_ns()-start)/200)
    return dict(samples_ns_per_call=samples,median_ns_per_call=statistics.median(samples))
before=measure(old);after=measure(p.candidate_id)
print(json.dumps(dict(original_source=baseline,python=sys.version,synthetic_input_count=len(inputs),
    outputs_identical=True,before=before,after=after,
    speedup=before['median_ns_per_call']/after['median_ns_per_call'],
    candidate_rule_sha256=hashlib.sha256((root/'scripts/h46_long_time_signal.py').read_bytes()).hexdigest()),indent=2))
