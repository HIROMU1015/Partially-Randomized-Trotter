"""Artificial negative control: candidate backend loaded before frame workers."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
TEST=ROOT/'tests/management/test_research_c_c1_frame_repair.py'
code='''import importlib.util,json
spec=importlib.util.spec_from_file_location('artificial_probe',%r)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import openfermion
m.stage.unthresholded_sparse([(((0,1),(0,0)),.3)],8)
print(json.dumps(m.parallel_probe(legacy=True)))
''' % str(TEST)
records=[]
for trial in range(4):
    child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,cwd=ROOT)
    record=json.loads(child.stdout);record['trial']=trial
    records.append(record)
    print(json.dumps(record),flush=True)
with (HERE/'preloaded_backend_audit.json').open('x') as f:
    json.dump(dict(artificial=True,legacy=True,backend_preloaded=True,records=records,
                   limitation='Observed cold-path failures do not establish a universal NumPy/BLAS defect or identify the prior molecular failure conclusively',molecular_scientific_calls=0),f,indent=2);f.write('\n')
