"""Check the real fixed accelerator constructor/AST guard; never call its kernel."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('_scheduler_constructor_adapter',ROOT/'scripts/run_rpe_optimizer_q3_job.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    adapter.route(a.view,'pauli_b_common_entry')  # unchanged fixed route SOURCE checks
    observed=adapter.COMMON['X'].L.R.Observed
    rows=[]
    with tempfile.TemporaryDirectory(prefix='d2-constructor-only-')as d:
        for n in (1,2,4,8):
            engine=observed(d,'CONSTRUCTOR_ONLY',workers=n,outcome_batch=8)
            rows.append(dict(workers=n,constructor='PASS',kernel_callable=callable(engine.kernel),kernel_calls=0))
    data=dict(scope='FIXED_REAL_ROUTE_CONSTRUCTOR_ONLY_NO_SCIENTIFIC_EVALUATION',
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        route_view=str(a.view),route_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.view,text=True).strip(),
        interpreter=sys.executable,python=platform.python_version(),rows=rows,
        guard='Original Accelerator._kernel AST hash and replacement-count guards executed unchanged',
        real_Q3_calls=0)
    a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))


if __name__=='__main__':main()
