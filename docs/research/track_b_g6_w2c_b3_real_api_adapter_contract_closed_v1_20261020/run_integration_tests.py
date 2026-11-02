"""Parallel independent artificial mutation processes; no AI delegation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')];sys.dont_write_bytecode=True


def ids(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from ids(test)
        else:yield test.id()


if __name__=='__main__':
    matrix,dest,jobs=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]);dest.mkdir(exist_ok=False)
    os.environ['B3_ADAPTER_MATRIX']=str(matrix)
    suite=unittest.defaultTestLoader.loadTestsFromName('test_g6_w2c_b3_real_adapter_integration_v1')
    names=list(ids(suite));env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]),
        PYTHONDONTWRITEBYTECODE='1',B3_ADAPTER_MUTATIONS=str(dest/'mutations'))
    def one(name):
        p=subprocess.run([sys.executable,'-m','unittest','-v',name],cwd=ROOT,env=env,capture_output=True,text=True)
        (dest/(name.rsplit('.',1)[-1]+'.log')).write_text(p.stdout+p.stderr)
        return dict(id=name,passed=p.returncode==0,exit_code=p.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:rows=list(pool.map(one,names))
    result=dict(run=len(rows),PASS=sum(x['passed'] for x in rows),FAIL=sum(not x['passed'] for x in rows),
                tests=rows,separate_reviewer=False,computational_process_parallelism=jobs)
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    sys.exit(result['FAIL']!=0)
