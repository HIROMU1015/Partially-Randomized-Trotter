"""Parallel computational workers, not delegated AI agents."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import independent as I

OUT = I.HERE / 'raw_audit'

def one_case(name):
    start=time.monotonic()
    result=I.verify(I.ROOT/'restored/cases'/name)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
    return dict(case=name,status=result['status'],issues=result['issues'],
                dispatch=result.get('independent_adapter'),calls=result.get('calls'),seconds=time.monotonic()-start)

def changes(a,b,path=''):
    if type(a) is not type(b):
        return [path]
    if type(a) is dict:
        return [path+'/'+k for k in set(a)^set(b)] + [p for k in a.keys()&b.keys() for p in changes(a[k],b[k],path+'/'+k)]
    if type(a) is list:
        if len(a)!=len(b):return [path+'/length']
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in changes(x,y,path+'/'+str(i))]
    return [] if a==b else [path]

def mutation(name):
    base=I.ROOT/'restored/cases/normal_n16'
    overlay=I.ROOT/'restored/integration/mutations'/name
    root=I.ROOT/'mutation_runtime'/name
    shutil.copytree(base,root,copy_function=os.link)
    for source in overlay.rglob('*'):
        if source.is_file() and source.name!='result.json':
            dest=root/source.relative_to(overlay)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():dest.unlink()
            shutil.copyfile(source,dest)
    touched=[]
    for source in (overlay/'output').glob('*.jsonl'):
        if source.name=='control.jsonl':continue
        normal=I.plain(I.plain((base/'output'/source.name).read_bytes().splitlines()[0])['payload'])
        changed=I.plain(I.plain(source.read_bytes().splitlines()[0])['payload'])
        touched.extend(changes(normal,changed))
    trusted=I.plain((base/'candidate_input.json').read_bytes())
    storage=I.STORAGE.verify(root,I.Reference(trusted))
    try:
        I.dispatch(root,storage)
        status='INCORRECTLY_ACCEPTED';reason=None
    except (I.Invalid,KeyError,ValueError,TypeError,OSError) as exc:
        status='REJECTED';reason=str(exc)
    result=dict(mutation=name,base='normal_n16',independently_observed_changed_paths=touched,
                storage_only_status=storage['status'],storage_issues=storage['issues'],
                independent_adapter_status=status,reason=reason)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    OUT.mkdir(exist_ok=False)
    # Eight independent scans distribute parsing/hash work across otherwise idle
    # CPUs; n256 remains one streaming scan, avoiding repeated large raw copies.
    jobs=min(8,len(os.sched_getaffinity(0)))
    cases=[];mutations=[]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        work={pool.submit(one_case,p.name):'case' for p in (I.ROOT/'restored/cases').iterdir()}
        work.update({pool.submit(mutation,p.name):'mutation' for p in (I.ROOT/'restored/integration/mutations').iterdir()})
        for future in as_completed(work):
            result=future.result()
            (cases if work[future]=='case' else mutations).append(result)
            print(json.dumps({k:result.get(k) for k in ('case','mutation','status','issues','independent_adapter_status','reason') if k in result}),flush=True)
    (I.HERE/'artificial_22_case_verification.json').write_text(json.dumps(dict(cases=sorted(cases,key=lambda x:x['case'])),indent=2)+'\n')
    (I.HERE/'independent_mutation_results.json').write_text(json.dumps(dict(mutations=sorted(mutations,key=lambda x:x['mutation'])),indent=2)+'\n')

if __name__=='__main__':main()
