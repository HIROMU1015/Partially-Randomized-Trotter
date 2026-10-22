"""DIAGNOSTICS binding: stdlib static inventory only, never science imports."""
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT.parent/'h2-diagnostics-binding-assets-20261023'
OUT=ROOT/'artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23'
PYTHON=Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python')
SITE=PYTHON.parent.parent/'lib/python3.11/site-packages'
OLD_PRIVATE=ROOT.parent/'h2-pr-resource-preparation-20261019/.h2_pr_resource_private/2026-10-19'
RUNNER_FILES=('scripts/h2_common_production_contract.py','scripts/h2_common_production_backend.py',
 'scripts/h2_common_production_diagnostics.py','scripts/run_h2_common_production.py',
 'scripts/launch_h2_diagnostics_binding.py','scripts/h2_diagnostics_binding.py',
 'src/trotterlib/h2_common_native_policy.py',
 'artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/saved_input_excerpts.json')
EXCLUDED={'__pycache__','tests','test','docs','doc','examples','benchmarks'}
ACQUISITION_FILES={
 'scripts/h46_new2_native.py','scripts/h46_new2_core.py','scripts/h46_m5_native.py',
 'scripts/h46_m5_contract.py','scripts/h46_pr_pilot_repair.py','scripts/h2_m5_compiler_phase_guard.py',
 'src/trotterlib/h46_m5_plan.py','references/h46_m5_best_formula_v1.json'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,reason):
    if not ok:raise ValueError(reason)


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def imports(paths):
    names=set()
    for p in paths:
        if p.suffix!='.py':continue
        tree=ast.parse(p.read_bytes(),filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:names.add(node.module.split('.')[0])
    return names


def package_files(path):
    if path.is_file():return [path]
    return [p for p in path.rglob('*') if p.is_file() and not any(x in EXCLUDED for x in p.relative_to(path).parts)
            and p.suffix not in {'.pyc','.pyo'}]


def materialize(commit,role,selector):
    target=ASSETS/role
    paths=[p for p in git('ls-tree','-r','--name-only',commit).decode().splitlines() if selector(p)]
    stream=git_batch(commit,paths);rows=[]
    for rel,(blob,raw) in zip(paths,stream):
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():require(p.read_bytes()==raw,'EXISTING_SOURCE_CAPSULE_CHANGED')
        else:p.write_bytes(raw)
        rows.append(dict(root=role,path=rel,sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=blob))
    return target,rows


def git_batch(commit,paths):
    require(bool(paths) and all('\n' not in p and '\r' not in p for p in paths),'FIXED_SOURCE_PATHS')
    raw=subprocess.check_output(
        ['git','-C',str(ROOT),'cat-file','--batch'],input=('\n'.join(commit+':'+p for p in paths)+'\n').encode())
    pos=0;result=[]
    for path in paths:
        end=raw.index(b'\n',pos);head=raw[pos:end].split();require(len(head)==3 and head[1]==b'blob','FIXED_BLOB')
        n=int(head[2]);pos=end+1;data=raw[pos:pos+n];pos+=n+1
        result.append((head[0].decode(),data))
    require(pos==len(raw),'SOURCE_STREAM_END')
    return result


def elf_info(path):
    with Path(path).open('rb') as f:
        if f.read(4)!=b'\x7fELF':return None
    r=subprocess.run(['readelf','-d',str(path)],capture_output=True,text=True)
    require(r.returncode==0,'ELF_HEADER_READ')
    needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',r.stdout)
    program=subprocess.run(['readelf','-l',str(path)],capture_output=True,text=True)
    require(program.returncode==0,'ELF_PROGRAM_HEADER_READ')
    interpreter=re.findall(r'Requesting program interpreter: (.*?)\]',program.stdout)
    search=[]
    for val in re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]',r.stdout):
        for item in val.split(':'):
            search.append(item.replace('${ORIGIN}',str(Path(path).parent)).replace('$ORIGIN',str(Path(path).parent)))
    return dict(path=str(Path(path).resolve()),needed=needed,search=search,interpreter=interpreter)


def elf_closure(paths):
    queue=sorted({p.resolve() for p in paths if p.suffix=='.so' or '.so.' in p.name or p==PYTHON.resolve()})
    seen=set();rows=[];missing=[]
    while queue:
        current=[p for p in queue if p not in seen];queue=[];seen.update(current)
        with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:infos=list(pool.map(elf_info,current))
        for info in infos:
            if info is None:continue
            joins=[]
            for name in info['interpreter']:
                found=Path(name).resolve()
                if not found.is_file():missing.append(dict(parent=info['path'],needed=name));continue
                joins.append(dict(needed='PT_INTERP',path=str(found)))
                if found not in seen:queue.append(found)
            for name in info['needed']:
                directories=info['search']+['/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu','/lib64','/usr/lib64']
                found=next((Path(d)/name for d in directories if (Path(d)/name).is_file()),None)
                if found is None:missing.append(dict(parent=info['path'],needed=name));continue
                found=found.resolve();joins.append(dict(needed=name,path=str(found)))
                if found not in seen:queue.append(found)
            rows.append(dict(path=info['path'],dependencies=joins,search=info['search']))
    return sorted(seen),rows,missing


def verify_mapped_runtime(cfg, maps_text=None):
    """Reject unsealed executable file mappings, including native dependencies."""
    if maps_text is None:maps_text=Path('/proc/self/maps').read_text()
    indexed={r['path']:r['sha256'] for r in cfg['files']}
    mapped=set()
    for line in maps_text.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or 'x' not in fields[1] or not fields[5].startswith('/'):continue
        require(not fields[5].endswith(' (deleted)'),'DELETED_EXECUTABLE_MAPPING')
        path=str(Path(fields[5]).resolve());mapped.add(path)
        require(path in indexed and sha(path)==indexed[path],'UNSEALED_EXECUTABLE_MAPPING:'+path)
    return sorted(mapped)


def verify_runtime_binding(cfg):
    r=cfg['runtime'];m=read(r['closure_manifest'])
    require(r['closure_mode']=='STATIC_CLOSED_IMPORT_SET_V1' and m['scientific_imports_observed'] is False,
            'STATIC_CLOSURE_NOT_DYNAMIC_ATTESTATION')
    indexed={x['path']:x for x in cfg['files']}
    require(str(Path(r['closure_manifest']).resolve()) in indexed,'RUNTIME_MANIFEST_NOT_SEALED')
    require(digest(m['files'])==r['inventory_sha256'],'RUNTIME_CLOSURE_DIGEST')
    require(not m['unresolved_elf_dependencies'],'UNRESOLVED_ELF_CLOSURE')
    for row in m['files']:
        require(row['path'] in indexed and indexed[row['path']]['sha256']==row['sha256'],'INCOMPLETE_RUNTIME_CLOSURE')
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_RUNTIME_LAUNCH')
    require(str(Path(sys.executable).resolve())==r['python_target'],'SELECTED_PYTHON_TARGET')
    require(sys._xoptions.get('pycache_prefix')==r['cache_prefix'] and not Path(r['cache_prefix']).exists(),
            'UNUSED_BYTECODE_CACHE_PREFIX')
    require(os.environ.get('LD_LIBRARY_PATH','')=='' and not os.environ.get('LD_PRELOAD'),'DYNAMIC_LOADER_ENVIRONMENT')
    require(r['loaded_closure_complete'] is False,'NO_FABRICATED_IMPORT_OBSERVATION')
    for package,version in [('numpy','1.26.4'),('qiskit','1.3.0')]:
        entry=m['package_metadata'][package]
        text=Path(entry['path']).read_text()
        require('Version: '+version+'\n' in text,'PACKAGE_METADATA_VERSION')
    require(not any(x.split('.')[0] in {'numpy','scipy','qiskit','trotterlib','trottertracks'} for x in sys.modules),
            'NO_SCIENTIFIC_IMPORT_DURING_PREFLIGHT')
    verify_mapped_runtime(cfg)


def accepted_diagnostics_receipt(root,expected):
    """Metadata/hash acceptance only. Never decode or evaluate saved vectors."""
    root=Path(root).resolve();completion=read(root/'completion.json')
    require(completion['status']=='CONTRACT_COMPLETE_NOT_SCIENTIFIC_GO' and
            completion['result']['status']=='DIAGNOSTICS_ONLY','COMPLETED_DIAGNOSTICS_REQUIRED')
    require(not (root/'STOP.json').exists(),'PARTIAL_DIAGNOSTICS_FORBIDDEN')
    require(digest(completion['used'])==digest(expected['budget']),'DIAGNOSTIC_BUDGET_IDENTITY')
    for key in ('source_commit','binding','physical','native_policy','allowance_policy','reference_policy','runtime'):
        require(completion[key]==expected[key],'DIAGNOSTIC_IDENTITY:'+key)
    require(sha(root/'CONSUMED.json')==completion['marker_sha256'],'ACCEPTED_CONSUMED_MARKER')
    require(read(root/'CONSUMED.json')['grant']['contract_sha256']==completion['contract_sha256'],
            'ORIGINAL_GRANT_CONTRACT_BINDING')
    require(completion['contract_sha256']==digest(expected),'DIAGNOSTIC_CONTRACT_DIGEST')
    files={r['path']:r['sha256'] for r in completion['result_files']}
    ids=[r['id'] for r in expected['candidates']]
    required={'results/preparation_1.json','results/preparation_3.json'}
    for cid in ids:required|={'results/'+cid+'_diagnostics.json','results/'+cid+'_vectors.npz'}
    require(required<=set(files),'COMPLETE_DIAGNOSTIC_BUNDLE')
    for rel,value in files.items():
        p=(root/rel).resolve();require(p.is_relative_to(root) and sha(p)==value,'DIAGNOSTIC_RESULT_HASH')
    for cid in ids:
        row=read(root/'results'/(cid+'_diagnostics.json'))
        require(row['candidate']==cid and row['native_policy']==expected['native_policy'], 'DIAGNOSTIC_CANDIDATE_POLICY')
        require(row['strict_diagnostic_limit']==1e-9 and row['phase_alignment'] is False and
                row['postcompile_correction'] is False,'UNCHANGED_STRICT_DIAGNOSTIC_GATE')
        require(type(row['scalar_applications']) is int and row['scalar_applications']==1,'SCALAR_ONCE_RECEIPT')
        archive='results/'+cid+'_vectors.npz'
        require(row['vector_archive']==dict(path=cid+'_vectors.npz',sha256=files[archive]),'ACCEPTED_VECTOR_ARCHIVE_BINDING')
    return dict(status='ACCEPTED_METADATA_ONLY',files=files,source_contract_sha256=digest(expected),
                prior_grant_reusable=False,science_GO=False,numerical_values_evaluated=False)


def build(source_commit):
    import h2_common_production_contract as c
    ASSETS.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    acquisition,arows=materialize(c.ACQUISITION,'acquisition',lambda p:p in ACQUISITION_FILES)
    provider,prows=materialize(c.PROVIDER,'provider',lambda p:p.startswith('src/') and p.endswith(('.py','.json','.txt')))
    source_rows=arows+prows
    for p,(blob,raw) in zip(RUNNER_FILES,git_batch(source_commit,RUNNER_FILES)):
        require((ROOT/p).read_bytes()==raw,'COMMITTED_RUNNER_BYTES')
        source_rows.append(dict(root='runner',path=p,sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=blob))
    probe='import sys,sysconfig,json; print(json.dumps(dict(executable=sys.executable,base_prefix=sys.base_prefix,stdlib=sysconfig.get_path("stdlib"),sys_path=sys.path,loaded=list(sys.modules))))'
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    info=json.loads(subprocess.check_output([str(PYTHON),'-I','-S','-B','-c',probe],env=env))
    require(not any(x.split('.')[0] in {'numpy','scipy','qiskit'} for x in info['loaded']),'STDLIB_ONLY_PROBE')
    # Conservative package-level transitive import closure, not an observed graph.
    selected={};pending=imports([Path(acquisition/r['path']) for r in arows if r['path'].endswith('.py')]+[Path(provider/r['path']) for r in prows if r['path'].endswith('.py')])
    while pending:
        top=pending.pop()
        if top in selected or top in {'pip','setuptools','pytest','IPython','sphinx'}:continue
        p=SITE/top
        if not p.exists():p=SITE/(top+'.py')
        if not p.exists():
            found=list(SITE.glob(top+'.*.so'));p=found[0] if found else None
        if p is None or not p.exists():continue
        files=package_files(p);selected[top]=files;pending|=imports(files)-set(selected)
    runtime_paths={p.resolve() for files in selected.values() for p in files}
    stdlib=Path(info['stdlib'])
    runtime_paths|={p.resolve() for p in package_files(stdlib) if not p.is_relative_to(stdlib/'site-packages')}
    metadata={}
    for p in SITE.iterdir():
        if not p.name.endswith('.dist-info'):continue
        text=(p/'METADATA').read_text();name=next((s[6:].lower().replace('-','_') for s in text.splitlines() if s.startswith('Name: ')),p.name)
        if name in {s.lower().replace('-','_') for s in selected} or name in {'numpy','qiskit','scipy'}:
            runtime_paths|={q.resolve() for q in package_files(p)}
            metadata[name]=dict(path=str((p/'METADATA').resolve()),sha256=sha(p/'METADATA'))
    for p in SITE.iterdir():
        if p.is_dir() and p.name.endswith('.libs') and p.name[:-5].replace('_','-').lower() in {s.replace('_','-').lower() for s in selected}:
            runtime_paths|={q.resolve() for q in package_files(p)}
    runtime_paths.add(PYTHON.resolve())
    elf_files,elf_rows,missing=elf_closure(runtime_paths);runtime_paths.update(elf_files)
    require(not missing,'ELF_DEPENDENCIES_REQUIRE_RESOLUTION:'+str(missing[:5]))
    def row(p):return dict(path=str(p),sha256=sha(p),role='runtime:static_import_or_elf')
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:runtime_rows=list(pool.map(row,sorted(runtime_paths)))
    runtime_manifest=ASSETS/'runtime_closure.json'
    write(runtime_manifest,dict(mode='STATIC_CLOSED_IMPORT_SET_V1',scientific_imports_observed=False,
        package_roots=sorted(selected),package_metadata=metadata,files=runtime_rows,
        elf_dependencies=elf_rows,unresolved_elf_dependencies=missing,probe=info))
    spec=read(OLD_PRIVATE/'D_activation01/D_spec_EXECUTION.json');inputs={}
    for entry in spec['inputs']:
        if entry['role'] not in {'F_snapshot','F_receipt'}:continue
        path=Path(entry['path']);require(sha(path)==entry['sha256'],'ORIGINAL_INPUT_BYTES')
        dest=ASSETS/path.name
        if dest.exists():require(sha(dest)==entry['sha256'],'PRESERVED_INPUT_COPY')
        else:shutil.copyfile(path,dest)
        inputs['snapshot' if entry['role']=='F_snapshot' else 'snapshot_receipt']=str(dest.resolve())
    inputs['old_reference']=str((ROOT/RUNNER_FILES[-1]).resolve())
    files=runtime_rows+[dict(path=str(runtime_manifest.resolve()),sha256=sha(runtime_manifest),role='runtime_manifest')]
    for r in source_rows:
        base={'runner':ROOT,'acquisition':acquisition,'provider':provider}[r['root']]
        files.append(dict(path=str((base/r['path']).resolve()),sha256=r['sha256'],role='SOURCE:'+r['root']))
    for name,p in inputs.items():
        if not any(r['path']==p for r in files):files.append(dict(path=p,sha256=sha(p),role='INPUT:'+name))
    files=sorted({r['path']:r for r in files}.values(),key=lambda r:r['path'])
    cache=ASSETS/'BYTECODE_CACHE_MUST_REMAIN_ABSENT'
    cfg=dict(schema='h2_common_production_contract_v1',series=c.SERIES,branch='DIAGNOSTICS',source_commit=source_commit,
        physical=c.policy.PHYSICAL,compiler=c.policy.COMPILER,native_policy=c.policy.NATIVE_POLICY,
        allowance_policy=c.policy.ALLOWANCE_POLICY,reference_policy='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC',
        candidates=[asdict(x) for x in c.policy.CANDIDATES],budget=c.counts('DIAGNOSTICS'),
        files=files,source_files=[{k:r[k] for k in ('root','path','sha256')} for r in source_rows],
        acquisition_root=str(acquisition),provider_root=str(provider),inputs=inputs,binding=c.BINDING,
        runtime=dict(python_sha256=sha(PYTHON.resolve()),python_target=str(PYTHON.resolve()),qiskit='1.3.0',numpy='1.26.4',
            loaded_closure_complete=False,closure_mode='STATIC_CLOSED_IMPORT_SET_V1',
            closure_manifest=str(runtime_manifest),inventory_sha256=digest(runtime_rows),cache_prefix=str(cache),
            sys_path=[str(p) for p in map(Path,info['sys_path']) if p.is_dir()]+[str(SITE)]),
        run_root=str(ASSETS/'DIAGNOSTICS_run01'),reuse_witnesses=[],retry=False,mandatory_STOP=True)
    contract=ASSETS/'DIAGNOSTICS_contract.json';write(contract,cfg)
    write(OUT/'DIAGNOSTICS_contract.json',cfg)
    write(OUT/'source_index.json',source_rows)
    write(OUT/'runtime_closure_summary.json',dict(files=len(runtime_rows),bytes=sum(Path(r['path']).stat().st_size for r in runtime_rows),
        package_roots=sorted(selected),inventory_sha256=digest(runtime_rows),manifest_sha256=sha(runtime_manifest),
        canonical_manifest=str(runtime_manifest),scientific_imports_observed=False,elf_entries=len(elf_rows),
        python_target=str(PYTHON.resolve()),python_sha256=sha(PYTHON.resolve()),package_metadata=metadata,
        not_a_historical_runtime_equivalence_certificate=True))
    command=[str(PYTHON),'-I','-S','-B','-X','pycache_prefix='+str(cache),str(ROOT/'scripts/launch_h2_diagnostics_binding.py'),'--contract',str(contract)]
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    write(OUT/'preflight.json',dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        science_calls=0,scientific_imports=0,grant_issued=False,contract_sha256=digest(cfg)))
    require(result.returncode==0,'REAL_BINDING_PREFLIGHT_FAILED:'+result.stderr[-1500:])
    print(json.dumps(dict(status='SEALED_TECHNICAL_BINDING_REQUIRES_USER_APPROVAL',files=len(files),
        packages=len(selected),contract_sha256=digest(cfg),science_calls=0)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);a=p.parse_args()
    build(a.source_commit)
