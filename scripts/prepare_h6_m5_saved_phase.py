"""Create a sealed INVALID proposal using byte hashes/JSON metadata only."""
import argparse
import base64
import copy
import csv
import hashlib
from pathlib import Path
import subprocess
import h6_m5_phase_contract as c


def installed_runtime_files(origin):
    """Known fixed package roots only; never search project/history directories."""
    files={row['path']:copy.deepcopy(row) for row in origin['sealed_files']}
    record=next(Path(p) for p in files if p.endswith('/qiskit-1.3.0.dist-info/RECORD'))
    site=record.parent.parent
    for package,version in (('qiskit','1.3.0'),('numpy','1.26.4')):
        record=site/(package+'-'+version+'.dist-info')/'RECORD'
        entries={row[0]:row for row in csv.reader(record.open())}
        candidates=list((site/package).rglob('*'))
        if (site/(package+'.libs')).is_dir():candidates+=list((site/(package+'.libs')).rglob('*'))
        for path in sorted(candidates):
            if not path.is_file() or not (path.suffix=='.py' or '.so' in path.name):continue
            rel=str(path.relative_to(site));row=entries[rel]
            expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
            c.require(row[1]==expected and int(row[2])==path.stat().st_size,'INSTALLED_RECORD_BYTES:'+rel)
            item=c.seal(path,'saved_phase_runtime:'+rel);files[item['path']]=item
        for path in (record,record.parent/'METADATA'):
            item=c.seal(path,'package_identity:'+package);files[item['path']]=item
    # Resolve the fixed interpreter and the venv identity. No package installation.
    executable=Path(origin['runtime']['python'])
    for path in (executable.resolve(),executable.parent.parent/'pyvenv.cfg'):
        item=c.seal(path,'fixed_python_runtime');files[item['path']]=item
    # Import identities only: no QPY, state, molecule or numerical computation.
    # Seal other modules/libraries actually loaded by the two selected runtimes.
    probe="""import json,sys,pathlib
import numpy,qiskit
from qiskit.quantum_info import Statevector
paths={str(pathlib.Path(m.__file__).resolve()) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).is_file()}
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    item=line.split()[-1]
    if item.startswith('/') and '.so' in item and pathlib.Path(item).is_file(): paths.add(str(pathlib.Path(item).resolve()))
print(json.dumps(sorted(paths)))
"""
    import json
    paths=json.loads(subprocess.check_output([str(executable),'-B','-c',probe],text=True))
    for path in paths:
        if path.endswith('.pyc') and Path(path[:-1]).is_file():path=path[:-1]
        item=c.seal(path,'imported_runtime_or_loaded_library');files[item['path']]=item
    return files


def prepare(input_paths,destination,source):
    paths=c.read(input_paths)
    expected={r['role']:r for r in c.read(c.BASE/'input_manifest_metadata.json')['roles']}
    c.require(set(paths)==set(expected),'EXACT_TWELVE_INPUT_ROLES')
    inputs={role:c.seal(path,role) for role,path in paths.items()}
    for role,row in inputs.items():
        c.require(all(row[k]==expected[role][k] for k in ('bytes','sha256')),'FIXED_INPUT_BYTES:'+role)
    origin=c.read(paths['recovery_config'])
    files=installed_runtime_files(origin)
    files.update({row['path']:row for row in inputs.values()})
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel
        c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_SOURCE:'+rel)
        row=c.seal(p,'analysis_SOURCE:'+rel);files[row['path']]=row
    dest=Path(destination).resolve();c.require(not dest.exists(),'NEW_PROPOSAL_DIRECTORY')
    run=dest/'run01';witness=c.read(paths['saved_exit'])
    resources=dict(per_worker_RSS_proxy_bytes=2*witness['ru_maxrss_KiB']*1024,
        disk_proxy_bytes=16*sum(r['bytes'] for r in inputs.values()),
        proxy_reason='Twice previous one-circuit peak for QPY loader and replay; 16 times saved input bytes for output/workspace admission only.',
        runtime_caps=False,other_worktree_freeze=False,transient_PSI_kill=False)
    cfg=dict(SOURCE=source,inputs=inputs,runtime=copy.deepcopy(origin['runtime']),sealed_files=list(files.values()),
        run_root=str(run),output=str(run/'quarantine'),ledger=str(run/'ledger'),marker=str(run/'CONSUMED.json'),
        authorized=False,resources=resources)
    cfg['supervisor']={k:str(dest/name) for k,name in
        [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    plan=dict(status='SEALED_DRAFT_EXECUTION_UNAPPROVED',SOURCE=source,origin_SOURCE=c.ORIGIN,
        scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),inputs_digest=c.digest(inputs),
        runtime_digest=c.digest(cfg['runtime']),run_root=str(run),resources=resources,
        supervisor=cfg['supervisor'],original_STOP_preserved=True,AD03='DEFERRED')
    dest.mkdir(parents=True,exist_ok=False);plan_path=dest/'plan_PRIVATE.json';c.save(plan_path,plan)
    cfg.update(plan=plan,plan_path=str(plan_path),plan_sha256=c.sha(plan_path))
    grant=dict(valid=False,approval_reference=None,grant_id=None,scope_id=c.SCOPE,SOURCE=source,
        plan_sha256=cfg['plan_sha256'],maximum_replays=4,maximum_saved_initial_states=1,new_compile=0,
        new_sampling=0,new_state_generation=0,accounting=False,retry=False,acceptance_override=False,
        next_phase_authorized=False,supervisor=cfg['supervisor'],**{k:cfg[k] for k in ('output','ledger','marker')})
    cfg['grant']=grant;c.verify_seals(cfg)
    c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID.json',grant)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('input-paths','destination','source'):p.add_argument('--'+key,required=True)
    a=p.parse_args();prepare(a.input_paths,a.destination,a.source)
