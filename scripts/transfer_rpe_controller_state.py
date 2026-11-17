"""Snapshot/import controller metadata and reusable outcomes, without science.

Large raw artifacts remain at their original host. SQLite is copied through a
read-only backup; requests, results, budgets and STOP files are never rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile

JOB_FILES = {'REQUEST.json','OUTCOME.json','COLLECTED_OUTCOME.json','SOURCE.json',
             'SUPERVISION.json','COLLECT_SUPERVISION.json','PROCESS_RETURN.json',
             'LAUNCH.json','STARTING.json','COLLECT_ATTEMPT.json','JOB_STAGES.jsonl',
             'SCOPED_HOLD.json','UNKNOWN_SCOPE.json','EXECUTION_HOLD.json'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):value.update(data)
    return value.hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def active_controllers():
    rows=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if p.stat().st_uid != os.getuid():continue
            command=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if any('/scripts/'+name in command for name in ('run_rpe_schedule_search.py',
                'run_rpe_method_optimizer.py','run_rpe_kappa_neighbourhood.py')):
            rows.append(dict(pid=int(p.parent.name),command=command))
    return rows


def export(service_pointer, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    pointer=json.loads(Path(service_pointer).read_text())
    service_path=Path(pointer.get('service_config',service_pointer))
    service=json.loads(service_path.read_text())
    active=active_controllers()
    if active:raise ValueError('Controller process is active; stop intake and reconcile before transfer: '+str(active))
    roots={str(Path(service_pointer).resolve().parent):'service'}
    state=Path(service['state_directory']);roots[str(state)]='state'
    rounds=[Path(service['round'])]
    if service.get('absolute_kappa_route'):rounds.append(Path(service['absolute_kappa_route']['round']))
    model_stores=set()
    if (Path(service['code_root'])/'.git').exists():
        common=subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=service['code_root'],text=True).strip()
        model_stores.add((Path(service['code_root'])/common/'rpe-price-validations.sqlite3').resolve())
    for round_path in rounds:
        spec=json.loads(round_path.read_text())
        if spec.get('price_validation',{}).get('store'):
            model_stores.add(Path(spec['price_validation']['store']).resolve())
        for value in spec.get('executor',{}).get('prior_run_dirs',[]):
            path=Path(value)
            if not path.is_absolute():path=Path(service['code_root'])/path
            roots.setdefault(str(path),'prior-'+hashlib.sha256(str(path).encode()).hexdigest()[:12])
    inventory=[];missing=[];deferred=[];runs=[]
    for original,label in roots.items():
        root=Path(original)
        if not root.exists():missing.append(original);continue
        copied=0;raw_bytes=0;jobs=0;outcomes=0;unresolved=[]
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(root)
            raw_bytes+=p.stat().st_size
            if p.name=='REQUEST.json' and 'jobs' in rel.parts:
                jobs+=1
                if (p.parent/'OUTCOME.json').exists():outcomes+=1
                elif any((p.parent/f).exists() for f in ('LAUNCH.json','STARTING.json')):
                    unresolved.append(str(p.parent))
            selected=(label in ('service','state') and p.suffix not in ('.bundle','.zip')) or (
                len(rel.parts)==1 and p.suffix in ('.json','.jsonl','.csv','.md')) or (
                'jobs' in rel.parts and len(rel.parts)==4 and p.name in JOB_FILES) or p.name in ('QUEUE.sqlite3',)
            if not selected:continue
            dest=output/'files'/label/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix in ('.sqlite','.sqlite3'):
                source=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
                target=sqlite3.connect(dest)
                try:source.backup(target)
                finally:target.close();source.close()
            else:shutil.copyfile(p,dest)
            inventory.append(dict(original=str(p),path=str(dest.relative_to(output)),sha256=sha(dest),bytes=dest.stat().st_size))
            copied+=dest.stat().st_size
        runs.append(dict(original_root=original,snapshot_root='files/'+label,selected_bytes=copied,
            original_total_bytes=raw_bytes,jobs=jobs,completed_outcomes=outcomes,unresolved_started_jobs=unresolved))
        if raw_bytes>copied:deferred.append(dict(original_host='server2',original_root=original,
            bytes_not_copied=raw_bytes-copied,reason='Raw scientific artifacts stay protected at original location; retrieve exact needed files by hash before reuse requiring raw'))
    stores=[]
    for path in sorted(model_stores):
        if not path.is_file():
            stores.append(dict(original=str(path),exists=False));continue
        dest=output/'model-stores'/hashlib.sha256(str(path).encode()).hexdigest()[:12]/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        source=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);target=sqlite3.connect(dest)
        try:source.backup(target)
        finally:target.close();source.close()
        row=dict(original=str(path),path=str(dest.relative_to(output)),sha256=sha(dest),bytes=dest.stat().st_size)
        inventory.append(row);stores.append(dict(row,exists=True))
    manifest=dict(schema='RPE_CONTROLLER_METADATA_TRANSFER_V1',service_pointer=str(service_pointer),
        service=service,active_controllers_at_snapshot=active,roots=runs,files=inventory,
        missing_roots=missing,deferred_raw=deferred,model_stores=stores,scientific_calls=0,old_controller_started=False)
    put(output/'MANIFEST.json',manifest)
    return dict(files=len(inventory),bytes=sum(r['bytes']for r in inventory),roots=runs,missing=missing)


def verify(directory):
    root=Path(directory).resolve();manifest=json.loads((root/'MANIFEST.json').read_text())
    for row in manifest['files']:
        p=root/row['path']
        if not p.is_relative_to(root) or sha(p)!=row['sha256']:
            raise ValueError('Snapshot hash differs: '+row['path'])
    return dict(status='PASS',files=len(manifest['files']),
        bytes=sum(r['bytes'] for r in manifest['files']),
        original_requests_results_and_STOP_preserved=True,
        active_controllers_at_snapshot=manifest['active_controllers_at_snapshot'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-service',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    value=verify(args.verify)if args.verify else export(args.export_service,args.output)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
