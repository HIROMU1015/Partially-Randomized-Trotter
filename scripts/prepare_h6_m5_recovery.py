"""Seal a fresh INVALID one-point proposal; metadata and byte hashing only."""
import argparse
import base64
import copy
import csv
import hashlib
from pathlib import Path
import subprocess
import h6_m5_recovery_contract as c


def seal(path, role):
    p=Path(path).resolve()
    return dict(path=str(p), role=role, bytes=p.stat().st_size, sha256=c.sha(p))


def prepare(origin_config, origin_stop, origin_completion, origin_exit, destination, source):
    origin=c.read(origin_config);c.base.verify_seals(origin)
    dest=Path(destination).resolve();c.require(not dest.exists(),'FRESH_PREPARATION_DIRECTORY')
    for rel in c.SOURCE_PATHS:
        c.require((c.ROOT/rel).read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT), 'COMMITTED_SOURCE:'+rel)
    files={x['path']:copy.deepcopy(x) for x in origin['sealed_files']}
    for rel in c.SOURCE_PATHS:
        row=seal(c.ROOT/rel,'scientific_SOURCE:'+rel);files[row['path']]=row
    for p,role in [(origin_config,'origin_config'),(origin_stop,'origin_STOP'),
        (origin_completion,'origin_completion'),(origin_exit,'origin_exit')]:
        row=seal(p,role);files[row['path']]=row
    for name in ('reference.npz','receipt.json'):
        row=seal(Path(origin['reference_root'])/name,'saved_full_H_reference:'+name);files[row['path']]=row
    # The evidence path adds QPY serialization to the old runtime closure.
    record=next(Path(x['path']) for x in origin['sealed_files'] if x['path'].endswith('/qiskit-1.3.0.dist-info/RECORD'))
    site=record.parent.parent
    entries={row[0]:row for row in csv.reader(record.open())}
    for p in sorted((site/'qiskit/qpy').rglob('*.py')):
        relative=str(p.relative_to(site));item=entries[relative]
        expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        c.require(item[1]==expected and int(item[2])==p.stat().st_size,'QPY_INSTALLED_RECORD_BYTES')
        row=seal(p,'QPY_runtime:'+relative);files[row['path']]=row
    cfg={k:copy.deepcopy(origin[k]) for k in ('binding','inputs','runtime','provider_root','reference_root')}
    root=dest/'run01'
    cfg.update(lane=c.LANE,system='H6',SOURCE=source,origin_SOURCE=c.ORIGIN_SOURCE,native_policy=c.POLICY,
        origin_config_path=str(Path(origin_config).resolve()),origin_STOP_path=str(Path(origin_stop).resolve()),
        origin_completion_path=str(Path(origin_completion).resolve()),origin_exit_path=str(Path(origin_exit).resolve()),
        sealed_files=list(files.values()),run_root=str(root),output=str(root/'output'),
        ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),authorized=False)
    # The same point's observed old peak, multiplied for circuit/QPY working data.
    # These are admission estimates, never runtime RSS caps or accuracy guarantees.
    peak=c.read(origin_exit)['ru_maxrss_KiB']*1024
    cfg.update(RSS_proxy_bytes=3*peak,disk_free_planning_proxy_bytes=3*peak)
    plan=dict(status='SEALED_FIXED_POINT_DIAGNOSTIC_PROPOSAL',SOURCE=source,scope=c.scope(),
        closure_digest=c.core.digest(cfg['sealed_files']),origin_SOURCE=c.ORIGIN_SOURCE,
        native_policy=c.POLICY,AD03='DEFERRED',phase3_authorized=False)
    plan.update({k+'_digest':c.core.digest(cfg[k]) for k in ('binding','inputs','runtime','provider_root','reference_root')})
    dest.mkdir(parents=True,exist_ok=False);path=dest/'plan_PRIVATE.json';c.save(path,plan)
    cfg.update(plan=plan,plan_path=str(path),plan_sha256=c.sha(path))
    grant=dict(valid=False,approval_reference=None,grant_id=None,lane=c.LANE,SOURCE=source,
        plan_sha256=cfg['plan_sha256'],maximum_axes=2,maximum_trajectories=0,maximum_reference_jobs=0,
        retry=False,phase3_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker')})
    cfg['grant']=grant
    c.verify_seals(cfg)
    c.save(dest/'grant_INVALID.json',grant);c.save(dest/'config_DRAFT_PRIVATE.json',cfg)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('origin-config','origin-stop','origin-completion','origin-exit','destination','source'):p.add_argument('--'+key,required=True)
    a=p.parse_args();prepare(a.origin_config,a.origin_stop,a.origin_completion,a.origin_exit,a.destination,a.source)
