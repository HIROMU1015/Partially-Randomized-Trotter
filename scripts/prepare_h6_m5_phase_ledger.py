"""Seal new scalar ledger DRAFT without evaluating the real ledger."""
import argparse
import copy
import csv
import base64
import hashlib
from pathlib import Path
import subprocess
import h6_m5_phase_ledger_contract as c


def prepare(origin_config,destination,source,approval):
    origin=c.read(origin_config);c.original.verify_seals(origin)
    dest=Path(destination).resolve();c.require(not dest.exists(),'NEW_PREPARATION_ROOT')
    run=dest/'run01';original_root=Path(origin['run_root'])
    paths=dict(wrapper_qpy=origin['inputs']['wrapper_qpy']['path'],compiled_qpy=origin['inputs']['compiled_qpy']['path'],
        origin_config=str(Path(origin_config).resolve()),static_metadata=str(c.STATIC_ROOT/'saved_circuit_metadata.json'),
        saved_report=str(original_root/'diagnostic_report.json'),saved_completion=str(original_root/'completion_witness.json'),
        saved_exit=origin['supervisor']['exit'])
    inputs={role:c.seal(path,role) for role,path in paths.items()}
    files={r['path']:copy.deepcopy(r) for r in origin['sealed_files']}
    for r in inputs.values():files[r['path']]=r
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_SOURCE:'+rel)
        r=c.seal(p,'scalar_ledger_SOURCE:'+rel);files.setdefault(r['path'],r)
    approval=Path(approval).resolve();r=c.seal(approval,'new_user_approval');files[r['path']]=r
    site=Path(origin['runtime']['python']).parent.parent/'lib/python3.12/site-packages'
    record=site/'mpmath-1.3.0.dist-info/RECORD';entries={row[0]:row for row in csv.reader(record.open())}
    for p in sorted((site/'mpmath').rglob('*.py')):
        row=entries[str(p.relative_to(site))]
        expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        c.require(row[1]==expected and int(row[2])==p.stat().st_size,'MPMATH_INSTALLED_RECORD')
        r=c.seal(p,'mpmath_runtime');files[r['path']]=r
    for p in (record,record.parent/'METADATA'):
        r=c.seal(p,'mpmath_package_identity');files[r['path']]=r
    previous=c.read(paths['saved_exit']);resources=dict(per_worker_RSS_proxy_bytes=2*previous['ru_maxrss_KiB']*1024,
        disk_proxy_bytes=4*sum(r['bytes'] for r in inputs.values()),runtime_caps=False,transient_PSI_kill=False)
    supervisor={k:str(dest/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    cfg=dict(SOURCE=source,inputs=inputs,runtime=origin['runtime'],numerical_runtime=dict(mpmath_version='1.3.0',backend='python'),
        sealed_files=list(files.values()),run_root=str(run),output=str(run/'output'),ledger=str(run/'ledger'),marker=str(run/'CONSUMED.json'),
        supervisor=supervisor,resources=resources,authorized=False,approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval))
    plan=dict(status='NEW_FIXED_SCALAR_LEDGER_PLAN',SOURCE=source,scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),
        inputs_digest=c.digest(inputs),runtime_digest=c.digest(cfg['runtime']),numerical_runtime_digest=c.digest(cfg['numerical_runtime']),
        run_root=str(run),resources=resources,supervisor=supervisor,approval_receipt_sha256=cfg['approval_receipt_sha256'])
    dest.mkdir(parents=True,exist_ok=False);path=dest/'plan_PRIVATE.json';c.save(path,plan)
    cfg.update(plan=plan,plan_path=str(path),plan_sha256=c.sha(path))
    cfg['grant']=dict(valid=False,approval_reference=None,grant_id=None,scope_id=c.SCOPE,SOURCE=source,plan_sha256=cfg['plan_sha256'],
        saved_QPY_roles=2,scalar_phase_ledgers=1,new_compile=0,new_statevector_replays=0,new_sampling=0,
        retry=False,acceptance_override=False,next_phase_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.verify_seals(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID.json',cfg['grant'])
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('origin-config','destination','source','approval'):p.add_argument('--'+key,required=True)
    a=p.parse_args();prepare(a.origin_config,a.destination,a.source,a.approval)
