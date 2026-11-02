"""Build a new metadata-only closure around the saved first-gate PASS."""
import argparse
import base64
import copy
import csv
import hashlib
from pathlib import Path
import subprocess
import h6_m5_preserve_oneq_contract as c


def prepare(origin_config,destination,source,approval):
    old=c.read(origin_config);c.prior.verify_seals(old);c.prior.consumed(old)
    dest=Path(destination).resolve();c.require(not dest.exists(),'UNUSED_PRESERVE_ONEQ_PROPOSAL')
    root=Path(old['run_root']);out=Path(old['output'])
    paths=dict(origin_repair_config=str(Path(origin_config).resolve()),origin_completion=str(root/'completion_witness.json'),
        origin_exit=old['supervisor']['exit'],origin_end_audit=str(c.ROOT/c.AUDIT),
        lowered_qpy=str(out/'lowered.qpy'),lowered_vector=str(out/'lowered_vector.npz'),
        lowering_receipt=str(out/'lowering_receipt.json'),lowered_gate=str(out/'lowered_vs_saved_wrapper_vector.json'),
        initial_npz=old['inputs']['initial_npz']['path'],ordinary_vector=old['inputs']['ordinary_vector']['path'])
    inputs={role:c.seal(path,role) for role,path in paths.items()}
    files={r['path']:copy.deepcopy(r) for r in old['sealed_files']}
    for row in inputs.values():files[row['path']]=row
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_PRESERVE_ONEQ_SOURCE:'+rel)
        row=c.seal(p,'preserve_oneq_SOURCE:'+rel);files.setdefault(row['path'],row)
    p=Path(approval).resolve();row=c.seal(p,'new_preserve_oneq_approval');files[row['path']]=row
    # ParameterExpression dispatch uses the same installed SymEngine; seal its
    # complete recorded distribution including the native wrapper library.
    site=Path(old['runtime']['python']).parent.parent/'lib/python3.12/site-packages'
    records=list(site.glob('symengine-*.dist-info/RECORD'));c.require(len(records)==1,'ONE_FROZEN_SYMENGINE_DISTRIBUTION')
    record=records[0]
    for rel,hashvalue,size in csv.reader(record.open()):
        if not rel.startswith('symengine/') or rel.endswith('.pyc') or not hashvalue:continue
        path=site/rel;c.require(path.resolve().is_relative_to(site.resolve()),'SYMENGINE_FILE_INSIDE_VENV')
        actual='sha256='+base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
        c.require(actual==hashvalue and path.stat().st_size==int(size),'SYMENGINE_RECORD:'+rel)
        row=c.seal(path,'full_symengine_runtime');files.setdefault(row['path'],row)
    for path in (record,record.parent/'METADATA'):
        row=c.seal(path,'symengine_distribution_identity');files.setdefault(row['path'],row)
    peak=c.read(paths['origin_exit'])['ru_maxrss_KiB']*1024
    resources=dict(per_worker_RSS_proxy_bytes=2*peak+2*inputs['lowered_qpy']['bytes'],
        disk_proxy_bytes=2*inputs['lowered_qpy']['bytes']+4*92549282,
        admission='TWICE_MEASURED_PRIOR_PEAK_PLUS_TWO_SAVED_QPY_BYTES',runtime_caps=False,host_PSI_kill=False)
    run=dest/'run01';supervisor={k:str(dest/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    cfg=dict(SOURCE=source,runtime=old['runtime'],inputs=inputs,sealed_files=list(files.values()),resources=resources,
        approval_receipt_path=str(p),approval_receipt_sha256=c.sha(p),run_root=str(run),output=str(run/'output'),
        ledger=str(run/'ledger'),marker=str(run/'CONSUMED.json'),supervisor=supervisor,authorized=False)
    cfg['pipeline_identity']=c.digest(dict(SOURCE=source,scope=c.scope(),runtime=cfg['runtime']))
    plan=dict(status='FIXED_PRESERVE_ONEQ_COSINE_PLAN',SOURCE=source,scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),
        inputs_digest=c.digest(inputs),runtime_digest=c.digest(cfg['runtime']),run_root=str(run),supervisor=supervisor,
        resources=resources,pipeline_identity=cfg['pipeline_identity'],approval_receipt_sha256=cfg['approval_receipt_sha256'])
    dest.mkdir(exist_ok=False,parents=True);plan_path=dest/'plan_PRIVATE.json';c.save(plan_path,plan)
    cfg.update(plan=plan,plan_path=str(plan_path),plan_sha256=c.sha(plan_path))
    cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,scope_id=c.SCOPE,SOURCE=source,plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
        maximum_new_compile=1,maximum_statevector_replays=1,sine=0,sampling=0,accepted_cost_rows=0,retry=False,next_phase_authorized=False,
        **{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.verify_seals(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID.json',cfg['grant'])
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('origin-config','destination','source','approval'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.origin_config,a.destination,a.source,a.approval)
