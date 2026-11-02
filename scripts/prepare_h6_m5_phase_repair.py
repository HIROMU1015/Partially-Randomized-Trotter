"""Metadata-only new contract closure; never restore molecular QPY or arrays."""
import argparse
import base64
import copy
import csv
import hashlib
from pathlib import Path
import subprocess
import h6_m5_phase_repair_contract as c


def prepare(origin_config,destination,source,approval):
    prior=c.read(origin_config);c.prior.verify_seals(prior)
    replay=c.read(prior['inputs']['origin_config']['path'])
    destination=Path(destination).resolve();c.require(not destination.exists(),'UNUSED_REPAIR_PROPOSAL')
    rr=Path(replay['output'])/'WRAPPER_ORDINARY'
    paths=dict(origin_ledger_config=str(Path(origin_config).resolve()),wrapper_qpy=replay['inputs']['wrapper_qpy']['path'],
        initial_npz=replay['inputs']['initial_npz']['path'],saved_vectors_npz=replay['inputs']['vectors_npz']['path'],
        ordinary_result=str(rr/'result.json'),ordinary_vector=str(rr/'vector.npz'),
        static_metadata=prior['inputs']['static_metadata']['path'],recovery_exit=replay['inputs']['saved_exit']['path'])
    inputs={role:c.seal(path,role) for role,path in paths.items()}
    files={r['path']:copy.deepcopy(r) for r in prior['sealed_files']}
    for row in inputs.values():files[row['path']]=row
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_REPAIR_SOURCE:'+rel)
        row=c.seal(p,'repair_SOURCE:'+rel);files.setdefault(row['path'],row)
    receipt=Path(approval).resolve();row=c.seal(receipt,'new_repair_approval');files[row['path']]=row
    # Seal the full installed Qiskit distribution, rather than assuming the
    # saved replay's imported-module subset covers the transpiler.
    site=Path(prior['runtime']['python']).parent.parent/'lib/python3.12/site-packages'
    record=site/'qiskit-1.3.0.dist-info/RECORD'
    for entry in csv.reader(record.open()):
        rel,hashvalue,size=entry
        if not rel.startswith('qiskit/') or rel.endswith('.pyc') or not hashvalue:continue
        p=site/rel;c.require(p.resolve().is_relative_to(site.resolve()),'QISKIT_RUNTIME_INSIDE_VENV')
        actual='sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        c.require(hashvalue==actual and p.stat().st_size==int(size),'QISKIT_DISTRIBUTION_RECORD:'+rel)
        row=c.seal(p,'full_qiskit_runtime');files.setdefault(row['path'],row)
    for p in (record,record.parent/'METADATA'):
        row=c.seal(p,'qiskit_distribution_identity');files.setdefault(row['path'],row)
    metadata=c.read(inputs['static_metadata']['path'])['circuits']['WRAPPER'];counts=metadata['gate_counts']
    bound=3*counts.get('h',0)+counts.get('p',0)+counts.get('rz',0)+3*counts.get('rzz',0)+counts.get('cx',0)+31*counts.get('xx_plus_yy',0)+counts.get('measure',0)
    old_peak=c.read(inputs['recovery_exit']['path'])['ru_maxrss_KiB']*1024
    multiplier=(bound+metadata['operation_count']-1)//metadata['operation_count']+1
    resources=dict(candidate_instruction_bound=bound,per_worker_RSS_proxy_bytes=old_peak*multiplier,
        disk_proxy_bytes=16*(inputs['wrapper_qpy']['bytes']+replay['inputs']['compiled_qpy']['bytes']),
        admission='PRIOR_MEASURED_RSS_TIMES_INPUT_EXPANSION_PLUS_ONE',runtime_caps=False,host_PSI_kill=False)
    root=destination/'run01';supervisor={k:str(destination/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    cfg=dict(SOURCE=source,runtime=prior['runtime'],inputs=inputs,sealed_files=list(files.values()),resources=resources,
        approval_receipt_path=str(receipt),approval_receipt_sha256=c.sha(receipt),run_root=str(root),
        output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),supervisor=supervisor,authorized=False)
    cfg['pipeline_identity']=c.digest(dict(SOURCE=source,scope=c.scope(),runtime=cfg['runtime']))
    plan=dict(status='FIXED_REPAIR_COSINE_PILOT_PLAN',SOURCE=source,scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),
        inputs_digest=c.digest(inputs),runtime_digest=c.digest(cfg['runtime']),run_root=str(root),supervisor=supervisor,
        resources=resources,pipeline_identity=cfg['pipeline_identity'],approval_receipt_sha256=cfg['approval_receipt_sha256'])
    destination.mkdir(exist_ok=False,parents=True);plan_path=destination/'plan_PRIVATE.json';c.save(plan_path,plan)
    cfg.update(plan=plan,plan_path=str(plan_path),plan_sha256=c.sha(plan_path))
    cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,scope_id=c.SCOPE,SOURCE=source,
        plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],maximum_new_compile=1,
        maximum_statevector_replays=2,sine=0,sampling=0,retry=False,accepted_cost_rows=0,next_phase_authorized=False,
        **{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.verify_seals(cfg);c.save(destination/'config_DRAFT_PRIVATE.json',cfg);c.save(destination/'grant_INVALID.json',cfg['grant'])
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('origin-config','destination','source','approval'):p.add_argument('--'+key,required=True)
    args=p.parse_args();prepare(args.origin_config,args.destination,args.source,args.approval)
