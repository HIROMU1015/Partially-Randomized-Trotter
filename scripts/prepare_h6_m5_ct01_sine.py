"""Metadata-only acceptance of saved cosine and a fresh INVALID sine proposal."""
import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import h6_m5_ct01_sine_contract as c


def prepare(origin_config,runtime_seal_config,destination,source,approval):
    old=c.read(origin_config);c.old.verify_seals(old);c.old.consumed(old)
    frozen=c.read(runtime_seal_config)
    c.require(frozen['SOURCE']=='0d47f59c351798434f7f29cdcbda993f60b5462c' and frozen['runtime']==old['runtime'],'FROZEN_FULL_RUNTIME_CLOSURE')
    dest=Path(destination).resolve();c.require(not dest.exists(),'UNUSED_CT01_PROPOSAL')
    root=Path(old['run_root']);task=Path(old['output'])/c.old.TASK['id'];q=task/'quarantine/cosine'
    paths=dict(old_config=Path(origin_config).resolve(),old_stop=task/'STOP.json',old_completion=root/'completion_witness.json',
      old_exit=c.ROOT/'artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/external_exit_witness.json',saved_signal=task/'signal.json')
    paths.update({key:q/name for key,name in [('wrapper_qpy','wrapper.qpy'),('compiled_cosine_qpy','compiled.qpy'),('initial_npz','initial.npz'),('saved_vectors_npz','vectors.npz'),('wrapper_receipt','wrapper_receipt.json'),('compiled_receipt','compiled_receipt.json'),('vectors_receipt','vectors_receipt.json'),('diagnostics','diagnostics.json')]})
    inputs={role:c.seal(path,role) for role,path in paths.items()}
    files={r['path']:copy.deepcopy(r) for r in frozen['sealed_files']}
    for row in old['sealed_files']:
        if row['path'] in files:c.require(row['sha256']==files[row['path']]['sha256'],'MATCHING_PRIOR_CLOSURES')
        else:files[row['path']]=copy.deepcopy(row)
    for r in inputs.values():files[r['path']]=r
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_CT01_SOURCE:'+rel)
        row=c.seal(p,'CT01_SOURCE:'+rel)
        if row['path'] in files:c.require(row['sha256']==files[row['path']]['sha256'],'PRIOR_SOURCE_UNCHANGED:'+rel)
        else:files[row['path']]=row
    receipt=Path(approval).resolve();row=c.seal(receipt,'future_CT01_sine_approval');files[row['path']]=row
    acceptance=c.assess_saved_cosine(inputs)
    dest.mkdir(parents=True,exist_ok=False);acceptance_path=dest/'saved_cosine_acceptance_PRIVATE.json';c.save(acceptance_path,acceptance)
    row=c.seal(acceptance_path,'saved_cosine_CT01_acceptance');files[row['path']]=row
    resources=dict(per_worker_RSS_proxy_bytes=old['RSS_proxy_bytes'],disk_proxy_bytes=2*inputs['compiled_cosine_qpy']['bytes']+2*inputs['wrapper_qpy']['bytes']+4*inputs['saved_vectors_npz']['bytes'],
      admission='SAME_POINT_PRIOR_MEASURED_PEAK_PROXY;NOT_RUNTIME_CAP',runtime_caps=False,host_PSI_kill=False)
    run=dest/'run01';supervisor={k:str(dest/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    cfg=dict(SOURCE=source,runtime=old['runtime'],provider_root=old['provider_root'],inputs=inputs,sealed_files=list(files.values()),resources=resources,approval_receipt_path=str(receipt),approval_receipt_sha256=c.sha(receipt),run_root=str(run),output=str(run/'output'),ledger=str(run/'ledger'),marker=str(run/'CONSUMED.json'),supervisor=supervisor,authorized=False,cosine_acceptance=acceptance,cosine_acceptance_path=str(acceptance_path))
    cfg['pipeline_identity']=c.digest(dict(SOURCE=source,scope=c.scope(),runtime=cfg['runtime']))
    protocol=subprocess.check_output(['git','show',c.PROTOCOL_COMMIT+':'+c.PROTOCOL],cwd=c.ROOT)
    plan=dict(status='FIXED_CT01_ONE_SINE_PLAN_EXECUTION_UNAPPROVED',SOURCE=source,scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),inputs_digest=c.digest(inputs),runtime_digest=c.digest(cfg['runtime']),run_root=str(run),supervisor=supervisor,resources=resources,pipeline_identity=cfg['pipeline_identity'],protocol_sha256=hashlib.sha256(protocol).hexdigest(),cosine_acceptance_sha256=c.sha(acceptance_path),approval_receipt_sha256=cfg['approval_receipt_sha256'])
    plan_path=dest/'plan_PRIVATE.json';c.save(plan_path,plan);cfg.update(plan=plan,plan_path=str(plan_path),plan_sha256=c.sha(plan_path))
    cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,scope_id=c.SCOPE,SOURCE=source,plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],maximum_new_compile=1,maximum_compiled_state_actions=1,new_cosine_compile=0,new_sampling=0,compile_vector_strictly_below=1e-7,other_checks_strictly_below=1e-9,retry=False,next_phase_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.verify_seals(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID.json',cfg['grant']);return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('origin-config','runtime-seal-config','destination','source','approval'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.origin_config,a.runtime_seal_config,a.destination,a.source,a.approval)
