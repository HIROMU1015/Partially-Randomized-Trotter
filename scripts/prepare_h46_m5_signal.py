"""Seal existing saved inputs/reference and six raw signal records; no arrays read."""
import argparse
import copy
import concurrent.futures.thread
import concurrent.futures._base
import queue
import _queue
import threading
from pathlib import Path
import subprocess
import h46_m5_signal_contract as c


def prepare(origin_directory,runtime_config,destination,source):
    dest=Path(destination).resolve();c.require(not dest.exists(),'UNUSED_SIGNAL_PROPOSAL')
    frozen=c.read(runtime_config);files={r['path']:copy.deepcopy(r) for r in frozen['sealed_files']}
    # Seal the added pool implementation without importing numerical libraries.
    for module in (concurrent.futures.thread,concurrent.futures._base,queue,_queue,threading):
        row=c.seal(Path(module.__file__).resolve(),'signal_thread_runtime:'+module.__name__)
        if row['path'] in files:c.require(row['sha256']==files[row['path']]['sha256'],'SAME_THREAD_RUNTIME')
        else:files[row['path']]=row
    systems={}
    for system in ('H4','H6'):
        path=Path(origin_directory)/(system+'_TOL_PILOT_DET_config_APPROVED_PRIVATE.json')
        old=c.verify_original_pilot(path)
        c.require(old['runtime']==frozen['runtime'],'SAME_FROZEN_SIGNAL_RUNTIME')
        for row in old['sealed_files']:
            if row['path'] in files:c.require(row['sha256']==files[row['path']]['sha256'],'UNCHANGED_PRIOR_CLOSURES')
            else:files[row['path']]=copy.deepcopy(row)
        info=dict(origin_config=str(path.resolve()),binding=old['binding'],inputs=old['inputs'],reference_root=old['reference_root'],reuse={})
        used=[path]+[Path(x['path']) for x in old['inputs'].values()]+[Path(old['reference_root'])/n for n in ('reference.npz','receipt.json')]
        for row in c.rows(system):
            if row['kind']!='DET_SIGNAL' or row['q']!=2:continue
            tasks=[t for t in old['plan']['scope']['tasks'] if all(t['group'][k]==row[k] for k in ('method','formula','prefix','q'))]
            if not tasks:continue
            c.require(len(tasks)==1,'UNIQUE_PILOT_SIGNAL')
            signal=Path(old['output'])/tasks[0]['id']/'signal.json'
            info['reuse'][row['id']]=c.seal(signal,'saved_signal:'+row['id']);used.append(signal)
        for p in used:
            row=c.seal(p,'saved_signal_input:'+p.name);files[row['path']]=row
        systems[system]=info
    for rel in c.SOURCE_PATHS:
        p=c.ROOT/rel;c.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT),'COMMITTED_SIGNAL_SOURCE:'+rel)
        row=c.seal(p,'signal_SOURCE:'+rel)
        if row['path'] in files:c.require(row['sha256']==files[row['path']]['sha256'],'OLD_SIGNAL_SOURCE_UNCHANGED')
        else:files[row['path']]=row
    dest.mkdir(parents=True)
    approval=dest/'approval_REQUIRED_PRIVATE.json'
    c.save(approval,dict(SOURCE=source,scope=c.scope(),preparation_authorized=True,execution_authorized=False,allowance_policy_accepted_for_this_run=False,
        user_instruction='その範囲でよいので作業を開始して',interpretation='Domain accepted and technical preparation authorized; real numerical execution requires separate approval'))
    row=c.seal(approval,'future_signal_execution_approval');files[row['path']]=row
    root=dest/'future_run01';supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    resources=dict(system_RSS_proxy_bytes={s:c.read(i['origin_config'])['RSS_proxy_bytes'] for s,i in systems.items()},
        proxy_kind='PRIOR_SAME_INPUT_CONTEXT_PILOT_OBSERVATION_FOR_ADMISSION_NOT_A_RUNTIME_BOUND',
        disk_proxy_bytes=sum(Path(r['path']).stat().st_size for i in systems.values() for r in i['inputs'].values()),
        host_PSI_kill=False,runtime_caps=False,worker_count='FRESH_SYSTEM_PARALLELISM_WITH_CANDIDATE_THREAD_POOL')
    cfg=dict(SOURCE=source,systems=systems,runtime=frozen['runtime'],provider_root=frozen['provider_root'],sealed_files=list(files.values()),resources=resources,
        run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),supervisor=supervisor,
        authorized=False,approval_receipt_path=str(approval),approval_receipt_sha256=c.sha(approval))
    cfg['pipeline_identity']=c.digest(dict(SOURCE=source,scope=c.scope(),runtime=cfg['runtime'],systems_digest=c.digest(systems)))
    plan=dict(status='SIGNAL_ONLY_FIXED_PREPARATION_EXECUTION_UNAPPROVED',SOURCE=source,scope=c.scope(),closure_digest=c.digest(cfg['sealed_files']),
        systems_digest=c.digest(systems),runtime_digest=c.digest(cfg['runtime']),resources=resources,run_root=str(root),supervisor=supervisor,
        pipeline_identity=cfg['pipeline_identity'],approval_receipt_sha256=c.sha(approval))
    pp=dest/'plan_DRAFT_PRIVATE.json';c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
    cfg['grant']=dict(valid=False,grant_id=None,approval_reference=None,scope_id=c.SCOPE_ID,SOURCE=source,plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
        maximum_new_signal=82,maximum_compile=0,maximum_sampling=0,retry=False,next_phase_authorized=False,**{k:cfg[k] for k in ('output','ledger','marker','supervisor')})
    c.verify(cfg);c.save(dest/'config_DRAFT_PRIVATE.json',cfg);c.save(dest/'grant_INVALID_PRIVATE.json',cfg['grant']);return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('origin-directory','runtime-config','destination','source'):p.add_argument('--'+name,required=True)
    a=p.parse_args();cfg=prepare(a.origin_directory,a.runtime_config,a.destination,a.source)
    print(c.digest(dict(plan_sha256=cfg['plan_sha256'],SOURCE=cfg['SOURCE'])))
