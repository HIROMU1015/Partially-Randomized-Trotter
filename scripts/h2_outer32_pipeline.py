"""Preauthorized H2 signal -> pilot -> design -> exploration acquisition only."""
import argparse
import copy
import datetime
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import h2_outer32_contract as c

ACTIVE_PROCESS=None
STOP_SIGNALS=[]


def other_reservation(cfg):
    r=cfg['resources'];p=Path(r['other_controller_finished'])
    if p.exists():
        try:
            done=c.read(p)
            if len(done['stages'])==2 and all(v['reaped'] for v in done['stages'].values()):return 0
        except (KeyError,ValueError):pass
    return r['other_fixed_growth_reserve_bytes']

def admission(cfg,count):
    from run_h46_m5_signal import fresh_resources
    r=cfg['resources'];root=Path(cfg['run_root']).parent
    obs=fresh_resources(dict(run_root=str(root/'unused_probe'),systems={'H2':{}},
        resources=dict(system_RSS_proxy_bytes={'H2':r['worker_RSS_proxy_bytes']},disk_proxy_bytes=0)))
    final=count*r['per_trajectory_growth_proxy_bytes'];raw=r['raw_pair_proxy_bytes'];other=other_reservation(cfg)
    scratch_budget=obs['disk_free_bytes']-other-final-raw
    workers=min(count,math.floor(obs['idle_CPU_equivalents']),obs['effective_available_RAM_bytes']//r['worker_RSS_proxy_bytes'],scratch_budget//(2*raw))
    c.require(workers>=1,'H2_FRESH_CPU_RAM_SCRATCH_ADMISSION')
    fs=os.statvfs(root);c.require(fs.f_favail>=r['inode_proxy'],'H2_FRESH_INODE_ADMISSION')
    return dict(workers=workers,observation=obs,other_fixed_growth_reserve_bytes=other,
        own_final_growth_proxy_bytes=final,raw_scratch_proxy_bytes=2*workers*raw,
        runtime_caps=False,proxy_is_upper_bound=False,host_PSI_kill=False)

def make_stage(parent,stage,signal_path=None,reference_path=None,preparation_hash=None,evaluation_path=None):
    dest=Path(parent['private'])/'stagepacks'/stage;dest.mkdir(parents=True,exist_ok=False);root=dest/'run01'
    cfg={k:copy.deepcopy(parent[k]) for k in ('SOURCE','runtime','provider_root','inputs','binding','resources','storage','compiler_phase_policy','approval_path')}
    cfg.update(stage=stage,tasks=[],campaign_id='H2_OUTER32_'+stage+'_RUN01_'+parent['SOURCE'],
        registry_path=None,excluded_seeds_path=None,signal_path=signal_path,reference_path=reference_path,
        preparation_hash=preparation_hash,evaluation_path=evaluation_path,run_root=str(root),output=str(root/'output'),
        ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),authorized=True,systems={'H2':{'binding':cfg['binding']}},
        supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]})
    closure={r['path']:r for r in parent['sealed_files']}
    for p,role in [(signal_path,'new_H2_outer32_signal'),(reference_path,'H2_full_H_reference'),(evaluation_path,'H2_pilot_design')]:
        if p:
            item=c.seal(p,role);closure[item['path']]=item
    if stage=='H2_PILOT_DESIGN':
        pilot=c.read(Path(parent['private'])/'stagepacks/H2_DESIGN_PILOT/config_APPROVED_PRIVATE.json')
        audit_path=Path(pilot['run_root'])/'end_audit.json';audit=c.read(audit_path)
        c.require(audit['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP','H2_PILOT_END_AUDIT_BEFORE_DESIGN')
        accepted={v['id']:v['result_sha256'] for v in audit['results']}
        for t in pilot['tasks']:
            out=Path(pilot['output'])/t['id'];p=out/'result.json'
            c.require(c.sha(p)==accepted[t['id']],'H2_ACCEPTED_PILOT_SHA')
            for path in [p,*[out/('piece_law_'+str(i)+'.json') for i in range(c.task_limits(t)['request_sampling_calls'])]]:
                item=c.seal(path,'accepted_H2_pilot_data');closure[item['path']]=item
        for path in [audit_path,Path(pilot['run_root'])/'completion_witness.json',Path(pilot['supervisor']['exit']),Path(pilot['run_root'])/'compiler_policy_end_audit.json']:
            item=c.seal(path,'accepted_H2_pilot_end_witness');closure[item['path']]=item
    if stage in ('H2_DESIGN_PILOT','H2_PR_EXPLORATION'):

        counts={r['id']:33 for r in c.groups()} if stage=='H2_DESIGN_PILOT' else c.counts_from_evaluation(c.read(evaluation_path))[0]
        cfg['tasks']=c.tasks_for(stage,counts)
        excluded=set(c.read(parent['initial_exclusions'])['values'])
        if stage=='H2_PR_EXPLORATION':
            old=Path(parent['private'])/'stagepacks/H2_DESIGN_PILOT/registry_PRIVATE.json'
            excluded.update(v for row in c.read(old).values() for v in row.values())
        ep=dest/'excluded_seeds_PRIVATE.json';rp=dest/'registry_PRIVATE.json'
        c.save(ep,dict(values=sorted(excluded),finite_known_registry_coverage=True,all_research_seeds_guaranteed=False))
        c.save(rp,c.registry(cfg['tasks'],cfg['SOURCE'],cfg['campaign_id'],c.cohort(stage),sorted(excluded)))
        cfg.update(registry_path=str(rp),excluded_seeds_path=str(ep))
        for p,role in [(ep,'H2_seed_exclusions'),(rp,'H2_fresh_cohort_registry')]:
            item=c.seal(p,role);closure[item['path']]=item
        cfg['resources']['inode_proxy']=2*sum(sum(c.task_limits(t).values())+100 for t in cfg['tasks'])
    cfg['resources']['disk_proxy_bytes']=0 # admission() computes own scratch + unchanged H4/H6 reservation.
    cfg['sealed_files']=list(closure.values());cfg['call_scope']=c.scope(stage,cfg['tasks'])
    pp=dest/'plan_APPROVED_PRIVATE.json';plan=c.plan_fields(cfg);c.save(pp,plan);cfg.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
    cfg['grant']=dict(valid=True,SOURCE=cfg['SOURCE'],stage=stage,plan_sha256=cfg['plan_sha256'],call_scope=cfg['call_scope'],
        grant_id=stage+'_'+c.digest(plan),retry=False,shot_accounting=False,ranking=False,independent_confirmation=False)
    c.authorize(cfg);path=dest/'config_APPROVED_PRIVATE.json';c.save(path,cfg);c.save(dest/'grant_APPROVED_PRIVATE.json',cfg['grant'])
    return path,cfg

def design_stage(cfg,pilot_cfg):
    c.authorize(cfg);root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_H2_DESIGN_ROOT');root.mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));Path(cfg['ledger']).mkdir();Path(cfg['output']).mkdir()
    audit=c.read(Path(pilot_cfg['run_root'])/'end_audit.json')
    c.require(audit['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP','H2_PILOT_AUDIT_REQUIRED')
    import h46_pr_pilot_evaluation as evaluation
    from concurrent.futures import ThreadPoolExecutor
    def one(candidate):
        c.save(Path(cfg['ledger'])/(candidate['id']+'.json'),dict(groups=1,pilot_trajectories=33,new_sampling=0,new_compile=0))
        indexed=[]
        for t in pilot_cfg['tasks']:
            if t['candidate']!=candidate:continue
            p=Path(pilot_cfg['output'])/t['id'];r=c.read(p/'result.json')
            indexed.append(dict(task=t,result_path=str(p/'result.json'),result_sha256=c.sha(p/'result.json'),
                SOURCE=pilot_cfg['SOURCE'],plan_sha256=pilot_cfg['plan_sha256'],input_binding=r['input_binding'],
                preparation_hash=r['preparation_hash'],law_paths=[str(p/('piece_law_'+str(i)+'.json')) for i in range(c.task_limits(t)['request_sampling_calls'])]))
        return evaluation.evaluate_group(dict(candidate=candidate,tasks=indexed),pilot_cfg['campaign_id'])
    with ThreadPoolExecutor(max_workers=min(12,len(os.sched_getaffinity(0)))) as pool:groups=list(pool.map(one,c.groups()))
    value=dict(status='H2_PILOT_DESIGN_PASS',SOURCE=cfg['SOURCE'],groups=groups,pilot_cohort='DESIGN_PILOT',
        independent_new_exploration=True,pooling=False,relative_MCSE_target=.02,variance_multiplier=4,minimum_n=33,
        zero_observed_variance_policy='FRESH33_FINITE_SAMPLE_MINIMUM_WITH_MCSE_DESIGN_UNRESOLVED_NOT_VARIANCE_ZERO_GUARANTEE',
        shot_accounting=False,ranking=False,population_tail_cost_bound=None,AD03='DEFERRED')
    counts,unresolved=c.counts_from_evaluation(value);value.update(n_E_adopted_by_preapproved_rule=counts,
        zero_variance_MCSE_design_unresolved=unresolved,exploration_trajectories=sum(counts.values()),exploration_compile_axes=2*sum(counts.values()))
    c.save(Path(cfg['output'])/'design.json',value);c.verify(cfg)
    c.save(root/'completion.json',dict(status=value['status'],result_sha256=c.sha(Path(cfg['output'])/'design.json'),mandatory_STOP=True))
    return value

def execute_one(parent,path,cfg):
    log=Path(cfg['run_root']).parent/'controller_supervisor_PRIVATE.log'
    global ACTIVE_PROCESS
    c.require(not STOP_SIGNALS,'H2_OWN_CONTROLLER_STOP')
    with log.open('xb') as f:
        process=subprocess.Popen([cfg['runtime']['python'],'-B',str(c.ROOT/'scripts/run_h2_outer32_stage.py'),
            '--supervise','--config',str(path)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        ACTIVE_PROCESS=process
        c.save(Path(cfg['run_root']).parent/'dispatch_PRIVATE.json',dict(pid=process.pid,SOURCE=cfg['SOURCE'],one_shot=True))
        code=process.wait();ACTIVE_PROCESS=None
    c.require(not STOP_SIGNALS,'H2_OWN_CONTROLLER_STOP_AFTER_REAP')
    c.require(code==0,'H2_STAGE_NATIVE_STOP:'+cfg['stage'])
    witness=c.read(cfg['supervisor']['exit'])
    c.require(witness['native_exit']==0 and witness['reaped'],'H2_STAGE_NATIVE_EXIT_REAP')
    if cfg['tasks']:
        # New auditor process avoids inheriting another stage's context/cohort.
        with (Path(cfg['run_root']).parent/'audit_PRIVATE.log').open('xb') as f:
            p=subprocess.Popen([cfg['runtime']['python'],'-B',str(c.ROOT/'scripts/run_h2_outer32_stage.py'),'--audit','--config',str(path)],
                stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);ACTIVE_PROCESS=p;code=p.wait();ACTIVE_PROCESS=None
        c.require(code==0,'H2_STAGE_END_AUDIT_STOP')
    else:
        result=c.read(Path(cfg['run_root'])/'completion.json');c.require(result['status'] in ('H2_OUTER32_SIGNAL_PASS','H2_PILOT_DESIGN_PASS'),'H2_METADATA_STAGE_COMPLETE')
    return dict(stage=cfg['stage'],native_exit=0,reaped=True,plan_sha256=cfg['plan_sha256'])

def finalize(parent,done,error=None):
    out=c.ROOT/'artifacts/resource_applicability/track_a_h2_outer32_run01/2026-10-21';out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in done:
        cfg=c.read(Path(parent['private'])/'stagepacks'/row['stage']/'config_APPROVED_PRIVATE.json')
        p=Path(cfg['run_root'])/'end_audit.json';audit=c.read(p) if p.exists() else {}
        ext=c.read(cfg['supervisor']['exit'])
        rows.append(dict(stage=row['stage'],plan_sha256=cfg['plan_sha256'],native_exit=row['native_exit'],reaped=True,
            planned_trajectories=len(cfg['tasks']),completed_trajectories=audit.get('completed_trajectories'),
            completed_axes=audit.get('accepted_compile_axes'),end_audit_status=audit.get('status'),
            end_audit_sha256=c.sha(p) if p.exists() else None,wall_seconds=ext.get('wall_seconds'),
            compressed_bytes=audit.get('compressed_byte_total'),archive_count=audit.get('archives'),
            byte_SHA_roundtrip_PASS=audit.get('byte_SHA_roundtrip_PASS'),result_private=True))
    summary=dict(status='H2_OUTER32_THROUGH_EXPLORATION_ACQUISITION_COMPLETE' if not error and len(done)==4 else 'STOP',
        SOURCE=parent['SOURCE'],stages=rows,error_details_private=bool(error),current_stage_failed=parent.get('current_stage') if error else None,
        post_exploration_statistics=False,shot_accounting=False,ranking=False,independent_confirmation=False,
        old_H2_samples_pooled=False,AD03='DEFERRED',retry=False,mandatory_STOP=True)
    c.save(out/'summary.json',summary)
    report=c.ROOT/'docs/research/track_a_h2_outer32_run01_results_20261021.md'
    report.write_text('# H2高次PF outer32：取得終了記録\n\n状態：`'+summary['status']+'`。SOURCE `'+parent['SOURCE']+'`。\n\n'
        'H2直鎖1 Å/STO-3G/tol-only1e-8 rank3、保存状態、T=.8/P=0、prefix1、S2/Yoshida4/m5_best、q1/2/4/8、K4/outer32。\n\n'
        '[技術summary](../../artifacts/resource_applicability/track_a_h2_outer32_run01/2026-10-21/summary.json)に各段階の終了状態を保存した。STOP時は欠測をPASSと扱わず、再試行しない。\n\n'
        'pilotだけを標本数設計に評価し、新探索標本は混合しない。本探索後の統計、shot会計、順位、独立確認、AD03採用は未実施。\n\n'
        '**mandatory STOP。次の担当：ユーザーが新探索結果の限定統計評価へ進めるか判断する。Codexは新しい承認まで後続解析を開始しない。\n')
    paths=[str(out.relative_to(c.ROOT)/'summary.json'),str(report.relative_to(c.ROOT))]
    brief='## 2026-10-21 H2 outer32：'+summary['status']+'\n\n[取得終了記録](docs/research/'+report.name+')。4段階の終了状態を区別し、旧H2/進行H4H6は不変。本探索後統計・shot・順位・P・AD03は未実施、mandatory STOP。以下は履歴。\n\n'
    for name in ('PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md'):
        p=c.ROOT/name;p.write_text((brief.replace('(docs/research/','(') if name.startswith('docs/research/') else brief)+p.read_text());paths.append(name)
    c.require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT),'EMPTY_OWN_INDEX')
    for rel in paths:c.require('/home/' not in (c.ROOT/rel).read_text(),'PUBLIC_METADATA_ONLY')
    subprocess.run(['git','add','--',*paths],cwd=c.ROOT,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m','Record H2 outer32 acquisition and mandatory STOP'],cwd=c.ROOT,check=True)


def launch(config_path):
    parent=c.read(config_path);require_parent(parent)
    root=Path(parent['run_root']);c.require(not root.exists(),'UNUSED_H2_PIPELINE_ROOT');root.mkdir()
    c.save(root/'CONSUMED.json',dict(SOURCE=parent['SOURCE'],approval_sha256=c.sha(parent['approval_path']),one_shot=True,retry=False))
    done=[];error=None
    def interrupted(sig,frame):
        STOP_SIGNALS.append(sig)
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:ACTIVE_PROCESS.send_signal(sig)
    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        p,cfg=make_stage(parent,c.STAGES[0]);parent['current_stage']=cfg['stage'];done.append(execute_one(parent,p,cfg))
        signals=c.read(Path(cfg['output'])/'signals.json');signal_path=str(Path(cfg['output'])/'signals.json');reference_path=str(Path(cfg['output'])/'reference.npz')
        prep=signals['preparation_hash']
        p,pilot=make_stage(parent,c.STAGES[1],signal_path,reference_path,prep);parent['current_stage']=pilot['stage'];done.append(execute_one(parent,p,pilot))
        p,cfg=make_stage(parent,c.STAGES[2],signal_path,reference_path,prep);parent['current_stage']=cfg['stage'];done.append(execute_one(parent,p,cfg))
        evaluation=str(Path(cfg['output'])/'design.json')
        p,cfg=make_stage(parent,c.STAGES[3],signal_path,reference_path,prep,evaluation);parent['current_stage']=cfg['stage'];done.append(execute_one(parent,p,cfg))
    except BaseException as exc:
        error=str(exc);c.save(root/'STOP.json',dict(stage=parent.get('current_stage'),reason=error,retry=False,mandatory_STOP=True))
    c.save(root/'completion.json',dict(SOURCE=parent['SOURCE'],stages=done,status='COMPLETE' if not error and len(done)==4 else 'STOP',
        error=error,post_exploration_statistics=False,mandatory_STOP=True))
    try:finalize(parent,done,error)
    except Exception as exc:c.save(root/'PUBLIC_FINALIZATION_STOP_PRIVATE.json',dict(reason=str(exc),science_retry=False))
    for sig,handler in handlers.items():signal.signal(sig,handler)
    return int(bool(error))

def require_parent(parent):
    c.require(parent.get('authorized') is True and parent.get('authorized_stages')==list(c.STAGES),'EXPLICIT_H2_FOUR_PHASE_AUTHORITY')
    c.require(parent['plan']==c.read(parent['plan_path']) and c.sha(parent['plan_path'])==parent['plan_sha256'],'H2_PIPELINE_PLAN_SHA')
    c.require(parent['plan']=={k:v for k,v in parent.items() if k not in ('plan','plan_path','plan_sha256')},'H2_PIPELINE_BINDINGS')
    c.require(parent.get('authorized') is True and parent.get('authorized_stages')==list(c.STAGES),'EXPLICIT_H2_FOUR_PHASE_AUTHORITY')
    c.require(c.read(parent['approval_path'])['no_intermediate_reapproval'] is True,'USER_H2_STAGED_EXECUTION')
    for r in parent['sealed_files']:
        p=Path(r['path']);c.require(p.stat().st_size==r['bytes'] and c.sha(p)==r['sha256'],'PIPELINE_SEALED_BYTES:'+r['role'])
    for rel,blob in c.source_blobs(parent['SOURCE']).items():c.require((c.ROOT/rel).read_bytes()==blob,'PIPELINE_SOURCE:'+rel)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();return launch(a.config)

if __name__=='__main__':raise SystemExit(main())
