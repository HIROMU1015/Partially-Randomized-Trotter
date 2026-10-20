"""Technical end audit only. No cost mean, variance, MCSE, shots or ranking."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import h46_pr_pilot_contract as c

def task_audit(cfg,task):
    root=Path(cfg['output'])/task['id'];result=c.read(root/'result.json')
    c.require(result['task']==task and result['SOURCE']==cfg['SOURCE']
      and result['plan_sha256']==cfg['plan_sha256'] and result['stage']==cfg['stage'], 'RESULT_IDENTITY')
    c.require(result['status']=='UNIT_COST_PASS_NOT_ACCOUNTING' and result['native_scientific_PASS']
      and result['byte_restoration_PASS'],'BOTH_NATIVE_AND_BYTE_PASS')
    c.require(len(result['checks'])==8 and all(r['PASS'] and not r['phase_alignment']
      and 0<=r['error']<r['strictly_below'] for r in result['checks']),'ALL_EIGHT_NUMERIC_GATES')
    c.require(result['seed_cohort']==c.design.COHORT and not result['conditional_sampling']
      and not result['cohort_pooling'] and not result['shot_accounting'] and not result['cost_ranking'],'ACQUISITION_ONLY')
    events=c.read(root/'events_PRIVATE.json')
    c.require(events['event_digest']==result['event_digest']==c.digest(events['events']),'EVENT_DIGEST')
    c.require(c.design.event_stratum(events['events'])==result['events'],'STRATUM_METADATA')
    c.require(sum(map(len,events['events']))==32*task['candidate']['q'],'REGISTERED_EVENT_COUNT')
    c.require(len(events['events'])==c.task_limits(task)['request_sampling_calls'],'REGISTERED_PIECE_COUNT')
    registry=c.read(cfg['registry_path'])[task['id']]
    import h46_m5_native as native
    schedule=native.c.core.schedule(task['candidate']['formula'],task['candidate']['q'],
      task['candidate']['r_policy'],task['candidate']['r_value'])
    for i,piece in enumerate(events['events']):
        law=c.read(root/('piece_law_'+str(i)+'.json'))
        c.require(law['piece_request_index']==i and law['event_count']==len(piece)
          and law['cohort']==c.design.COHORT and not law['conditioned'],'PIECE_LAW_BINDING')
        step=schedule[i]
        c.require(law['registry_seed_PRIVATE']==registry[str(step['outer'])+':'+str(step['piece'])]
          and law['signed_step_time']==step['signed_time'] and law['rte_steps']==step['rte_steps'],
          'REGISTERED_SEED_SIGNED_SCHEDULE')
    sa=c.storage.audit_task(cfg,task)
    stored=c.read(root/'storage_audit.json')
    c.require(sa==stored and c.sha(root/'storage_audit.json')==result['storage_audit_sha256'],'FINAL_BYTE_AUDIT_REPRODUCED')
    c.require([r['axis'] for r in result['axis_costs']]==['cosine','sine'],'PAIRED_AXES')
    for receipt in result['axis_costs']:
        m,_=c.storage.manifest_for(cfg,root/('quarantine_'+receipt['axis'])/'compiled.qpy',verify=False)
        c.require(receipt['compiled_SHA']==m['codec_receipt']['original']['sha256']
          and receipt['compiler_settings_hash']==c.design.COMPILER_HASH
          and receipt['native_policy']==c.design.NATIVE_POLICY
          and receipt['event_digest']==result['event_digest'],'RAW_COMPILED_SHA_AND_PROFILE')
    return dict(id=task['id'],result_sha256=c.sha(root/'result.json'),archives=sa['archive_count'],
      original_bytes=sa['original_bytes'],stored_bytes=sa['stored_bytes'],
      peak_RSS_KiB=result['peak_RSS_KiB'],native_gates=len(result['checks']))

def audit(cfg):
    c.authorize(cfg);c.consumed(cfg);root=Path(cfg['run_root'])
    completion=c.read(root/'completion_witness.json') if (root/'completion_witness.json').exists() else dict(status='MISSING',results=[],errors={'missing':True},workers={})
    external=c.read(cfg['supervisor']['exit']) if Path(cfg['supervisor']['exit']).exists() else dict(native_exit=None,reaped=False)
    errors={};rows=[]
    try:
        c.require(completion['status']=='UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP'
          and not completion['errors'],'COMPLETE_STAGE_REQUIRED')
        c.require(external['SOURCE']==cfg['SOURCE'] and external['plan_sha256']==cfg['plan_sha256']
          and external['native_exit']==0 and external['reaped'],'NATIVE_EXIT_REAP')
        c.require(completion['workers'] and all(w['exit']==0 and w['reaped'] for w in completion['workers'].values()),'ALL_WORKERS_EXIT_REAP')
        # Streaming SHA/decompression, no QPY/NPZ decode or state replay.
        count=min(len(cfg['tasks']),len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=count) as pool:rows=list(pool.map(lambda t:task_audit(cfg,t),cfg['tasks']))
        c.require(sorted(completion['results'],key=lambda r:r['id'])==sorted([
          {k:r[k] for k in ('id','result_sha256','peak_RSS_KiB')} for r in rows],key=lambda r:r['id']),
          'COMPLETION_RESULT_SHA_SET')
        expected={t['id']+'__'+kind+'__'+str(i)+'.json' for t in cfg['tasks']
          for kind,n in c.task_limits(t).items() for i in range(n)}
        c.require({p.name for p in Path(cfg['ledger']).glob('*.json')}==expected,'EXACT_SCIENTIFIC_CALL_LEDGER')
        c.verify(cfg)
    except Exception as exc:errors['end_audit']=str(exc)
    # Byte sizes and technical resource maxima only; no scientific aggregation.
    files=[p for p in root.rglob('*') if p.is_file()]
    ram_peak=scratch_peak=0
    observations=root/'resource_observation.jsonl'
    if observations.exists():
        for line in observations.read_text().splitlines():
            r=__import__('json').loads(line)
            ram_peak=max(ram_peak,sum(x.get('VmRSS',0) for x in r.get('own_process_RAM',{}).values()))
            scratch_peak=max(scratch_peak,r.get('owned_scratch_allocated_bytes_observed',0))
    report=dict(status='STOP' if errors else 'ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP',
      SOURCE=cfg['SOURCE'],stage=cfg['stage'],plan_sha256=cfg['plan_sha256'],errors=errors,
      completed_trajectories=len(completion['results']),accepted_compile_axes=2*len(completion['results']),
      planned_trajectories=len(cfg['tasks']),planned_compile_axes=cfg['call_scope']['compile_axes'],
      failed_or_incomplete_trajectories=len(cfg['tasks'])-len(completion['results']),
      archives=sum(r['archives'] for r in rows),original_byte_total=sum(r['original_bytes'] for r in rows),
      compressed_byte_total=sum(r['stored_bytes'] for r in rows),
      byte_SHA_roundtrip_PASS=not errors,results=rows,
      logical_run_bytes=sum(p.stat().st_size for p in files),allocated_run_bytes=sum(p.stat().st_blocks*512 for p in files),
      worker_peak_RSS_KiB=max((r['peak_RSS_KiB'] for r in rows),default=None),external_exit=external,
      own_aggregate_RAM_peak_sampled_bytes=ram_peak,owned_scratch_peak_sampled_bytes=scratch_peak,
      peaks_are_sampled_not_exact_bounds=True,
      reserved_compile_calls=len(list(Path(cfg['ledger']).glob('*__compile_axes__*.json'))),
      completion_sha256=c.sha(root/'completion_witness.json') if (root/'completion_witness.json').exists() else None,
      scientific_statistics_executed=False,n_E=None,shot_policy_adopted=False,AD03='DEFERRED',
      retry=False,next_phase_started=False,mandatory_STOP=True)
    c.save(root/'end_audit.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    r=audit(c.read(a.config));print(__import__('json').dumps({k:v for k,v in r.items() if k not in ('results','external_exit')}))
    return 1 if r['errors'] else 0

if __name__=='__main__':raise SystemExit(main())
