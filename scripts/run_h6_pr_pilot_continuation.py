"""One approved continuation only, durable exit/byte audit and mandatory STOP."""
import argparse
import json
from pathlib import Path
from types import FunctionType
import h6_pr_pilot_continuation_contract as c

def finalize(cfg,auditor):
    report=auditor.audit(cfg)
    result=dict(status='STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
      old_SOURCE=c.OLD_SOURCE,old_whole_run_STOP_retained=True,retry=False,next_phase_started=False,
      pilot_statistics=False,n_E=None,shot_accounting=False,ranking=False,AD03='DEFERRED',mandatory_STOP=True,
      new_stage_audit_status=report['status'],new_complete_pairs=report['completed_trajectories'],
      new_accepted_compile_axes=report['accepted_compile_axes'])
    if report['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP':
        # Recheck the accepted originals byte-for-byte before joining references.
        from audit_h6_pr_pilot_cutoff_repair import audit
        acceptance=c.acceptance(cfg)[0];saved,part=audit(acceptance['old_config_path'])
        c.require(saved['result_set_sha256']==c.ACCEPTED_RESULT_SET
          and part['completed_ids']==acceptance['completed_ids']
          and part['remaining_ids']==acceptance['remaining_ids'],'FINAL_SAVED74_ACCEPTANCE')
        refs=[dict(id=r['id'],SOURCE=c.OLD_SOURCE,path=r['path'],sha256=r['sha256'],kind='SAVED74_SEPARATE_ACCEPTANCE')
          for r in acceptance['accepted_results']]
        for task in cfg['tasks']:
            p=Path(cfg['output'])/task['id']/'result.json'
            refs.append(dict(id=task['id'],SOURCE=cfg['SOURCE'],path=str(p),sha256=c.sha(p),kind='CONTINUATION322_DIRECT'))
        c.require(len(refs)==396 and len({r['id'] for r in refs})==396,'COMPLETE_ORIGINAL396_COVERAGE')
        c.save(Path(cfg['run_root'])/'combined_result_references_PRIVATE.json',dict(
          results=refs,original_registry_sha256=acceptance['original_registry_sha256'],
          old_run_STOP_retained=True,no_cohort_pooling=True,statistics=False))
        result.update(status='ORIGINAL396_COVERAGE_COMPLETE_SEPARATE74_ACCEPTANCE_PLUS322_CONTINUATION',
          saved_accepted_pairs=74,covered_pairs=396,covered_compile_axes=792,
          new_archives=report['archives'],new_compressed_bytes=report['compressed_byte_total'],
          new_native_and_byte_PASS=True,old_saved_byte_PASS=True,
          wall_seconds=c.read(Path(cfg['run_root'])/'completion_witness.json')['wall_seconds'],
          new_end_audit_sha256=c.sha(Path(cfg['run_root'])/'end_audit.json'))
    c.save(Path(cfg['run_root'])/'continuation_summary_PRIVATE.json',result)
    return result

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true')
    p.add_argument('--config');a=p.parse_args()
    if not (a.execute or a.supervise):
        print(json.dumps(dict(status='SCOPE_ONLY',accepted_saved_pairs=74,pending_pairs=322,
          maximum_compile_axes=644,science_started=False,mandatory_STOP=True)));return 0
    try:
        c.require(a.config,'CONTINUATION_CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        runner,auditor=c.bind()
        if a.supervise:
            supervise=FunctionType(runner.supervise.__code__,dict(runner.supervise.__globals__,c=c,__file__=__file__),
              'continuation_supervise',runner.supervise.__defaults__,runner.supervise.__closure__)
            code=supervise(cfg,a.config)
            try:summary=finalize(cfg,auditor)
            except Exception as exc:
                c.save(Path(cfg['run_root']).parent/'END_AUDIT_STOP_PRIVATE.json',dict(
                  status='STOP',reason=str(exc),retry=False,next_phase_started=False,mandatory_STOP=True))
                raise
            return code or (0 if summary['status'].startswith('ORIGINAL396_COVERAGE_COMPLETE') else 1)
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
          grant_id=cfg['grant']['grant_id'],retry=False),'EXTERNAL_COST_SUPERVISOR_REQUIRED')
        return runner.execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED_OR_STOP',reason=str(exc),retry=False)));return 2
if __name__=='__main__':raise SystemExit(main())
