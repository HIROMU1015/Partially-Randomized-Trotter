"""Run both separately approved acquisitions once, then technical audits only.

Durable controller may outlive the chat. No retry/top-up/statistics branch.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import h46_pr_pilot_contract as c

def common_integrity_failure(text):
    return any(word in text for word in ('SEALED_BYTES','COMMITTED_',
      'ARCHIVE_FORMAT','STORED_BYTES_OR_SHA','RAW_BYTES_OR_SHA','RAW_CHANGED_BEFORE_RELEASE',
      'ARCHIVE_MANIFEST_BINDING','STORAGE_POLICY','SAME_SIGNAL_INPUT_RUNTIME','FIXED_ORIGINAL_COMPILER'))

def launch(config_paths,controller_root):
    cfgs=[c.read(p) for p in config_paths]
    c.require({v['stage'] for v in cfgs}==set(c.STAGES) and len(cfgs)==2,'TWO_SEPARATE_STAGE_APPROVALS')
    c.require(len({v['SOURCE'] for v in cfgs})==1 and len({v['campaign_id'] for v in cfgs})==1,'COMMON_SOURCE_ONLY')
    for cfg in cfgs:c.authorize(cfg)
    root=Path(controller_root).resolve();root.mkdir(exist_ok=False)
    c.save(root/'CONTROLLER_ENTERED.json',dict(SOURCE=cfgs[0]['SOURCE'],
      stages=[v['stage'] for v in cfgs],retry=False,statistics=False))
    pending=list(zip(config_paths,cfgs));active={};done={};signals=[];handlers={}
    def interrupt(sig,frame):
        signals.append(sig)
        for p in active.values():
            if p.poll() is None:p.send_signal(sig)
    for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
    files=[];common_stop=False
    try:
        # Complete both static gates before either grant can be consumed.
        for path,cfg in pending:
            if signals or common_stop:break
            log=(root/(cfg['stage']+'_supervisor_PRIVATE.log')).open('xb');files.append(log)
            p=subprocess.Popen([cfg['runtime']['python'],'-B',str(c.ROOT/'scripts/run_h46_pr_design_pilot.py'),
              '--supervise','--config',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[cfg['stage']]=p
            c.save(root/(cfg['stage']+'_dispatch.json'),dict(pid=p.pid,SOURCE=cfg['SOURCE'],
              plan_sha256=cfg['plan_sha256'],one_shot=True))
        while active:
            # A worker may expose a common integrity error before its pool has
            # finished cooperative cleanup. Observe only our known stage journal.
            if not common_stop:
                for cfg in cfgs:
                    journal=Path(cfg['run_root'])/'task_completion.jsonl'
                    if not journal.exists():continue
                    with journal.open('rb') as f:
                        f.seek(0,2);size=f.tell();f.seek(max(0,size-65536))
                        lines=f.read().splitlines()
                    if lines and common_integrity_failure(lines[-1].decode(errors='replace')):
                        common_stop=True
                        for other in active.values():
                            if other.poll() is None:other.send_signal(signal.SIGTERM)
                        c.save(root/'COMMON_INTEGRITY_STOP.json',dict(origin_stage=cfg['stage'],
                          owned_stages_only=True,no_retry=True))
                        break
            for stage,p in list(active.items()):
                code=p.poll()
                if code is None:continue
                p.wait();done[stage]=dict(exit=code,reaped=True);del active[stage]
                cfg=next(v for v in cfgs if v['stage']==stage)
                cp=Path(cfg['run_root'])/'completion_witness.json'
                reason=json.dumps(c.read(cp).get('errors',{})) if cp.exists() else ''
                reason+=(root/(stage+'_supervisor_PRIVATE.log')).read_text()[-10000:]
                # Local numeric failure stops its own pool. Shared identity or
                # storage corruption requests cooperative STOP of our other
                # launcher, never an unrelated process or a PSI-based kill.
                if code and common_integrity_failure(reason) and not common_stop:
                    common_stop=True
                    for other in active.values():
                        if other.poll() is None:other.send_signal(signal.SIGTERM)
                    c.save(root/(stage+'_common_integrity_STOP.json'),dict(reason=reason,
                      owned_stages_only=True,retry=False,old_STOPs_unchanged=True))
            if active:time.sleep(2)
        for log in files:log.close()
        from audit_h46_pr_pilot_execution import audit
        audits={}
        for cfg in cfgs:
            try:
                report=audit(cfg)
                audits[cfg['stage']]=dict(status=report['status'],
                  end_audit_sha256=c.sha(Path(cfg['run_root'])/'end_audit.json'))
            except Exception as exc:
                audits[cfg['stage']]=dict(status='STOP_END_AUDIT_INCOMPLETE',reason=str(exc))
        result=dict(SOURCE=cfgs[0]['SOURCE'],stages=done,audits=audits,signals=signals,
          common_integrity_STOP=common_stop,retry=False,next_phase_started=False,
          statistics=False,mandatory_STOP=True)
        c.save(root/'CONTROLLER_FINISHED.json',result)
        from finalize_h46_pr_pilot_acquisition import finalize
        try:finalize(cfgs,root)
        except Exception as exc:
            c.save(root/'PUBLIC_FINALIZATION_STOP_PRIVATE.json',dict(reason=str(exc),
              science_retry=False,acquisition_results_preserved=True,mandatory_STOP=True))
        return (0 if len(done)==2 and all(v['exit']==0 for v in done.values())
          and all(v['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP' for v in audits.values()) else 1)
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',action='append',required=True)
    p.add_argument('--controller-root',required=True);a=p.parse_args()
    return launch(a.config,a.controller_root)

if __name__=='__main__':raise SystemExit(main())
