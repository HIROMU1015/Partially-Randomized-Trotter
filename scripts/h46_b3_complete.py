"""Check and locally commit the study after its science service exits.

No scientific work is launched here. Any failed scientific stage stays stopped.
Only the dedicated branch and explicit study/report/index paths are committed.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import h46_b3 as b
from h46_b3_durable import save

UNIT='prt-h46-b3-20261026-run03.service'
BRANCH='codex/h46-long-time-b3-20261026'
STATUS=b.PRIVATE/'final_status.json'


def command(args,**kw):
    return subprocess.run(args,cwd=b.ROOT,text=True,capture_output=True,**kw)


def state():
    r=command(['systemctl','--user','show',UNIT,'--property=ActiveState','--property=SubState','--property=Result','--property=ExecMainStatus'])
    if r.returncode: raise RuntimeError(r.stderr)
    return dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)


def main():
    while True:
        unit=state();progress={}
        for cohort in ('B3_EXPLORATION','B3_CONFIRMATION'):
            path=b.PRIVATE/(cohort+'_progress.json')
            if path.exists():
                r=b.read(path);progress[cohort]={k:r.get(k) for k in ('total','complete','active','pending')}
                progress[cohort]['errors']=len(r.get('errors',[]))
        save(STATUS,dict(stage='SCIENTIFIC_ACQUISITION_RUNNING',unit=unit,progress=progress,updated_epoch=time.time(),
            follow_on='After successful acquisition: accounting, independent checks, plots, completion checks and dedicated-branch local commit; concrete payload review and direct push remain'))
        if unit.get('ActiveState') not in ('active','activating','deactivating'): break
        time.sleep(30)
    if unit.get('Result')!='success' or unit.get('ExecMainStatus')!='0' or not (b.PUBLIC/'stages_complete.json').exists():
        save(STATUS,dict(stage='SCIENTIFIC_STAGE_FAILED_STOP_PRESERVED',unit=unit,progress=progress,
            stderr=str(b.PRIVATE/'run03_stderr.log'),new_scientific_stage_started=False))
        raise RuntimeError('SCIENCE_SERVICE_NOT_COMPLETE; no automatic retry')
    save(STATUS,dict(stage='COMPLETION_CHECKS',unit=unit))
    import h46_b3_finalize
    h46_b3_finalize.main()
    for name in ('cost_vs_time','selected_q_delta','R_outer_B_N_C_G','expanded_best_method_map'):
        for suffix in ('png','svg','pdf'):
            path=b.PUBLIC/(name+'.'+suffix)
            if path.stat().st_size<100: raise ValueError('MISSING_FIGURE:'+str(path))
        with (b.PUBLIC/(name+'.png')).open('rb') as stream:
            header=stream.read(24)
            if header[:8]!=b'\x89PNG\r\n\x1a\n' or min(struct.unpack('>II',header[16:24]))<500:
                raise ValueError('INVALID_PNG_EXPORT')
    current=command(['git','branch','--show-current']).stdout.strip()
    remote=command(['git','remote','get-url','origin'])
    if current!=BRANCH or remote.returncode: raise ValueError('DEDICATED_BRANCH_OR_REMOTE')
    paths=[str(b.PUBLIC.relative_to(b.ROOT)),'REPORT.md','PROJECT_MAP.md','VALIDATION_STATUS.md',
        'docs/README.md','docs/research/研究概要・現状.md','scripts/README.md','src/trotterlib/README.md',
        'artifacts/validation_manifest.json','docs/research/研究ノート/2026-10-26.md']
    paths += [str(p.relative_to(b.ROOT)) for p in (b.ROOT/'scripts').glob('h46_b3*.py')]
    paths += [str(p.relative_to(b.ROOT)) for p in (b.ROOT/'tests').glob('test_h46_b3*.py')]
    r=command(['git','add','--',*paths])
    if r.returncode: raise RuntimeError(r.stderr)
    env=dict(os.environ,GIT_AUTHOR_NAME='AbeHiromu',GIT_AUTHOR_EMAIL='AbeHiromu@users.noreply.github.com',
        GIT_COMMITTER_NAME='AbeHiromu',GIT_COMMITTER_EMAIL='AbeHiromu@users.noreply.github.com',GIT_TERMINAL_PROMPT='0')
    if command(['git','diff','--cached','--quiet']).returncode:
        r=command(['git','commit','-m','Complete H4 H6 long-time B3 benchmark with independent costs and preserved recovery provenance'],env=env)
        if r.returncode: raise RuntimeError(r.stderr)
    sha=command(['git','rev-parse','HEAD']).stdout.strip()
    retry='git -C '+str(b.ROOT)+' push -u origin '+BRANCH
    result=dict(stage='LOCAL_COMPLETE_PUBLICATION_PENDING',
        local_commit=sha,branch=BRANCH,remote='origin',push_attempted=False,push_command=retry,
        scientific_report=str(b.PUBLIC/'REPORT.md'),mandatory_scientific_stop_reached=True,next_scientific_work=None,
        next_technical_work='Codex: inspect the actual committed payload and directly push to the verified requested repository',
        completed_epoch=time.time())
    save(STATUS,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        current=b.read(STATUS) if STATUS.exists() else {}
        if current.get('stage')!='SCIENTIFIC_STAGE_FAILED_STOP_PRESERVED':
            save(STATUS,dict(stage='TECHNICAL_COMPLETION_FAILED',reason=str(exc),scientific_results_retained=True,
                new_scientific_work_started=False))
        raise
