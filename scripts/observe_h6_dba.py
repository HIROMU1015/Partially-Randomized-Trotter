"""Read-only scientific-job observer. Never stop, restart, fit or reaccount."""
import argparse
import json
import os
from pathlib import Path
import time
import h6_dba_contract as c


def maybe(path):
    try:return c.read(path)
    except (FileNotFoundError,ValueError):return None


def snapshot(private):
    scopes=[('D',private,private/'D_run01'),
            ('B',private/'B_attempt02',private/'B_attempt02/output'),
            ('A',private/'A_attempt02',private/'A_attempt02/output')]
    stages={}
    for stage,base,out in scopes:
        started=maybe(base/(stage+'_native_started.json'))
        exit=maybe(base/(stage+'_native_exit.json'));terminal=maybe(out/'terminal.json')
        stop=maybe(out/'STOP.json');result=maybe(out/'result.json')
        status='LAUNCH_PENDING';cpu=None;rss=None;threads=None;alive=False
        if started:
            proc=Path('/proc')/str(started['pid'])
            try:
                fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                command=(proc/'cmdline').read_bytes()
                alive=(b'run_h6_dba.py' in command and f'--stage\0{stage}\0'.encode() in command)
                if alive:
                    cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
                    rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
                    threads=int(fields[17])
                    status='RUNNING'
            except (FileNotFoundError,ProcessLookupError):pass
        if exit:
            status=('COMPLETE_MANDATORY_STOP' if exit['exit_code']==0 and result and terminal
                    and terminal['status']=='COMPLETE_MANDATORY_STOP' else 'STOP')
        elif stop:status='STOP_NATIVE_EXIT_PENDING'
        workers=[];measurements=[]
        for name in ['D','first_control_workers','remaining_workers','signal_workers']:
            parent=out/name
            if not parent.exists():continue
            dirs=[parent] if name=='D' else [p for p in parent.iterdir() if p.is_dir()]
            for work in dirs:
                w=maybe(work/'started.json')
                if w:
                    row=dict(task=w['task'],pid=w['pid'],complete=(work/'task_result.json').exists())
                    proc=Path('/proc')/str(w['pid'])
                    try:
                        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                        if int(fields[19])==w['identity'][1]:
                            row.update(cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                                       RSS_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'),threads=int(fields[17]))
                    except (FileNotFoundError,ProcessLookupError):pass
                    workers.append(row)
                for axis in ['cosine','sine']:
                    value=maybe(work/('measured_'+axis+'.json'))
                    if value:measurements.append(dict(task=work.name,**value))
        stages[stage]=dict(status=status,SOURCE=started['SOURCE'] if started else None,
            driver_alive=alive,driver_cpu_seconds=cpu,driver_RSS_bytes=rss,driver_threads=threads,
            workers=workers,measurements=measurements,
            first_control_gate=(out/'first_control_gate.json').exists(),
            primitive_gate=(out/'primitive_gate.json').exists(),
            native_exit=exit,STOP_reason=stop['reason'] if stop else None)
    return dict(time_unix=time.time(),stages=stages,observer_never_controls_workers=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--public',type=Path)
    a=p.parse_args()
    while True:
        value=snapshot(a.private)
        if not a.watch:
            print(json.dumps(value,indent=2));return
        temporary=a.private/'progress.tmp'
        temporary.write_text(json.dumps(value,allow_nan=False));os.replace(temporary,a.private/'progress.json')
        with (a.private/'progress_history.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        if all(x['native_exit'] is not None for x in value['stages'].values()):
            c.save(a.private/'completion_summary.json',value)
            if a.public:
                a.public.mkdir(parents=True,exist_ok=True)
                for stage,parent,out in [('D',a.private,a.private/'D_run01'),
                    ('B',a.private/'B_attempt02',a.private/'B_attempt02/output'),
                    ('A',a.private/'A_attempt02',a.private/'A_attempt02/output')]:
                    result=maybe(out/'result.json');exit=maybe(parent/(stage+'_native_exit.json'))
                    stop=maybe(out/'STOP.json')
                    c.save(a.public/(stage+'_terminal_result.json'),dict(
                        SOURCE=value['stages'][stage]['SOURCE'],native_exit=exit,
                        terminal=maybe(out/'terminal.json'),result=result,
                        STOP_reason=stop['reason'] if stop else None,
                        scientific_reaccounting_performed=False))
                c.save(a.public/'completion_status.json',value)
            return
        time.sleep(10)


if __name__=='__main__':main()
