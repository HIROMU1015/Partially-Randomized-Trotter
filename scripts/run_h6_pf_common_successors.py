"""Advance successful approved H6 stages once; never retry a scientific command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def run(private):
    private=Path(private).resolve();root=Path(__file__).resolve().parents[1]
    save(private/'SUCCESSOR_DRIVER_ENTERED.json',dict(status='ONE_SHOT',stages=['qualify','cost','account'],retry=False))
    # The only already-running stage is SIGNAL. Poll its immutable receipt;
    # no repeated launch or scientific function is called while waiting.
    while not (private/'SIGNAL/completion.json').exists():
        if (private/'SIGNAL/STOP.json').exists():raise RuntimeError('SIGNAL_STOP_NO_SUCCESSORS')
        time.sleep(2)
    assert read(private/'SIGNAL/completion.json')['status']=='PASS'
    for stage,folder in [('qualify','QUALIFY'),('cost','COST'),('account','ACCOUNT')]:
        assert not (private/folder).exists(),'Never repeat a stage command'
        with (private/(stage+'_stdout.log')).open('x') as out,(private/(stage+'_stderr.log')).open('x') as err:
            result=subprocess.run([sys.executable,'-B',str(root/'scripts/run_pf_common_run.py'),stage,'--private',str(private)],cwd=root,stdout=out,stderr=err)
        save(private/(stage+'_external_exit.json'),dict(exit_code=result.returncode,child_reaped=True,launched_once=True,retry=False))
        if result.returncode!=0:raise RuntimeError(stage.upper()+'_STOP_NO_SUCCESSORS')
        receipt=read(private/folder/'completion.json');assert receipt['status']=='PASS'
        if stage=='qualify':assert receipt['unresolved_endpoints']==0,'UNRESOLVED_POLICY_STOP'
    save(private/'SUCCESSOR_DRIVER_COMPLETE.json',dict(status='PASS',next_scientific_stage_started=False,retry=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private',required=True);args=parser.parse_args()
    try:run(args.private)
    except BaseException as exc:
        path=Path(args.private)/'SUCCESSOR_DRIVER_STOP.json'
        if not path.exists():save(path,dict(status='MANDATORY_STOP',reason=str(exc),retry=False))
        raise
