"""Artificial process/partial-output fixture; imports no scientific modules."""
import os
import signal
import sys
from pathlib import Path
from common import journal,save

if __name__=='__main__':
    role,task,folder=sys.argv[1:];folder=Path(folder)
    journal(folder/(task+'.partial.jsonl'),event='TRIAL_COMMITTED',trial=task+':0')
    if role=='exit7':os._exit(7)
    if role=='signal':os.kill(os.getpid(),signal.SIGTERM)
    if role=='exception':raise RuntimeError('ARTIFICIAL_EXCEPTION')
    save(folder/(task+'.complete.json'),dict(task=task,trials=[task+':0',task+':1'],complete=True))
