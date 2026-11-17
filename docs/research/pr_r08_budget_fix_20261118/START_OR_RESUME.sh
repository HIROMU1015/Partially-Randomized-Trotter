#!/usr/bin/env bash
set -euo pipefail
PR_BUDGET_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-budget-continuity-20261118
PR_BUDGET_STUDY=docs/research/pr_r08_budget_fix_20261118
PR_BUDGET_PYTHON=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
cd "$PR_BUDGET_ROOT"
# Retain the previous R08 owners' existing locks as well as this epoch's lock.
exec 8>>/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117/docs/research/h4_pr_r08_recovery_20261117/HANDOVER.lock
flock -n 8 || { echo 'Previous R08 owner is active'; exit 75; }
exec 9>>"$PR_BUDGET_STUDY/run02/CONTROLLER.lock"
flock -n 9 || { echo 'R08 budget-recovery controller is active'; exit 75; }
exec 7>>/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117/docs/research/h4_pr_candidate_continuity_20261117/run01/CONTROLLER.lock
flock -n 7 || { echo 'Original R08 owner is active'; exit 75; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PR_BUDGET_ROOT/src"
exec "$PR_BUDGET_PYTHON" -B - <<'PY'
import datetime,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
sys.set_int_max_str_digits(0)
study=Path('docs/research/pr_r08_budget_fix_20261118');run=study/'run02'
source=json.loads((run/'SOURCE.json').read_text());state=json.loads((run/'KAPPA_STATE.json').read_text())
subprocess.run(['git','diff','--exit-code',source['scientific_code_source'],'--','src','scripts'],check=True)
for filename,expected in source['fixed_input_sha256'].items():
    assert hashlib.sha256((study/filename).read_bytes()).hexdigest()==expected,'Fixed input changed: '+filename
if state.get('stop')and state['stop']['reason']not in ('MAX_CYCLES','WAITING_FOR_INPUTS'):
    print('Preserved STOP: '+json.dumps(state['stop']),flush=True);sys.exit(0)
remaining=source['original_cycle_ceiling']-state['cycle']
if remaining<=0:print('Original cycle ceiling reached',flush=True);sys.exit(0)
def write(path,row):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)
def event(kind,**fields):
    row=dict(event=kind,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),**fields)
    with(run/'CONTROLLER_INVOCATIONS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
    return row
command=[sys.executable,'-B','scripts/run_rpe_schedule_search.py','--problem',str(study/'PROBLEM.json'),
         '--round',str(study/'ROUND.json'),'--run-dir',str(run),'--batch','64','--max-cycles',str(remaining)]
input_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
with(run/'CONTROLLER.log').open('a')as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    row=event('CONTROLLER_START',pid=process.pid,command=command,source=source['scientific_code_source'],
              input_commit=input_commit,executable=sys.executable,python=sys.version,remaining_cycles=remaining)
    write(run/'CONTROLLER_PID.json',row)
    code=process.wait()
row=event('CONTROLLER_RETURN',returncode=code,automatic_retry=False,stop=json.loads((run/'KAPPA_STATE.json').read_text()).get('stop'))
write(run/'CONTROLLER_STATUS.json',row)
sys.exit(code if code>=0 else 128-code)
PY
