#!/usr/bin/env bash
# Ordinary integrated entry, one owned controller, no worker retry loop.
set -euo pipefail
PR_RUN_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-resume-20261117
PR_STUDY=docs/research/h4_pr_multiwidth_resume_20261117
PR_RUN_DIRECTORY="$PR_RUN_ROOT/$PR_STUDY/run01"
PR_CONTROLLER_PYTHON=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
cd "$PR_RUN_ROOT"
test -f "$PR_STUDY/SOURCE.json"
test -f "$PR_STUDY/ROUND.json"
mkdir -p "$PR_RUN_DIRECTORY"
exec 9>>"$PR_RUN_DIRECTORY/CONTROLLER.lock"
flock -n 9 || { echo 'Existing controller/owned children still hold the run lock'; exit 75; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PR_RUN_ROOT/src"
PR_CONTROLLER_ARGS=(-B scripts/run_rpe_schedule_search.py --problem "$PR_STUDY/PROBLEM.json" --round "$PR_STUDY/ROUND.json" --run-dir "$PR_STUDY/run01" --batch 64 --max-cycles 199)
python3 - "$PR_RUN_DIRECTORY" "$$" "$PR_CONTROLLER_PYTHON" "${PR_CONTROLLER_ARGS[@]}" <<'PY'
import datetime,json,os,socket,sys,subprocess
from pathlib import Path
row=dict(event='CONTROLLER_START',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),pid=int(sys.argv[2]),command=sys.argv[3:],cwd=os.getcwd(),input_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scientific_code_source=json.loads((Path(sys.argv[1]).parent/'SOURCE.json').read_text())['scientific_code_source'])
with (Path(sys.argv[1])/'CONTROLLER_INVOCATIONS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
PY
set +e
"$PR_CONTROLLER_PYTHON" "${PR_CONTROLLER_ARGS[@]}"
PR_CONTROLLER_RETURN=$?
set -e
python3 - "$PR_RUN_DIRECTORY" "$$" "$PR_CONTROLLER_RETURN" <<'PY'
import datetime,json,sys
from pathlib import Path
row=dict(event='CONTROLLER_RETURN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=int(sys.argv[2]),shell_returncode=int(sys.argv[3]),automatic_retry=False)
with (Path(sys.argv[1])/'CONTROLLER_INVOCATIONS.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
PY
exit "$PR_CONTROLLER_RETURN"
