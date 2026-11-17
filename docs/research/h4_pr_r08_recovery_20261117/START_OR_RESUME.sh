#!/usr/bin/env bash
set -euo pipefail
PR_RECOVERY_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117
PR_RECOVERY_STUDY=docs/research/h4_pr_r08_recovery_20261117
PR_RECOVERY_PYTHON=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
cd "$PR_RECOVERY_ROOT"
exec 9>>"$PR_RECOVERY_STUDY/HANDOVER.lock"
flock -n 9 || { echo 'R08 handover/controller already owns this run'; exit 75; }
python3 - <<'CHECK'
import hashlib
from pathlib import Path
p=Path('docs/research/h4_pr_r08_recovery_20261117/HANDOVER_PLAN.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='3a7f1003952205794520ef3cca424346dbc8d4ce246e488e6ee8776b774796cb', 'Fixed R08 handover plan changed'
CHECK
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PR_RECOVERY_ROOT/src"
exec "$PR_RECOVERY_PYTHON" -B scripts/resume_pr_r08_after_drain.py --plan "$PR_RECOVERY_STUDY/HANDOVER_PLAN.json" >>"$PR_RECOVERY_STUDY/HANDOVER.log" 2>&1
