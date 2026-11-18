#!/usr/bin/env bash
set -euo pipefail
PR_SWITCH_STUDY=$(cd "$(dirname "$0")" && pwd)
cd "$PR_SWITCH_STUDY/../../.."
exec 9>>"$PR_SWITCH_STUDY/HANDOVER.lock"
flock -n 9 || { echo 'This R08 continuation already has an owner'; exit 75; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python \
  -B scripts/resume_pr_r08_after_drain.py --plan "$PR_SWITCH_STUDY/PLAN.json" >>"$PR_SWITCH_STUDY/HANDOVER.log" 2>&1
