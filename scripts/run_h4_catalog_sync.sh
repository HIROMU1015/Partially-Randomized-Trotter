#!/bin/bash
# Own-user cron entry. Only the dedicated C repository may be updated.
set -euo pipefail

task_python=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
if [ "$#" -eq 2 ] && [ "$1" = --python ]; then
    task_python=$2
elif [ "$#" -ne 0 ]; then
    printf '%s\n' 'usage: run_h4_catalog_sync.sh [--python /absolute/interpreter]' >&2
    exit 2
fi
case "$task_python" in
    /*) ;;
    *) printf '%s\n' 'An absolute interpreter path is required' >&2; exit 2 ;;
esac

task_repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$task_repo"
task_local=artifacts/local_h4_catalog_sync
mkdir -p -- "$task_local"
exec >>"$task_local/cron.log" 2>&1
exec 9>"$task_local/scheduler.lock"
if ! flock -n 9; then
    printf '%s\n' 'SKIPPED: prior H4 publication sync still holds the scheduler lock'
    exit 0
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$task_repo/src"
printf '%s START H4 saved-publication sync\n' "$(date -u +%FT%TZ)"
task_status=0
"$task_python" -B scripts/sync_saved_rpe_publications.py \
    --config docs/research/h4_saved_rpe_cost_catalog_20261116/SYNC_CONFIG.json \
    --publish || task_status=$?
printf '%s END H4 saved-publication sync status=%s\n' "$(date -u +%FT%TZ)" "$task_status"
exit "$task_status"
