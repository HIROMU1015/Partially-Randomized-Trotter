#!/usr/bin/env bash
set -euo pipefail

study=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$study" rev-parse --show-toplevel)
relative_study=${study#"$root/"}
run="$study/run01"
predecessor='/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-df-fixed-signal-worker-20261118/docs/research/pr_r08_df_worker_cutover_20261118'
source_commit='e25179a74b5b2dad281ec2c73b0d26cb151425b6'
python='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'
cd "$root"

exec 9>"$study/CONTROLLER.lock"
flock -n 9
exec 8<"$predecessor/CONTROLLER.lock"
flock -n 8
exec 7<"$predecessor/run01/FAST_DRIVER.lock"
flock -n 7

git diff --quiet "$source_commit" -- scripts src
for name in PROBLEM.json ROUND.json PREFLIGHT.json; do
    cmp -- "$study/$name" <(git show "HEAD:$relative_study/$name")
done
remaining=$("$python" -B -c 'import json,sys; print(200-json.load(open(sys.argv[1]))["cycle"])' "$run/KAPPA_STATE.json")
if (( remaining <= 0 )); then
    printf '%s R08 reached its existing 200-cycle limit\n' "$(date -Is)"
    exit 0
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src"
printf '%s CONTROLLER_START source=%s remaining_cycles=%s\n' "$(date -Is)" "$source_commit" "$remaining" >>"$run/CONTROLLER.log"
code=0
"$python" -B scripts/run_rpe_schedule_search.py --problem "$study/PROBLEM.json" --round "$study/ROUND.json" --run-dir "$run" --batch 64 --max-cycles "$remaining" >>"$run/CONTROLLER.log" 2>&1 || code=$?
printf '%s CONTROLLER_RETURN code=%s\n' "$(date -Is)" "$code" >>"$run/CONTROLLER.log"
exit "$code"
