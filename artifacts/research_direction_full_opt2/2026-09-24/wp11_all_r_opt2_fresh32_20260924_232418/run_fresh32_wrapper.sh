#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-wp11
OUTPUT_REL=artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418
OUTPUT="$ROOT/$OUTPUT_REL"
MANIFEST="$ROOT/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_extension_20260924_222811.manifest.json"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

COMMAND=(
  "$ROOT/.venv-server-opt2/bin/python"
  "$ROOT/scripts/run_research_direction_full_opt2_compute.py"
  --task-manifest "$MANIFEST"
  --output-dir "$OUTPUT"
  --run
  --max-workers 8
  --memory-budget-gib 32
)

cd "$ROOT"
started_at=$(date --iso-8601=seconds)
started_at_utc=$(date -u --iso-8601=seconds)
started_epoch=$(date +%s)
command_text=$(printf '%q ' "${COMMAND[@]}")
"${COMMAND[@]}" >"$OUTPUT/compute_driver.log" 2>&1 &
runner_pid=$!

jq -n \
  --arg started_at "$started_at" \
  --arg started_at_utc "$started_at_utc" \
  --arg command "$command_text" \
  --argjson wrapper_pid "$$" \
  --argjson runner_pid "$runner_pid" \
  '{schema_version:"wp11_fresh32_wrapper_start_v1",started_at:$started_at,started_at_utc:$started_at_utc,wrapper_pid:$wrapper_pid,runner_pid:$runner_pid,command:$command}' \
  >"$OUTPUT/.wrapper_start.json.tmp"
mv "$OUTPUT/.wrapper_start.json.tmp" "$OUTPUT/wrapper_start.json"

set +e
wait "$runner_pid"
exit_code=$?
set -e
finished_at=$(date --iso-8601=seconds)
finished_at_utc=$(date -u --iso-8601=seconds)
finished_epoch=$(date +%s)
elapsed_seconds=$((finished_epoch - started_epoch))

jq -n \
  --arg started_at "$started_at" \
  --arg started_at_utc "$started_at_utc" \
  --arg finished_at "$finished_at" \
  --arg finished_at_utc "$finished_at_utc" \
  --arg command "$command_text" \
  --argjson wrapper_pid "$$" \
  --argjson runner_pid "$runner_pid" \
  --argjson exit_code "$exit_code" \
  --argjson elapsed_seconds "$elapsed_seconds" \
  '{schema_version:"wp11_fresh32_wrapper_status_v1",started_at:$started_at,started_at_utc:$started_at_utc,finished_at:$finished_at,finished_at_utc:$finished_at_utc,wrapper_pid:$wrapper_pid,runner_pid:$runner_pid,command:$command,exit_code:$exit_code,elapsed_seconds:$elapsed_seconds}' \
  >"$OUTPUT/.wrapper_status.json.tmp"
mv "$OUTPUT/.wrapper_status.json.tmp" "$OUTPUT/wrapper_status.json"
exit "$exit_code"
