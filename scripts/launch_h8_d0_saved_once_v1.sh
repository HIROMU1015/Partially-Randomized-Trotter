#!/usr/bin/env bash
# One numerical Python process. The second invocation only records the shell's
# actual wait status after the first has exited; no retry or numerical work.
set -u
if [[ $# -ne 6 ]]; then
  printf '%s\n' 'usage: launch_h8_d0_saved_once_v1.sh ROOT PYTHON CONTRACT AUTH INPUT_MAPPING OUTPUT' >&2
  exit 2
fi
repo_root=$1
analysis_python=$2
contract_path=$3
authorization_path=$4
input_mapping=$5
output_root=$6
"$analysis_python" -P -B "$repo_root/scripts/h8_d0_saved_execution_v1.py" \
  --contract "$contract_path" --authorization "$authorization_path" \
  --inputs "$input_mapping" --output "$output_root"
driver_wait_status=$?
"$analysis_python" -P -B "$repo_root/scripts/h8_d0_saved_execution_v1.py" \
  --contract "$contract_path" --authorization "$authorization_path" \
  --inputs "$input_mapping" --output "$output_root" --record-wait-status "$driver_wait_status"
receipt_wait_status=$?
if [[ $receipt_wait_status -ne 0 ]]; then
  exit 2
fi
exit "$driver_wait_status"
