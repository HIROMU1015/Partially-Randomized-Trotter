#!/usr/bin/env bash
# Foreground one-shot and same-parent native wait receipt. No retry.
set -u
if [[ $# -ne 6 ]]; then
  printf '%s\n' 'usage: ROOT PYTHON CONTRACT AUTH INPUT_MAPPING OUTPUT' >&2
  exit 2
fi
exec "$2" -P -B "$1/scripts/h6_b0i_family_launcher_v1.py" "$3" "$4" "$5" "$6"
