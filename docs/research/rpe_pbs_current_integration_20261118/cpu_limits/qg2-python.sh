#!/bin/sh
# All RPE mailbox operations and detached child processes use one CPU set.
exec /usr/bin/taskset --cpu-list 0-29 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python "$@"
