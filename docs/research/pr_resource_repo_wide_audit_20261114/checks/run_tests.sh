#!/bin/bash
# usage: run_tests.sh <label> ; runs tests/check_*.py at current HEAD in parallel, records rc and wall time
S=/tmp/claude-30038/-home-AbeHiromu-projects/6d04e01a-9916-4daa-81ba-7452201ae135/scratchpad
PY=/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python
L=$1; O=$S/testlogs/$L; mkdir -p $O
cd $S/clone
ls tests/check_*.py | grep -vE "track_d|h8_|h6_df19_rpe_(m|d)" > $O/list.txt
one() { f=$1; n=$(basename $f .py); s=$(date +%s.%N); OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1200 $PY $f > $O/$n.log 2>&1; rc=$?; e=$(date +%s.%N); printf "%s %s %.1f %s\n" $n $rc $(echo "$e - $s" | bc) "$(grep -E '^Ran [0-9]+ tests' $O/$n.log | tail -1)" ; }
export -f one; export S PY O
cat $O/list.txt | xargs -P 24 -I{} bash -c 'one {}' > $O/summary.txt
sort -k3 -n -r $O/summary.txt
