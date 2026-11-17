#!/usr/bin/env bash
# Placement only, inside Ubuntu WSL. Starts no worker, relay, controller or Q3.
set -euo pipefail
RpeWslRepo=/root/projects/Partially-Randomized-Trotter-local0-wsl
RpeWslSource=1c4710384f11d4b262d120a62c6a00161d622105
RpeWslTransport="$RpeWslRepo/.local0-wsl/transports/$RpeWslSource"
RpeWslState="$RpeWslRepo/.local0-wsl/central-${RpeWslSource:0:9}"
if ! git -C "$RpeWslRepo" cat-file -e "$RpeWslSource^{commit}" 2>/dev/null; then
    git -C "$RpeWslRepo" fetch origin "$RpeWslSource"
fi
if [ ! -e "$RpeWslTransport" ]; then
    git -C "$RpeWslRepo" worktree add --detach --no-checkout "$RpeWslTransport" "$RpeWslSource"
    git -C "$RpeWslTransport" sparse-checkout set --cone scripts src
    git -C "$RpeWslTransport" checkout --detach "$RpeWslSource"
fi
test "$(git -C "$RpeWslTransport" rev-parse HEAD)" = "$RpeWslSource"
test -z "$(git -C "$RpeWslTransport" diff HEAD --name-only -- scripts src)"
python3 - "$RpeWslRepo" "$RpeWslSource" "$RpeWslState" <<'PY'
import json,pathlib,subprocess,sys
repo,source,directory=sys.argv[1:]
locator=source+':docs/research/rpe_q3_host_eligibility/optional_wsl/PROFILE.json'
profile=json.loads(subprocess.check_output(['git','-C',repo,'show',locator]))
assert profile['host_id']=='local0-wsl'
profile.update(slots=1,work_dir=directory+'/spool')
raw=(json.dumps(profile,indent=2)+'\n').encode()
path=pathlib.Path(directory)/'PROFILE.json'
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    if path.read_bytes()!=raw:raise ValueError('Existing next-run profile differs; preserved')
else:
    with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(profile=str(path),transport_source=source,started_jobs=0,participation_changed=False)))
PY
