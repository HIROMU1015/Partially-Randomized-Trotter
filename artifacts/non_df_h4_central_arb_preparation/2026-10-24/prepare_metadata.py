"""Metadata only: hash checks, exact saved quotes, no numerical backend."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
BASE_COMMIT = '36e993e698953f24415f4e90c4cac199a693ee16'


def read(path): return json.loads((ROOT/path).read_text())
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(name, value):
    data = json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    path = BASE/name
    if path.exists(): assert path.read_text() == data, 'EXISTING_PREPARATION_RECORD_DIFFERS:'+name
    else: path.write_text(data)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == BASE_COMMIT
    prior = 'artifacts/non_df_h4_central_r1_sine/2026-10-24/'
    old = read(prior+'preservation_before.json')
    paths = subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE_COMMIT,'--',
        prior,'docs/non_df_h4_central_r1_sine_20261024.md','docs/research/研究ノート/2026-10-24_H4_R1_sine.md'],cwd=ROOT).decode().split('\0')
    for path in filter(None,paths): old[path] = sha(path)
    assert all(sha(p) == s for p,s in old.items())
    save('preservation_before.json',old)
    manifest = read('artifacts/validation_manifest.json')
    save('prior_manifest_rows.json',{k:digest(v) for k,v in manifest.items()})
    inputs = read(prior+'inputs.json')['files']
    for path in ('artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/PF_C_target_saved.json',
                 'artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json',
                 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json',
                 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json'):
        inputs[path] = sha(path)
    assert all(sha(p) == s for p,s in inputs.items())
    save('inputs.json',dict(kind='PREPARATION_ONLY_SAVED_EVIDENCE_INPUTS',files=inputs,real_H4_state_actions=0))
    target = next(t for t in read('artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json')['targets'] if t['key']=='PF0_cosine')
    save('PF_cosine_saved_target.json',target)
    save('PF_C_target_interval_QUOTED.json',read('artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/PF_C_target_saved.json'))
    save('state_identity_QUOTED.json',read('artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json'))
    print(json.dumps(dict(preserved=len(old),inputs=len(inputs),prior_rows=len(manifest),real_H4_state_actions=0)))


if __name__ == '__main__': main()
