"""Read-only hashes/quotes; no Arb, QPY, state action or science authority."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
PF=Path('artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25')
BASE_COMMIT='78d7f24c80b500fc63c908efc768e683406571b3'


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,value):
    p=BASE/name;data=json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==data,'PREEXISTING_METADATA:'+name
    else:p.write_text(data)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE_COMMIT
    old=read(PF/'preservation_before.json')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE_COMMIT,'--',str(PF),
        'docs/non_df_h4_central_arb_pf_cosine_20261025.md','docs/research/研究ノート/2026-10-25_H4_PF_cosine_Arb.md'],cwd=ROOT).decode().split('\0')
    for p in filter(None,paths):old[p]=sha(p)
    for p,h in read(PF/'seal.json')['code_files'].items():
        assert sha(p)==h,'PF_SOURCE_MODIFIED:'+p
        old[p]=h
    assert all(sha(p)==h for p,h in old.items());save('preservation_before.json',old)
    save('prior_manifest_rows.json',{k:digest(v) for k,v in read('artifacts/validation_manifest.json').items()})
    target_path='artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json'
    t=next(t for t in read(target_path)['targets'] if t['key']=='PR0_cosine')
    folder=str(Path(t['input_qpy']).parent)
    files=[target_path,t['state_model'],t['price_join'],t['source_body'],t['controlled_body'],t['input_qpy'],t['compiled_qpy'],
        folder+'/input.json',folder+'/metrics.json',
        'artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json',
        'artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json']
    save('inputs.json',dict(kind='PR0_PREPARATION_SAVED_INPUTS_ONLY',files={p:sha(p) for p in files},real_H4_state_actions=0))
    save('PR0_target_QUOTED.json',t)
    g=next(g for g in read(files[-1])['axes'] if g['case']=='pauli_jw_PR' and g['axis']=='cosine' and g['epsilon_sig']=='0.0005')
    save('PR0_guard_QUOTED.json',g)
    policy=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-23.md')
    save('policy_observation.json',dict(path=str(policy),sha256=sha(policy),applied=['8.4','10','10.1','10.2'],
        section_10_3_present='### 10.3' in policy.read_text(),version_mismatch='requested10.3 absent in supplied latest2026-10-23 file',
        subagents_used=0,arbitrary_resource_ceilings=False,shared_environment_changes=False))
    print(json.dumps(dict(preserved=len(old),inputs=len(files),science_state_actions=0)))


if __name__=='__main__':main()
