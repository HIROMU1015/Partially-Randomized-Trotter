"""Saved-value/hash preparation only; no scientific numerical backend."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
PREP=Path('artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25')
BASE_COMMIT='06ca7c838808cbde9e1c1b5030c46f260d929c32'


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,value):
    p=BASE/name;data=json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==data,'PREEXISTING_METADATA:'+name
    else:p.write_text(data)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE_COMMIT
    old=read(PREP/'preservation_before.json')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE_COMMIT,'--',str(PREP),
        'docs/non_df_h4_central_arb_pr1_sine_20261025.md','docs/research/研究ノート/2026-10-25_H4_PR1_Arb_sine.md'],cwd=ROOT).decode().split('\0')
    for p in filter(None,paths):old[p]=sha(p)
    for p in read(PREP/'seal.json')['code_files']:old[p]=sha(p)
    assert all(sha(p)==s for p,s in old.items());save('preservation_before.json',old)
    save('prior_manifest_rows.json',{k:digest(v) for k,v in read('artifacts/validation_manifest.json').items()})
    sine='artifacts/non_df_h4_central_r1_sine/2026-10-24/inputs.json'
    m=read(sine);files=[sine,'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json',
        'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json',
        'artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json',
        'artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json']
    for t,ct in zip(m['targets'][:2],m['cosine_targets'][:2]):
        folder=Path(t['input_qpy']).parent;cf=Path(ct['input_qpy']).parent
        files += [t['price_join'],t['source_body'],t['controlled_body'],ct['input_qpy'],t['input_qpy'],t['compiled_qpy'],
            str(folder/'input.json'),str(folder/'metrics.json'),str(folder/'source_binding.json'),str(cf/'input.json'),str(cf/'metrics.json')]
    save('inputs.json',dict(kind='H4_PF_PR0_SINE_ARB_FIXED_INPUTS_V1',files={p:sha(p) for p in files},
        scope='only actual computation inputs; historical preservation inventory checked separately, not a worker cancellation gate'))
    policy=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-23.md')
    text=policy.read_text()
    save('policy_observation.json',dict(path=str(policy),sha256=sha(policy),applied=['8.4','10','10.1','10.2'],
        section_10_3_present='### 10.3' in text,note='No10.3 in supplied latest file; apply actual sections and explicit user scope',
        arbitrary_resource_ceiling=False,subagents=0,shared_environment_changes=False))
    print(json.dumps(dict(preserved=len(old),inputs=len(files),science_state_actions=0)))


if __name__=='__main__':main()
