"""Diagnosed recovery metadata/SHA only. No numerical evaluation or retry."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    assert read(PRIOR/'run01/STOP.json')['status']=='FAILED_NO_AUTOMATIC_RETRY'
    used=read(PRIOR/'run01/call_ledger.json')['used']
    assert used['pauli_10_conditions_20_axes_with_guard_faces']==used['paired_G_statistics_batch']==used['comparisons_15_with_guard_boxes']==0
    for name in ('approval.txt','github_verification.json'):
        shutil.copyfile(PRIOR/name,BASE/name)
    shutil.copyfile('/tmp/h4_central_artificial_recovery02.log',BASE/'artificial_adapter16.log')
    preservation=read(PRIOR/'preservation_before.json')
    files=dict(read(PRIOR/'inputs.json')['files'])
    previous_files=[p for p in PRIOR.rglob('*') if p.is_file() and BASE not in p.parents and
                    'mplconfig' not in p.parts and '__pycache__' not in p.parts and p.name!='README.md']
    for p in previous_files:
        rel=str(p.relative_to(ROOT));preservation[rel]=sha(p);files[rel]=sha(p)
    src=read(PRIOR/'seal.json')['SOURCE']; archives={}
    for p in ('src/trotterlib/h4_central_cng.py','scripts/run_h4_central_cng.py','tests/test_h4_central_cng.py'):
        data=subprocess.check_output(['git','show',src+':'+p],cwd=ROOT)
        dst=BASE/'prior_SOURCE'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
        assert hashlib.sha256(data).hexdigest()==read(PRIOR/'seal.json')['code_files'][p]
        archives[p]=dict(SOURCE=src,archive=str(dst.relative_to(ROOT)),sha256=sha(dst))
        files[str(dst.relative_to(ROOT))]=sha(dst)
    for p,h in preservation.items():assert sha(ROOT/p)==h,p
    save(BASE/'preservation_before.json',preservation)
    save(BASE/'inputs.json',dict(kind='H4_CENTRAL_RECOVERY_SAVED_INPUTS_V1',files=files,
        original_input_manifest_sha256=sha(PRIOR/'inputs.json'),science_input_values_changed=False))
    diagnosis=dict(classification='TECHNICAL_ADAPTER_GATE_SCOPE_ERROR',
        failed_SOURCE=src,failed_namespace=str((PRIOR/'run01').relative_to(ROOT)),
        cause='SHARED_REFERENCE required exact cross-representation reference identity; original adapters intentionally use their own represented full Hamiltonian',
        evidence=['src/trotterlib/h4_audited_evaluation.py:44', 'src/trotterlib/h4_pilot_science.py:52',
                  'artifacts/non_df_h4_pilot/2026-10-23/run01/run/*/result.json:reference',
                  str((PRIOR/'run01/native/stderr.log').relative_to(ROOT))],
        fixed_gate='same saved ray/T/method across representations; PF/PR exact reference equality within each representation; retain distinct cross-representation reference/bias',
        scientific_input_or_condition_change=False,reference_values_changed=False,scientific_tolerance_added=False,
        original_grant_reused=False,failed_run_and_STOP_preserved=True,prior_source_archives=archives,
        prior_used=used,completed_scientific_conditions=0,successful_scientific_results_to_reuse=[],
        remaining_scientific_scope='same 10 Pauli conditions/20 axes+guard faces/15 comparisons/one paired G batch',
        new_SOURCE_seal_contract_namespace_grant_required=True,authorization='attachment section 10')
    save(BASE/'recovery_history.json',diagnosis)
    print(json.dumps(dict(status='DIAGNOSED_RECOVERY_METADATA_READY',inputs=len(files),preserved=len(preservation),prior_new_N_G_calls=0)))
