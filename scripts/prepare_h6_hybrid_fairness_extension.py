#!/usr/bin/env python3
"""Phase A saved-byte/metadata checks only. No science imports or run command."""
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/h6_hybrid_fairness_readiness_20261027'
BASE='f7757c4320ba34dd36a64de4250c29a69bf38390'
OLD='8fd1ceb283309a7e515ff4a533a253260bdd0f78'
PREP='25d248a1f2ed88de62b0e621c2d1cea8532804c4'
EXT='10cb3e5cca7834ebf75301bd91bd093ca51d4a6b'
P='artifacts/pauli_h2_h6_comparison/2026-10-25/'

def raw(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def read(ref,path):return json.loads(raw(ref,path),parse_float=str,parse_int=str)
def rows(ref,path):return list(csv.DictReader(io.StringIO(raw(ref,path).decode())))
def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def csvsave(path,data):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fixed=[]
    for prefix,q,r,K in [(78,8,16,2),(78,8,32,2),(78,8,64,2),(684,1,16,4)]:
        fixed.append(dict(candidate_id=f'PR_p{prefix:04d}_q{q}_r{r}_K{K}',prefix=prefix,q=q,r=r,K=K,s=q*r,
                          execute_enabled='false',signal_status='NOT_EXECUTED',price_status='NOT_ACQUIRED',
                          exploration_n='NOT_FIXED',confirmation_n='NOT_FIXED',new_signal_N_G='null',
                          source_base=BASE,phase='A_PREPARATION_ONLY'))
    csvsave(OUT/'FIXED_CANDIDATES.csv',fixed)
    source=read(EXT,'artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json')
    bindings=[]
    for fn in ['model.json','reference.json']:
        path=P+'H6/inputs/'+fn;b=raw(OLD,path)
        assert hashlib.sha256(b).hexdigest()==source['input_sha256'][path]
        # The later copied input lives in its reversible archive, not a loose
        # Git blob. Compare its saved SOURCE binding; do not restore huge raw.
        copy_path='artifacts/pauli_rte_event_extension/2026-10-26/inputs/H6_'+fn
        assert hashlib.sha256(b).hexdigest()==source['input_sha256'][copy_path]
        bindings.append(dict(ref=OLD,path=path,sha256=hashlib.sha256(b).hexdigest(),
                             url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+OLD+'/'+path))
    model=read(OLD,P+'H6/inputs/model.json');ref=read(OLD,P+'H6/inputs/reference.json')
    results=json.loads(gzip.decompress(raw(OLD,P+'H6/candidate_results.json.gz')),parse_float=str,parse_int=str)
    norms={r['target']['raw_norm_exact'] for r in results if r['config']['method']=='PR'}
    assert norms=={ref['raw_norm_exact']} and model['n_qubits']=='12' and model['nonidentity_terms']=='918'
    assert model['all_Z_prefix']=='78'
    assert all(set(s)<=set('IZ') for s,c in model['hamiltonian']['terms'][:78])
    # Do not recompute the norm, law, B, N, mean, variance, SE or any ranking.
    inventory=rows(PREP,'artifacts/pauli_hybrid_rte_preparation/2026-10-25/candidate_evidence_inventory.csv')
    baseline=[]
    for r in inventory:
        if r['molecule']=='H6' and r['candidate_id'] in ('PR_p0078_q8_r8_K2','PR_p0684_q1_r8_K4') and r['cohort']=='confirmation':
            fields=('candidate_id','epsilon_sig','cohort','price_n','B_exact','N_cos','N_sin','C_cos_exact','C_sin_exact','C_cos_SE','C_sin_SE','G_exact','G_SE','eligibility')
            baseline.append({k:r[k] for k in fields}|dict(source_commit=PREP,source_file='artifacts/pauli_hybrid_rte_preparation/2026-10-25/candidate_evidence_inventory.csv',variance_covariance='NOT_SEPARATELY_SAVED_HERE; RAW_PAIRED_VALUES_EXIST; NO_RECOMPUTATION'))
    for r in rows(EXT,'artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv'):
        if r['molecule']=='H6':
            x={k:'' for k in baseline[0]};x.update(candidate_id=r['RTE_id'],epsilon_sig=r['epsilon_sig'],cohort='confirmation',price_n=r['RTE_n'],G_exact=r['RTE_G_exact'],G_SE=r['RTE_G_SE'],source_commit=EXT,source_file='artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv',variance_covariance='SEE_SAVED_PRICE_STATISTICS; NO_NEW_VARIANCE_OR_COVARIANCE')
            baseline.append(x)
    assert len(baseline)==15
    csvsave(OUT/'SAVED_COMPARATOR_STATISTICS.csv',baseline)
    code=['src/trotterlib/'+p for p in ['h6_hybrid_fairness_extension.py','h4_pauli_pr_search.py','pauli_signal_adapter.py','rte.py','rte_event_audit.py','h4_d2_pauli_control.py','fixed_signal_evaluation.py','rte_compiled_cost.py','pauli_vector_enclosure.py','pauli_vector_tight_enclosure.py','h4_conditional_cng.py','h6_rte_fairness.py','h4_pauli_b0_comparison.py','h4_ball_evidence.py','h4_pauli_pr_mean.py']]
    code+=['scripts/pauli_h2_h6_study.py','scripts/run_pauli_rte_event_extension.py','scripts/prepare_h6_hybrid_fairness_extension.py','tests/test_h6_hybrid_fairness_extension.py','tests/test_h6_hybrid_fairness_extension_mean.py']
    save(OUT/'SOURCE_INPUTS.json',dict(status='PHASE_A_SOURCE_PREPARED_NOT_A_SCIENCE_RUN',base_commit=BASE,
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},inputs=bindings,
        saved_Hamiltonian_hash=model['Hamiltonian_hash'],saved_raw_norm_exact=ref['raw_norm_exact'],
        preserved_prefix_order=True,prefix78_all_Z=True,raw_state_or_reference_regenerated=False,
        runtime_versions={p:importlib.metadata.version(p) for p in ['qiskit','numpy','scipy','python-flint']},
        current_scientific_execution_enabled=False,future_science_output='artifacts/h6_hybrid_fairness_extension/2026-10-27',
        future_science_cache='artifacts/h6_hybrid_fairness_extension/2026-10-27/cache',
        future_batch_preconditions='Separate Phase B permission; fix n_E, n_C rule/caps and total call limits before sampling; fresh SOURCE/input/runtime/output; no old STOP resume'))
    print(json.dumps(dict(status='PASS_STATIC_IDENTITY_AND_LITERAL_QUOTATION',fixed_candidates=4,
                         saved_comparator_rows=15,input_blobs_match=True,saved_norm_strings_match=True,
                         new_scientific_evaluation=0,phase_B_started=False)))

if __name__=='__main__':main()
