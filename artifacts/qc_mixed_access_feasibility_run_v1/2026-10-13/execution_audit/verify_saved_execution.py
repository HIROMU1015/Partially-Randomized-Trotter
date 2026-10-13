"""Stdlib-only audit of bindings, caps, saved eligibility and cost arithmetic.

Does not regenerate molecules, PF matrices, functions, eig/SVD, or resource
templates. Norm/CAR verification is the separate frozen checker replay.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
FOLDER=Path(__file__).resolve().parent
OUT=FOLDER.parent/'attempt_1'
BASE='18a47a4b0052e1c443e97f0317aa7e574ddcf562'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def from_git(commit,path):
    return subprocess.run(['git','show',commit+':'+path],cwd=ROOT,capture_output=True,check=True).stdout


def main():
    contract_path='artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json'
    contract=read(ROOT/contract_path); approval=read(FOLDER/'user_authorization.json')
    manifest_path=ROOT/'artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json'
    manifest=read(manifest_path)
    assert approval['type']=='USER_EXPLICIT_SCIENCE_APPROVAL'
    assert sha(manifest_path)==approval['source_manifest_SHA256']
    assert sha(ROOT/contract_path)==approval['contract_SHA256']
    for name,digest in manifest['files'].items():
        assert sha(ROOT/name)==digest
        assert hashlib.sha256(from_git(approval['source_payload_commit'],name)).hexdigest()==digest
    assert read(OUT/'launch_receipt.json')['source_payload_commit']==approval['source_payload_commit']
    assert read(OUT/'launch_receipt.json')['user_approval_reference']==approval['user_approval_reference']
    assert sha(OUT/'input.npz')==read(OUT/'input_seal.json')['input_SHA256']
    seal=read(OUT/'raw_seal.json')
    assert seal['status']=='SEALED_BEFORE_SELECTION'
    for name,digest in seal['files'].items(): assert sha(OUT/name)==digest
    result=read(OUT/'result.json'); metrics=read(OUT/'execution_metrics.json')
    assert not (OUT/'failure.json').exists() and result['mandatory_stop']
    assert read(OUT/'independent_audit.json')['status']=='PASS'
    assert read(FOLDER/'independent_checker_replay.json')['status']=='PASS'
    limits=contract['limits']
    for key,count in result['counts'].items(): assert count<=limits[key],key
    for key,reservation in metrics['reservation'].items():
        assert reservation==sum(w['counts'].get(key,0) for w in metrics['worker_results']),key
    assert metrics['aggregate_RSS_peak_sampled']<=limits['aggregate_RSS_bytes']
    assert (FOLDER/'science_run.log').stat().st_size<=limits['log_bytes']
    bytes_total=sum(p.stat().st_size for p in FOLDER.parent.rglob('*') if p.is_file())
    assert bytes_total<limits['output_bytes']
    probes=read(OUT/'probes.json'); grid=contract['grid']; methods=contract['methods']
    expected={(m,t,r) for m in methods for t in grid['times_atomic'] for r in grid['steps']}
    assert len(probes)==154 and {(p['method'],p['T'],p['r']) for p in probes}==expected
    observed={(p['method'],p['T'],p['r']):p['error'] for p in probes}
    resources=read(OUT/'resource_evaluation.json'); rows=resources['rows']; by_id={r['id']:r for r in rows}
    assert len(rows)==1386 and len(by_id)==1386
    for row in rows:
        assert row['status']=='PARTIAL_ENVELOPE'
        key=(row['method'],row['T'],row['r'])
        eps=row['PF_fraction']*grid['operator_error_target']
        eligible=(2*row['r'] in grid['steps']) and all(observed[(row['method'],row['T'],r)]+grid['numerical_margin']<=eps for r in grid['steps'] if r>=row['r'])
        assert row['eligible']==eligible
        price=math.ceil(row['kappa']*math.log2(1/row['eta_RZ']))
        assert price==row['T_RZ_hypothesis']
        assert row['known_T_hypothesis']==row['Q']+row['r']*(7*row['per_step']['Toffoli']+row['per_step']['RZ']*price)
        assert row['word_bits']<=128 and row['angle_bits']<=128
        assert all(0<=v<=128 for p in resources['precision_catalog'][row['precision_id']]['by_j'] for v in p)
        if row['method'][0]=='R':
            assert row['F_remaining']=='UNKNOWN' and row['depth']=='UNKNOWN' and row['total_workspace']=='UNKNOWN'
    baseline_candidates={}
    for t in grid['times_atomic']:
        for k in contract['resource_model']['rotation_sensitivity_kappa']:
            candidates=[x for x in rows if x['T']==t and x['kappa']==k and x['eligible'] and x['method'][0]!='R']
            baseline_candidates[(t,k)]=min(candidates,key=lambda x:(x['known_T_hypothesis'],x['r'],methods.index(x['method']),x['id']))
    assert len(resources['frontiers'])==sum(row['eligible'] for row in rows if row['method'][0]=='R')
    for frontier in resources['frontiers']:
        row=by_id[frontier['R_id']]; baseline=baseline_candidates[(row['T'],row['kappa'])]
        assert frontier['BASE_id']==baseline['id']
        slack=Fraction(baseline['known_T_hypothesis']-row['known_T_hypothesis'],row['r'])
        assert float(slack)==frontier['remaining_T_per_step']
        assert float(slack/row['relative_blocks_per_step'])==frontier['uniform_F_ceiling']
        assert slack<0 and frontier['status']=='NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL'
    assert not resources['numerical_ambiguities']
    assert result['status']==resources['status']=='NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL'
    old= json.loads(from_git(BASE,'artifacts/validation_manifest.json'))
    current=read(ROOT/'artifacts/validation_manifest.json')
    assert current['result_sets'][:len(old['result_sets'])]==old['result_sets']
    assert {k:v for k,v in current.items() if k!='result_sets'}=={k:v for k,v in old.items() if k!='result_sets'}
    assert 'description' not in current['relevant_commits'][7]
    report={'status':'PASS','locked_files':len(manifest['files']),'old_result_sets_preserved':len(old['result_sets']),
            'raw_file_hashes_verified':len(seal['files']),'PF_queries_verified':len(probes),'resource_rows_verified':len(rows),
            'frontiers_exact_Fraction_verified':len(resources['frontiers']),'numerical_ambiguities':0,
            'new_molecular_jobs':0,'new_PF_queries':0,'new_eig_SVD':0,'output_bytes_before_audit_report':bytes_total,
            'global_description_failure_preserved':True,'mandatory_stop':True,
            'limitation':'arithmetic/binding audit;does not prove circuit templates optimal or kappa hypothesis achieved'}
    (FOLDER/'saved_execution_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
