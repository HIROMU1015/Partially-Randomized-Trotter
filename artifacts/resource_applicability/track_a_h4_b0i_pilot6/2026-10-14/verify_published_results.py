"""Public-only result audit: binding, Git source blobs and independent Fraction costs."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import subprocess
import sys


def load(path):
    return json.loads(path.read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def check_files(repo,rows):
    for row in rows:
        data=(repo/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']


def shot_counts(saved,e):
    hs=[e/math.sqrt(2)-saved['bias'][a] for a in ('cosine','sine')]
    return None if min(hs)<=0 else [math.ceil(2*saved['normalization']**2/h**2*math.log(2/.025)) for h in hs]


def fraction_work_se(pairs,counts):
    sample=[counts[0]*a+counts[1]*b for a,b in pairs];n=len(sample)
    work=Fraction(sum(sample),n)
    variance=Fraction(sum(v*v for v in sample),n*(n-1))-Fraction(sum(sample)**2,n*n*(n-1)) if n>1 else Fraction(0)
    return work,math.sqrt(float(variance))


def verify(bundle):
    repo=bundle.parents[3];sys.path.insert(0,str(repo/'src'))
    from trottertracks.resource_applicability.h4_b0i.contract import authorize,digest
    from trottertracks.resource_applicability.h4_b0i.adapter import bound_fingerprint
    check_files(repo,load(bundle/'launch_artifact_manifest.json')['files'])
    if (bundle/'result_artifact_manifest.json').exists():check_files(repo,load(bundle/'result_artifact_manifest.json')['files'])
    p=load(bundle/'plan.json');a=load(bundle/'authorization.json');r=load(bundle/'review.json');binding=load(bundle/'binding.json')
    authorize(p,a,r)
    assert binding['plan_digest']==digest(p) and binding['authorization_digest']==digest(a) and binding['review_digest']==digest(r)
    assert hashlib.sha256((bundle/'user_approval_evidence.json').read_bytes()).hexdigest()==a['user_approval_evidence_sha256']
    assert hashlib.sha256((bundle/'scientific_review.md').read_bytes()).hexdigest()==r['scientific_review_sha256']
    check_files(repo,p['source_closure']+p['bound_public_files'])
    for row in p['source_closure']:
        raw=subprocess.check_output(['git','show',p['source_commit']+':'+row['path']],cwd=repo)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    check_files(repo,load(bundle/'result_provenance.json')['public_analysis_inputs'])
    native=load(bundle/'native_resource_receipt.json');terminal=native['terminal'];metrics=load(bundle/'compiled_wrapper_metrics.json')['results'];summary=load(bundle/'result_summary.json')
    assert terminal['status']=='PILOT6_COMPLETE_STOP' and terminal['records']==terminal['actual_consumed_or_reserved']==6
    assert terminal['cleanup_complete'] and terminal['first_stop_reason'] is None and native['driver_native_exit_code']==terminal['native_worker_exit_code']==0
    assert terminal['native_observer_exit_code']==-15 and all(not v['same_run_identity_present'] for v in native['run_residual_audit'])
    assert native['first_failure'] is None and summary['next_stage_authorized'] is False and summary['mandatory_stop'] is True
    assert [v['task_id'] for v in metrics]==[v['task_id'] for v in p['tasks']]
    for t,m in zip(p['tasks'],metrics):
        assert m['source_commit']==p['source_commit'] and m['plan_digest']==digest(p) and m['old_cache_reuse'] is False
        assert m['validation_wrapper_fingerprint']==bound_fingerprint(t['template'],t['tail_identity'],t['axis'],t['input_identity'],p['source_commit'],m['numeric_circuit_fingerprint'])
        assert m['delta_RZ']==m['metrics']['rz_count']-t['old_B0_reference_RZ']==0
    root=repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13'
    original={(v['geometry'],v['template_id']):v for v in load(root/'saved_candidates_v1.json')['candidates']}
    with (bundle/'measured_fixed3_precision_906.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==906 and len({(v['geometry_A'],v['epsilon']) for v in rows})==906
    counts_by_geometry={};selected=[]
    for row in rows:
        g=row['geometry_A'];e=float(row['epsilon']);t=next(t for t in p['tasks'] if t['geometry']==g);ms=[v for v in metrics if v['geometry_A']==g]
        saved=dict(bias=t['saved_bias_B0I'],normalization=1.);ns=shot_counts(saved,e)
        assert (ns is not None)==(row['B0I_eligible']=='True')
        b2=original[g,row['B2_point_min_template']];n2=shot_counts(b2['saved'],e);work2,se2=fraction_work_se(b2['saved']['paired_costs'],n2)
        assert math.isclose(float(row['B2_point_RZ_work']),float(work2),rel_tol=1e-12)
        assert math.isclose(float(row['B2_descriptive_SE']),se2,rel_tol=1e-12)
        if ns:
            work=sum(n*v['metrics']['rz_count'] for n,v in zip(ns,ms));assert work==int(row['B0I_RZ_work'])
            assert ns==[int(row['N_cos']),int(row['N_sin'])]
            assert math.isclose(float(row['B0I_to_B2_point_ratio']),work/float(work2),rel_tol=1e-12)
            below=work<work2
        else:below=False;assert row['B0I_RZ_work']==''
        assert below==(row['B0I_below_B2_point']=='True')
        counts_by_geometry[g]=counts_by_geometry.get(g,0)+below
        if e==.05:selected.append((g,int(row['B0I_RZ_work'])))
    for g,n in counts_by_geometry.items():assert summary['fixed3_precision_projection'][g]['points_below_B2_point_min']==n
    assert selected==[(v['geometry_A'],v['B0I_RZ_work']) for v in summary['selected_epsilon_0p05']]
    budget=native['budget'];assert budget['historical_carry']==p['carry'] and budget['cumulative_actual_consumed_or_reserved']==73029 and budget['actual_cap']==74805 and budget['remaining_actual']==1776
    assert budget['charge_at_receipt_bytes']==p['carry']['charged_bytes']+budget['new_runtime_allocated_bytes']<budget['charge_cap_bytes']
    assert budget['failed_calls_refunded'] is False
    return dict(status='PASS_PUBLIC_BINDING_GIT_SOURCE_AND_FRACTION_COST_AUDIT',source_files=len(p['source_closure']),real_compile_receipts=6,fixed_configuration_precision_rows=906,points_below_saved_B2_point_min=counts_by_geometry,private_NPZ_or_runtime_access=0,new_compile_or_signal_calls=0,CI_or_external_reproduction=False)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30));signal.alarm(60)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=Path(__file__).absolute().parent);args=parser.parse_args();print(json.dumps(verify(args.bundle),ensure_ascii=False))


if __name__=='__main__':main()
