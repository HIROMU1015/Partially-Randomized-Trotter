"""Bounded public-only audit with Fraction paired-cost arithmetic; no science calls."""
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


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    check(math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=1e-6), f'arithmetic mismatch {a}/{b}')


def shots(bias, B, epsilon):
    result=[]
    for a in ('cosine','sine'):
        h=epsilon/math.sqrt(2)-bias[a]
        if h<=0:return None
        result.append(math.ceil(2*B**2/h**2*math.log(80)))
    return result


def exact_work(pairs, counts):
    return Fraction(sum(counts[0]*a+counts[1]*b for a,b in pairs),len(pairs))


def exact_se(pairs, counts):
    n=len(pairs)
    if n==1:return 0.
    samples=[counts[0]*a+counts[1]*b for a,b in pairs]
    total=sum(samples)
    variance=Fraction(sum(x*x for x in samples),n*(n-1))-Fraction(total*total,n*n*(n-1))
    return math.sqrt(float(variance))


def verify(repo, bundle, manifest=None):
    if manifest is not None:
        for path, expected in manifest['files'].items():
            data=(repo/path).read_bytes()
            check(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'], 'asset hash mismatch: '+path)
    provenance=json.loads((bundle/'provenance_v1.json').read_text())
    for item in provenance['public_analysis_inputs']:
        data=(repo/item['path']).read_bytes()
        check(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'], 'input hash mismatch')
    # Verify actual immutable Git blobs, not only the prose source labels.
    for item in provenance['static_source_inspection']+provenance['fixed_comparison_references']:
        data=subprocess.check_output(['git','show',item['commit']+':'+item['path']],cwd=repo)
        check(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'], 'reference Git blob mismatch')
    root=repo/'artifacts/resource_applicability'
    projection=json.loads((root/'track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json').read_text())
    original={(r['geometry'],r['template_id']):r for r in projection['candidates']}
    grid=read_csv(bundle/'precision_grid_1812.csv')
    scalar=read_csv(bundle/'scalar_classification_1812.csv')
    check(len(grid)==len(scalar)==1812 and len(original)==1308, 'coverage')
    check(len({(r['geometry'],r['epsilon']) for r in grid})==1812,'duplicate grid point')
    for row in grid:
        candidate=original[row['geometry'],row['winner']]
        ns=shots(candidate['saved']['bias'],candidate['saved']['normalization'],float(row['epsilon']))
        check(ns==[int(row['N_cos']),int(row['N_sin'])],'winner shot count')
        pairs=candidate['saved']['paired_costs']
        near(row['G'],exact_work(pairs,ns));near(row['SE'],exact_se(pairs,ns))
        near(row['G'],float(row['N_total'])*float(row['C_effective']))
        near(row['ratio_det'],float(row['ratio_N_det'])*float(row['ratio_C_effective_det']))
        check(row['winner'].startswith('B2-rank3-') and int(row['overlap_count'])>0,'winner/uncertainty')
    phases=json.loads((root/'track_a_h4_geometry_scalar_diagnostic/2026-10-13/saved_b0_phase_sensitivity_v1.json').read_text())
    aux={}
    for phase in phases:
        saved=original[phase['geometry'],phase['original_template_id']]['saved']
        aux[phase['geometry'],phase['original_template_id']+'+I']={'bias':phase['counterfactual_bias'],'pairs':saved['paired_costs']}
    counts={d:0 for d in (0,1,10,100,200,500,1000,2000,3000,5000,10000)}
    fixed=dict(counts);reversals=0
    for row in scalar:
        g=row['geometry'];e=float(row['epsilon']);G2=float(row['B2_G'])
        values=[]
        for (geometry,tid),v in aux.items():
            if geometry!=g:continue
            ns=shots(v['bias'],1.,e)
            if ns is not None:values.append((tid,exact_work(v['pairs'],ns),sum(ns)))
        tid,G0,n=min(values,key=lambda x:(x[1],x[0]))
        check(tid==row['B0_I_template'],'auxiliary minimum')
        near(row['B0_I_G0'],G0)
        near(row['threshold_for_zero_delta_winner'],(G2-float(G0))/n)
        near(row['max_threshold_any_B0_I'],max((G2-float(v))/s for t,v,s in values))
        reversals+=G0<G2
        for d in counts:
            counts[d]+=min(v+s*d for t,v,s in values)<G2
            fixed[d]+=G0+n*d<G2
    check(reversals==397,'397-point sensitivity changed')
    for row in read_csv(bundle/'delta_sensitivity.csv'):
        d=int(row['common_delta_RZ_per_axis_shot'])
        check(counts[d]==int(row['reoptimized_auxiliary_points_below_B2']),'reoptimized delta count')
        check(fixed[d]==int(row['fixed_zero_delta_candidate_points_below_B2']),'fixed delta count')
    grouped=read_csv(bundle/'scalar_reversal_groups.csv')
    check(sum(int(r['grid_points']) for r in grouped)==397,'reversal grouping')
    nearby=read_csv(bundle/'nearby_candidates.csv')
    check(len(nearby)==sum(int(r['overlap_count']) for r in grid),'nearby coverage')
    for row in nearby:
        v=original[row['geometry'],row['candidate']]['saved']
        ns=shots(v['bias'],v['normalization'],float(row['epsilon']))
        near(row['G'],exact_work(v['paired_costs'],ns));near(row['SE'],exact_se(v['paired_costs'],ns))
    rare=read_csv(bundle/'rare_order_law_diagnostics.csv')
    check(len(rare)==1164,'rare-law coverage')
    unresolved=0
    for row in rare:
        R=int(row['R_per_trajectory']);K=int(row['K'])
        check(row['observed_order2']=='NOT_SAVED','unsaved observations relabeled')
        check(row['observed_order4']==('NOT_SAVED' if K==4 else 'OUTSIDE_CUTOFF'),'order4 scope')
        expected=original[row['geometry'],row['template']]['saved']['paired_costs']
        near(row['mean_RZ_cos'],Fraction(sum(p[0] for p in expected),32))
        near(row['mean_RZ_sin'],Fraction(sum(p[1] for p in expected),32))
        if row['normalization_resolution']=='ROUNDED_TO_ONE_UNRESOLVED':
            check(float(row['B'])==1 and row['tau_inferred_from_saved_B']=='' and row['probability_all32_zero']=='','unresolved rounded B')
            unresolved+=1
            continue
        tau=float(row['tau_inferred_from_saved_B'])
        weights=[tau**n/math.factorial(n)*math.hypot(1,tau/(n+1)) for n in range(0,K+1,2)]
        normal=math.fsum(weights);near(normal**R,row['B'])
        p2=weights[1]/normal;near(row['p2'],p2)
        if K==4:near(row['p4'],weights[2]/normal)
    summary=json.loads((bundle/'analysis_summary_v1.json').read_text())
    check(summary['artificial_arithmetic_checks_passed']==14,'limited arithmetic tests')
    check(unresolved==summary['rare_law_normalization_unresolved_rows']==172,'rounded normalization scope')
    check(not summary['B0_I_formal_baseline_adopted'] and not summary['H8_modified'],'scientific scope')
    return {'status':'PASS_PUBLIC_SAVED_VALUE_TECHNICAL_AUDIT','grid_rows':1812,
            'winner_Fraction_work_and_paired_SE_rows':1812,'auxiliary_envelope_rows':1812,
            'nearby_Fraction_work_and_paired_SE_rows':len(nearby),'rare_law_rows':1164,
            'reversal_group_rows':len(grouped),'artificial_arithmetic_checks':14,'rare_law_unresolved_rows':unresolved,
            'source_and_reference_Git_bytes_SHA':'PASS','new_science_compile_sampling':0,
            'private_runtime_NPZ_access':0,'scope':'local public-only technical arithmetic, not scientific review or CI'}


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-manifest',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.bundle/'artifact_manifest_v1.json').read_text()) if args.verify_manifest else None
    result=verify(args.repo,args.bundle,manifest)
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
