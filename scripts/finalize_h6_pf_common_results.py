"""Reference-join completed H6 with immutable H2/H4 results; no science calls."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OLD='artifacts/resource_applicability/track_a_pf_common_results/2026-10-22'
META='artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22'
H6='artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22'
OUTPUT='artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22'
BASE='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
FORMULAS=('2nd','4th','4th(m5_best)')
EPS=(.05,.01,.005,.001,.0005)
NAMES=dict(zip(FORMULAS,('S2','Yoshida4','m5_best')))|{'ALL_FORMULAS':'全公式'}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cid(row):return '_'.join(str(row[k]) for k in ('system','method','prefix','q','formula'))


def export_controls(private,out):
    p=Path(private);plan=read(p/'plan_APPROVED.json');controls={}
    for name in ('SIGNAL','QUALIFY','COST','ACCOUNT'):
        root=p/name;controls[name]={n:read(root/n) for n in ('grant.json','CONSUMED.json','reserved_counts.json','completion.json')}
        controls[name]['ledger']=[read(q) for q in sorted((root/'ledger').glob('*.json'))]
    for name,value in [('controls_and_ledgers.json.gz',controls),('signal_records.json.gz',read(p/'SIGNAL/output/all_signals.json')),
                       ('native_results.json.gz',[read(q) for q in sorted((p/'COST/output').glob('*/result.json'))])]:
        (out/name).write_bytes(gzip.compress((json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode(),compresslevel=1,mtime=0))
    projection={k:v for k,v in plan.items() if k not in ('private_root','provider','input_paths','closure')}
    projection.update(original_plan_sha256=sha(p/'plan_APPROVED.json'),read_only_projection=True,not_a_replay_grant=True,
        input_paths={k:Path(v).name for k,v in plan['input_paths'].items()},closure=[{k:v for k,v in e.items() if k!='path'} for e in plan['closure']])
    write(out/'execution_contract_projection.json',projection);write(out/'approval_receipt.json',read(p/'approval_receipt.json'))
    write(out/'resource_receipts.json',{s:read(p/s/'resource_receipt.json') for s in ('SIGNAL','COST')})


def archive_executed_source(private,out):
    plan=read(Path(private)/'plan_APPROVED.json');paths=sorted({e['role'][7:] for e in plan['closure'] if e['role'].startswith('SOURCE:')})
    with tarfile.open(out,'w:gz') as tar:
        for rel in paths:
            data=subprocess.check_output(['git','show',plan['SOURCE']+':'+rel],cwd=ROOT)
            entries=[e for e in plan['closure'] if e['role']=='SOURCE:'+rel]
            assert all(hashlib.sha256(data).hexdigest()==e['sha256'] for e in entries)
            item=tarfile.TarInfo(rel);item.size=len(data);item.mtime=0;tar.addfile(item,io.BytesIO(data))


def build(common,metadata):
    out=ROOT/OUTPUT;assert not out.exists();out.mkdir(parents=True)
    meta=read(ROOT/META/'ending_audit.json');h6=read(ROOT/H6/'end_audit.json')
    assert meta['status']=='PASS' and meta['records']==63 and h6['status']=='PASS_MANDATORY_STOP' and h6['completed_candidates']==276
    export_controls(common,ROOT/H6)
    archive_executed_source(metadata,ROOT/META/'executed_SOURCE.tar.gz')
    archive_executed_source(common,ROOT/H6/'executed_SOURCE.tar.gz')
    refs={};candidates=[];comparisons=[]
    for system in ('H2','H4','H6'):
        b=f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22'
        rows=read(ROOT/b/'candidate_status.json');comps=read(ROOT/b/'comparisons.json');audit=read(ROOT/b/'end_audit.json')
        assert len(rows)=={'H2':84,'H4':228,'H6':276}[system] and all(r['status']=='COMPLETE' for r in rows)
        candidates.extend(rows);comparisons.extend(comps)
        refs[system]=dict(bundle=b,SOURCE=audit['SOURCE'],candidate_sha256=sha(ROOT/b/'candidate_status.json'),comparison_sha256=sha(ROOT/b/'comparisons.json'),
            end_audit_sha256=sha(ROOT/b/'end_audit.json'),recomputed_in_H6_track=system=='H6')
    assert len(candidates)==588 and len({r['id'] for r in candidates})==588 and len(comparisons)==60
    for r in comparisons:assert r['PF']['status']=='COMPLETE_REGISTERED_GRID' and r['PF']['best'] and r['PR_saved']
    write(out/'source_result_references.json',refs);write(out/'candidate_status_588.json',candidates);write(out/'comparisons_60.json',comparisons)
    with (out/'candidate_status_588.csv').open('x',newline='') as f:
        cols=['id','system','formula','method','prefix','q','signal','status','cost_kind','eligible_epsilons'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(candidates)
    def flat(r):
        pf=r['PF']['best'];pr=r['PR_saved'];return dict(system=r['system'],epsilon=r['epsilon'],scope=r['formula'],PF_id=cid(pf['candidate']),PF_total_RZ=pf['total_RZ_exact'],PR_candidate_id=pr['candidate_id'],PR_formula=pr['formula'],PR_q=pr['q'],
            PR_n=pr['n'],PR_mean_RZ=pr['total_RZ'],PR_unbiased_variance=pr['total_RZ_variance'],PR_SE=pr['total_RZ_SE'],PR_mean_exact=str(Fraction(int(pr['total_RZ_numerator']),int(pr['total_RZ_denominator']))),
            PR_reduction_percent=r['PR_reduction_percent'],direction=r['point_direction'])
    with (out/'comparisons_60.csv').open('x',newline='') as f:
        rows=[flat(r) for r in comparisons];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ranks=[]
    for s in ('H2','H4','H6'):
        for ep in EPS:
            items=[]
            for r in comparisons:
                if r['system']!=s or r['epsilon']!=ep or r['formula']=='ALL_FORMULAS':continue
                pf=r['PF']['best'];pr=r['PR_saved'];items.extend([
                    dict(method='PF',formula=r['formula'],candidate=pf['candidate'],total_RZ_exact=pf['total_RZ_exact'],Monte_Carlo_SE=None),
                    dict(method='PR_saved_exploration',formula=r['formula'],candidate_id=pr['candidate_id'],q=pr['q'],n=pr['n'],SE=pr['total_RZ_SE'],unbiased_variance=pr['total_RZ_variance'],total_RZ_exact=str(Fraction(int(pr['total_RZ_numerator']),int(pr['total_RZ_denominator']))))])
            items.sort(key=lambda r:Fraction(r['total_RZ_exact']))
            ranks.append(dict(system=s,epsilon=ep,point_ranks=[dict(r,rank=i+1) for i,r in enumerate(items)],population_ranking_guaranteed=False))
    write(out/'precision_rankings_15.json',ranks)
    old=[r for r in csv.DictReader((ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/formula_comparison_30.csv').open()) if r['mode']=='USUAL_SIGNED']
    changes=[]
    for r in comparisons:
        if r['formula']!='ALL_FORMULAS':continue
        before=next(v for v in old if v['system']==r['system'] and float(v['epsilon'])==r['epsilon'])
        assert r['PR_saved']['candidate_id']==before['best_PR_candidate']
        changes.append(dict(system=r['system'],epsilon=r['epsilon'],old=before,new=flat(r),saved_PR_selection_unchanged=True,
            difference_factors=['deterministic prefix/formula search domain','common empirical PF allowance policy','direct common native prices; legacy profile prices excluded'],no_search_only_causal_effect_claim=True))
    write(out/'old_vs_common_15.json',changes)
    # Compare nested deterministic subsets using completed saved endpoints only.
    # No new qualification/shot/compile calls or PR statistics are evaluated.
    oldcost=list(csv.DictReader((ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv').open()))
    breakdown=[]
    for system in ('H4','H6'):
        saved=[r for r in oldcost if r['system']==system and r['method']!='PR' and r['mode']=='USUAL_SIGNED']
        ids={cid(dict(r,prefix=int(r['prefix']),q=int(r['q']))) for r in saved}
        with gzip.open(ROOT/f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22/PF_endpoint_costs.json.gz','rt') as f:endpoints=json.load(f)
        assert len(ids)==(28 if system=='H4' else 36)
        for epsilon in EPS:
            common=[r for r in endpoints if r['epsilon']==epsilon]
            sparse=[r for r in common if cid(r['candidate']) in ids]
            assert len(sparse)==len(ids) and all(r['status'] in ('ELIGIBLE','INELIGIBLE') for r in sparse)
            best=lambda rows:min((r for r in rows if r['status']=='ELIGIBLE'),key=lambda r:Fraction(r['total_RZ_exact']))
            a=best(sparse);b=best([r for r in common if r['candidate']['formula']=='4th(m5_best)']);d=best(common)
            before=next(r for r in old if r['system']==system and float(r['epsilon'])==epsilon)
            c0=Fraction(before['m5_det_subset_total_RZ']);c1=Fraction(a['total_RZ_exact']);c2=Fraction(b['total_RZ_exact']);c3=Fraction(d['total_RZ_exact'])
            factors=[c1/c0,c2/c1,c3/c2];assert factors[0]*factors[1]*factors[2]==c3/c0
            breakdown.append(dict(system=system,epsilon=epsilon,old_sparse_saved_cost=str(c0),old_sparse_under_common=a,new_m5_common_best=b,all_formula_common_best=d,
                policy_and_saved_arithmetic_factor_exact=str(factors[0]),m5_prefix_expansion_factor_exact=str(factors[1]),formula_expansion_factor_exact=str(factors[2]),
                exact_product=str(c3/c0),unchanged_native_cost_series_for_sparse_m5=True,
                interpretation='Nested saved-point comparison under fixed direct native prices; not a population or search-only causal guarantee'))
    write(out/'nested_PF_cost_change_decomposition_10.json',breakdown)
    calls={s:read(ROOT/f'artifacts/resource_applicability/track_a_pf_common_{s}_results/2026-10-22/end_audit.json') for s in ('H2','H4','H6')}
    summary=dict(status='ALL588_EVALUATED_PASS_MANDATORY_STOP',result_join_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),result_join_file_sha256=sha(Path(__file__)),registered=588,complete=588,unresolved=0,systems={s:{k:a[k] for k in ('SOURCE','registered_candidates','completed_candidates','new_signal_candidates','qualification_endpoints','new_compile_candidates','new_compile_axes','used_saved_compiled_pairs')} for s,a in calls.items()},
        H6_metadata_extra=dict(observed_candidates=63,old_signals_preserved=52,new_raw_signals=11,source=meta['SOURCE'],actual_calls=meta['actual_calls']),
        new_common_signal_candidates=sum(a['new_signal_candidates'] for a in calls.values()),new_common_compile_candidates=sum(a['new_compile_candidates'] for a in calls.values()),new_common_compile_axes=sum(a['new_compile_axes'] for a in calls.values()),
        common_qualification_endpoints=2940,same_native_price_pairs_used=sum(a['used_saved_compiled_pairs'] for a in calls.values()),old_legacy217_prices_reused=0,
        old_failed_snapshot_restore_nonrefunded=1,new_PR_sampling=0,old_PR_prices_statistics_samples_changed=False,H2_H4_recomputed_in_H6_track=False,
        empirical_allowance_not_certificate=True,population_ranking_guarantee=False,unobserved_rare_event_bound=None,full_energy_RPE_optimization=False,
        detailed_scientific_review=False,GitHub_push=False,mandatory_STOP=True)
    write(out/'summary.json',summary)
    # Protect every existing H2/H4 bundle and historical H6 STOP/partial results.
    protected=[]
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT).decode().splitlines()
    for rel in paths:
        if rel.startswith(('artifacts/resource_applicability/track_a_pf_common_H2_results/','artifacts/resource_applicability/track_a_pf_common_H4_results/',OLD+'/')):
            data=subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT);assert (ROOT/rel).read_bytes()==data;protected.append(dict(path=rel,sha256=sha(ROOT/rel)))
    write(out/'prior_results_preserved.json',dict(base=BASE,files=protected))
    return summary


def doc(comparisons):
    def pfname(r):
        c=r['PF']['best']['candidate'];return f"{NAMES[c['formula']]} {c['method']} p{c['prefix']}/q{c['q']}"
    head='| 系 | ε | PF最良 | PF RZ | 保存PR最良 | PR平均RZ ± SE | PR削減率 |\n|---|---:|---|---:|---|---:|---:|\n'
    table=[];within=[]
    for r in comparisons:
        p=r['PF']['best'];s=r['PR_saved']
        if r['formula']=='ALL_FORMULAS':
            table.append(f"| {r['system']} | {r['epsilon']:g} | {pfname(r)} | {int(p['total_RZ_exact']):.7e} | {NAMES[s['formula']]} q{s['q']} | {float(s['total_RZ']):.7e} ± {float(s['total_RZ_SE']):.3e} | {r['PR_reduction_percent']:+.4f}% |")
        else:
            within.append(f"| {r['system']} | {r['epsilon']:g} | {NAMES[r['formula']]} | {pfname(r)} | {int(p['total_RZ_exact']):.7e} | q{s['q']} / {float(s['total_RZ']):.7e} ± {float(s['total_RZ_SE']):.3e} | {r['PR_reduction_percent']:+.4f}% | {s['n']} |")
    return head+'\n'.join(table), '| 系 | ε | 同公式 | PF最良 | PF RZ | 保存PR q / 平均RZ ± SE | PR削減率 | n |\n|---|---:|---|---|---:|---|---:|---:|\n'+'\n'.join(within)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--common',required=True);p.add_argument('--metadata',required=True);args=p.parse_args()
    print(json.dumps(build(args.common,args.metadata),indent=2))
