"""Saved-value arithmetic for the H4 B0+I manuscript; no science imports."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import signal

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO/'artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14'
RESULT = REPO/'artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14'
OLD = REPO/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13'
AXES = ('cosine','sine')
GEOMETRIES = ('0.70','0.80','0.90','1.10','1.40','1.60')
BASE_COMMIT = '98cdb35cd08469714f20d6cc3b982d9c80908f46'
SOURCE = '924b779c2bc6f6e844d6283c355f91c73aed6445'


def load(path):
    return json.loads(path.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def csv_read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def identity(path):
    raw=path.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def check_result_manifest():
    manifest=load(RESULT/'artifact_manifest.json')
    for name,expected in manifest['files'].items():assert identity(RESULT/name)==expected,name
    for name,expected in manifest['dependencies'].items():assert identity(REPO/name)==expected,name
    prior=load(RESULT/'verification.json')
    assert prior['status']=='PASS_PUBLIC_SAVED_VALUE_REPRODUCTION'
    assert prior['exact_fraction_B2_comparisons']==1812 and prior['new_compile_calls']==0


def counts(bias,normalization,epsilon):
    h=[epsilon/math.sqrt(2)-bias[a] for a in AXES]
    if min(h)<=0:return None
    return [math.ceil(2*normalization**2/v**2*math.log(80)) for v in h],h


def evaluate(bias,normalization,pairs,epsilon):
    calculated=counts(bias,normalization,epsilon)
    if calculated is None:return None
    ns,h=calculated;n=len(pairs)
    assert n in (1,32)
    means=[Fraction(sum(p[i] for p in pairs),n) for i in range(2)]
    values=[ns[0]*p[0]+ns[1]*p[1] for p in pairs]
    work=Fraction(sum(values),n)
    variance=(Fraction(sum(v*v for v in values),n*(n-1))-
              Fraction(sum(values)**2,n*n*(n-1))) if n>1 else Fraction(0)
    return dict(N_cos=ns[0],N_sin=ns[1],N_total=sum(ns),
        bias_cos=bias['cosine'],bias_sin=bias['sine'],B=normalization,B_squared=normalization**2,
        headroom_cos=h[0],headroom_sin=h[1],C_cos=float(means[0]),C_sin=float(means[1]),
        C_effective=float(work)/sum(ns),G=float(work),G_numerator=work.numerator,
        G_denominator=work.denominator,paired_SE=math.sqrt(variance))


def factor_ratio(a,b):
    value=dict(ratio_shots=a['N_total']/b['N_total'],
        ratio_effective_RZ=a['C_effective']/b['C_effective'],ratio_work=a['G']/b['G'])
    assert math.isclose(value['ratio_shots']*value['ratio_effective_RZ'],value['ratio_work'],rel_tol=1e-12)
    return value


def grid_segments(rows,label):
    """Runs of observed grid labels, with adjacent brackets, never interpolated roots."""
    result=[];start=0
    for stop in range(1,len(rows)+1):
        if stop<len(rows) and label(rows[start])==label(rows[stop]):continue
        result.append(dict(label=label(rows[start]),points=stop-start,
            epsilon_first=float(rows[start]['epsilon']),epsilon_last=float(rows[stop-1]['epsilon']),
            previous_grid_epsilon=float(rows[start-1]['epsilon']) if start else None,
            next_grid_epsilon=float(rows[stop]['epsilon']) if stop<len(rows) else None,
            continuous_boundary_certified=False))
        start=stop
    return result


def review_tables(tables,summary):
    text='# v0.4 review tables：保存値の代表点・grid bracket・因子・SE\n\nH4 linear/STO-3G/DF12、T=.8、prefix3/4/5/6/9、q1/2/4/8、δ=.1〜.8、S2/ordinary/structured/fixed compiler/P=0。εはcomplex signal。\n\n## Method領域と隣接grid点\n\n|距離 Å|点最小method|点数|最初ε|最後ε|直前ε|直後ε|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in tables['method_grid_regions']:
        vals=[r['geometry'],r['label'],str(r['points'])]+[format(r[k],'.9f') if r[k] is not None else '—' for k in ('epsilon_first','epsilon_last','previous_grid_epsilon','next_grid_epsilon')]
        text+='|'+'|'.join(vals)+'|\n'
    text+='\n隣接点のbracketであり、区間内部の連続ε切替点を求めたものではない。\n\n## ε=.05の費用と因子（B0+I/B2）\n\n|距離 Å|I百万G|B2百万G|I shot|B2 shot|shot比|C_eff比|G比|B2 paired SE百万|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in tables['selected_factors18']:
        if r['epsilon']!=.05:continue
        vals=[r['geometry'],f"{r['B0I_G']/1e6:.6f}",f"{r['B2_G']/1e6:.6f}",str(r['B0I_N_total']),str(r['B2_N_total'])]+[f'{r[k]:.6f}' for k in ('ratio_shots','ratio_effective_RZ','ratio_work')]+[f"{r['B2_paired_SE']/1e6:.6f}"]
        text+='|'+'|'.join(vals)+'|\n'
    text+='\n双方で構成を再選択した最小。G比=shot比×C_eff比。これは同じprefix/qの位相効果とは別である。\n\n## 全1812点の記述的2SE分類\n\n|距離 Å|IがB2点−2SE未満|I点最小・差2SE以内|B2がI−2SE未満|B2点最小・差2SE以内|\n|---|---:|---:|---:|---:|\n'
    keys=('B0I_BELOW_B2_MINUS_2SE','B0I_LOWER_WITHIN_2SE','B2_BELOW_B0I_MINUS_2SE','B2_LOWER_WITHIN_2SE')
    for g,c in summary['per_geometry_engineering_counts'].items():text+='|'+'|'.join([g]+[str(c.get(k,0)) for k in keys])+'|\n'
    return text+'|合計|'+'|'.join(str(summary['total_engineering_counts'][k]) for k in keys)+'|\n\n同じ32 paired trajectoriesの選抜後推定であり、正式confidence intervalや多重比較・希少母平均保証ではない。\n'


def derive():
    check_result_manifest()
    old=load(OLD/'saved_candidates_v1.json')
    original={(r['geometry'],r['template_id']):r for r in old['candidates']}
    measured=load(RESULT/'measured_candidates.json')
    b0i={(r['geometry'],r['template_id']):r for r in measured}
    grid=csv_read(RESULT/'precision_grid_1812.csv')
    assert len(grid)==1812 and len(b0i)==120
    factors=[];uncertainty=[];segments=[];config_segments=[];summary_counts=defaultdict(Counter)
    for row in grid:
        g,e=row['geometry'],float(row['epsilon']);i=b0i[(g,row['B0I_template'])];r=original[(g,row['B2_template'])]
        a=evaluate(i['bias'],1.,[[i['costs'][k] for k in AXES]],e)
        b=evaluate(r['saved']['bias'],r['saved']['normalization'],r['saved']['paired_costs'],e)
        assert a is not None and b is not None
        assert a['N_cos']==int(row['B0I_N_cos']) and a['N_sin']==int(row['B0I_N_sin'])
        assert a['G']==float(row['B0I_work']) and b['G']==float(row['B2_work'])
        assert math.isclose(b['paired_SE'],float(row['B2_paired_SE']),rel_tol=1e-12)
        factor=dict(geometry=g,epsilon=e,B0I_template=i['template_id'],B2_template=r['template_id'],
            point_min_method=row['winner'],selection_scope='separate family minima reselected at each saved precision')
        factor.update({'B0I_'+k:v for k,v in a.items()});factor.update({'B2_'+k:v for k,v in b.items()})
        factor.update(factor_ratio(a,b));factors.append(factor)
        gap=b['G']-a['G'];two_se=2*b['paired_SE']
        lower=gap>0;below=gap>two_se
        category=('B0I_BELOW_B2_MINUS_2SE' if below else
            'B0I_LOWER_WITHIN_2SE' if lower else
            'B2_BELOW_B0I_MINUS_2SE' if -gap>two_se else 'B2_LOWER_WITHIN_2SE')
        summary_counts[g][category]+=1
        uncertainty.append(dict(geometry=g,epsilon=e,B0I_template=i['template_id'],B2_template=r['template_id'],
            B0I_G=a['G'],B2_G=b['G'],B2_paired_SE=b['paired_SE'],gap_B2_minus_B0I=gap,
            B0I_over_B2=a['G']/b['G'],relative_gap=gap/b['G'],two_SE_over_B2=two_se/b['G'],
            B0I_lower_point=lower,B0I_below_B2_point_minus_2SE=below,
            engineering_category=category,formal_confidence_interval=False,
            independent_selection_sample=False,rare_event_mean_guarantee=False))
    for g in GEOMETRIES:
        rows=[r for r in grid if r['geometry']==g]
        assert len(rows)==302
        segments += [dict(geometry=g,**r) for r in grid_segments(rows,lambda x:x['winner'])]
        config_segments += [dict(geometry=g,**r) for r in grid_segments(rows,
            lambda x:x['B0I_template'] if x['winner']=='B0_PLUS_I' else x['B2_template'])]
    fixed=[]
    templates=[('B0_PLUS_I',f'B0I-validation-p{p}-q{q}') for p,q in ((3,1),(4,1),(5,1),(6,1),(9,2),(9,4))]
    templates += [('B2',f'B2-rank3-q{q}-r{r}-K{k}') for q,r,k in ((1,4,2),(1,4,4),(1,8,4),(2,4,2),(4,2,2),(4,2,4))]
    for method,t in templates:
        for g in GEOMETRIES:
            candidate=b0i[(g,t)] if method=='B0_PLUS_I' else original[(g,t)]
            bias=candidate['bias'] if method=='B0_PLUS_I' else candidate['saved']['bias']
            norm=1. if method=='B0_PLUS_I' else candidate['saved']['normalization']
            pairs=[[candidate['costs'][a] for a in AXES]] if method=='B0_PLUS_I' else candidate['saved']['paired_costs']
            for e in (.005,.05,.1):
                result=evaluate(bias,norm,pairs,e)
                fixed.append(dict(method=method,fixed_template=t,geometry=g,epsilon=e,
                    eligible=result is not None,selection_scope='posthoc display panel held fixed, not independent confirmatory sampling',
                    **{k:result[k] if result else None for k in ('bias_cos','bias_sin','headroom_cos','headroom_sin','B','N_cos','N_sin','N_total','C_cos','C_sin','C_effective','G','paired_SE')}))
    matched=[]
    wrapper_map={(r['geometry'],r['prefix'],r['q'],r['axis']):r for r in load(RESULT/'compiled_wrapper_metrics.json')}
    metric_rows=[]
    for c in measured:
        representative=wrapper_map[(c['geometry'],c['prefix'],c['q'],'cosine')]
        old_candidate=original[(c['geometry'],representative['original_B0_template'])]
        a=evaluate(c['bias'],1.,[[c['costs'][a] for a in AXES]],.05)
        o=evaluate(old_candidate['saved']['bias'],1.,old_candidate['saved']['paired_costs'],.05)
        row=dict(geometry=c['geometry'],prefix=c['prefix'],q=c['q'],epsilon=.05,
            B0I_template=c['template_id'],B0_template=old_candidate['template_id'],
            B0I_eligible=a is not None,B0_eligible=o is not None,
            selection_scope='same prefix/q and frozen geometry, phase-only difference')
        for method,candidate,bias in (('B0I',a,c['bias']),('B0',o,old_candidate['saved']['bias'])):
            row.update({method+'_bias_'+axis:bias[axis] for axis in AXES})
            row.update({method+'_headroom_'+axis:.05/math.sqrt(2)-bias[axis] for axis in AXES})
            row.update({method+'_'+k:candidate[k] if candidate else None for k in ('N_cos','N_sin','N_total','C_effective','G')})
        row['ratio_work_same_configuration']=a['G']/o['G'] if a and o else None;matched.append(row)
        for axis in AXES:
            w=wrapper_map[(c['geometry'],c['prefix'],c['q'],axis)]
            for metric,new_value in w['metrics'].items():
                old_value=old_candidate['paired_metric_axis_means'][axis][metric]
                assert new_value==old_value
                metric_rows.append(dict(geometry=c['geometry'],prefix=c['prefix'],q=c['q'],axis=axis,
                    metric=metric,B0_value=int(old_value),B0I_value=new_value,delta=0,
                    numeric_circuit_fingerprint=w['numeric_circuit_fingerprint'],source_commit=SOURCE))
    total=Counter()
    for c in summary_counts.values():total.update(c)
    assert total['B0I_BELOW_B2_MINUS_2SE']==374 and total['B0I_LOWER_WITHIN_2SE']==23
    assert sum(total.values())==1812
    chosen=[r for r in factors if r['epsilon'] in (.005,.05,.1)]
    summary=dict(schema_version='h4-b0i-manuscript-integration-v1',base_result_commit=BASE_COMMIT,
        source_commit=SOURCE,dataset_identity_source=old['dataset_identity_source'],
        point_minimum={'B0_PLUS_I':397,'B2':1415,'ties':0},
        per_geometry_engineering_counts={g:dict(c) for g,c in summary_counts.items()},
        total_engineering_counts=dict(total),method_grid_segments=len(segments),
        configuration_grid_segments=len(config_segments),selected_factors=chosen,
        matched_configuration_rows=len(matched),metric_comparisons=len(metric_rows),
        fixed_display_rows=len(fixed),old_result_verification_reused=True,
        old_B0I_measurements_repeated=False,new_signal_compile_sampling_calls=0,NPZ_reads=0,
        H8_shared_venv_changes=0,formal_confidence_interval=False,
        scientific_claim_decision='GPT_REVIEW_PENDING',mandatory_stop=True)
    return dict(cost_factors_1812=factors,uncertainty_comparison_1812=uncertainty,
        method_grid_regions=segments,configuration_grid_regions=config_segments,
        fixed_configuration_factors=fixed,matched_B0_B0I_120=matched,
        compiled_metric_deltas_1440=metric_rows,selected_factors18=chosen),summary


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));signal.alarm(90)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    tables,summary=derive()
    if args.verify:
        for name,rows in tables.items():
            assert csv_read(BUNDLE/(name+'.csv'))==[{k:'' if v is None else str(v) for k,v in r.items()} for r in rows],name
        assert load(BUNDLE/'analysis_summary.json')==summary
        assert (BUNDLE/'tables_for_review.md').read_text()==review_tables(tables,summary)
        print(json.dumps(dict(status='PASS_SAVED_VALUE_INTEGRATION',points=1812,metric_comparisons=1440,
            B0I_point_min=397,B0I_below_B2_minus_2SE=374,NPZ_reads=0,science_compile_sampling_calls=0)))
    else:
        BUNDLE.mkdir(parents=True,exist_ok=True)
        for name,rows in tables.items():table(BUNDLE/(name+'.csv'),rows)
        dump(BUNDLE/'analysis_summary.json',summary)
        (BUNDLE/'tables_for_review.md').write_text(review_tables(tables,summary))
        print(json.dumps(dict(status='POSTHOC_TABLES_WRITTEN',rows={k:len(v) for k,v in tables.items()},
            engineering_counts=summary['total_engineering_counts'],new_science_calls=0)))


if __name__=='__main__':main()
