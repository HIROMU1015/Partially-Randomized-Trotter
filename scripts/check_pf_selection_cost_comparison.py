"""Author validation with integer cross-products and saved-record identity."""
from collections import defaultdict
from datetime import datetime,timezone
from decimal import Decimal,localcontext
from fractions import Fraction as F
from functools import cmp_to_key
from math import lcm

from run_pf_selection_cost_comparison import OUT,input_data,read,sha,verify,write


def rat(value):
    v=F(value);return v.numerator,v.denominator


def cmp_rat(a,b):
    an,ad=rat(a);bn,bd=rat(b);return (an*bd>bn*ad)-(an*bd<bn*ad)


def integer_total(shots,prices):
    a,b=rat(prices['cosine']);c,d=rat(prices['sine']);common=lcm(b,d)
    return shots['cosine']*a*(common//b)+shots['sine']*c*(common//d),common


def same_pair(total,value):
    a,b=total;c,d=rat(value);return a*d==c*b


def main():
    contract=read(OUT/'sealed_contract.json');verify(contract);data=input_data()
    costs=read(OUT/'candidate_totals.json')['rows'];comparisons=read(OUT/'selection_comparison.json')['rows']
    ranks=read(OUT/'candidate_ranks.json')['rows'];changes=read(OUT/'rank_relation_changes.json')['rows']
    original={r['row_id']:r for r in data['saved_condition_rows.json']['rows']}
    units={u['task_id']:u for u in data['axis_price_lineage.json']['rows']}
    endpoints={(c['system'],c['bias_mode'],c['epsilon'],c['allowance_factor']) for c in contract['conditions']}
    def ep(r):return (r['system'],r['bias_mode'],r['epsilon'],r['allowance_factor'])
    expected={r['row_id'] for r in original.values() if r['kind']=='PF' and ep(r) in endpoints}
    assert {r['row_id'] for r in costs}==expected and len(costs)==len(expected)
    assert len(comparisons)==len({r['condition_id'] for r in comparisons})==70
    assert all(r['bias_mode']=='signed_signal' for r in costs if r['system']=='H6_DF19')
    eligible=0
    with localcontext() as ctx:
        ctx.prec=80
        for row in costs:
            src=original[row['row_id']]
            for key in ('saved_shots','eligible','saved_total_RZ_literal','candidate_id','source_file_sha256'):
                assert row[key]==src[key],(row['row_id'],key)
            if row['eligible'] is not True:
                assert row['baseline_G_exact'] is row['replacement_G_exact'] is None
                continue
            eligible+=1
            old={a:units[tid]['baseline_RZ_literal'] for a,tid in src['unit_ids'].items()}
            new={a:units[tid]['replacement_RZ_literal'] for a,tid in src['unit_ids'].items()}
            assert same_pair(integer_total(src['saved_shots'],old),row['baseline_G_exact'])
            assert same_pair(integer_total(src['saved_shots'],old),src['saved_total_RZ_literal'])
            assert same_pair(integer_total(src['saved_shots'],new),row['replacement_G_exact'])
            for name,prices in [('baseline_G_exact',old),('replacement_G_exact',new)]:
                value=sum(Decimal(src['saved_shots'][axis])*Decimal(rat(price)[0])/Decimal(rat(price)[1]) for axis,price in prices.items())
                a,b=rat(row[name]);expected_decimal=Decimal(a)/Decimal(b)
                assert abs(value-expected_decimal)<=max(Decimal('1e-40'),abs(expected_decimal)*Decimal('1e-60'))
    by_ep=defaultdict(list);rank_index=defaultdict(dict);change_index=defaultdict(list)
    for r in costs:by_ep[ep(r)].append(r)
    for r in ranks:rank_index[r['condition_id']][r['candidate_id']]=r
    for r in changes:change_index[r['condition_id']].append(r)
    condition_index={('|'.join(ep(c)+(c['original_scope'],))):c for c in contract['conditions']}
    checked_pairs=0
    for row in comparisons:
        c=condition_index[row['condition_id']];ids=set(c['candidate_ids'])
        selected=[r for r in by_ep[ep(row)] if r['candidate_id'] in ids and r['eligible'] is True]
        selected_by_id={r['candidate_id']:r for r in selected}
        for side,field in [('baseline','baseline_G_exact'),('replacement','replacement_G_exact')]:
            def comparison(a,b):
                value=cmp_rat(a[field],b[field])
                return value if value else (a['candidate_id']>b['candidate_id'])-(a['candidate_id']<b['candidate_id'])
            order=sorted(selected,key=cmp_to_key(comparison));best=order[0]
            assert row[side]['candidate_id']==best['candidate_id'] and cmp_rat(row[side]['G_exact'],best[field])==0
            assert row[side]['exact_ties']==sorted(r['candidate_id'] for r in selected if cmp_rat(r[field],best[field])==0)
            for ordinal,r in enumerate(order,1):
                saved=rank_index[row['condition_id']][r['candidate_id']][side]
                assert saved['ordinal']==ordinal and saved['G_exact']==r[field]
                assert saved['cost_tie_rank']==1+sum(cmp_rat(peer[field],r[field])<0 for peer in selected)
        pr=original[row['saved_PR_row_id']]
        assert ep(pr)==ep(row) and pr['saved_total_RZ_literal']==row['saved_PR_total_exact']
        assert pr['saved_PR_SE_literal']==row['saved_PR_SE_literal'] and pr['cohort']==row['saved_PR_cohort']
        for side in ('baseline','replacement'):
            rn,rd=rat(row['PR_over_PF_'+side+'_exact']);pn,pd=rat(row['saved_PR_total_exact']);gn,gd=rat(row[side]['G_exact'])
            assert rn*pd*gn==rd*pn*gd
        actual=[];sorted_ids=sorted(selected_by_id)
        for i,left in enumerate(sorted_ids):
            for right in sorted_ids[i+1:]:
                a=selected_by_id[left];b=selected_by_id[right]
                old=cmp_rat(a['baseline_G_exact'],b['baseline_G_exact']);new=cmp_rat(a['replacement_G_exact'],b['replacement_G_exact']);checked_pairs+=1
                if old!=new:actual.append((left,right,old,new,old*new<0,(old==0)!=(new==0)))
        stored=[(r['left_candidate'],r['right_candidate'],r['before_relation'],r['after_relation'],r['strict_inversion'],r['tie_change']) for r in change_index[row['condition_id']]]
        assert stored==actual
        assert row['strict_pairwise_inversions']==sum(r[4] for r in actual)
        assert row['pairwise_tie_changes']==sum(r[5] for r in actual)
    excluded=read(OUT/'excluded_unresolved30.json')
    assert excluded['rows']==contract['excluded_unresolved_conditions'] and not excluded['numeric_analysis_performed']
    for name,value in read(OUT/'numeric_completion.json')['result_sha256'].items():assert sha(OUT/name)==value
    native=read(OUT/'native_exit_evidence.json');assert native['native_exit_code']==0 and native['same_parent_wait4'] and native['reaped']
    rendering=read(OUT/'rendering_completion.json');assert len(rendering['figures'])==6
    for figure in rendering['figures']:
        path=OUT/figure['path'];raw=path.read_bytes();assert sha(path)==figure['sha256']
        assert raw.startswith(b'\x89PNG\r\n\x1a\n') if path.suffix=='.png' else raw.startswith(b'%PDF-') if path.suffix=='.pdf' else b'<svg' in raw
    figure_data=read(OUT/'figure_data.json');assert len(figure_data['points'])==70
    assert sha(OUT/'figure_data.json')==rendering['figure_data_sha256']
    for name in ('PF_cost_change','PR_PF_ratios'):
        ids=[v for panel,values in figure_data['panel_condition_ids'].items() if panel.startswith(name) for v in values]
        assert len(ids)==len(set(ids))==70 and set(ids)==set(contract['condition_ids'])
    write('structural_validation.json',dict(status='PASS_AUTHOR_EXACT_NUMERIC_AND_STRUCTURE_CHECKS',
        SOURCE=contract['SOURCE'],checked_UTC=datetime.now(timezone.utc).isoformat(),
        baseline_candidate_replays=eligible,integer_cross_product_and_Decimal_candidate_checks=eligible,
        approved_candidate_condition_rows=len(costs),approved_scope_conditions=70,unique_endpoints=30,
        integer_comparator_scope_rank_checks=70,pairwise_relations_checked=checked_pairs,
        PR_cost_SE_cohort_verbatim=True,excluded_unresolved30_untouched=True,
        input_SOURCE_runtime_hashes='PASS',same_parent_wait4='PASS',figures=2,exports=6,
        independent_scientific_review=False,checks_by='Primary author agent',
        detailed_scientific_review=False,final_evidence_freeze=False,mandatory_STOP=True))
    print(dict(status='PASS',conditions=70,candidate_rows=len(costs),eligible_replays=eligible,pair_checks=checked_pairs))


if __name__=='__main__':main()
