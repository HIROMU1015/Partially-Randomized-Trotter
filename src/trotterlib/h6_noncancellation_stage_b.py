"""Stage A literal join and stage B exact saved-shot comparison. No shot/model API."""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction as F
from functools import cmp_to_key
import hashlib
import json
from math import lcm

import pf_selection_cost_comparison as old

IDS = {f'B0I_p{p}_q{q}_2nd' for p in (5,10,15) for q in (1,2,4,8)}
BIAS = 'empirical_no_cancellation'
ep = old.endpoint
require = old.require


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def authorize(contract, grant, *, artificial=False):
    if artificial:
        require(contract.get('kind')=='ARTIFICIAL','ARTIFICIAL_ONLY')
        return
    require(contract.get('scope')=='H6_NONCANCELLATION_STAGE_B_30' and
        contract.get('scope_conditions')==30 and contract.get('strategy')=='ALL_SAVED458', 'EXACT_STAGE_B_SCOPE')
    require(grant.get('scientific_execution_authorized') is True and grant.get('SOURCE')==contract.get('SOURCE')
        and grant.get('retry') is False and grant.get('scope')==contract['scope'], 'NEW_STAGE_B_GRANT_REQUIRED')


def akey(row):
    return row['candidate_id'],str(row['epsilon']),str(row['allowance_factor'])


def join_stage_a(rows, stage_a, stage_a_evidence, *, artificial=False):
    """Only originally unknown rows can change; all PR and known rows stay verbatim."""
    before=digest(rows)
    target=[r for r in rows if r['system']=='H6_DF19' and r['bias_mode']==BIAS]
    require(len({r['row_id'] for r in target})==len(target),'DUPLICATE_ORIGINAL_ROW')
    amap={akey(r):r for r in stage_a}
    require(len(amap)==len(stage_a),'DUPLICATE_STAGE_A_ENDPOINT')
    unknown=[r for r in target if r['eligible'] is None]
    require({akey(r) for r in unknown}==set(amap),'PATCH_ONLY_EXACT_UNKNOWN_ENDPOINTS')
    if not artificial:
        require(len(target)==1440 and len(unknown)==len(stage_a)==120,'EXACT1440_120_JOIN')
        require({r['candidate_id'] for r in unknown}==IDS,'EXACT12_B0I')
    result=[];lineage=[]
    for source in target:
        row=deepcopy(source)
        if source['eligible'] is None:
            a=amap[akey(source)]
            require(source['kind']=='PF' and source['saved_total_RZ_literal'] is None and
                source['saved_bias_and_shot_policy'] is None, 'OLD_UNKNOWN_SCHEMA')
            require(type(a['eligible']) is bool and a['bias_mode']==BIAS and a['T']==.8 and a['PF']=='2nd',
                    'UNRESOLVED_OR_CHANGED_STAGE_A')
            for k in ('prefix','q'):require(str(a[k])==source[k],'STAGE_A_CANDIDATE_IDENTITY')
            require(a['accounting']['cost_acquisition_eligible']==a['eligible'],'STAGE_A_ACCOUNTING_FLAG')
            if a['eligible'] is True:
                require(a['shots']=={axis:a['accounting']['axes'][ch]['shots'] for axis,ch in
                    (('cosine','real'),('sine','imag'))},'STAGE_A_SAVED_SHOTS')
                require(all(type(v) is int and v>0 for v in a['shots'].values()),'STAGE_A_SHOTS_POSITIVE')
            else:require(a['shots'] is None,'INELIGIBLE_STAGE_A_SHOTS')
            row.update(eligible=a['eligible'],saved_shots=deepcopy(a['shots']),
                saved_bias_and_shot_policy=deepcopy(a['accounting']),
                unit_ids={axis:':'.join(('H6_DF19',source['candidate_id'],axis)) for axis in old.AXES}
                    if a['eligible'] is True else {},
                saved_cost_status='STAGE_A_NEW_ACCOUNTING_BASELINE_COST_NOT_PREVIOUSLY_SAVED',
                stage_A_provenance=dict(stage_a_evidence,endpoint_sha256=digest(a)))
            lineage.append(dict(row_id=row['row_id'],candidate_id=row['candidate_id'],
                previous_eligible=None,eligible=a['eligible'],stage_A_provenance=row['stage_A_provenance'],
                old_total_literal_retained=None,new_cost_lineage=True))
        require(type(row['eligible']) is bool,'UNRESOLVED_FAIL_CLOSED:'+row['row_id'])
        result.append(row)
    require(digest(rows)==before,'ORIGINAL_ROWS_MUTATED')
    return result,lineage


def baseline(rows,units):
    costs=[];replayed=0;new=0
    for src in rows:
        if src['kind']!='PF':continue
        require(type(src['eligible']) is bool,'UNKNOWN_BASELINE_ELIGIBILITY')
        r={k:deepcopy(src[k]) for k in ('row_id',*old.ENDPOINT,'candidate_id','method','prefix','q','PF',
            'eligible','saved_shots','saved_total_RZ_literal','source_id','source_locator','source_file_sha256')}
        recovered='stage_A_provenance' in src
        r.update(baseline_G_exact=None,replacement_G_exact=None,baseline_replay_matches=None,unit_provenance={},
            baseline_cost_provenance='NEW_STAGE_A_ACCOUNTING' if recovered else 'EXACT_REPLAY_SAVED_COST',
            stage_A_provenance=src.get('stage_A_provenance'))
        if r['eligible'] is True:
            prices={}
            for axis in old.AXES:
                tid=src['unit_ids'].get(axis);require(tid in units,'MISSING_BASELINE_AXIS')
                prices[axis]=units[tid]['baseline_RZ_literal'];r['unit_provenance'][axis]=tid
            total=old.weighted(src['saved_shots'],prices)
            if recovered:
                require(src['saved_total_RZ_literal'] is None,'NEW_COST_NOT_HISTORICAL_REPLAY');new+=1
            else:
                require(src['saved_total_RZ_literal'] is not None and total==F(src['saved_total_RZ_literal']),
                    'BASELINE_COST_MISMATCH:'+src['row_id'])
                r['baseline_replay_matches']=True;replayed+=1
            r.update(baseline_G_exact=str(total),baseline_axis_RZ_exact=prices)
        costs.append(r)
    return costs,dict(status='PASS_ALL_KNOWN_COSTS_EXACT',known_costs_replayed=replayed,
        newly_constructed_eligible_baseline_costs=new,historical_full_domain_minimum_was_null=True)


def baseline_selections(costs,rows,comparisons,conditions):
    pr_index={(ep(r),r['candidate_id']):r for r in rows if r['kind']=='PR'}
    saved_index={ep(r)+(r['scope'],):r['saved_result'] for r in comparisons}
    gates=[]
    for c in conditions:
        original=saved_index[ep(c)+(c['original_scope'],)]
        require(original['deterministic']['best'] is None and
                original['deterministic']['status']=='UNRESOLVED_FULL_DOMAIN','HISTORICAL_BASELINE_NOT_NULL')
        chosen=original['confirmed_PR']['best'];pr=pr_index.get((ep(c),chosen['cell_id']))
        require(pr is not None and pr['eligible'] is True and pr['paired_samples']==32 and
                pr['cohort']=='CONFIRMATION_UNION8_N32' and pr['saved_PR_SE_literal'] is not None and
                F(pr['saved_total_RZ_literal'])==F(chosen['G_exact']), 'SAVED_PR_BINDING')
        selected=[r for r in costs if ep(r)==ep(c) and r['candidate_id'] in c['candidate_ids']]
        require(len(selected)==len(c['candidate_ids']) and {r['candidate_id'] for r in selected}==set(c['candidate_ids']),
                'EXACT_REGISTERED_DOMAIN_REQUIRED')
        require(all(type(r['eligible']) is bool for r in selected),'UNKNOWN_NOT_EXCLUDED')
        best=old.choose(selected,'baseline_G_exact')
        gates.append(dict(condition_id=old.condition_id(c),baseline=best,saved_PR=deepcopy(pr),
                         PR_over_PF_baseline_exact=str(F(pr['saved_total_RZ_literal'])/F(best['G_exact'])),
                         baseline_saved_winner_reproduced=False,new_completed_baseline=True))
    return gates


def compare(costs,conditions,gates):
    rows,ranks,changes=old.compare(costs,conditions,gates)
    for row in rows:
        row.update(baseline_saved_result_matches=False,
            baseline_provenance='NEW_COMPLETED_BASELINE_AFTER_STAGE_A;KNOWN_COSTS_EXACTLY_REPLAYED',
            historical_full_domain_winner=None,numeric_origin='NEW_STAGE_B_30')
    return rows,ranks,changes


def integer_pair(value):
    f=F(value);return f.numerator,f.denominator


def ratio_cmp(a,b):
    an,ad=integer_pair(a);bn,bd=integer_pair(b)
    return (an*bd>bn*ad)-(an*bd<bn*ad)


def int_total(shots,prices):
    an,ad=integer_pair(prices['cosine']);bn,bd=integer_pair(prices['sine']);d=lcm(ad,bd)
    return shots['cosine']*an*(d//ad)+shots['sine']*bn*(d//bd),d


def audit(costs,rows,units,conditions,results,ranks,changes):
    """Author check via integer cross-products and a distinct rank comparator."""
    original={r['row_id']:r for r in rows};candidate_checks=0
    for r in costs:
        s=original[r['row_id']]
        require(r['eligible'] is s['eligible'] and r['saved_shots']==s['saved_shots'],'AUDIT_SHOTS_OR_ELIGIBILITY')
        if r['eligible'] is False:
            require(r['baseline_G_exact'] is r['replacement_G_exact'] is None,'AUDIT_INELIGIBLE_NOT_ZERO');continue
        for name,unit_field in [('baseline_G_exact','baseline_RZ_literal'),('replacement_G_exact','replacement_RZ_literal')]:
            n,d=int_total(s['saved_shots'],{a:units[t][unit_field] for a,t in s['unit_ids'].items()})
            x,y=integer_pair(r[name]);require(n*y==x*d,'AUDIT_INTEGER_WEIGHTED_COST');candidate_checks+=1
    rank_index={(r['condition_id'],r['candidate_id']):r for r in ranks}
    change_index=defaultdict(list)
    for r in changes:change_index[r['condition_id']].append(r)
    result_index={r['condition_id']:r for r in results};tested=0
    for c in conditions:
        cid=old.condition_id(c);result=result_index[cid]
        selected=[r for r in costs if ep(r)==ep(c) and r['candidate_id'] in c['candidate_ids'] and r['eligible'] is True]
        for side,field in [('baseline','baseline_G_exact'),('replacement','replacement_G_exact')]:
            def cmp(a,b):
                v=ratio_cmp(a[field],b[field]);return v or ((a['candidate_id']>b['candidate_id'])-(a['candidate_id']<b['candidate_id']))
            ordered=sorted(selected,key=cmp_to_key(cmp));best=ordered[0]
            require(best['candidate_id']==result[side]['candidate_id'] and ratio_cmp(best[field],result[side]['G_exact'])==0,
                    'AUDIT_PF_SELECTION')
            require(result[side]['exact_ties']==sorted(r['candidate_id'] for r in selected if ratio_cmp(r[field],best[field])==0),
                    'AUDIT_EXACT_TIES')
            for i,r in enumerate(ordered,1):
                rank=rank_index[(cid,r['candidate_id'])][side]
                require(rank['ordinal']==i and rank['cost_tie_rank']==1+sum(ratio_cmp(peer[field],r[field])<0 for peer in selected),
                        'AUDIT_INTEGER_RANK')
            rn,rd=integer_pair(result['PR_over_PF_'+side+'_exact']);pn,pd=integer_pair(result['saved_PR_total_exact']);gn,gd=integer_pair(result[side]['G_exact'])
            require(rn*pd*gn==rd*pn*gd,'AUDIT_PR_PF_RATIO')
            require(result['PR_PF_relation_'+side]==('PR_LOWER' if rn<rd else 'PF_LOWER' if rn>rd else 'TIE'),
                    'AUDIT_DIRECTION')
        pr=original[result['saved_PR_row_id']]
        require(all(result[k]==pr[v] for k,v in [('saved_PR_candidate','candidate_id'),
            ('saved_PR_total_exact','saved_total_RZ_literal'),('saved_PR_SE_literal','saved_PR_SE_literal'),('saved_PR_cohort','cohort')]),
            'AUDIT_PR_VERBATIM')
        pairs=[];by_id=sorted(selected,key=lambda r:r['candidate_id'])
        for i,a in enumerate(by_id):
            for b in by_id[i+1:]:
                before=ratio_cmp(a['baseline_G_exact'],b['baseline_G_exact']);after=ratio_cmp(a['replacement_G_exact'],b['replacement_G_exact']);tested+=1
                if before!=after:pairs.append((a['candidate_id'],b['candidate_id'],before,after,before*after<0,(before==0)!=(after==0)))
        saved=[tuple(r[k] for k in ('left_candidate','right_candidate','before_relation','after_relation','strict_inversion','tie_change')) for r in change_index[cid]]
        require(saved==pairs,'AUDIT_ALL_PAIR_RELATIONS')
    return dict(status='PASS_INTEGER_COST_SELECTION_RANK_RATIO_AUDIT',weighted_cost_checks=candidate_checks,
                scope_checks=len(conditions),pair_relations_checked=tested,PR_cost_SE_cohort_verbatim=True,
                independent_reviewer=False,second_primary_accounting_pass=False)


def aggregate(new_rows,old_rows):
    require(len({r['condition_id'] for r in new_rows+old_rows})==len(new_rows)+len(old_rows),'DISJOINT70_AND30')
    rows=[dict(r,numeric_origin='VERBATIM_PREVIOUS70') for r in deepcopy(old_rows)]+deepcopy(new_rows)
    return rows,dict(new30=old.summarize(new_rows),previous70=old.summarize(old_rows),
        all100=old.summarize(rows),scope_counts_overlap=True,unique_endpoints=len({ep(r) for r in rows}),
        old70_costs_choices_ratios_or_ranks_recalculated=False,conditional_only=True)
