"""Exact saved-shot PF accounting. No signal, shot, model, fit or compile API."""
from collections import Counter, defaultdict
from fractions import Fraction as F

from h8_six_axis_reaccount import minimum

AXES = ('cosine', 'sine')
ENDPOINT = ('system', 'bias_mode', 'epsilon', 'allowance_factor')


def endpoint(row):
    return tuple(row[k] for k in ENDPOINT)


def condition_id(row):
    return '|'.join(endpoint(row)+(row['original_scope'],))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def authorize(contract, grant, *, synthetic=False):
    if synthetic:
        require(contract.get('kind') == 'ARTIFICIAL', 'ARTIFICIAL_ONLY')
        return
    require(grant.get('scientific_execution_authorized') is True and
            grant.get('strategy') == contract.get('strategy') == 'ALL_SAVED458' and
            contract.get('scope_conditions') == 70 and
            grant.get('SOURCE') == contract.get('SOURCE'), 'NEW_SEALED_70_CONDITION_GRANT_REQUIRED')


def weighted(shots, prices):
    require(set(shots) == set(prices) == set(AXES), 'COMPLETE_PAIR_REQUIRED')
    require(all(type(v) is int and v > 0 for v in shots.values()), 'POSITIVE_SAVED_INTEGER_SHOTS')
    require(all(v is not None and F(v) > 0 for v in prices.values()), 'POSITIVE_PRICE_NO_ZERO_FILL')
    return sum(shots[a]*F(prices[a]) for a in AXES)


def replay_pf(rows, units, selected_endpoints):
    """Complete baseline gate before any direct replacement is evaluated."""
    result=[]
    for row in rows:
        if row['kind'] != 'PF' or endpoint(row) not in selected_endpoints:
            continue
        require(row['eligible'] is not None, 'UNRESOLVED_BASELINE_ELIGIBILITY:'+row['row_id'])
        result_row={k:row[k] for k in ('row_id', *ENDPOINT, 'candidate_id', 'method', 'prefix', 'q', 'PF',
            'eligible', 'saved_shots', 'saved_total_RZ_literal', 'source_id', 'source_locator', 'source_file_sha256')}
        result_row.update(baseline_G_exact=None, replacement_G_exact=None,
                          baseline_replay_matches=None, unit_provenance={})
        if row['eligible']:
            prices={}
            for axis in AXES:
                unit=units.get(row['unit_ids'].get(axis))
                require(unit is not None, 'MISSING_BASELINE_AXIS:'+row['row_id'])
                prices[axis]=unit['baseline_RZ_literal']
                result_row['unit_provenance'][axis]=unit['task_id']
            total=weighted(row['saved_shots'], prices)
            require(row['saved_total_RZ_literal'] is not None and total == F(row['saved_total_RZ_literal']),
                    'BASELINE_COST_MISMATCH:'+row['row_id'])
            result_row.update(baseline_G_exact=str(total), baseline_replay_matches=True,
                              baseline_axis_RZ_exact=prices)
        result.append(result_row)
    require(len({r['row_id'] for r in result}) == len(result), 'DUPLICATE_BASELINE_ROW')
    return result


def choose(rows, field):
    chosen=minimum([dict(cell_id=r['candidate_id'], eligible=r['eligible'], G_exact=r[field]) for r in rows])
    require(chosen['best'] is not None, 'NO_ELIGIBLE_PF_IN_APPROVED_CONDITION')
    cid=chosen['best']['cell_id']
    record=next(r for r in rows if r['candidate_id'] == cid)
    return dict(candidate_id=cid, G_exact=chosen['best']['G_exact'], exact_ties=chosen['exact_ties'],
                **{k:record[k] for k in ('method', 'q', 'prefix', 'PF')})


def saved_comparison_key(row):
    return endpoint(row)+(row['scope'],)


def baseline_gate(costs, saved_rows, comparisons, conditions):
    cost_index=defaultdict(list); pr_index={}; comparison_index={saved_comparison_key(r):r for r in comparisons}
    for row in costs:
        cost_index[endpoint(row)].append(row)
    selected_endpoints={endpoint(c) for c in conditions}
    for row in saved_rows:
        if row['kind'] == 'PR' and endpoint(row) in selected_endpoints:
            k=endpoint(row)+(row['candidate_id'],)
            require(k not in pr_index, 'DUPLICATE_SAVED_PR'); pr_index[k]=row
    result=[]
    for c in conditions:
        key=endpoint(c)+(c['original_scope'],)
        saved=comparison_index[key]['saved_result']
        candidate_ids=set(c['candidate_ids'])
        selected=[r for r in cost_index[endpoint(c)] if r['candidate_id'] in candidate_ids]
        require({r['candidate_id'] for r in selected} == candidate_ids, 'SAVED_CANDIDATE_DOMAIN_CHANGED')
        best=choose(selected, 'baseline_G_exact')
        expected=saved['deterministic'] if c['system'] == 'H6_DF19' else saved['after']
        require(best['candidate_id'] == expected['best']['cell_id'] and
                F(best['G_exact']) == F(expected['best']['G_exact']) and
                best['exact_ties'] == expected['exact_ties'], 'SAVED_BASELINE_WINNER_OR_TIES_MISMATCH:'+condition_id(c))
        pr_saved=saved['confirmed_PR'] if c['system'] == 'H6_DF19' else saved['saved_PR']
        pr_id=pr_saved['best']['cell_id']; pr=pr_index[endpoint(c)+(pr_id,)]
        require(pr['eligible'] is True and pr['paired_samples'] == 32 and
                pr['saved_total_RZ_literal'] is not None and
                F(pr['saved_total_RZ_literal']) == F(pr_saved['best']['G_exact']), 'SAVED_PR_BINDING_MISMATCH')
        saved_se=pr_saved['best'].get('saved_paired_SE', pr_saved['best'].get('G_standard_error'))
        if saved_se is not None:
            require(pr['saved_PR_SE_literal'] == saved_se, 'SAVED_PR_SE_MISMATCH')
        require(pr['saved_PR_SE_literal'] is not None, 'SAVED_PR_SE_REQUIRED')
        # This validates a saved choice; no PR mean, SE, shots or choice is recalculated.
        for (ep_sys,ep_bias,ep_eps,ep_factor,cid), peer in pr_index.items():
            if (ep_sys,ep_bias,ep_eps,ep_factor) != endpoint(c):
                continue
            require(peer['eligible'] is not None, 'UNRESOLVED_SAVED_PR')
            if peer['eligible'] is True:
                require(peer['saved_total_RZ_literal'] is not None and
                        (F(peer['saved_total_RZ_literal']),cid) >= (F(pr['saved_total_RZ_literal']),pr_id), 'SAVED_PR_CHOICE_CONFLICT')
        ratio=F(pr['saved_total_RZ_literal'])/F(best['G_exact'])
        if c['system'] == 'H6_DF19':
            require(ratio == F(saved['PR_over_deterministic_exact']), 'SAVED_BASELINE_RATIO_MISMATCH')
        else:
            require(100*(1-ratio) == F(saved['reduction_after_exact']), 'SAVED_BASELINE_REDUCTION_MISMATCH')
        result.append(dict(condition_id=condition_id(c), baseline=best, saved_PR=pr,
            PR_over_PF_baseline_exact=str(ratio), baseline_saved_winner_reproduced=True,
            candidate_domain_sha256=None))
    return result


def replace_costs(costs, units):
    used=set()
    for row in costs:
        if row['eligible'] is not True:
            continue
        prices={}
        for axis, tid in row['unit_provenance'].items():
            unit=units[tid]
            require(unit['ALL_SAVED458'] in ('REPLACE_NEW450','REPLACE_EXISTING_EXTRA8','RETAIN_OLD_MEASURED'),
                    'MISSING_SAVED_DIRECT_PRICE:'+tid)
            prices[axis]=unit['replacement_RZ_literal']
            if unit['ALL_SAVED458'] == 'RETAIN_OLD_MEASURED':
                require(prices[axis] == unit['baseline_RZ_literal'], 'OLD_MEASURED_UNIT_CHANGED')
            used.add(tid)
        total=weighted(row['saved_shots'], prices)
        row.update(replacement_G_exact=str(total), replacement_axis_RZ_exact=prices,
            total_change_RZ_exact=str(total-F(row['baseline_G_exact'])),
            total_change_percent_exact=str(100*(total/F(row['baseline_G_exact'])-1)))
    return used


def ordering(rows, field):
    valid=[r for r in rows if r['eligible'] is True]
    ordered=sorted(valid,key=lambda r:(F(r[field]),r['candidate_id']))
    result={}; block_rank=0;last=None
    for ordinal, row in enumerate(ordered,1):
        value=F(row[field])
        if last is None or value != last:
            block_rank=ordinal
        result[row['candidate_id']]=dict(ordinal=ordinal, cost_tie_rank=block_rank, G_exact=row[field])
        last=value
    return result


def pair_changes(old, new):
    require(set(old) == set(new), 'ELIGIBLE_RANK_DOMAIN_CHANGED')
    ids=sorted(old);changes=[];tested=0
    def sign(value): return (value>0)-(value<0)
    for i,left in enumerate(ids):
        for right in ids[i+1:]:
            a=sign(F(old[left]['G_exact'])-F(old[right]['G_exact']))
            b=sign(F(new[left]['G_exact'])-F(new[right]['G_exact']))
            tested+=1
            if a != b:
                changes.append(dict(left_candidate=left,right_candidate=right,before_relation=a,after_relation=b,
                    strict_inversion=a*b<0,tie_change=(a==0)!=(b==0)))
    return changes, tested


def relation(ratio):
    return 'PR_LOWER' if ratio<1 else 'PF_LOWER' if ratio>1 else 'TIE'


def compare(costs, conditions, gates):
    by_endpoint=defaultdict(list);gate_index={g['condition_id']:g for g in gates}
    for row in costs:
        by_endpoint[endpoint(row)].append(row)
    results=[];ranks=[];changes=[]
    for c in conditions:
        cid=condition_id(c);gate=gate_index[cid]
        rows=[r for r in by_endpoint[endpoint(c)] if r['candidate_id'] in c['candidate_ids']]
        old=gate['baseline']; new=choose(rows,'replacement_G_exact'); pr=gate['saved_PR']
        old_order=ordering(rows,'baseline_G_exact');new_order=ordering(rows,'replacement_G_exact')
        pairwise, tested=pair_changes(old_order,new_order)
        base_ratio=F(gate['PR_over_PF_baseline_exact']);direct_ratio=F(pr['saved_total_RZ_literal'])/F(new['G_exact'])
        common=dict(condition_id=cid,**{k:c[k] for k in (*ENDPOINT,'original_scope')})
        row=dict(common, strategy='ALL_SAVED458',domain_kind='ORIGINAL_REGISTERED_SCOPE',
            baseline=old,replacement=new,saved_PR_candidate=pr['candidate_id'],
            saved_PR_total_exact=pr['saved_total_RZ_literal'],saved_PR_SE_literal=pr['saved_PR_SE_literal'],
            saved_PR_cohort=pr['cohort'],saved_PR_row_id=pr['row_id'],
            PR_over_PF_baseline_exact=str(base_ratio),PR_over_PF_replacement_exact=str(direct_ratio),
            PR_over_PF_baseline=float(base_ratio),PR_over_PF_replacement=float(direct_ratio),
            PR_PF_relation_baseline=relation(base_ratio),PR_PF_relation_replacement=relation(direct_ratio),
            PR_PF_relation_changed=relation(base_ratio)!=relation(direct_ratio),
            best_candidate_changed=old['candidate_id']!=new['candidate_id'],q_changed=old['q']!=new['q'],
            prefix_changed=old['prefix']!=new['prefix'],PF_changed=old['PF']!=new['PF'],
            winner_exact_ties_changed=old['exact_ties']!=new['exact_ties'],
            winner_total_changed=F(old['G_exact'])!=F(new['G_exact']),
            winner_total_change_percent_exact=str(100*(F(new['G_exact'])/F(old['G_exact'])-1)),
            ordering_changed=list(old_order)!=list(new_order),
            strict_pairwise_inversions=sum(p['strict_inversion'] for p in pairwise),
            pairwise_tie_changes=sum(p['tie_change'] for p in pairwise),pairwise_comparisons=tested,
            baseline_saved_result_matches=True,registered_PF_candidates=len(rows),
            eligible_PF_candidates=len(old_order),conditional_comparison=True,
            old_profile_pending_axes_in_scope=c['old_profile_pending_axes_in_scope'])
        results.append(row)
        for candidate in old_order:
            ranks.append(dict(common,candidate_id=candidate,baseline=old_order[candidate],replacement=new_order[candidate]))
        changes.extend(dict(common,**p) for p in pairwise)
    return results,ranks,changes


def summarize(rows):
    flags=('best_candidate_changed','q_changed','prefix_changed','PF_changed','winner_total_changed',
           'PR_PF_relation_changed','ordering_changed','winner_exact_ties_changed')
    return dict(scope_conditions=len(rows),**{k:sum(r[k] for r in rows) for k in flags},
        strict_pairwise_inversions=sum(r['strict_pairwise_inversions'] for r in rows),
        pairwise_tie_changes=sum(r['pairwise_tie_changes'] for r in rows),
        conditions_with_strict_inversion=sum(r['strict_pairwise_inversions']>0 for r in rows),
        conditions_with_tie_change=sum(r['pairwise_tie_changes']>0 for r in rows),
        relation_before=dict(Counter(r['PR_PF_relation_baseline'] for r in rows)),
        relation_after=dict(Counter(r['PR_PF_relation_replacement'] for r in rows)))
