"""Finite R proposals and normalization-only prioritization, never N/C/G.

metadata_plan is passive. screen_normalization on real saved tail inputs is a
scientific reanalysis needing its own permission. Current tests are artificial.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import pr_r_dependent_n as A
from . import pf_pr_rpe_pipeline as p

B_COMMIT='7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae'
B_PATH='src/trottertracks/rpe_r_selection.py'
B_FUNCTIONS=('build_menu','request_context','_validate','_candidate','coverage')


def b_metadata(root):
    """Unchanged pinned B metadata bodies only; no cost evaluator is exposed."""
    source=subprocess.check_output(['git','show',B_COMMIT+':'+B_PATH],cwd=root)
    tree=ast.parse(source);nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in B_FUNCTIONS:nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED_DELTA' for t in node.targets):nodes.append(node)
    if len(nodes)!=len(B_FUNCTIONS)+1:raise ValueError('Pinned B metadata SOURCE differs')
    scope=dict(p=p,deepcopy=deepcopy)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),B_COMMIT+':'+B_PATH,'exec'),scope)
    return SimpleNamespace(**{k:scope[k] for k in ('build_menu','request_context','coverage')},
        source=dict(commit=B_COMMIT,path=B_PATH,sha256=hashlib.sha256(source).hexdigest()))


def _factors(values):
    if not isinstance(values,(list,tuple)) or not values:raise ValueError('Explicit finite positive scale factors required')
    if any(not isinstance(x,(str,int)) or isinstance(x,bool) for x in values):raise ValueError('Use exact rational strings/integers for factors')
    result=[Fraction(v) for v in values]
    if min(result)<=0 or len(set(result))!=len(result):raise ValueError('Distinct positive factors required')
    return sorted(result)


def metadata_plan(root,base,rounds,*,factors,sampling_law,source,reference_R=None):
    """Structural proposal only: no tail lambda, B, attenuation or N is read."""
    system=base['binding']['system']
    if system not in ('H4','H6'):raise ValueError('Fixed H4/H6 task required')
    allowed=range(13) if system=='H4' else range(4,13)
    if not isinstance(rounds,(list,tuple)) or not rounds or len(set(rounds))!=len(rounds) or any(type(m) is not int or m not in allowed for m in rounds):raise ValueError('Explicit H4 m0..12 / H6 m4..12 rounds required')
    anchor_m=min(allowed);anchor=next(r for r in base['candidates'] if r['m']==anchor_m and p.arm(r['candidate'])=='PR')
    # A checks the pinned scientific task and R admissibility unchanged.
    A.with_round_R(base,{anchor_m:anchor['candidate']['R_outer']},source)
    scales=_factors(factors);extras=reference_R or {}
    if any(type(m) is not int or m not in rounds for m in extras):raise ValueError('Reference R outside explicit rounds')
    R_map={};seeds={};minimum=1 if system=='H4' else 11
    for m in sorted(rounds):
        row=next(r for r in base['candidates'] if r['m']==m and p.arm(r['candidate'])=='PR')
        seed=Fraction(anchor['candidate']['R_outer']*row['q_PF'],anchor['q_PF'])
        values=[row['candidate']['R_outer']]+[max(minimum,math.ceil(seed*f)) for f in scales]
        extra=extras.get(m,[])
        if not isinstance(extra,(list,tuple)) or any(type(R) is not int or R<minimum for R in extra):raise ValueError('Explicit valid integer reference R required')
        R_map[m]=sorted(set(values+list(extra)));seeds[str(m)]=dict(anchor_m=anchor_m,R_times_q_ratio=str(seed),scale_factors=[str(f) for f in scales])
    B=b_metadata(root);menu=B.build_menu(base,R_map,sampling_law=sampling_law);rows=[]
    for row in menu['candidates']:
        c=row['candidate']
        if p.arm(c)!='PR' or row['m'] not in rounds:continue
        old=next(v for v in base['candidates'] if v['m']==row['m'] and p.arm(v['candidate'])=='PR')
        rows.append(dict(system=system,m=row['m'],T=c['T'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],delta=c['delta'],
            formula=c['formula'],prefix=c['prefix'],K=c['K'],R_outer=c['R_outer'],candidate_id=c['id'],
            baseline=c['id']==old['candidate']['id'],structure=deepcopy(row['structure']),request=B.request_context(menu,row),
            normalization=dict(status='NOT_EVALUATED',method='EXISTING_K2_FINITE_DISTRIBUTION',log_B=None,lambda_source=None),
            detailed_N_required=True,priority=None,scientific_status='NOT_EVALUATED',N=None,C=None,G=None,
            unverified=['corrected/raw signal','phase/radius/disk','R-aware allowance baseline parity','actual implementation/law','C full R/q/law domain','R-range completeness']))
    return dict(status='STRUCTURAL_CANDIDATE_PROPOSAL_ONLY',base=deepcopy(base),menu=menu,rows=rows,
        seeds=seeds,source=deepcopy(source),B_metadata_source=B.source,R_upper_bound=None,
        range_complete=False,scientific_R_adopted=False,science_calls=0,mandatory_stop=True)


def screen_normalization(root,proposal,tail,*,shot_inflation_budget,budget_source,finite_distribution=None):
    """Scientific on real inputs. Scores only normalization, never actual N.

    The budget is an explicit caller sensitivity/resource choice, not a PASS
    threshold. Deferred rows remain in full_menu and require N for full scope.
    """
    no_budget=shot_inflation_budget is None
    if not no_budget and (isinstance(shot_inflation_budget,bool) or not isinstance(shot_inflation_budget,(int,float))
            or not math.isfinite(shot_inflation_budget) or shot_inflation_budget<=1 or not budget_source):raise ValueError('Explicit sourced inflation budget >1 required; no default')
    task=proposal['menu'];binding=task['binding'];B=b_metadata(root)
    B.coverage(task,[],[],cohort_selection={})
    expected={r['candidate']['id']:r for r in task['candidates']}
    for row in proposal['rows']:
        actual=expected.get(row['candidate_id'])
        if actual is None or not p.same(row['request'],B.request_context(task,actual)) or not p.same(row['structure'],actual['structure']):raise ValueError('Candidate/law/recipe screening binding differs')
        c=actual['candidate']
        if any(row[k]!=c[k] for k in ('system','T','delta','formula','prefix','K','R_outer')) or any(row[k]!=actual[k] for k in ('m','q_RPE','q_PF')):raise ValueError('Screen row differs from B menu')
    if not p.same(tail['binding'],binding) or not tail.get('source'):raise ValueError('Saved tail binding/SOURCE required')
    lam=tail['lambda_R'];prefix=1 if binding['system']=='H4' else 7
    if isinstance(lam,bool) or not isinstance(lam,(int,float)) or not math.isfinite(lam) or lam<=0 or tail['prefix']!=prefix:raise ValueError('Positive saved tail lambda and fixed prefix required')
    if finite_distribution is None:
        from trotterlib.rte import finite_rte_distribution
        finite_distribution=finite_rte_distribution
    limit=None if no_budget else .5*math.log(shot_inflation_budget);out=deepcopy(proposal);priorities={};all_R={}
    for row in out['rows']:
        st=row['structure'];q=row['q_PF'];times=[lam*row['delta']*w/r for w,r in zip(st['signed_weights'],st['allocation'])]
        normal=row['normalization'];normal['lambda_source']=deepcopy(tail['source'])
        try:
            logs=[];sensitivities=[]
            for tau,r in zip(times,st['allocation']):
                b=finite_distribution(tau,row['K']).exact_finite_distribution
                if tau==0 or not math.isfinite(b) or b<=1:raise ValueError('Unresolved finite-normalizer binary64 precision')
                logs.append(r*math.log(b))
                sensitivities.append(r*(math.log(math.nextafter(b,math.inf))-math.log(math.nextafter(b,0.))))
            logB=q*math.fsum(logs);leading=q*(lam*row['delta'])**2*math.fsum(w*w/r for w,r in zip(st['signed_weights'],st['allocation']))
            if not math.isfinite(logB) or not math.isfinite(leading):raise ValueError('Numerical range unresolved')
            sensitivity=q*math.fsum(sensitivities)
            boundary=not no_budget and abs(logB-limit)<=sensitivity
            above=not no_budget and logB>limit and not boundary
            normal.update(status='NORMALIZATION_SCORED_NOT_N_QUALIFIED',log_B=logB,optimistic_shot_inflation_log=2*logB,
                leading_small_tau_term=leading,leading_term_is_used_to_accept=False,max_abs_event_tau=max(abs(t) for t in times),
                one_ulp_accumulation_log_sensitivity=sensitivity,sensitivity_is_rigorous_bound=False,
                relation_to_explicit_budget='NOT_CLASSIFIED_NO_BUDGET' if no_budget else ('NUMERIC_BOUNDARY_RETAIN' if boundary else ('ABOVE' if above else 'WITHIN')),log_B_budget=limit)
            row['priority']=row['baseline'] or not above
            row['priority_reason']='PRESERVE_BASELINE' if row['baseline'] else ('NORMALIZATION_BAND' if not above else 'DEFER_EXTREME_ATTENUATION_ONLY')
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            normal.update(status='NUMERICAL_NORMALIZATION_UNRESOLVED',reason=str(exc));row['priority']=row['baseline']
            row['priority_reason']='BASELINE_NEEDS_PRECISION_DIAGNOSIS' if row['baseline'] else 'DEFER_NUMERICAL_DIAGNOSIS'
        all_R.setdefault(row['m'],[]).append(row['R_outer'])
        if row['priority']:priorities.setdefault(row['m'],[]).append(row['R_outer'])
    for row in out['rows']:
        row['upper_R_frontier_open']=row['R_outer']==max(all_R[row['m']])
        row['detailed_N_required']=True  # Deferral never creates scientific FAIL/dominance.
    B=b_metadata(root)
    out.update(status='NORMALIZATION_SCREEN_ONLY_NO_N_C_G',priority_menu=B.build_menu(out['base'],priorities,sampling_law=task['sampling_law']),
        screening_policy=dict(shot_inflation_budget=shot_inflation_budget,source=deepcopy(budget_source),mode='SCORES_ONLY_NO_THRESHOLD' if no_budget else 'EXPLICIT_SENSITIVITY_BAND',scientific_acceptance_threshold=False),
        deferred_ids=[r['candidate_id'] for r in out['rows'] if not r['priority']],
        needs_range_extension_review=True,N_C_optimum_known=False)
    return out


def A_request(root,proposal,candidate_id,source):
    """One B menu row -> unchanged A one-R task. Does not produce law evidence."""
    row=next(r for r in proposal['menu']['candidates'] if r['candidate']['id']==candidate_id)
    if p.arm(row['candidate'])!='PR' or row['m'] not in proposal['menu']['variable_R_rounds']:raise ValueError('Explicit variable PR row required')
    a=A.with_round_R(proposal['base'],{row['m']:row['candidate']['R_outer']},source)
    ar=next(r for r in a['task']['candidates'] if r['candidate']['id']==candidate_id)
    request=b_metadata(root).request_context(proposal['menu'],row)
    if not p.same(p.identity(a['task'],ar),request['identity']):raise ValueError('A/B identity differs')
    return dict(A_proposal=a,A_row=ar,B_request=request,law_echo_requires_matching_scientific_evidence=True)
