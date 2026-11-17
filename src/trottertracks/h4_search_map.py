"""Saved H4 search coverage, independent of the scientific solvers and roadmap.

Counts identify evaluations, not jobs, file copies, or execution hosts. Prices
used for minima come only from the existing C comparison consumer.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath

from .saved_rpe_cost_catalog import PASS, FAIL, display, enc, frac

METHODS = ('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF')


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()[:24]


def unique(items):
    return list({json.dumps(x, sort_keys=True): x for x in items if x is not None}.values())


def dimensions(row):
    width = row.get('requested_width', row.get('delta_pf'))
    if width is None:
        wkey, label = 'UNKNOWN', '幅未同定'
    else:
        try:
            unresolved_ratio=False
            if isinstance(width,dict) and 'delta_over_h' in width:
                ratio=frac(width['delta_over_h'])
                if row.get('h') is None:
                    wkey,label='ratio:'+str(ratio),str(ratio)+'h [h未取込]'
                    unresolved_ratio=True;value=ratio
                else:value=ratio*frac(row['h'])
            else:value = Fraction(width) if isinstance(width,float) else frac(width)
            if not unresolved_ratio:
                wkey = str(value)
                h = frac(row.get('h') or '0.1')
                ratio = value / h
                label = (str(ratio) if ratio.denominator<1000000 else format(float(ratio),'.8g')+'≈') + 'h'
            # Exact keys retain binary64 .2 versus exact 1/5; do not alias.
            if not unresolved_ratio and value.denominator > 10**8:
                label += ' [binary64保存値]'
            elif not unresolved_ratio and (not isinstance(width,dict)or'delta_over_h'not in width):
                label += ' [Δ='+str(value)+']'
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            wkey, label = json.dumps(width, sort_keys=True), str(width)
    method = row['method']
    p = row.get('p')
    if p is None and method == 'TRUNCATED_PF':
        split = row.get('split') or {}
        p = split.get('prefix', split.get('L_D', row.get('L_D')))
    formula = row.get('formula')
    if method == 'FULL_PF':
        rkey = str(formula or 'UNKNOWN')
        rlabel = {'2nd': 'S2', '4th': '標準Yoshida4', '4th(m5_best)': 'M5'}.get(formula, rkey)
    else:
        v = p if method == 'TRUNCATED_PF' else row.get('L_D')
        rkey = str(v if v is not None else 'UNKNOWN')
        rlabel = ('p=' if method == 'TRUNCATED_PF' else 'L_D=') + rkey
    return dict(width_key=wkey, width_label=label, row_key=rkey, row_label=rlabel,
                requested_width=deepcopy(width), p=p)


def identity(row, comparison):
    """Prices, run, job, host and SOURCE locator are not evaluation identities."""
    return key([row['method'], row.get('configuration_id'), row.get('N_X'), row.get('N_Y'),
                row.get('adopted_evaluation') or 'UNPUBLISHED_EVALUATION_CONTRACT'])


def evaluation_key(row):
    return (row['method'], row.get('configuration_id'),
            tuple(row.get('N_X') or []), tuple(row.get('N_Y') or []))


def brief_bound(value, source):
    # Long exact numerators remain in the original immutable record, rather
    # than duplicating megabytes or silently treating display bounds as exact.
    if len(json.dumps(value)) > 700:
        return dict(exact_source=source, display='元結果のexact上下界を参照')
    return deepcopy(value)


def saved_schedule(value):
    if value is None:return None
    result=[]
    for r in value:
        item=dict(m=r['m'],q=r.get('q',r.get('q_PF')),R=r.get('R',r.get('R_outer')),
                  physical_id=r.get('canonical_id',r.get('physical_id')))
        for name,source in [('T','T'),('actual_delta','actual_delta'if'actual_delta'in r else'actual_delta_pf')]:
            if r.get(source)is not None:item[name]=enc(frac(r[source]))
        if r.get('structure')is not None:item['structure']=deepcopy(r['structure'])
        result.append(item)
    return result


def _validation_price_table(application, point):
    """Return supplied unit prices only; never predict or dereference a producer."""
    table = {}
    prices = application.get('prices')
    if isinstance(prices, list) and len(prices) == 2 * len(point.get('N_X') or []):
        for i, value in enumerate(prices):
            if value is not None:
                table[i // 2, ('X', 'Y')[i % 2]] = frac(value)
    rows = application.get('price_rows') or (point.get('price_evidence') or {}).get('price_axes') or []
    for row in rows:
        value = next((row[k] for k in ('unit_price_RZ', 'C_ref_exact', 'C_ref', 'unit_price_RZ_exact', 'C_RZ')
                      if row.get(k) is not None), None)
        if value is not None and row.get('axis') in ('X', 'Y') and type(row.get('m')) is int:
            table[row['m'], row['axis']] = frac(value)
    return table


def _validation_view(model, application, point):
    """Allocation-specific display arithmetic on the published price table.

    It is a contribution diagnostic, not a new E[C], G, ranking or model fit.
    Missing prices stay unknown. A changed (q,R) is displayed as a different
    application point even when it uses the same frozen model.
    """
    result = dict(model_id=model['model_id'], status=model.get('status'), label=model.get('label'),
                  application_points=deepcopy(application.get('points') or []),
                  source=application.get('source'), validation_round_contributions=[])
    x, y = point.get('N_X'), point.get('N_Y')
    prices = _validation_price_table(application, point)
    total = None
    if x and y and len(x) == len(y) and all(type(n) is int and n >= 0 for n in x + y):
        if set(prices) == {(m, a) for m in range(len(x)) for a in ('X', 'Y')} and all(v >= 0 for v in prices.values()):
            total = sum(prices[m, a] * n for a, vector in (('X', x), ('Y', y)) for m, n in enumerate(vector))
    for observed in model.get('validation_points') or []:
        m = observed.get('m')
        applied = next((v for v in application.get('points') or [] if v.get('m') == m), None)
        share = None
        if total and type(m) is int and 0 <= m < len(x):
            share = 100 * (x[m] * prices[m, 'X'] + y[m] * prices[m, 'Y']) / total
        result['validation_round_contributions'].append(dict(m=m,
            validation_q=observed.get('q'), validation_R=observed.get('R'),
            application_q=None if applied is None else applied.get('q'),
            application_R=None if applied is None else applied.get('R'),
            same_application_point=bool(applied and all(applied.get(k) == observed.get(k) for k in ('q', 'R'))),
            percent_of_model_total=enc(share) if share is not None else None,
            scope='SUPPLIED_PRICE_TABLE_AND_THIS_ALLOCATION_ONLY_NO_GLOBAL_RESIDUAL_CORRECTION'))
    return result


def _attach_price_validations(points, runs, issues):
    models = {}; applications = {}
    priority = {'CALIBRATION_ONLY': 0, 'PENDING': 1, 'INCOMPLETE': 2, 'INDEPENDENT_MEASURED': 3}
    for run in runs:
        rejected = set()
        document = run.get('price_model_validations') or {}
        source = run.get('price_validation_source')
        for i, row in enumerate(document.get('models') or []):
            mid = row.get('model_id')
            if not mid:
                continue
            candidate = deepcopy(row)
            if source:
                candidate.setdefault('publication_source', dict(source, selector=list(source.get('selector') or []) + ['models', i]))
            old = models.get(mid)
            if old and old.get('identity') != row.get('identity'):
                issues.append(dict(source=source, reason='PRICE_VALIDATION_MODEL_IDENTITY_CONFLICT:' + str(mid)))
                rejected.add(mid)
                continue
            # Shared independent observations are stronger display evidence than
            # an older pending view. This does not change price applicability.
            score = lambda x: (bool(x.get('validation_points')), priority.get(x.get('status'), -1), len(x.get('validation_points') or []))
            if old is None or score(candidate) > score(old):
                models[mid] = candidate
        for application in document.get('applications') or []:
            cid, mid = application.get('configuration_id'), application.get('model_id')
            if cid and mid and mid not in rejected:
                applications.setdefault(cid, {})[mid] = application
    for point in points:
        views = []
        for mid, application in sorted(applications.get(point.get('configuration_id'), {}).items()):
            if mid not in models:
                continue
            try:
                views.append(_validation_view(models[mid], application, point))
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                issues.append(dict(source=application.get('source'), reason='PRICE_VALIDATION_DISPLAY_UNKNOWN:' + str(exc)))
                views.append(dict(model_id=mid, status=models[mid].get('status'), label=models[mid].get('label'),
                    application_points=deepcopy(application.get('points') or []), validation_round_contributions=[]))
        if views:
            point['price_validations'] = views
    return models


def build_index(catalog, runs, *, sources=None, issues=None):
    comparison = catalog['comparison']; points = {}; targets = {}; problems = list(issues or [])
    accepted = {}; configurations={}
    for r in catalog['candidates']:
        if r['method'] not in METHODS:
            continue
        accepted.setdefault(evaluation_key(r), []).append(r)
        configurations.setdefault(r['configuration_id'],r)
    def add(row, occurrence=None, catalog_row=None):
        if row.get('method') not in METHODS:
            return
        r = deepcopy(row)
        r['id'] = identity(r, comparison)
        r.update(dimensions(r))
        r['sources'] = unique(r.get('sources', []) + [r.get('source')])
        r['occurrences'] = [occurrence] if occurrence else []
        r['issues'] = list(r.get('issues') or [])
        r['MSE_lower'] = brief_bound(r.get('MSE_lower'), r.get('source'))
        r['MSE_upper'] = brief_bound(r.get('MSE_upper'), r.get('source'))
        r['q3_evaluated'] = r.get('q3_status') is not None and r.get('N_X') is not None
        # A public summary supports a coverage count. It cannot bypass C's
        # existing physical/price/evaluation binding checks for a cost minimum.
        if catalog_row is None:
            r['saved_summary_cost'] = r.pop('conditional_cost', None)
            r['conditional_cost'] = None
            r['status'] = ('PRICE_WAITING' if r.get('q3_status') == PASS else
                           'INFEASIBLE' if r.get('q3_status') == FAIL else
                           'PENDING' if r['q3_evaluated'] else 'Q3_WAITING')
            r['cost_admission'] = 'SUMMARY_ONLY_NOT_C_PRICE_ADMITTED'
            r['evidence_level'] = 'PUBLIC_CONTROLLER_SUMMARY'
        else:
            r['cost_admission'] = 'EXISTING_C_CATALOG'
            r['evidence_level'] = 'C_ACCEPTED_BINDING'
        if r['id'] in points:
            old = points[r['id']]
            if old.get('q3_status') and r.get('q3_status') and old['q3_status'] != r['q3_status']:
                old['issues'].append('Conflicting saved classification for same evaluation')
                old.update(status='PENDING', conditional_cost=None)
            if catalog_row is not None and not old.get('conditional_cost'):
                for k in ('conditional_cost', 'status', 'price_evidence', 'price_axes', 'cost_admission'):
                    if k in r: old[k] = r[k]
            old['occurrences'] = unique(old['occurrences'] + r['occurrences'])
            old['sources'] = unique(old['sources'] + r['sources'])
        else:
            points[r['id']] = r
    # Canonical evaluated contract and physical recipe are read by C once.
    for r in catalog['candidates']:
        if r['method'] not in METHODS: continue
        row = {k: deepcopy(r.get(k)) for k in ('method', 'configuration_id', 'formula', 'L_D', 'split',
            'h', 'kappa', 'delta_pf', 'N_X', 'N_Y', 'q3_status', 'status', 'MSE_lower', 'MSE_upper',
            'conditional_cost', 'adopted_evaluation', 'schedule', 'signal_source', 'binding_id',
            'total_configuration_id', 'issues', 'profile', 'law')}
        row.update(source=(r.get('result_sources') or [r.get('configuration_source')])[0],
            sources=r.get('result_sources', []), configuration_source=r.get('configuration_source'),
            price_evidence=dict(price_axes=r.get('price_axes', []), model_coverage_axes=r.get('model_coverage_axes', []),
                qualified_for_candidate=r.get('qualified_for_candidate'), scenario=r.get('price_scenario'),
                source=r.get('price_evidence'), catalog_point_id=r.get('point_id')))
        add(row, catalog_row=r)
    clean_runs = []
    for run in runs:
        rid = run['id']; run_issues = list(run.get('missing', []))
        run_issues+=list(run.get('issues',[]))
        for reason in run_issues:problems.append(dict(source=run['source'],reason=reason))
        for t in run.get('targets', []):
            if t.get('method') not in METHODS: continue
            binding=t.get('physical_binding')
            if binding and any(binding.get(k)!=v for k,v in comparison.get('binding',{}).items()):continue
            t = deepcopy(t)
            known=configurations.get(t.get('configuration_id'))
            if known:
                for k in ('formula','L_D','split','h','delta_pf','schedule','kappa'):
                    if t.get(k)is None:t[k]=deepcopy(known.get(k))
                if t.get('requested_width')is None:t['requested_width']=deepcopy(known.get('delta_pf'))
            t.update(dimensions(t)); t['run_id'] = rid
            t['id'] = key([rid, t['method'], t.get('configuration_id'), t['row_key'], t['width_key'], t.get('formula')])
            targets[t['id']] = t
        for e in run.get('evaluations', []):
            if e.get('method')not in METHODS:continue
            if e.get('controller_adopted')is False:
                problems.append(dict(source=e.get('source'),reason='WORKER_OR_UNADOPTED_RESULT_NOT_COUNTED'))
                continue
            binding=e.get('physical_binding')
            if binding and any(binding.get(k)!=v for k,v in comparison.get('binding',{}).items()):
                problems.append(dict(source=e.get('source'),reason='DIFFERENT_HAMILTONIAN_STATE_OR_SNAPSHOT_NOT_JOINED'))
                continue
            matches = accepted.get(evaluation_key(e), [])
            # A unique catalog record has the checked contract/signal/view.
            # Ambiguous joins are kept separate and explicitly reported.
            choices = {identity(r, comparison): r for r in matches}
            chosen = next(iter(choices.values())) if len(choices) == 1 else None
            merged = deepcopy(e)
            known=configurations.get(e.get('configuration_id'))
            if known:
                for k in ('formula','L_D','delta_pf','h','schedule','kappa','split'):
                    if merged.get(k)is None:merged[k]=deepcopy(known.get(k))
                if merged.get('requested_width')is None:merged['requested_width']=deepcopy(known.get('delta_pf'))
            if chosen:
                if e.get('adopted_evaluation')and e['adopted_evaluation']!=chosen['adopted_evaluation']:
                    chosen=None;merged.setdefault('issues',[]).append('EVALUATION_CONTRACT_MISMATCH')
            if chosen:
                for k in ('adopted_evaluation', 'signal_source', 'binding_id', 'total_configuration_id',
                          'formula', 'L_D', 'delta_pf', 'h', 'schedule', 'kappa', 'split'):
                    merged[k] = deepcopy(chosen.get(k))
            elif len(choices) > 1:
                merged.setdefault('issues', []).append('Ambiguous C evaluation-contract join')
            add(merged, dict(run_id=rid, new_evaluation=e.get('new_evaluation'),
                source=e.get('source'), origin=e.get('origin'), job=e.get('job'),
                outcome_sha256=e.get('outcome_sha256'), hosts=e.get('hosts') or run.get('hosts', [])))
        saved_run={k: deepcopy(v) for k,v in run.items() if k not in ('targets', 'evaluations')}
        saved_run.pop('price_model_validations', None)  # shared once in INDEX.price_validation_models
        saved_run['missing']=run_issues
        clean_runs.append(saved_run)
    points = sorted(points.values(), key=lambda r:r['id']); targets = sorted(targets.values(), key=lambda r:r['id'])
    price_validation_models = _attach_price_validations(points, runs, problems)
    details={}
    for p in sorted(points,key=lambda p:p['cost_admission']!='EXISTING_C_CATALOG'):
        k=p['method']+':'+str(p.get('configuration_id'))
        detail=details.setdefault(k,{})
        for name in ('schedule','profile','law','split','physical_binding','configuration_source'):
            value=p.pop(name,None)
            if value is not None:
                if name=='schedule':value=saved_schedule(value)
                if name=='schedule'and name in detail:
                    def physical(schedule):return [{k:v for k,v in s.items()if k!='structure'}for s in schedule]
                    if physical(detail[name])==physical(value):
                        for old,new in zip(detail[name],value):
                            if 'structure'in new:old.setdefault('structure',new['structure'])
                        continue
                if name in detail and detail[name]!=value:
                    if name=='schedule':p['issues'].append('CONFLICTING_PHYSICAL_SCHEDULE')
                    detail.setdefault('metadata_variants',[])
                    variant=dict(field=name,value=value)
                    if variant not in detail['metadata_variants']:detail['metadata_variants'].append(variant)
                else:detail[name]=value
        p['configuration_detail_key']=k
    views = [make_view('all', '全run累計（重複なし）', points, targets)]
    for run in clean_runs:
        rid=run['id']
        for mode in ('RUN','NEW'):
            selected=[p for p in points if any(o['run_id']==rid and
                (mode=='RUN' or o.get('new_evaluation') is True) for o in p['occurrences'])]
            rt=[t for t in targets if t['run_id']==rid]
            views.append(make_view(mode.lower()+':'+rid, run['label']+('：採用・再利用を含む' if mode=='RUN' else '：このrunの新規評価のみ'),
                selected,rt,run_id=rid,mode=mode))
    return dict(schema='H4_SEARCH_MAP_V1', comparison=deepcopy(comparison), sources=sources or [],
        runs=clean_runs, points=points,configuration_details=details, targets=targets, views=views, issues=problems,
        price_validation_models=price_validation_models,
        count_policy=dict(evaluation='method/configuration/decoder-contract/X-Y-vector; unknown contract marked, jobs/hosts/copies not counted',
            new='Explicit NEW_Q3_JOB or controller-owned evaluation index; SAVED/PRIOR_ROUND_JOB excluded; unknown not counted as new',
            minima='Only existing C-admitted conditional costs, no new price/RMSE evaluation',
            scope='Published saved candidate set only; gaps do not prove scientific inferiority'),
        new_scientific_calls=0)


def make_view(vid,label,points,targets,run_id=None,mode='ALL'):
    cells={}
    for r in points+targets:
        if r['method'] not in METHODS:continue
        k=(r['method'],r['row_key'],r['width_key'])
        if k not in cells:
            cells[k]={s:r[s] for s in ('method','row_key','row_label','width_key','width_label')}
            cells[k].update(point_ids=[],target_ids=[])
        cells[k]['point_ids' if 'occurrences'in r else 'target_ids'].append(r['id'])
    pmap={p['id']:p for p in points};tmap={t['id']:t for t in targets}
    for c in cells.values():
        ps=[pmap[i]for i in set(c['point_ids'])];ts=[tmap[i]for i in set(c['target_ids'])]
        ev=[p for p in ps if p['q3_evaluated']]
        feasible=[p for p in ev if p.get('q3_status')==PASS]
        priced=[p for p in feasible if p.get('conditional_cost') is not None and p.get('status')=='FEASIBLE']
        cost=min((frac(p['conditional_cost'])for p in priced),default=None)
        missing=sum(t.get('status')=='SOURCE_NOT_IMPORTED' or any(
            any(word in str(reason)for word in ('NOT_PUBLIC','NOT_IMPORTED','SOURCE_MISSING','OUTCOME_NOT','RESULT_NOT'))
            for reason in t.get('missing',[])) for t in ts)
        c.update(physical_evaluated=len({p['configuration_id']for p in ev}), evaluated_allocations=len(ev),
            feasible_allocations=len(feasible),min_cost=enc(cost)if cost is not None else None,
            min_point_ids=[p['id']for p in priced if frac(p['conditional_cost'])==cost],
            price_validation_labels=sorted({v.get('label') or v.get('status') for p in ps
                for v in p.get('price_validations', []) if v.get('label') or v.get('status')}),
            missing_count=missing,target_count=len(ts),
            state='MIXED'if missing and ev else'SOURCE_NOT_IMPORTED'if missing else
                'EVALUATED_NO_FEASIBLE'if ev and not feasible else'EVALUATED'if ev else'TARGET_UNEVALUATED',
            point_ids=sorted(set(c['point_ids'])),target_ids=sorted(set(c['target_ids'])))
    return dict(id=vid,label=label,run_id=run_id,mode=mode,cells=sorted(cells.values(),key=lambda c:(c['method'],c['row_key'],c['width_key'])))


def report(index):
    lines=['# H₄ 保存探索マップ','',
        '[閲覧用HTML（保存してブラウザで開く）](index.html) ／ [機械可読索引](INDEX.json) ／ [固定入力](SOURCES.json) ／ [運用・公開手順](HANDOFF.md)','',
        'Full PF・B0I・PR-Pauli・PR-DFについて、公開した探索対象と実評価を表示する。',
        '結論の範囲は**収録した保存候補集合内**。ロードマップとは別のcoverage成果物であり、新しい科学資格・最安幅・winnerを採択しない。','',
        '|方式|Q3評価済み構成|評価配分（重複なし）|保存判定が適合|価格接続済み適合点の最安RZ|',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        ps=[p for p in index['points']if p['method']==method and p['q3_evaluated']]
        eligible=[p for p in ps if p.get('status')=='FEASIBLE'and p.get('conditional_cost')is not None]
        value=min((frac(p['conditional_cost'])for p in eligible),default=None)
        lines.append('|'+method+'|'+str(len({p['configuration_id']for p in ps}))+'|'+str(len(ps))+'|'+
            str(sum(p.get('q3_status')==PASS for p in ps))+'|'+display(value)+'|')
    lines+=['','公開controller summaryで評価された配分を数える。exact原本や価格のC側受入が不足する点は、',
        'summary由来の表示と元SOURCEを残し、最安価格の選択には昇格しない。共通snapshot・比較条件はINDEX.comparison。','',
        '同一構成・配分・評価契約の再同期や別run再利用は累計で1件。新規表示は元NEW_Q3_JOBかcontroller自身の取得記録がある点だけ。',
        '要求幅の厳密表現を保持し、binary64 0.2と厳密1/5を丸めて統合しない。unknownはunknownのまま。','',
        '|収録run|停止・進行状態|そのrunの新規評価|実行host（公開metadata）|固定結果|','|---|---|---:|---|---|']
    for r in index['runs']:
        v=next(v for v in index['views']if v['id']=='new:'+r['id'])
        count=sum(c['evaluated_allocations']for c in v['cells'])
        src=r['source'];url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+src['commit']+'/'+src['path']
        lines.append('|'+r['label'].replace('|','/')+'|'+json.dumps(r.get('status'),ensure_ascii=False).replace('|','/')+'|'+str(count)+'|'+
            ', '.join(r.get('hosts',[]))+'|['+src['commit'][:9]+']('+url+')|')
    lines+=['','## 未取込・未確認の範囲','']
    if not index['issues']:lines.append('parse不能な公開索引なし。各点のcontract・価格・scheduleの不足はINDEX.points[*].issuesに保持。')
    for issue in index['issues']:lines.append('- '+str(issue.get('reason'))+'：`'+str(issue.get('source'))+'`')
    lines+=['','最新公開scopeの進捗件数だけからshot配分やQ3点を捏造しない。各マスで「対象・未評価」「評価済み適合0」「元記録未取込」を区別。',
        '新規信号・sampling・compile・fit・Q3は全0件。既存の10分同期で更新する。検査と公開方法はHANDOFFを参照。','']
    return '\n'.join(lines)
