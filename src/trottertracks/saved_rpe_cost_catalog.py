"""Read-only catalog of published Q3 outcomes and their saved conditional costs.

No signal, probability, price, allocation or risk evaluation is performed here.
Supported producer summaries are normalized to one small comparison record.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess
import sys

from .baseline_signal_pipeline import blob

# Published exact Q3 bounds contain integers longer than Python's default limit.
# This is process-local JSON I/O, as in B2's saved-result summarizer.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
LABELS = {'FEASIBLE': '適合', 'INFEASIBLE': 'モデル不適合', 'Q3_WAITING': 'Q3待ち',
          'PRICE_WAITING': 'Q3適合・価格待ち', 'PENDING': '判定保留', 'MISMATCH': '条件・対応不一致'}


def frac(v):
    return Fraction(int(v['numerator']), int(v['denominator'])) if isinstance(v, dict) else Fraction(str(v))


def enc(v):
    v = frac(v)
    return dict(numerator=v.numerator, denominator=v.denominator)


def saved_bound(value):
    """Preserve exact bounds while keeping JSON readable by default Python parsers."""
    if isinstance(value,dict) and 'numerator'in value:
        return dict(numerator=str(value['numerator']),denominator=str(value['denominator']))
    return value


def display(v):
    if v is None:
        return '—'
    with localcontext() as ctx:
        ctx.prec = 30
        v = frac(v)
        return f'{Decimal(v.numerator) / Decimal(v.denominator):,.2f}'


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(commit, path, *selector):
    return dict(commit=commit, path=path, selector=list(selector))


class Reader:
    """Reuse the existing Git-blob reader; cache each published document once."""
    def __init__(self, root):
        self.root = root

    @cache
    def commit(self, value):
        return subprocess.check_output(['git', 'rev-parse', value+'^{commit}'], cwd=self.root, text=True).strip()

    @cache
    def document(self, commit, path):
        return json.loads(blob(self.root, self.commit(commit), path))

    def read(self, locator):
        value = self.document(locator['commit'], locator['path'])
        for key in locator.get('selector', []):
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value


def shots(allocation):
    rows = allocation['shots'] if isinstance(allocation, dict) else allocation
    require(len(rows) == 13 and {int(r['m']) for r in rows} == set(range(13)), '13 unique allocation rounds required')
    rows = sorted(rows, key=lambda r: int(r['m']))
    result = {a: [int(r['N_'+a]) for r in rows] for a in ('X', 'Y')}
    require(all(type(n) is int and n >= 0 for v in result.values() for n in v), 'Invalid saved shot vector')
    return result


def case_ref(case):
    return case.get('configuration') or case.get('original_source') or case['configuration_source']


def case_for(cases, allocation, label=None):
    same = [c for c in cases if shots(c['allocation']) == shots(allocation)]
    if isinstance(allocation,dict) and allocation.get('allocation_id'):
        identified=[c for c in same if c['allocation'].get('allocation_id')==allocation['allocation_id']]
        if identified:return identified[0]
    if label:
        exact = [c for c in same if c.get('label') == label]
        if exact:
            return exact[0]
    require(len(same) == 1, 'Saved case/allocation is not unique')
    return same[0]


class Catalog:
    def __init__(self, reader, comparison):
        self.r = reader
        self.comparison = comparison
        self.rows = []
        self.visited = set()
        self.inputs = []

    def add(self, locator):
        commit = self.r.commit(locator['commit']); path = locator['path']
        if (commit, path) in self.visited:
            return
        self.visited.add((commit, path)); locator = ref(commit, path)
        self.inputs.append(locator)
        if path.endswith('.md'):
            text = blob(self.r.root, commit, path).decode()
            links = re.findall(r'\]\(([^)]+(?:SUMMARY|C_RESULT_JOIN|C_HANDOFF)\.json)\)', text)
            # The usual HANDOFF links use bare relative filenames.
            links += re.findall(r'\]\((SUMMARY\.json|C_RESULT_JOIN\.json|C_HANDOFF\.json)\)', text)
            require(bool(links), 'HANDOFF has no supported saved-result index: '+path)
            for link in set(links):
                if not link.startswith(('http:', 'https:')):
                    self.add(ref(commit, str(PurePosixPath(path).parent/link)))
            return
        d = self.r.read(locator)
        parent = str(PurePosixPath(path).parent)
        if path.endswith('C_RESULT_JOIN.json') and d.get('price_evaluations')==0 and 'rows' in d:
            self.q3_without_prices(locator,d);return
        if isinstance(d.get('summary'), str):
            self.add(ref(commit, d['summary'])); return
        if path.endswith('C_RESULT_JOIN.json') and ('candidates' in d or 'price_terms' in d):
            self.add(ref(commit, parent+'/SUMMARY.json')); return
        if d.get('schema') == 'PR_DF_LD_SPLIT_SIX_ENTRY_Q3_COMPARISON_V1':
            self.pr_df(locator, d)
        elif d.get('schema')=='PR_DF_THREE_SPLITS_SAME_V0_Q3_V1':
            inp=self.r.document(commit,parent+'/Q3_INPUTS.json')['inputs']
            case=inp['cases'][0]
            own=next(r for r in d['rows']if r['L_D']==self.r.read(case['configuration'])['spec']['L_D'])
            require(shots(own['shots'])==shots(case['allocation']),'LD7 V0 saved Q3 allocation mismatch')
            self.make(dict(case,_source=ref(commit,parent+'/Q3_INPUTS.json','inputs')),case['allocation'],
                own['status'],None,own['MSE_lower'],own['MSE_upper'],inp['adopted_evaluation'],locator,[])
            self.rows[-1]['q3_source']=own['Q3_source']
        elif d.get('kind') == 'CONDITIONAL_FIXED_ALLOCATION_COST_WITH_MODEL_REFERENCE_PRICES':
            self.pauli(locator, d)
        elif d.get('schema') == 'FULL_H_ZERO_ZZ_ADOPTED_EMPIRICAL_MODEL_SCENARIO_V1':
            self.full_h(locator, d)
        elif d.get('kind') == 'A_PAULI_LD3_ALL_ROUND_SIGNAL_HANDOFF':
            for i, packet in enumerate(d['packets']):
                configuration=d.get('configuration_source') or packet['configuration']
                canonical=self.r.read(configuration)if 'commit'in configuration else configuration
                require(canonical['id']==packet['configuration']['id'],'Signal packet/configuration mismatch')
                require([r['candidate_id']for r in packet['rounds']]==[r['candidate']['id']for r in canonical['rounds']],
                        'Signal packet physical round mismatch')
                self.make(dict(configuration=configuration,signal_packet=ref(commit,path,'packets',i)), None, None, None, None, None, {},
                          ref(commit,path,'packets',i), [], signal_only=True)
        elif 'conditional_minima' in d and 'adopted_evaluation' in d:
            self.pf(locator, d)
        elif 'achieved_conditional_cost_RZ' in d and 'rows' in d:
            self.p7(locator, d)
        else:
            raise ValueError('Unsupported saved index schema: '+commit+':'+path)

    def make(self, case, allocation, status, cost, lower, upper, adopted, origin, terms, signal_only=False):
        config_locator = case_ref(case)
        conf = self.r.read(config_locator) if 'commit' in config_locator else config_locator
        spec = conf['spec']; binding = spec['binding']
        method = 'PR_PAULI' if (spec.get('split') or {}).get('tail_dictionary') else (
            'FULL_H' if spec.get('L_D') == 0 else 'PR_DF')
        if spec['method'] == 'PF':
            method = 'FULL_PF' if spec['split']['deterministic_scope'] == 'FULL_H' else 'TRUNCATED_PF'
        vector = shots(allocation) if allocation is not None else None
        row = dict(method=method, label=case.get('label') or conf['id'], configuration_id=conf['id'],
            labels=[case.get('label') or conf['id']],
            total_configuration_id=case.get('total_configuration_id') or case.get('total_view_id'),
            binding_id=case.get('binding_id'), formula=spec['formula'], delta_pf=spec.get('delta_pf'),
            L_D=spec.get('L_D'), finite_K=spec.get('finite_K'), kappa=spec.get('kappa'),
            split=deepcopy(spec.get('split')),h=deepcopy(spec['h']),M=spec['M'],
            profile=deepcopy(spec['profile']), law=deepcopy(spec.get('law')), allocation=deepcopy(allocation),
            N_X=vector['X'] if vector else None,N_Y=vector['Y'] if vector else None,
            q3_status=status,MSE_lower=saved_bound(lower),MSE_upper=saved_bound(upper),conditional_cost=enc(cost) if cost is not None else None,
            assessment_basis='SAVED_CLASSIFICATION_NOT_REEVALUATED',
            q3_failure_kind='MODEL_THRESHOLD_EXCEEDED_NOT_NUMERICAL_FAILURE' if status==FAIL else None,
            status='Q3_WAITING' if status is None else 'PRICE_WAITING' if status==PASS and cost is None else
                'FEASIBLE' if status==PASS else 'INFEASIBLE' if status==FAIL else 'PENDING',
            signal_source=case.get('signal_packet') or case.get('signal_packet_source') or case.get('original_source') or config_locator,
            configuration_source=config_locator if 'commit' in config_locator else origin,
            result_sources=[origin], case_source=case.get('_source'),adopted_evaluation=deepcopy(adopted),
            historical_budget=deepcopy(case.get('budget')),historical_generation=deepcopy(case.get('generation')),
            price_axes=[],issues=[],E_C=None,G_qualified=None,winner=None,
            scope='MINIMUM_ONLY_WITHIN_SPECIFIED_SAVED_CANDIDATE_SET')
        try:
            c=self.comparison
            require(all(binding.get(k)==v for k,v in c['binding'].items()),'Snapshot/H/state comparison mismatch')
            require(frac(spec['h'])==frac(c['h']) and spec['M']==c['M'],'RPE horizon mismatch')
            if 'energy_prior_ha' in c:
                require(spec['rmse_policy']['error_contract']['energy_prior_ha']==c['energy_prior_ha'],'Energy prior mismatch')
            require(spec['profile']['P']==0,'P=0 required')
            require(all(spec['profile'].get(k)==v for k,v in c.get('compiler_profile',{}).items()),'Compiler/control/wrapper profile mismatch')
            if not signal_only:
                require(all(adopted.get(k)==v for k,v in c['accuracy'].items() if k!='MSE_upper'),'Accuracy/decoder/target mismatch')
                require(frac(adopted['MSE_upper'])==frac(c['accuracy']['MSE_upper']),'Adopted MSE threshold mismatch')
                require(conf['id']==case.get('original_configuration_id',case.get('configuration_id',conf['id'])),'Configuration ID mismatch')
                if case.get('law') is not None:
                    require(case['law']==spec.get('law'),'Case/configuration law mismatch')
                if 'target' in c:
                    target=case['evaluation_target']
                    require(all(target.get(k)==v for k,v in c['target'].items()),'Energy target mismatch')
            schedule={int(r['m']):r for r in conf['rounds']}
            require(set(schedule)==set(range(13)),'13 signal rounds required')
            row['schedule']=[dict(m=m,q=schedule[m]['q_PF'],R=schedule[m].get('R_outer'),
                actual_delta=deepcopy(schedule[m]['actual_delta_pf']),T=deepcopy(schedule[m]['T']),
                canonical_id=schedule[m]['candidate']['id']) for m in range(13)]
            if not terms:
                if status==PASS:row['status']='PRICE_WAITING'
            else:
                require(vector is not None and len(terms)==26,'26 price axes and an evaluated allocation required')
                require({(int(t['m']),t['axis'])for t in terms}=={(m,a)for m in range(13)for a in ('X','Y')},'Duplicate/missing price axis')
                physical={int(r['m']):r['physical_candidate']['id'] for r in case.get('rows',[]) if 'physical_candidate' in r}
                for t in terms:
                    m,axis=int(t['m']),t['axis'];p=self.r.read(t['price_source'])
                    require(int(p['m'])==m and p['axis']==axis,'Price round/axis mismatch')
                    if 'N' in t:require(int(t['N'])==vector[axis][m],'Q3/price allocation mismatch')
                    canonical=p.get('original_candidate_id') or p.get('original_canonical_candidate_id')
                    if canonical:require(canonical==schedule[m]['candidate']['id'],'Price canonical candidate mismatch')
                    elif p['physical_id']!=schedule[m]['candidate']['id']:
                        require(p['physical_id']==physical.get(m),'Physical candidate mapping missing/mismatched')
                    if p.get('original_configuration_id'):
                        require(p['original_configuration_id']==conf['id'],'Price configuration mismatch')
                    if p.get('axis_map'):
                        mapping=self.r.read(p['axis_map'])
                        require(mapping['configuration_id']==conf['id'] and mapping['physical_id']==p['physical_id'] and mapping['axis']==axis,
                                'Pauli price law/configuration bridge mismatch')
                        require(mapping['law_target']==spec['law']['model'],'Pauli price law mismatch')
                    if 'q' in p:require(int(p['q'])==int(schedule[m]['q_PF']),'Price q mismatch')
                    if p.get('R_outer') is not None:require(p['R_outer']==schedule[m].get('R_outer'),'Price R mismatch')
                    row['price_axes'].append(dict(m=m,axis=axis,N=vector[axis][m],physical_id=p['physical_id'],
                        canonical_id=schedule[m]['candidate']['id'],price_source=t['price_source'],
                        price_kind=p.get('price_class') or p.get('price_origin') or p.get('model_application') or p.get('evidence_class'),
                        compiler_profile=spec['profile']))
            require(cost is None or frac(cost)>=0,'Invalid saved conditional total')
        except (KeyError,ValueError,TypeError) as error:
            row['status']='MISMATCH';row['issues'].append(str(error))
        row['status_display']=LABELS[row['status']]
        self.rows.append(row)

    def pf(self, loc, d):
        inp=self.r.read(d['execution_source']);cases=inp['cases']
        joins=self.r.document(loc['commit'],str(PurePosixPath(loc['path']).parent/'C_RESULT_JOIN.json'))['candidates']
        for i,r in enumerate(d['rows']):
            case=next(c for c in cases if c['base_label']==r['base_label'])
            if r['rule']!='BASELINE_N0':
                case=case_for(cases,r['allocation'],r['label'])
                join=next(j for j in joins if j['label']==r['label'])
                require(shots(join['allocation'])==shots(r['allocation']) and join['model_status']==r['status'],'PF result join mismatch')
                require(frac(case['conditional_cost'])==frac(r['conditional_cost']),'PF saved total/allocation mismatch')
            terms=[dict(m=t['m'],axis=t['axis'],price_source=t['price_source'])for t in case['price_terms']]
            if r['rule']!='BASELINE_N0':terms=case['price_terms']
            case=dict(case,_source=d['execution_source'],label=r['label'])
            self.make(case,r['allocation'],r['status'],r['conditional_cost'],r.get('MSE_lower'),r.get('MSE_upper'),
                inp['adopted_evaluation'],ref(loc['commit'],loc['path'],'rows',i),terms)

    def p7(self, loc, d):
        for i,r in enumerate(d['rows']):
            src=r['allocation'].get('source') or d['source'];inp=self.r.read(src)
            case=case_for(inp.get('cases',inp.get('single_cases',[])+inp.get('transfer_cases',[])),r['allocation'])
            require(frac(case['conditional_cost'])==frac(r['conditional_cost_RZ']),'Local PF saved total/allocation mismatch')
            case=dict(case,_source=src,label=case['label']+' (m12='+str(r['m12'])+')')
            self.make(case,r['allocation'],r['status'],r['conditional_cost_RZ'],r['MSE_lower'],r['MSE_upper'],
                inp['adopted_evaluation'],ref(loc['commit'],loc['path'],'rows',i),case['price_terms'])

    def pauli(self, loc, d):
        parent=str(PurePosixPath(loc['path']).parent)
        # INPUTS is fixed in the same published tree; no caller/solver import.
        inp=self.r.document(loc['commit'],parent+'/INPUTS.json')
        for i,r in enumerate(d['rows']):
            case=next(c for c in inp['cases']if c['label']==r['label'])
            require(r['allocation']==shots(case['allocation'])['X'] and r['allocation_Y_equals_X'] is True,'Pauli saved allocation mismatch')
            require(frac(r['T'])==frac(case['conditional_cost']),'Pauli saved cost mismatch')
            self.make(dict(case,_source=ref(loc['commit'],parent+'/INPUTS.json')),case['allocation'],r['status'],r['T'],r['MSE_lower'],r['MSE_upper'],
                inp['adopted_evaluation'],ref(loc['commit'],loc['path'],'rows',i),case['price_terms'])
            self.rows[-1].update(historical_PF_benchmarks=deepcopy(d['PF_benchmarks']),
                historical_PF_benchmark_source=deepcopy(inp['PF_minima']),historical_comparisons=deepcopy(r['comparisons']))

    def pr_df(self, loc, d):
        directory='docs/research/d_pr_df_ld_split_results_20261115'
        inp=self.r.document(d['fixed_Q3_input_source'],directory+'/Q3_INPUTS.json')
        for i,r in enumerate(d['menu']):
            saved_cost=self.r.read(r['cost_source'])
            require(saved_cost['shots']==r['shots'] and frac(saved_cost['cost'])==frac(r['cost']),'DF saved cost/allocation source mismatch')
            cases=inp['cases']
            if r['rule']=='V0':
                base=self.r.read(inp['fixed_Q3']);cases=base['cases']
                case=next(c for c in cases if c['label']=='PR_DF_LD'+str(r['L_D']))
                adopted=base['adopted_evaluation']
            else:
                case=next(c for c in cases if c['label']==r['label']);adopted=inp['adopted_evaluation']
            require(shots(case['allocation'])==shots(r['shots']),'DF Q3 allocation mismatch')
            weighted=self.r.read(r['weighted_terms_source'])
            terms=[dict(m=t['m'],axis=t['axis'],N=t['N'],price_source=t['price_source'])for t in weighted
                   if t['L_D']==r['L_D'] and t['rule']==r['rule']]
            self.make(dict(case,_source=ref(d['fixed_Q3_input_source'],directory+'/Q3_INPUTS.json')),
                case['allocation'],r['status'],r['cost'],r['MSE_lower'],r['MSE_upper'],adopted,
                ref(loc['commit'],loc['path'],'menu',i),terms)

    def full_h(self, loc, d):
        axis=d['rows'][0];j=axis['B_join'];join=self.r.read(dict(j,selector=j['selector'][:2]))
        inp=self.r.document(j['commit'],str(PurePosixPath(j['path']).parent/'INPUTS.json'))
        case=case_for(inp['cases'],join['allocation'])
        summary_path=str(PurePosixPath(j['path']).parent/'SUMMARY.json')
        summary=self.r.document(j['commit'],summary_path)
        observed=next(r for r in summary['rows']if r['allocation_id']==join['allocation']['allocation_id'])
        require(observed['status']==join['model_status'] and observed['original_configuration_id']==case['original_configuration_id'],
                'Full-H saved Q3 classification mismatch')
        require(all(int(r['N'])==shots(join['allocation'])[r['axis']][int(r['m'])] for r in d['rows']),'Full-H price/Q3 allocation mismatch')
        terms=[dict(m=r['m'],axis=r['axis'],N=r['N'],price_source=ref(loc['commit'],loc['path'],'rows',i))for i,r in enumerate(d['rows'])]
        self.make(dict(case,_source=ref(j['commit'],str(PurePosixPath(j['path']).parent/'INPUTS.json'))),
            join['allocation'],join['model_status'],d['T'],observed['MSE_lower'],observed['MSE_upper'],inp['adopted_evaluation'],loc,terms)
        self.rows[-1]['price_scenario']='EMPIRICAL_M0_PLUS_24_F1_MODEL_REFERENCES'
        self.rows[-1]['q3_source']=dict(j,selector=j['selector'][:2])

    def q3_without_prices(self,loc,join):
        """A published Q3/axis join with an explicitly missing price is not ranked."""
        parent=str(PurePosixPath(loc['path']).parent)
        inp=self.r.read(join['source']);summary=self.r.document(loc['commit'],parent+'/SUMMARY.json')
        case=inp['cases'][0];conf=self.r.read(case_ref(case))
        observed=next(r for r in summary['comparison']if r['configuration_id']==case['total_configuration_id'])
        require(shots(observed['shots'])==shots(case['allocation']),'Price-waiting Q3 allocation mismatch')
        require(len(join['rows'])==26 and {(r['m'],r['axis'])for r in join['rows']}=={(m,a)for m in range(13)for a in ('X','Y')},
                'Price-waiting Q3 needs 26 unique axis bindings')
        for r in join['rows']:
            require(r['price'] is None,'A new price publication requires its own price binding')
            require(r['original_configuration_id']==conf['id'] and r['total_configuration_id']==case['total_configuration_id'],
                    'Price-waiting configuration bridge mismatch')
            require(r['original_candidate_id']==conf['rounds'][r['m']]['candidate']['id'],'Price-waiting physical candidate mismatch')
            require(r['N']==shots(case['allocation'])[r['axis']][r['m']],'Price-waiting axis allocation mismatch')
        self.make(dict(case,_source=join['source']),case['allocation'],observed['status'],None,
            observed['MSE_lower'],observed['MSE_upper'],inp['adopted_evaluation'],loc,[])
        self.rows[-1]['q3_source']=ref(loc['commit'],parent+'/SUMMARY.json','comparison',summary['comparison'].index(observed))
        self.rows[-1]['q3_axis_sources']=[ref(loc['commit'],loc['path'],'rows',i)for i in range(26)]

    def result(self):
        # Signal-only evidence is retained as history once an evaluated allocation
        # for the same configuration is supplied. It must no longer say Q3待ち.
        evaluated={r['configuration_id']for r in self.rows if r['q3_status'] is not None}
        history=[r for r in self.rows if r['N_X']is None and r['configuration_id']in evaluated]
        unique={}
        for row in self.rows:
            if row in history:continue
            key=json.dumps([row['configuration_id'],row['N_X'],row['N_Y'],
                row['adopted_evaluation'],row['issues'],
                [(a['m'],a['axis'],a['price_source'])for a in row['price_axes']]],sort_keys=True)
            row_id=hashlib.sha256(key.encode()).hexdigest()[:24]
            row['point_id']=row_id
            if row_id in unique:
                old=unique[row_id]
                if old['conditional_cost']!=row['conditional_cost'] or old['q3_status']!=row['q3_status']:
                    old['status']='MISMATCH';old['issues'].append('Conflicting duplicate saved outcome/cost')
                    old['status_display']=LABELS['MISMATCH']
                old['result_sources']+=row['result_sources']
                old['labels']=sorted(set(old['labels']+row['labels']))
            else:unique[row_id]=row
        rows=list(unique.values());best={}
        for method in ('FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF','FULL_H'):
            eligible=[r for r in rows if r['method']==method and r['status']=='FEASIBLE']
            value=min((frac(r['conditional_cost'])for r in eligible),default=None)
            best[method]=dict(conditional_cost=enc(value)if value is not None else None,
                point_ids=[r['point_id']for r in eligible if frac(r['conditional_cost'])==value],
                scope='指定した保存候補集合内の最安適合点')
        return dict(schema='SAVED_H4_RPE_CONDITIONAL_COST_CATALOG_V1',comparison=self.comparison,
            adopted_inputs=self.inputs,candidates=rows,method_minima=best,
            signal_only_history=history,duplicates_removed=len(self.rows)-len(history)-len(rows),new_signal=0,new_sampling=0,new_compile=0,
            new_price_fit=0,new_allocation_search=0,new_Q3=0,formal_G=None,winner=None,mandatory_stop=True)


def report(result):
    lines=['# H4 保存RPE評価結果の比較','', '結論の範囲：**指定した保存候補集合内の最安適合点**。',
        '保存Q3判定と条件付き価格の比較であり、全候補最適性・正式E[C]/G・方式winnerの認定は行わない。','',
        '|方式|公式 / LD|要求幅|最安適合の条件付き費用 RZ|point ID|','|---|---|---|---:|---|']
    for method,item in result['method_minima'].items():
        points=[r for r in result['candidates']if r['point_id']in item['point_ids']]
        formula={'2nd':'S2','4th':'Yoshida4','4th(m5_best)':'M5',None:'pure RTE'}
        attrs=', '.join(formula.get(r['formula'],str(r['formula']))+' / '+str(r['L_D'])for r in points)or'—'
        def width(r):
            if r['delta_pf']is None:return '—'
            ratio=frac(r['delta_pf'])/frac(r['h'])
            return str(ratio.numerator)+'h'+('/'+str(ratio.denominator)if ratio.denominator!=1 else'')
        widths=', '.join(width(r)for r in points)or'—'
        lines.append('|'+method+'|'+attrs+'|'+widths+'|'+display(item['conditional_cost'])+'|'+', '.join(item['point_ids'])+'|')
    lines+=['','|候補|方式|LD|公式|判定|費用 RZ|X/Y配分|','|---|---|---:|---|---|---:|---|']
    for r in result['candidates']:
        lines.append('|'+str(r['label'])+'|'+r['method']+'|'+str(r['L_D'])+'|'+str(r['formula'])+'|'+r['status_display']+'|'+
            display(r['conditional_cost'])+'|'+str(r['N_X'])+' / '+str(r['N_Y'])+'|')
    lines+=['',f"重複除去：{result['duplicates_removed']}件。詳細・固定SOURCE・26軸の価格対応はCATALOG.jsonを参照。",
            'Q3適合・価格待ちは費用nullで選択対象外。信号だけの旧状態はsignal_only_historyへ保持する。',
            'Full-Hは同一固定配分のQ3を照合した経験的m0＋24モデル価格のシナリオ。',
            'PRの事前固定予算・旧PF基準はhistorical_*に元SOURCEとともに保持し、新PF最安値で再計算しない。',
            '新規signal・sampling・compile・fit・配分探索・Q3はすべて0。','',
            '## 比較条件','', '```json',json.dumps(result['comparison'],ensure_ascii=False,indent=2),'```','',
            '## 固定入力一覧','']
    for source in result['adopted_inputs']:
        lines.append('- `'+source['commit']+':'+source['path']+'`')
    lines+=['','再実行コマンドと追加方法は親directoryのHANDOFF.mdを参照。','']
    return '\n'.join(lines)
