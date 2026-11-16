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
        if d.get('schema')=='RPE_KAPPA_ABS_RESULTS_V1':
            self.kappa_search_results(locator,d);return
        if '/best_points/'in path and path.endswith('/REQUEST.json'):
            self.public_optimizer_point(locator,d);return
        if d.get('schema')=='RPE_OPTIMIZER_FIXED_RESULTS_FOR_C_V1':
            self.pf_optimizer_results(locator,d);return
        if d.get('schema')=='B2_INTERRUPTED_SAVED_TOTAL_ADMISSION_V1':
            self.interrupted_pauli_best(locator,d);return
        if d.get('status')=='CONDITIONAL_WIDTH_SPECIFIC_A_SHAPE_COST':
            self.coarse_pf_prices(locator,d);return
        if d.get('kind')=='LD5_FIXED_ALLOCATION_MODEL_COST_AND_Q3_NOT_QUALIFIED_G':
            self.pauli_ld5(locator,d);return
        if d.get('classification')=='DIRECT_FINITE_COHORT_DESCRIPTIVE_PRICE_ONLY' and 'LD9_configuration'in d:
            self.ld9_partial_price(locator,d);return
        if d.get('schema')=='PR_DF_LD7_PF_BUDGET_MODEL_COST_Q3_JOIN_FOR_C_V1':
            self.ld7_budget(locator,d);return
        if d.get('schema')=='PR_DF_LD9_CONFIGURATION_V0_Q3_JOIN_V1':
            self.pr_df_unpriced_v0(locator,d);return
        if d.get('kind')=='CONDITIONAL_MODEL_REFERENCE_COST_AND_Q3_NOT_QUALIFIED_G' and 'records'in d:
            self.pauli_budget(locator,d);return
        if d.get('schema')=='PR_DF_LD7_MODEL_COST_Q3_JOIN_FOR_C_V1':
            self.add(ref(commit,parent+'/SUMMARY.json'));return
        if d.get('schema')=='PR_DF_LD1_LD2_LD7_NINE_MODEL_MENU_V1':
            self.pr_df_ld7(locator,d);return
        if d.get('schema')=='A_PAULI_LD3_PRICE_HANDOFF_BC_V1':
            self.pauli_model_prices(locator,d);return
        if d.get('kind')=='FIXED_ALLOCATION_Q3_RESULT_JOIN_NOT_COST_OR_N_QUALIFICATION':
            self.recovered_q3(locator,d);return
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

    def pf_optimizer_results(self,loc,d):
        """Import a completed method's saved result, not its running siblings."""
        index=self.r.read(d['records'][0]['price_source_index'])
        baseline=ref(d['saved_catalog_inputs']['commit'],str(PurePosixPath(d['saved_catalog_inputs']['path']).parent/'run01/CATALOG.json'))
        old=self.r.read(baseline)
        for i,r in enumerate(d['records']):
            require(r['method'] in ('FULL_PF','TRUNCATED_PF'),'Unsupported completed PF method')
            entry=next(e for e in index['configurations']if e['configuration_id']==r['configuration_id'])
            base=next(e for e in old['candidates']if e['configuration_id']==r['configuration_id'])
            result=self.r.read(r['result_source']);api_rows=self.r.read(r['source'])
            require(len(api_rows)==1 and api_rows[0]==result['record'],'PF optimizer original API/result mismatch')
            api=api_rows[0];conf=self.r.read(result['configuration_source'])
            require(conf['id']==r['configuration_id']==entry['configuration_id'] and
                    result['configuration_source']==base['configuration_source']==entry['configuration_source'],
                    'PF optimizer configuration SOURCE mismatch')
            require(r['total_configuration_id']==api['configuration_id']==entry['total_configuration_id'] and
                    r['binding_id']==api['canonical_binding']==entry['binding_id'] and
                    shots(api['allocation'])==shots(r['allocation'])==shots(result['allocation']),
                    'PF optimizer total-view/allocation mismatch')
            risk=api['diagnostic']['deterministic_risk'];adopted=base['adopted_evaluation']
            require(r['status']==result['status']==result['adopted_q3_assessment']['status'] and
                    frac(r['MSE_lower'])==frac(risk['MSE_lower']) and frac(r['MSE_upper'])==frac(risk['MSE_upper']) and
                    frac(r['tau'])==frac(adopted['MSE_upper']) and risk['estimator']==adopted['estimator'],
                    'PF optimizer saved Q3/MSE mismatch')
            require(r['signal_packet']==result['signal_packet_source']==entry['signal_source'] and
                    api['statistical_contract']==r['statistical_contract'] and api['evaluation_target']==r['evaluation_target'],
                    'PF optimizer signal/contract mismatch')
            require(r['configuration']['formula']==conf['spec']['formula'] and
                    frac(r['configuration']['delta_over_h'])==frac(conf['spec']['delta_pf'])/frac(conf['spec']['h']),
                    'PF optimizer formula/width mismatch')
            axes={(p['m'],p['axis']):p for p in entry['axes']};terms=[]
            require(len(r['price_axes'])==26,'PF optimizer price coverage missing')
            for p in r['price_axes']:
                a=axes[p['m'],p['axis']];original=self.r.read(p['price_source']);m=p['m']
                require(p['physical_id']==a['physical_id']==original['physical_id'] and
                        a['canonical_id']==original['original_candidate_id']==conf['rounds'][m]['candidate']['id'] and
                        p['price_source']==a['price_source'] and p['price_kind']==a['price_kind']==original['price_class'],
                        'PF optimizer price physical/SOURCE/class mismatch')
                require(frac(p['unit_price_RZ_exact'])==frac(a['unit_price_RZ_exact'])==frac(original['C']) and
                        p['compiler_profile']==a['compiler_profile']==conf['spec']['profile'],
                        'PF optimizer unit price/compiler mismatch')
                terms.append(dict(m=m,axis=p['axis'],N=p['N'],price_source=p['price_source']))
            case=dict(configuration=result['configuration_source'],original_configuration_id=conf['id'],
                total_configuration_id=r['total_configuration_id'],binding_id=r['binding_id'],
                label=r['method']+'_OPTIMIZER_'+r['configuration']['formula']+'_'+r['configuration']['delta_over_h']+'h',
                signal_packet=r['signal_packet'],evaluation_target=r['evaluation_target'],_source=r['result_source'])
            self.make(case,r['allocation'],r['status'],r['conditional_total_RZ'],r['MSE_lower'],r['MSE_upper'],
                adopted,dict(loc,selector=['records',i]),terms)
            out=self.rows[-1]
            out.update(q3_source=r['source'],optimizer_result_source=r['optimizer_result_source'],
                search_status='COMPLETED_METHOD_FINITE_SEARCH',campaign_status=d['status'],
                price_binding_level='26_AXIS_SAVED_PRICE_RECORDS',price_kind_counts=deepcopy(r['price_kind_counts']),
                price_evidence=dict(source_index=r['price_source_index'],inherited=r.get('price_evidence'),
                    inherited_qualification=r['price_qualification']),price_scenario=r.get('price_scenario'),
                conditional_cost_source=dict(loc,selector=['records',i,'conditional_total_RZ']))
            for p in out['price_axes']:
                observed=next(x for x in r['price_axes']if (x['m'],x['axis'])==(p['m'],p['axis']))
                p.update(C_ref=deepcopy(observed['unit_price_RZ_exact']),saved_weighted_term=deepcopy(observed['N_times_C_exact']))

    def public_optimizer_point(self,loc,request):
        """Join a producer's published selected point, never its private jobs.

        REQUEST -> EVALUATIONS hash -> exact OUTCOME/CASE_INPUTS -> saved
        price coverage. A configuration-only match cannot admit a new vector.
        """
        require(request['method']=='PR_PAULI','Unsupported public selected-point method')
        path=PurePosixPath(loc['path']);base=path.parent.parent.parent
        require(path.parent.parent.name=='best_points','Public selected point must be in best_points')
        listing=subprocess.check_output(['git','ls-tree','-r','--name-only',loc['commit'],'--',str(base)],cwd=self.r.root,text=True)
        evaluations=[]
        for name in listing.splitlines():
            p=PurePosixPath(name)
            if p.name!='EVALUATIONS.json' or p.parent.parent!=base:continue
            for i,e in enumerate(self.r.document(loc['commit'],name)['rows']):
                if PurePosixPath(e['job']).name==request['request_id']:
                    evaluations.append((ref(loc['commit'],name,'rows',i),e))
        require(len(evaluations)==1,'Missing/ambiguous public EVALUATIONS hash for '+request['request_id'])
        ev_ref,ev=evaluations[0];run=PurePosixPath(ev_ref['path']).parent
        raw_ref=ref(loc['commit'],str(path.parent/'OUTCOME.json'))
        raw=blob(self.r.root,loc['commit'],raw_ref['path']);outcome=self.r.read(raw_ref)
        require(hashlib.sha256(raw).hexdigest()==ev['outcome_sha256'],'Public selected OUTCOME sha256 mismatch')
        identity=outcome['identity'];config_id=request['configuration_id']
        require(config_id==ev['configuration_id']==identity['configuration_id']==request['expected']['configuration_id'] and
                request['request_id']==outcome['request_id'] and request['route']['kind']==outcome['route'] and
                request['N_X']==ev['N_X'] and request['N_Y']==ev['N_Y'] and frac(request['cost'])==frac(ev['cost']) and
                outcome['status']==ev['status'],'Public selected request/evaluation/outcome mismatch')
        inputs_ref=ref(loc['commit'],str(path.parent/'CASE_INPUTS.json'));inputs=self.r.read(inputs_ref)
        case=case_for(inputs['cases'],identity['allocation']);conf=self.r.read(case_ref(case))
        route=request['route'];item=route['configurations'][config_id];packet=self.r.read(item['signal_packet'])['packets'][0]
        require(conf['id']==config_id==packet['configuration']['id'] and conf==packet['configuration'] and
                self.r.read(item['configuration'])==conf and case['signal_packet']==item['signal_packet']==identity['signal_packet'] and
                case['binding_id']==item['binding_id']==identity['binding_id']==request['expected']['binding_id'] and
                case['total_configuration_id']==item['total_configuration_id']==identity['total_configuration_id']==request['expected']['total_configuration_id'] and
                identity['contract_matches'] is True and shots(case['allocation'])==shots(identity['allocation'])==dict(X=ev['N_X'],Y=ev['N_Y']),
                'Public selected configuration/total/binding/signal/allocation mismatch')
        template=self.r.read(route['contract_template'])
        require(inputs['adopted_evaluation']==template['adopted_evaluation'] and
                case['evaluation_target']==identity['evaluation_target'],'Public selected evaluation contract mismatch')
        rr_ref=ref(loc['commit'],str(run/'ROUND_RESULT.json'));rr=self.r.read(rr_ref)
        coverage_ref=ref(loc['commit'],str(run/'NEEDS.json'));needs=self.r.read(coverage_ref)
        found=[(i,c)for i,c in enumerate(needs['candidates'])if c['configuration_id']==config_id]
        require(len(found)==1,'Missing/ambiguous public price coverage for configuration '+config_id)
        i,coverage=found[0];coverage_ref['selector']=['candidates',i];quote=coverage['price_quote']
        require(coverage['ready'] is True and coverage['needs']==[] and coverage['price_status']=='PRICE_RULES_MODEL_COMPLETE' and
                coverage['signal_source']==item['signal_packet'] and quote['complete'] is True and quote['policy']=='MODEL' and
                quote['status']=='QUOTED' and quote['registry']=='PAULI_LD%d'%conf['spec']['L_D'] and
                quote['classifications']=={'MODEL_REFERENCE':26},'Public selected saved price coverage mismatch')
        override=rr['round'].get('price_rules_overrides',{}).get('registry_file')
        rules_ref=ref(loc['commit'],override)if override else ref(quote['implementation_source'],'configs/pr_price_rules/h4_saved_rules.json')
        rules=[r for r in self.r.read(rules_ref)['entries']if r['name']==quote['registry']]
        require(len(rules)==1,'Public selected registry entry missing/ambiguous')
        rule=rules[0];base_ref=rule['configuration_source'];original=self.r.read(base_ref)
        require(all(original['spec'][k]==conf['spec'][k]for k in
                ('binding','profile','method','L_D','split','formula','delta_pf','h','M','finite_K','law')),
                'Public selected price model SOURCE/domain mismatch')
        # Read the published model, without predicting any new unit prices.
        self.r.read(rule['model_source'])
        absolute=frac(conf['spec']['kappa']['value']);base_absolute=frac(original['spec']['kappa']['value'])
        require(frac(coverage['kappa_value'])==absolute,'Public selected price coverage kappa mismatch')
        require(len(packet['rounds'])==13 and [r['m']for r in packet['rounds']]==list(range(13)),
                'Public selected signal round coverage mismatch')
        for r,saved in zip(conf['rounds'],packet['rounds']):
            physical=saved['mean_record']['physical_context']['physical_candidate']
            require(saved['candidate_id']==physical['id']==r['candidate']['id'] and saved['identity']==r['identity'] and
                    saved['law_binding']==r['law_binding'] and physical['q']==r['q_PF'] and physical['R_outer']==r['R_outer'] and
                    Fraction(physical['T'])==frac(r['T']) and Fraction(physical['delta'])==frac(r['actual_delta_pf']),
                    'Public selected signal physical round/law mismatch')
        self.make(dict(case,_source=inputs_ref,label=item['label']+'__'+request['request_id']),identity['allocation'],
            outcome['status'],request['cost'],outcome['MSE_lower'],outcome['MSE_upper'],inputs['adopted_evaluation'],loc,[])
        row=self.rows[-1]
        if row['status']=='PRICE_WAITING':row.update(status='FEASIBLE',status_display=LABELS['FEASIBLE'])
        vector=shots(identity['allocation'])
        row.update(q3_source=raw_ref,outcome_sha256=ev['outcome_sha256'],public_evaluation_source=ev_ref,
            conditional_cost_source=dict(loc,selector=['cost']),search_result_source=rr_ref,
            search_status=rr['methods']['PR_PAULI']['stop_reason'],search_converged=False,
            kappa_absolute=enc(absolute),kappa_multiplier=enc(absolute/base_absolute),kappa_base_source=base_ref,
            kappa_base_absolute=enc(base_absolute),price_binding_level='SAVED_TOTAL_AND_26_MODEL_COVERAGE_SUMMARY',
            price_axis_unit_values_persisted=False,price_scenario='MODEL_REFERENCE_PRICE_NOT_E_C',qualified_for_candidate=False,CI=None,
            price_evidence=dict(coverage_source=coverage_ref,summary=quote,rule_source=rules_ref,rule=rule,
                implementation_source=quote['implementation_source'],original_model=rule['model_source'],per_axis_unit_values=None),
            model_coverage_axes=[dict(m=r['m'],axis=a,N=vector[a][r['m']],canonical_id=r['candidate']['id'],
                q=r['q_PF'],R=r['R_outer'],law_binding=deepcopy(r['law_binding']),C_ref=None,
                classification='MODEL_REFERENCE',extrapolated=True if quote.get('extrapolated_axes')==26 else None,
                coverage_source=coverage_ref)for r in conf['rounds']for a in ('X','Y')])

    def kappa_search_results(self,loc,d):
        """Resolve final reported selections to public evidence, not ROUND best."""
        base=PurePosixPath(loc['path']).parent
        listing=subprocess.check_output(['git','ls-tree','-r','--name-only',loc['commit'],'--',str(base/'best_points')],cwd=self.r.root,text=True)
        requests=[ref(loc['commit'],p)for p in listing.splitlines()if p.endswith('/REQUEST.json')]
        for selector in ('pr_pauli_overall_best','kappa2_best'):
            selected=d[selector];job=(selected.get('best_job')or selected['job']).split(':')[-1]
            matches=[r for r in requests if self.r.read(r)['request_id']==job]
            require(len(matches)==1,'Missing/ambiguous public selected REQUEST for '+selector+':'+job)
            request_ref=matches[0];request=self.r.read(request_ref)
            require(frac(request['cost'])==frac(selected.get('best_cost',selected.get('cost'))) and
                    request['N_X']==selected.get('best_N_X',selected.get('N_X')) and
                    request['configuration_id']==selected.get('configuration_id',request['configuration_id']),
                    'Reported selection/request mismatch: '+selector)
            self.add(request_ref)
            rows=[r for r in self.rows if r.get('q3_source',{}).get('path')==str(PurePosixPath(request_ref['path']).parent/'OUTCOME.json') and
                  r.get('q3_source',{}).get('commit')==loc['commit']]
            require(len(rows)==1,'Selected public point not admitted: '+selector)
            row=rows[0]
            if selector=='kappa2_best':require(frac(row['kappa_absolute'])==2,'kappa2 selection is not absolute kappa=2')
            row.setdefault('reported_selection_sources',[]).append(dict(loc,selector=[selector]))
            if loc not in row['result_sources']:row['result_sources'].append(loc)

    def interrupted_pauli_best(self,loc,d):
        """Saved-total-only admission with explicit missing per-axis unit prices.

        The producer's fixed 26-model coverage summary is retained as such;
        no quote is regenerated and no base-kappa price record is relabelled.
        """
        rr=self.r.read(d['round_result']);method=rr['methods']['PR_PAULI'];best=method['best'];ev=self.r.read(d['evaluation'])
        require(method['stop_reason']=='INTERRUPTED_SEARCH_POLICY_CHANGE','B2 interrupted search reason differs')
        require(best['configuration_id']==ev['configuration_id']==d['configuration_id'] and
                best['N_X']==ev['N_X'] and best['N_Y']==ev['N_Y'] and best['cost']==ev['cost'] and
                best['status']==ev['status'] and ev['adopted_by_rule'] is True,'B2 saved best/evaluation mismatch')
        raw_ref=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/d['outcome_file']))
        raw=blob(self.r.root,self.r.commit(raw_ref['commit']),raw_ref['path'])
        require(hashlib.sha256(raw).hexdigest()==ev['outcome_sha256']==d['outcome_sha256'],
                'B2 single saved OUTCOME sha256 mismatch')
        outcome=self.r.read(raw_ref);identity=outcome['identity']
        route=next(x for x in rr['round']['routes_override']['PR_PAULI']if d['configuration_id']in x['configurations'])
        item=route['configurations'][d['configuration_id']];conf=self.r.read(item['configuration'])
        packet=self.r.read(item['signal_packet'])['packets'][0];template=self.r.read(route['contract_template'])
        require(conf['id']==packet['configuration']['id']==identity['configuration_id']==d['configuration_id'] and
                conf==packet['configuration'] and identity['contract_matches'] is True and
                identity['signal_packet']==item['signal_packet'] and
                identity['total_configuration_id']==item['total_configuration_id'] and identity['binding_id']==item['binding_id'],
                'B2 saved outcome physical/total/signal identity mismatch')
        vector=shots(identity['allocation'])
        require(vector['X']==ev['N_X'] and vector['Y']==ev['N_Y'] and outcome['status']==ev['status'] and
                outcome['request_id']==PurePosixPath(ev['job']).name and outcome['route']==route['kind'],
                'B2 saved outcome allocation/route mismatch')
        coverage=self.r.read(d['price_coverage']);quote=coverage['price_quote']
        require(coverage['configuration_id']==conf['id'] and coverage['ready'] is True and coverage['needs']==[] and
                coverage['price_status']=='PRICE_RULES_MODEL_COMPLETE' and quote['complete'] is True and quote['policy']=='MODEL' and
                quote['status']=='QUOTED' and quote['registry']=='PAULI_LD5' and quote['classifications']=={'MODEL_REFERENCE':26} and
                quote['extrapolated_axes']==26 and quote['implementation_source']==d['price_rules']['commit'],
                'B2 saved model coverage/implementation mismatch')
        rule=next(x for x in self.r.read(d['price_rules'])['entries']if x['name']==quote['registry'])
        base=self.r.read(d['base_configuration']);rule_base=self.r.read(rule['configuration_source'])
        require(base==rule_base and all(base['spec'][k]==conf['spec'][k]for k in
                ('binding','profile','method','L_D','split','formula','delta_pf','h','M','finite_K','law')) and
                rule['model_source']['commit']==d['original_model']['commit'] and
                rule['model_source']['path']==d['original_model']['path'],
                'B2 price model SOURCE/domain mismatch')
        multiplier=frac(best['configuration']['kappa_multiplier']);absolute=frac(conf['spec']['kappa']['value'])
        require(absolute==multiplier*frac(base['spec']['kappa']['value']),'B2 kappa absolute/multiplier mismatch')
        require(len(packet['rounds'])==13 and [r['m']for r in packet['rounds']]==list(range(13)),
                'B2 saved signal round coverage mismatch')
        for row,saved in zip(conf['rounds'],packet['rounds']):
            c=saved['mean_record']['physical_context']['physical_candidate']
            require(saved['candidate_id']==c['id']==row['candidate']['id'] and saved['identity']==row['identity'] and
                    saved['law_binding']==row['law_binding'] and c['q']==row['q_PF'] and c['R_outer']==row['R_outer'] and
                    Fraction(c['T'])==frac(row['T']) and Fraction(c['delta'])==frac(row['actual_delta_pf']),
                    'B2 signal physical round/law mismatch')
        case=dict(item,original_configuration_id=conf['id'],label='LD5_KAPPA_X2_INTERRUPTED_BEST',
            evaluation_target=identity['evaluation_target'],_source=raw_ref)
        self.make(case,identity['allocation'],outcome['status'],best['cost'],outcome['MSE_lower'],outcome['MSE_upper'],
            template['adopted_evaluation'],loc,[])
        out=self.rows[-1]
        # make() retains PRICE_WAITING without per-axis numeric records. This
        # narrowly scoped alternative admits the PRODUCER'S saved total only,
        # after the independent identity/hash/coverage checks above.
        if out['status']=='PRICE_WAITING':out.update(status='FEASIBLE',status_display=LABELS['FEASIBLE'])
        out.update(q3_source=raw_ref,original_outcome_path=d['original_outcome_path'],outcome_sha256=d['outcome_sha256'],
            public_evaluation_source=d['evaluation'],conditional_cost_source=dict(d['round_result'],selector=['methods','PR_PAULI','best','cost']),
            search_status='INTERRUPTED_SEARCH_POLICY_CHANGE',search_converged=False,
            kappa_absolute=enc(absolute),kappa_multiplier=enc(multiplier),kappa_base_source=d['base_configuration'],
            kappa_base_absolute=enc(base['spec']['kappa']['value']),price_binding_level='SAVED_TOTAL_AND_26_MODEL_COVERAGE_SUMMARY',
            price_axis_unit_values_persisted=False,price_scenario='MODEL_REFERENCE_PRICE_NOT_E_C',qualified_for_candidate=False,CI=None,
            price_evidence=dict(coverage_source=d['price_coverage'],summary=quote,rule_source=d['price_rules'],rule=rule,
                original_model=d['original_model'],per_axis_unit_values=None),
            model_coverage_axes=[dict(m=r['m'],axis=a,N=vector[a][r['m']],canonical_id=r['candidate']['id'],
                q=r['q_PF'],R=r['R_outer'],law_binding=deepcopy(r['law_binding']),C_ref=None,
                classification='MODEL_REFERENCE',extrapolated=True,coverage_source=d['price_coverage'])
                for r in conf['rounds']for a in ('X','Y')])

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

    def pauli_budget(self,loc,d):
        """Read saved budget allocations; reuse the already bound LD3 model prices."""
        inp=self.r.read(d['execution_source'])
        summary_loc=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/'SUMMARY.json'))
        summary=self.r.read(summary_loc)
        for i,r in enumerate(d['records']):
            case=next(c for c in inp['cases']if c['label']==r['label'])
            s=next(x for x in summary['rows']if x['label']==r['label'])
            require(all(r[k]==case[k] for k in ('original_configuration_id','total_configuration_id','binding_id','allocation')),
                    'Pauli budget configuration/allocation mismatch')
            require(r['status']==s['status'] and frac(r['conditional_reference_total'])==frac(s['conditional_cost'])==frac(case['conditional_cost']),
                    'Pauli budget saved verdict/total mismatch')
            require(r['price_qualified_for_candidate'] is False and all(r[k] is None for k in ('E_C','CI','G','G_qualified','sufficient_N')),
                    'Pauli budget reference must not acquire scientific qualification')
            q3_loc=dict(r['result_source'],selector=r['result_record_selector']);q3=self.r.read(q3_loc)
            require(q3['configuration_id']==case['total_configuration_id'] and q3['canonical_binding']==case['binding_id'] and
                    shots(q3['allocation'])==shots(r['allocation']),'Pauli budget raw Q3 identity/allocation mismatch')
            self.add(r['price_source'])
            price_handoff=self.r.read(r['price_source'])
            template=next(x for x in self.rows if x['configuration_id']==r['original_configuration_id'] and
                          x['N_X']==x['N_Y']==price_handoff['VP']['N'])
            conf=self.r.read(case_ref(case));terms=[]
            for t in r['axes']:
                p=self.r.read(t['original_price_source']);a=self.r.read(t['price_source']);context=self.r.read(t['physical_context_source'])
                m=t['m'];canonical=conf['rounds'][m]['candidate']['id']
                require(t['original_candidate_id']==a['original_candidate_id']==p['physical_id']==canonical and
                        t['total_candidate_id']==a['total_candidate_id']==context['total_candidate_id'],
                        'Pauli budget physical/total candidate mismatch')
                require(t['original_law_binding']==context['original_law_binding'] and t['total_law_binding']==context['total_law_binding'] and
                        t['original_law_binding']==conf['rounds'][m]['law_binding'],'Pauli budget law mismatch')
                require(t['price_class']==a['value_source']=='MODEL_APPLICATION_FROM_Q8R660_DECOMPOSITION' and
                        frac(t['C_ref'])==frac(a['C_ref_exact'])==frac(p['C_ref_exact']),
                        'Pauli budget saved model price mismatch')
                original=next(x for x in case['price_terms']if x['m']==m and x['axis']==t['axis'])
                require(all(t[k]==original[k]for k in ('N','C_ref','N_times_C_ref','original_price_source')),
                        'Pauli budget fixed weighted term mismatch')
                terms.append(dict(m=m,axis=t['axis'],N=t['N'],price_source=t['original_price_source']))
            self.make(dict(case,_source=d['execution_source']),r['allocation'],r['status'],r['conditional_reference_total'],
                s['MSE_lower'],s['MSE_upper'],inp['adopted_evaluation'],dict(loc,selector=['records',i]),terms)
            row=self.rows[-1]
            row.update(q3_source=q3_loc,qualified_for_candidate=False,price_scenario='MODEL_REFERENCE_PRICE_NOT_E_C',CI=None,
                price_evidence=deepcopy(template['price_evidence']),
                saved_PF_comparisons=dict(source=dict(summary_loc,selector=['rows',summary['rows'].index(s)]),
                    budgets=deepcopy(summary['budgets']),difference_T_minus_P7=deepcopy(s['difference_T_minus_P7']),
                    difference_T_minus_Full_PF=deepcopy(s['difference_T_minus_Full_PF'])))
            for p,t in zip(row['price_axes'],r['axes']):
                p.update(price_kind='MODEL',qualified_for_candidate=False,model_application=t['price_class'],C_ref=deepcopy(t['C_ref']),
                    physical_context_source=t['physical_context_source'],price_handoff_source=t['price_source'])

    def pauli_ld5(self,loc,d):
        """Read LD5's two saved model-price/Q3 joins, without importing LD3 rows."""
        inp=self.r.read(d['execution_source'])
        summary_loc=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/'SUMMARY.json'));summary=self.r.read(summary_loc)
        for i,r in enumerate(d['records']):
            case=case_for(inp['cases'],r['allocation']);conf=self.r.read(case_ref(case))
            s=next(x for x in summary['rows']if x['prefix']==5 and x['allocation_name']==case['comparison_name'])
            require(all(r[k]==case[k]for k in ('original_configuration_id','total_configuration_id','binding_id','allocation')) and
                    self.r.read(r['configuration_source'])==conf,'LD5 saved configuration/allocation mismatch')
            require(r['status']==s['status'] and frac(r['conditional_total'])==frac(s['conditional_cost'])==frac(case['conditional_cost']),
                    'LD5 saved verdict/total mismatch')
            require(r['qualified_for_candidate'] is False and r['model_axes']==26 and r['price_kind']=='MODEL_REFERENCE_PRICE_NOT_E_C',
                    'LD5 model reference qualification mismatch')
            require(r['dictionary_identity']==conf['spec']['split']['tail_dictionary'],'LD5 Pauli dictionary mismatch')
            q3_loc=dict(r['result_source'],selector=r['result_record_selector']);q3=self.r.read(q3_loc)
            require(q3['configuration_id']==case['total_configuration_id'] and q3['canonical_binding']==case['binding_id'] and
                    shots(q3['allocation'])==shots(r['allocation']),'LD5 raw Q3 binding mismatch')
            price=self.r.read(r['price_source']);binding_source=ref(r['price_source']['commit'],str(PurePosixPath(r['price_source']['path']).parent/'BINDING.json'))
            binding=self.r.read(binding_source)
            require(price['configuration_id']==conf['id'] and binding['request']['physical_cell']['profile']==conf['spec']['profile'] and
                    binding['request']['physical_cell']['split']==conf['spec']['split'],'LD5 model compiler/split SOURCE mismatch')
            terms=[]
            for t in r['axes']:
                p=self.r.read(t['price_source']);m=t['m'];context=self.r.read(t['physical_context_source'])
                view=self.r.document(t['physical_context_source']['commit'],t['physical_context_source']['path'])
                require(view['configuration_id']==case['total_configuration_id'] and view['means'][m]['identity']['candidate']['id']==t['total_candidate_id'],
                        'LD5 total-view axis candidate mismatch')
                require(p['physical_id']==t['original_candidate_id']==conf['rounds'][m]['candidate']['id'] and
                        context['physical_candidate']['id']==t['original_candidate_id'],'LD5 model physical candidate mismatch')
                require(p['q']==conf['rounds'][m]['q_PF'] and p['R']==conf['rounds'][m]['R_outer'] and
                        p['value_source']==t['value_source']=='MODEL_APPLICATION_FROM_Q8R4_DECOMPOSITION',
                        'LD5 model q/R/application mismatch')
                require(t['price_qualified_for_candidate'] is False and frac(t['C_ref'])==frac(p['C_ref_exact']),
                        'LD5 saved axis model price mismatch')
                original=next(x for x in case['price_terms']if x['m']==m and x['axis']==t['axis'])
                require(all(t[k]==original[k]for k in ('N','C_ref','N_times_C_ref','price_source')),'LD5 fixed weighted term mismatch')
                terms.append(dict(m=m,axis=t['axis'],N=t['N'],price_source=t['price_source']))
            comparison=self.r.read(r['A_comparison_source'])
            require(frac(comparison['T_LD5'])==frac(r['conditional_total']) and comparison['N_per_axis']==shots(r['allocation'])['X']==shots(r['allocation'])['Y'] and
                    frac(self.r.read(comparison['T_LD3_source']))==frac(comparison['T_LD3']),'LD5/LD3 saved comparison mismatch')
            self.make(dict(case,_source=d['execution_source']),r['allocation'],r['status'],r['conditional_total'],s['MSE_lower'],s['MSE_upper'],
                inp['adopted_evaluation'],dict(loc,selector=['records',i]),terms)
            row=self.rows[-1]
            row.update(q3_source=q3_loc,qualified_for_candidate=False,price_scenario=r['price_kind'],CI=None,
                price_evidence=dict(source=r['price_source'],binding_source=binding_source,rule=price['rule'],
                    coefficient_source=dict(r['price_source'],selector=['coefficients']),calibration_source=dict(r['price_source'],selector=['calibration_cell']),
                    scalar=deepcopy(binding['scalar']),direct_calibration_paths=binding['n_paths'],
                    round_model_prices=26,round_direct_prices=0,qualified_for_candidate=False),
                saved_split_comparison=dict(source=r['A_comparison_source'],T_LD3=deepcopy(comparison['T_LD3']),
                    T_LD3_source=comparison['T_LD3_source'],difference_LD5_minus_LD3=deepcopy(comparison['difference_LD5_minus_LD3']),
                    scope='NUMERICAL_DIFFERENCE_UNDER_THE_ADOPTED_REFERENCE_PRICE_RULES'))
            for p,t in zip(row['price_axes'],r['axes']):
                p.update(price_kind='MODEL',qualified_for_candidate=False,model_application=t['value_source'],
                    C_ref=deepcopy(t['C_ref']),total_candidate_id=t['total_candidate_id'],physical_context_source=t['physical_context_source'])

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

    def pr_df_ld7(self,loc,d):
        """Reuse the six old points and join three saved LD7 price/Q3 points."""
        old=self.r.read(d['LD1_LD2_saved_source'])
        require([r for r in d['menu']if r['L_D'] in (1,2)]==old['menu'],
                'LD1/LD2 republication differs from the original six results')
        self.add(d['LD1_LD2_saved_source'])
        handoff_loc=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/'C_HANDOFF.json'))
        h=self.r.read(handoff_loc);conf=self.r.read(h['configuration'])
        require(conf['id']==h['configuration_id'] and conf['spec']['L_D']==7,'LD7 configuration mismatch')
        require(h['compiled_price_quantity']==self.comparison['price_quantity'],'LD7 price quantity mismatch')
        require(h['original_law']==conf['spec']['law']['model'],'LD7 price law mismatch')
        require(d['model_commit']==h['models']['commit']==h['axis_prices']['commit'],'LD7 frozen model SOURCE mismatch')
        models=self.r.read(h['models'])
        require(models['finite_cohort_is_law_expectation'] is False,'LD7 model is not a law expectation')
        prices=self.r.read(h['axis_prices'])['rows']
        require(len(prices)==26 and {(p['m'],p['axis'])for p in prices}==
                {(m,a)for m in range(13)for a in ('X','Y')},'LD7 needs 26 unique model price axes')
        # Load the existing V0 publication, including when this input comes first.
        config_source=prices[0]['configuration_source']
        self.add(ref(config_source['commit'],str(PurePosixPath(config_source['path']).parents[1]/'SUMMARY.json')))
        for i,r in enumerate(d['menu']):
            if r['L_D']!=7:continue
            hr=next(x for x in h['rows']if x['rule']==r['rule'])
            require(hr==r,'LD7 handoff/summary mismatch')
            q3=self.r.read(r['Q3_source']);input_loc=q3['source']['inputs'];inp=self.r.read(input_loc)
            inp=inp.get('inputs',inp);case=case_for(inp['cases'],r['shots'],r['label'])
            require(self.r.read(case_ref(case))==conf,'LD7 Q3/price configuration mismatch')
            require(shots(q3['allocation'])==shots(case['allocation'])==shots(r['shots']),'LD7 Q3/price allocation mismatch')
            require(q3['status']==r['status'] and q3['record']['configuration_id']==case['total_configuration_id'],
                    'LD7 Q3 verdict/total view mismatch')
            require(frac(q3['adopted_q3_assessment']['q3']['model_MSE_upper'])==frac(r['MSE_upper']),
                    'LD7 saved MSE bound mismatch')
            terms=[]
            for t in r['weighted_terms']:
                require(t['price_source']['commit']==h['axis_prices']['commit'] and t['price_source']['path']==h['axis_prices']['path'],
                        'LD7 axis price SOURCE mismatch')
                p=self.r.read(t['price_source']);m=t['m'];round_=conf['rounds'][m]
                require(t['configuration_id']==p['configuration_id']==conf['id'] and
                        t['physical_id']==p['physical_id']==round_['physical_candidate']['id'],'LD7 physical price binding mismatch')
                require(p['law']==round_['law_binding'] and p['q']==round_['q_PF'] and p['R']==round_['R_outer'],
                        'LD7 model law/q/R mismatch')
                require(t['axis']==p['axis']==p['model_axis'] and t['m']==p['m'] and frac(t['C_ref'])==frac(p['C_ref']),
                        'LD7 saved price axis/value mismatch')
                require(t['application']==p['application'],'LD7 model application mismatch')
                terms.append(dict(m=m,axis=t['axis'],N=t['shots'],price_source=t['price_source']))
            self.make(dict(case,_source=input_loc,rows=conf['rounds'],label=r['label']),case['allocation'],r['status'],
                r['cost'],r['MSE_lower'],r['MSE_upper'],inp['adopted_evaluation'],dict(loc,selector=['menu',i]),terms)
            row=self.rows[-1]
            row.update(q3_source=r['Q3_source'],price_scenario='MODEL_REFERENCES_NOT_FORMAL_E_C',qualified_for_candidate=False,
                price_evidence=dict(model_source=h['models'],axis_prices_source=h['axis_prices'],holdout_source=h['holdout'],
                    model_holdout_verdict=h['model_holdout_verdict'],producer_price_acceptance=h['price_acceptance'],
                    calibration_sources={a:[dict(h['models'],selector=['models',a,'calibration',j])
                        for j in range(len(models['models'][a]['calibration']))]for a in ('X','Y')},
                    qualifications=deepcopy(d['qualifications'])),
                saved_cost_breakdown={k:deepcopy(r[k])for k in ('m0_m6','m7_m12','m7_m12_fraction','m7_m12_percent_display')})
            for p,t in zip(row['price_axes'],r['weighted_terms']):
                p.update(price_kind='MODEL',qualified_for_candidate=False,model_application=t['application'],C_ref=deepcopy(t['C_ref']))
            if r['rule']=='V0':
                prior=next(x for x in self.rows[:-1]if x['configuration_id']==row['configuration_id'] and x['N_X']==row['N_X'] and x['N_Y']==row['N_Y'])
                require(prior['q3_status']==row['q3_status'] and prior['total_configuration_id']==row['total_configuration_id'] and
                        prior['binding_id']==row['binding_id'],'LD7 existing V0 Q3 bridge mismatch')
                row['result_sources']=prior['result_sources']+row['result_sources']
                row['labels']=sorted(set(prior['labels']+row['labels']))
                row['price_availability_history']=[dict(status=prior['status'],sources=prior['result_sources'])]
                self.rows.remove(prior)

    def pr_df_unpriced_v0(self,loc,d):
        """Attach one saved Q3 allocation; a referenced split is not a new point."""
        published=self.r.read(d['fixed_Q3_inputs']);inp=published.get('inputs',published)
        case=case_for(inp['cases'],d['allocation']);conf=self.r.read(case_ref(case));q3=self.r.read(d['Q3_source'])
        require(self.r.read(d['configuration'])==conf and d['configuration_id']==conf['id'],
                'Unpriced V0 configuration mismatch')
        require(d['total_configuration_id']==case['total_configuration_id']==q3['record']['configuration_id'] and
                d['binding_id']==case['binding_id']==q3['record']['canonical_binding'],'Unpriced V0 total view/binding mismatch')
        require(shots(d['allocation'])==shots(case['allocation'])==shots(q3['allocation']),
                'Unpriced V0 allocation mismatch')
        require(d['law']==conf['spec']['law']['model'] and d['Q3_status']==q3['status'],
                'Unpriced V0 law/verdict mismatch')
        require(frac(d['MSE_upper'])==frac(q3['adopted_q3_assessment']['q3']['model_MSE_upper']),
                'Unpriced V0 saved MSE bound mismatch')
        require(d['price_status']=='NOT_EVALUATED' and all(d[k] is None for k in ('axis_prices','total_model_cost','formal_E_C','G')),
                'Do not import a different split price into unpriced V0')
        require(shots(d['same_V0_LD7']['shots'])==shots(d['allocation']),'Shared V0 shot vector mismatch')
        self.make(dict(case,_source=d['fixed_Q3_inputs']),d['allocation'],d['Q3_status'],None,
            d['MSE_lower'],d['MSE_upper'],inp['adopted_evaluation'],loc,[])
        self.rows[-1].update(q3_source=d['Q3_source'],price_status=d['price_status'],
            shared_V0_reference=dict(source=dict(loc,selector=['same_V0_LD7']),
                meaning='Same shot vector; LD7 remains a different physical candidate and price domain'))

    def ld7_budget(self,loc,d):
        """Keep saved failed budget allocations beside the achieved LD7 points."""
        self.add(d['index_source']);inp=self.r.read(d['source']['inputs'])
        template=next(r for r in self.rows if r['configuration_id']==d['configuration_id'] and 'price_evidence'in r)
        base={(p['m'],p['axis']):p['price_source']for p in template['price_axes']}
        for i,r in enumerate(d['rows']):
            case=next(c for c in inp['cases']if c['label']==r['label']);conf=self.r.read(case_ref(case))
            q3_loc=ref(loc['commit'],r['result_source']['result_path']);q3=self.r.read(q3_loc)
            require(conf['id']==d['configuration_id'] and case['total_configuration_id']==d['total_configuration_id']==q3['record']['configuration_id'] and
                    case['binding_id']==d['binding_id']==q3['record']['canonical_binding'],'LD7 budget identity mismatch')
            require(shots(r['allocation'])==shots(case['allocation'])==shots(q3['allocation']) and
                    r['model_status']==q3['status'],'LD7 budget Q3/allocation mismatch')
            require(frac(r['conditional_cost'])==frac(case['reference_cost']) and r['price_terms']==case['price_terms'],
                    'LD7 budget fixed price terms/total mismatch')
            for t in r['price_terms']:
                p=self.r.read(t['price_source'])
                require(t['price_source']==base[t['m'],t['axis']] and frac(t['C_ref'])==frac(p['C_ref']),
                        'LD7 budget model price SOURCE mismatch')
            self.make(dict(case,_source=d['source']['inputs'],rows=conf['rounds']),r['allocation'],r['model_status'],r['conditional_cost'],
                r['MSE_lower'],r['MSE_upper'],inp['adopted_evaluation'],dict(loc,selector=['rows',i]),r['price_terms'])
            row=self.rows[-1]
            row.update(q3_source=q3_loc,qualified_for_candidate=False,price_scenario='MODEL_REFERENCES_NOT_FORMAL_E_C',
                price_evidence=deepcopy(template['price_evidence']),historical_PF_benchmarks=deepcopy(d['PF_benchmarks']),
                historical_comparisons=deepcopy(r['comparisons']),saved_cost_breakdown=dict(m0_m6=r['cost_m0_m6'],
                    m7_m12=r['cost_m7_m12'],m7_m12_fraction=r['m7_m12_fraction']))
            for p,t in zip(row['price_axes'],r['price_terms']):
                p.update(price_kind='MODEL',qualified_for_candidate=False,model_application=t['application'],C_ref=deepcopy(t['C_ref']))

    def ld9_partial_price(self,loc,d):
        """Attach saved finite-cohort means without making a total-cost point."""
        conf=self.r.read(d['LD9_configuration'])
        if not any(r['configuration_id']==conf['id']for r in self.rows):
            source=d['LD9_configuration']
            self.add(ref(source['commit'],str(PurePosixPath(source['path']).parents[1]/'C_HANDOFF.json')))
        row=next(r for r in self.rows if r['configuration_id']==conf['id'])
        require(self.r.read(row['configuration_source'])==conf and row['conditional_cost'] is None,
                'LD9 partial price configuration/total mismatch')
        require(d['m']==6 and all(d[k] is None for k in ('fit','LD9_total_model_cost','formal_E_C','G')),
                'Partial LD9 direct price must not create an all-round cost')
        comparison_source=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/'COMPARISON.json'))
        comparison=self.r.read(comparison_source)
        require(comparison['comparisons']==d['comparisons'] and comparison['cohorts']==d['cohorts'] and
                comparison['quantity']==self.comparison['price_quantity'],
                'LD9 partial price comparison/quantity mismatch')
        require(len(d['comparisons'])==2 and {c['axis']for c in d['comparisons']}=={'X','Y'} and
                all(c['n_paths_LD7']==c['n_paths_LD9']==4 for c in d['comparisons']),'Partial comparison axes/path counts mismatch')
        require({c['L_D']for c in d['cohorts']}=={7,9} and len({c['cohort']for c in d['cohorts']})==2,
                'LD7/LD9 direct cohorts must remain separate')
        for co in d['cohorts']:
            request=co['request'];m=d['m']
            matched=next((r for r in self.rows if r['configuration_id']==co['configuration_id']),None)
            co_conf=self.r.read(matched['configuration_source'])if matched else None
            # The LD7 comparison remains a source reference if read before its catalog point.
            if co_conf is not None:
                require(request['identity']==co_conf['rounds'][m]['identity'] and request['law_binding']==co_conf['rounds'][m]['law_binding'],
                        'Partial direct price identity/law mismatch')
            profile=request['physical_context']['price_profile']
            require(all(profile[k]==v for k,v in self.comparison['compiler_profile'].items()),'Partial direct price compiler profile mismatch')
            require(len(co['rows'])==8 and {(r['index'],r['axis'])for r in co['rows']}=={(i,a)for i in range(4)for a in ('X','Y')},
                    'Partial direct price requires four complete X/Y paths')
            for r in co['rows']:
                receipt=self.r.read(r['receipt'])
                require(r['configuration_id']==co['configuration_id'] and r['cohort']==co['cohort'] and r['m']==m and
                        r['physical_id']==request['physical_context']['physical_candidate']['id'] and r['status']=='VALID',
                        'Partial direct price path binding mismatch')
                require(receipt['metrics']['rz_count']==r['C'] and receipt['fingerprint']==r['fingerprint'] and
                        receipt['compiler']==profile['compiler'],'Partial direct price receipt mismatch')
        own_index=next(i for i,c in enumerate(d['cohorts'])if c['L_D']==9);own=d['cohorts'][own_index]
        require(own['configuration_id']==conf['id'] and d['physical_id']==conf['rounds'][6]['physical_candidate']['id'],
                'LD9 partial price physical candidate mismatch')
        row.update(partial_price_axes=[dict(m=6,axis=c['axis'],physical_id=d['physical_id'],C_RZ=deepcopy(c['LD9_mean']),
            price_kind='DIRECT_FINITE_COHORT_MEAN',n_paths=c['n_paths_LD9'],cohort=own['cohort'],qualified_for_candidate=False,
            reference_C_RZ=deepcopy(c['LD7_mean']),difference_LD9_minus_LD7=deepcopy(c['difference_LD9_minus_LD7']),
            comparison_source=dict(comparison_source,selector=['comparisons',i]),
            path_sources=[dict(loc,selector=['cohorts',own_index,'rows',j])for j,r in enumerate(own['rows'])if r['axis']==c['axis']])
            for i,c in enumerate(d['comparisons'])],partial_price_source=loc,
            partial_price_scope=comparison['scope'],price_status='M6_DIRECT_FINITE_COHORT_ONLY_ALL_ROUND_COST_MISSING')

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

    def recovered_q3(self,loc,join):
        """A resolved model verdict supersedes an execution UNKNOWN, not its history."""
        parent=str(PurePosixPath(loc['path']).parent)
        published=self.r.document(loc['commit'],parent+'/INPUTS.json');inp=published['inputs']
        case=case_for(inp['cases'],join['allocation']);conf=self.r.read(case_ref(case))
        require(join['configuration_id']==conf['id'] and join['total_view_id']==case['total_view_id'],
                'Recovered Q3 configuration/decoder mismatch')
        require(shots(self.r.read(join['fixed_allocation_source']))==shots(case['allocation']),
                'Recovered Q3 fixed allocation mismatch')
        require(frac(join['exact_tau'])==frac(inp['adopted_evaluation']['MSE_upper']),'Recovered Q3 threshold mismatch')
        require(join['model_status']==join['assessment']['status'],'Recovered saved classification mismatch')
        require(join['coarse_cost_RZ'] is None,'Do not invent an unacquired coarse price')
        require(len(join['rows'])==26 and {(r['m'],r['axis'])for r in join['rows']}=={(m,a)for m in range(13)for a in ('X','Y')},
                'Recovered Q3 requires 26 unique axis bindings')
        for r in join['rows']:
            require(r['original_candidate_id']==conf['rounds'][r['m']]['candidate']['id'],'Recovered physical binding mismatch')
            require(r['N']==shots(case['allocation'])[r['axis']][r['m']],'Recovered axis allocation mismatch')
        self.make(dict(case,_source=ref(loc['commit'],parent+'/INPUTS.json','inputs')),case['allocation'],
            join['model_status'],None,join['MSE_lower'],join['MSE_upper'],inp['adopted_evaluation'],loc,[])
        old=self.r.read(published['previous_STOP'])
        self.rows[-1].update(q3_source=join['result_source'],q3_axis_sources=[ref(loc['commit'],loc['path'],'rows',i)for i in range(26)],
            execution_history=[dict(status=old['status'],source=published['previous_STOP'],errors=deepcopy(old['errors']))],
            saved_recovery_assessment_source=dict(loc,selector=['assessment']))

    def coarse_pf_prices(self,loc,d):
        """Enrich the one saved coarse-width allocation with its own price table."""
        self.add(d['Q3_acceptance']);join=self.r.read(d['Q3_acceptance'])
        q3_loc=d['Q3_acceptance'];input_loc=ref(q3_loc['commit'],str(PurePosixPath(q3_loc['path']).parent/'INPUTS.json'),'inputs')
        inp=self.r.read(input_loc);case=case_for(inp['cases'],d['allocation']);conf=self.r.read(case_ref(case))
        prior=next(r for r in self.rows if r['configuration_id']==conf['id'] and r['allocation']['allocation_id']==d['allocation_id'])
        require(d['configuration_id']==conf['id'] and d['allocation_id']==case['allocation']['allocation_id'] and
                shots(d['allocation'])==shots(case['allocation']),'Coarse PF price configuration/allocation mismatch')
        require(d['Q3_model_status']==prior['q3_status']==join['model_status'],'Coarse PF saved Q3 verdict mismatch')
        require(d['compiler_profile']==conf['spec']['profile'] and d['scalar_quantity']=='HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY' and
                d['scalar_application']=='ONCE_AT_TOTAL_T','Coarse PF compiler/scalar profile mismatch')
        require(d['quantity']=='FULL_MEASURED_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT_P0_EXCLUDED',
                'Coarse PF full-wrapper price quantity mismatch')
        q3_axes={(t['m'],t['axis']):t for t in join['rows']};terms=[]
        for i,p in enumerate(d['axes']):
            m=p['m'];q=q3_axes[m,p['axis']];round_=conf['rounds'][m]
            require(p['configuration_id']==conf['id'] and all(p[k]==q[k]for k in ('physical_id','original_candidate_id','total_candidate_id','N')),
                    'Coarse PF physical/axis/N binding mismatch')
            require(p['q']==round_['q_PF'] and float(frac(round_['T'])).hex()==p['T_hex']==p['T'].hex() and
                    float(frac(round_['actual_delta_pf'])).hex()==p['actual_delta_hex']==p['actual_delta'].hex(),
                    'Coarse PF actual time/width mismatch')
            require(p['kind'] in ('DIRECT_COMPILED_DETERMINISTIC','CONDITIONAL_AFFINE_PRICE_REFERENCE'),
                    'Unknown coarse PF price kind')
            if p['kind']=='CONDITIONAL_AFFINE_PRICE_REFERENCE':
                require(p['actual_delta_hex']==d['model_actual_delta_hex'] and p['model_error_bound_RZ'] is None,
                        'Coarse PF model domain/error qualification mismatch')
            terms.append(dict(m=m,axis=p['axis'],N=p['N'],price_source=dict(loc,selector=['axes',i])))
        require(sum(p['kind']=='DIRECT_COMPILED_DETERMINISTIC'for p in d['axes'])==d['accepted_direct_axes'] and
                sum(p['kind']=='CONDITIONAL_AFFINE_PRICE_REFERENCE'for p in d['axes'])==d['model_axes'],
                'Coarse PF direct/model coverage mismatch')
        self.make(dict(case,_source=input_loc),d['allocation'],prior['q3_status'],d['conditional_cost']['conditional_total_RZ'],
            prior['MSE_lower'],prior['MSE_upper'],prior['adopted_evaluation'],loc,terms)
        priced=self.rows.pop();merged=deepcopy(prior);merged.update(priced)
        merged.update(result_sources=prior['result_sources']+[loc],labels=deepcopy(prior['labels']),
            price_availability_history=[dict(status=prior['status'],sources=prior['result_sources'])],
            price_scenario=d['status'],qualified_for_candidate=False,
            price_evidence=dict(source=loc,quantity=d['quantity'],model_source=d['model_source'],
                native_source=d['native_source'],wrapper_source=d['wrapper_source'],phase_sources=deepcopy(d['phase_sources']),
                holdout_source=dict(loc,selector=['holdout']),direct_axes=d['accepted_direct_axes'],model_axes=d['model_axes']),
            saved_cost_breakdown=deepcopy(d['conditional_cost']))
        for p,saved in zip(merged['price_axes'],d['axes']):
            p.update(price_kind=saved['kind'],region=saved['region'],unit_price_RZ=deepcopy(saved['unit_price_RZ']),
                model_error_bound_RZ=deepcopy(saved['model_error_bound_RZ']),price_evidence=deepcopy(saved['price_evidence']))
        self.rows[self.rows.index(prior)]=merged

    def pauli_model_prices(self,loc,d):
        """Attach a published model total to its already evaluated allocation.

        The 26 saved prices and total are copied, never predicted or multiplied.
        Calibration receipts remain evidence for a separate non-round cell.
        """
        require(d['kind']=='MODEL_REFERENCE_PRICE_NOT_E_C' and d['qualified_for_candidate'] is False,
                'Pauli model reference must retain its unqualified status')
        require(all(d[k] is None for k in ('E_C','CI','G')),'Do not promote a Pauli model reference to E[C]/G')
        require(len(d['rows'])==26 and {(r['m'],r['axis'])for r in d['rows']}==
                {(m,a)for m in range(13)for a in ('X','Y')},'Pauli model needs 26 unique axes')
        q3_loc={k:v for k,v in d['rows'][0]['b_index_source'].items()if k!='selector'}
        self.add(q3_loc)  # Also works when the price input precedes the Q3 input.
        join=self.r.read(q3_loc);inp=self.r.read(join['source']);case=inp['cases'][0]
        conf=self.r.read(case_ref(case));vector=shots(case['allocation'])
        require(self.r.read(d['configuration_source'])==conf,'Pauli price configuration SOURCE mismatch')
        require(d['total_view']['id']==case['total_configuration_id'],'Pauli price total view mismatch')
        require(d['VP']['N']==vector['X']==vector['Y'],'Pauli model total belongs to a different allocation')
        prior=next(r for r in self.rows if r['configuration_id']==conf['id'] and
                   r['total_configuration_id']==case['total_configuration_id'] and r['N_X']==vector['X'] and r['N_Y']==vector['Y'])
        if prior['conditional_cost'] is not None:
            require(frac(prior['conditional_cost'])==frac(d['T_LD3']),'Conflicting Pauli saved model total')
        binding_source=ref(loc['commit'],str(PurePosixPath(loc['path']).parent/'BINDING.json'))
        binding=self.r.read(binding_source);physical=binding['request']['physical_cell']
        require(physical['profile']==conf['spec']['profile'] and physical['split']==conf['spec']['split'],
                'Pauli model calibration compiler/split binding mismatch')
        require(binding['configuration']['id']==conf['id'] and binding['total_view']['id']==case['total_configuration_id'],
                'Pauli calibration identity mismatch')
        terms=[]
        for r in d['rows']:
            q3=self.r.read(r['b_index_source']);p=self.r.read(r['price_source']);m=r['m']
            require(self.r.commit(r['b_index_source']['commit'])==self.r.commit(q3_loc['commit']) and
                    r['b_index_source']['path']==q3_loc['path'],'Pauli Q3 index SOURCE mismatch')
            require(all(r[k]==q3[k]for k in ('m','axis','N','original_configuration_id','total_configuration_id',
                    'original_candidate_id','total_candidate_id','binding_id','allocation_id')),'Pauli price/Q3 axis binding mismatch')
            require(r['binding_id']==case['binding_id'] and r['allocation_id']==case['allocation']['allocation_id'],
                    'Pauli price binding/allocation mismatch')
            require(r['q']==p['q']==conf['rounds'][m]['q_PF'] and r['R']==p['R']==conf['rounds'][m]['R_outer'],
                    'Pauli model q/R mismatch')
            require(r['original_candidate_id']==p['physical_id']==conf['rounds'][m]['candidate']['id'],
                    'Pauli model physical candidate mismatch')
            require(r['value_source']==p['value_source']=='MODEL_APPLICATION_FROM_Q8R660_DECOMPOSITION',
                    'Round model application is not a direct calibration receipt')
            require(all(frac(r[k])==frac(p[k])for k in ('C_ref_exact','N_times_C_ref_exact')),
                    'Pauli saved price/weighted-term mismatch')
            price_doc=self.r.document(r['price_source']['commit'],r['price_source']['path'])
            require(price_doc['configuration_id']==conf['id'] and frac(price_doc['T_LD3'])==frac(d['T_LD3']),
                    'Pauli saved total SOURCE mismatch')
            terms.append(dict(m=m,axis=r['axis'],N=r['N'],price_source=r['price_source']))
        self.make(dict(case,_source=join['source']),case['allocation'],prior['q3_status'],d['T_LD3'],
            prior['MSE_lower'],prior['MSE_upper'],prior['adopted_evaluation'],loc,terms)
        priced=self.rows.pop()
        for key in ('q3_source','q3_axis_sources','labels'):
            priced[key]=deepcopy(prior[key])
        priced['result_sources']=prior['result_sources']+[loc]
        priced['price_availability_history']=prior.get('price_availability_history',[])+(
            [dict(status='PRICE_WAITING',sources=prior['result_sources'])]if prior['conditional_cost'] is None else [])
        priced.update(price_scenario=d['kind'],qualified_for_candidate=False,CI=None,
            price_evidence=dict(source=loc,coefficient_source=d['coefficient_source'],rule=d['rule'],
                binding_source=binding_source,lowering=physical['lowering'],scalar=deepcopy(binding['scalar']),
                calibration_cell=deepcopy(d['calibration_cell']),
                direct_calibration_paths=len({r['path_id']for r in d['calibration_cell']['rows']}),
                round_direct_prices=0,round_model_prices=26,qualified_for_candidate=False))
        by_axis={(r['m'],r['axis']):(i,r)for i,r in enumerate(d['rows'])}
        for p in priced['price_axes']:
            i,r=by_axis[p['m'],p['axis']]
            p.update(price_kind='MODEL',qualified_for_candidate=False,model_application=r['value_source'],
                C_ref_exact=r['C_ref_exact'],q=r['q'],R=r['R'],total_candidate_id=r['total_candidate_id'],
                binding_id=r['binding_id'],allocation_id=r['allocation_id'],
                q3_axis_source=r['b_index_source'],price_handoff_source=dict(loc,selector=['rows',i]))
        self.rows[self.rows.index(prior)]=priced

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
        differences={method:{benchmark:enc(frac(item['conditional_cost'])-frac(best[benchmark]['conditional_cost']))
            for benchmark in ('FULL_PF','TRUNCATED_PF') if best[benchmark]['conditional_cost'] is not None}
            for method,item in best.items() if item['conditional_cost'] is not None}
        kappa2=[r for r in rows if r['method']=='PR_PAULI'and r['status']=='FEASIBLE'and
                frac(r.get('kappa_absolute',r['kappa']['value']))==2]
        kappa2_cost=min((frac(r['conditional_cost'])for r in kappa2),default=None)
        restricted=dict(PR_PAULI_KAPPA_ABSOLUTE_2=dict(method='PR_PAULI',kappa_absolute=enc(2),
            conditional_cost=enc(kappa2_cost)if kappa2_cost is not None else None,
            point_ids=[r['point_id']for r in kappa2 if frac(r['conditional_cost'])==kappa2_cost],
            scope='指定した保存候補集合内・絶対κ=2限定の最安適合点'))
        return dict(schema='SAVED_H4_RPE_CONDITIONAL_COST_CATALOG_V1',comparison=self.comparison,
            adopted_inputs=self.inputs,candidates=rows,method_minima=best,
            restricted_minima=restricted,
            current_minimum_differences=differences,
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
    lines+=['','現在の保存集合内最安同士の差（方式の費用−各基準、正なら基準より高額）：','',
        '|方式|−Full PF RZ|−B0I p7 RZ|','|---|---:|---:|']
    for method,diff in result.get('current_minimum_differences',{}).items():
        lines.append('|'+method+'|'+display(diff.get('FULL_PF'))+'|'+display(diff.get('TRUNCATED_PF'))+'|')
    for name,item in result.get('restricted_minima',{}).items():
        lines+=['',name+'：'+display(item['conditional_cost'])+' RZ（絶対κ=2限定。元κの倍率×2や方式全体の最安とは別）。',
            'point ID：'+', '.join(item['point_ids'])+'。']
    for r in result['candidates']:
        if r.get('search_status')=='INTERRUPTED_SEARCH_POLICY_CHANGE':
            lines+=['',r['label']+'：探索方針変更による途中終了（INTERRUPTED_SEARCH_POLICY_CHANGE）。収束値ではない。',
                'κ絶対値='+str(frac(r['kappa_absolute']))+'、元κ倍率='+str(frac(r['kappa_multiplier']))+'。',
                '保存総額と26軸MODEL適用summaryを受入。個別26単価は未保存/null、全26軸外挿、qualified_for_candidate=false。',
                'exact MSE上下界は公開hashと一致した最安点1件のOUTCOMEを保持し、再評価していない。']
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
    if any(r['L_D']==3 and r.get('price_evidence',{}).get('round_model_prices')==26 for r in result['candidates']):
        lines+=['','PR-Pauli LD3 VPは保存Q3とAの26軸モデル価格を接続した条件付き参照総額。直接取得は別較正cellの2pathのみ、qualified_for_candidate=falseを保持する。']
    for r in result['candidates']:
        if r.get('price_scenario')=='CONDITIONAL_WIDTH_SPECIFIC_A_SHAPE_COST':
            c=r['saved_cost_breakdown']
            lines+=['','p7 Yoshida4・要求幅32h/5 A_SHAPE：直接16軸の部分額 '+display(c['direct_part_RZ'])+
                ' RZ、モデル10軸の部分額 '+display(c['model_part_RZ'])+' RZ。モデル価格全体の割合 '+c['model_percent_display']+
                '%と、総額中のq>80未検証外挿の割合 '+c['unvalidated_q_above_80_percent_display']+
                '%を区別する。価格誤差上界・CIは未取得。旧S2適合点とPR探索時の固定予算は変更しない。']
        if r.get('partial_price_axes'):
            lines+=['','LD9 V0の保存m6直接価格（別cohortの各軸4path平均、全13round費用はnull）：',
                '','|軸|LD9 RZ/shot|LD7参照 RZ/shot|差 LD9−LD7 RZ/shot|','|---|---:|---:|---:|']
            for p in r['partial_price_axes']:
                lines.append('|'+p['axis']+'|'+display(p['C_RZ'])+'|'+display(p['reference_C_RZ'])+'|'+display(p['difference_LD9_minus_LD7'])+'|')
        if r['point_id']in result['method_minima']['PR_DF']['point_ids'] and 'saved_cost_breakdown'in r:
            lines+=['',r['label']+'の費用の'+r['saved_cost_breakdown']['m7_m12_percent_display']+
                '%はq>64のm7〜m12へのモデル外挿（元SUMMARYの保存内訳）。qualified_for_candidate=false。価格モデル・配分・Q3・較正receiptの固定locatorはCATALOG.jsonに保持。']
        if r['point_id']in result['method_minima']['PR_PAULI']['point_ids'] and 'saved_PF_comparisons'in r:
            comparison=r['saved_PF_comparisons']
            lines+=['','PR-Pauli '+r['label']+'の保存費用差（T−PF基準）：Full PF S2 '+display(comparison['difference_T_minus_Full_PF'])+
                ' RZ、p7 S2 '+display(comparison['difference_T_minus_P7'])+' RZ。全26価格はモデル参照、qualified_for_candidate=false。']
        if r['point_id']in result['method_minima']['PR_PAULI']['point_ids'] and 'saved_split_comparison'in r:
            lines+=['','PR-Pauli LD5 '+r['label']+'のLD3同名配分に対する保存差（LD5−LD3）は '+
                display(r['saved_split_comparison']['difference_LD5_minus_LD3'])+
                ' RZ。採用したモデル参照価格規則での数値差であり、価格精度を保証する差ではない。全26軸MODEL・qualified_for_candidate=falseを保持する。']
    lines+=['','再実行コマンドと追加方法は親directoryのHANDOFF.mdを参照。','']
    return '\n'.join(lines)
