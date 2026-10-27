"""Saved winner transcription and parameter metadata only; no scientific analysis.

Never computes G/N/B/SE, calls a study runner, or ranks costs. Only explicit saved
winner rows are used. Min/max below apply to discrete PARAMETER VALUES, not costs.
"""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = dict(common='f75a7c25d584b69d90de454731a7f343a421e059',
           long='155529e20a8fac39e00786c40b57830b71203c4f',
           screening='2e518b86e7c052dafb20dece0de7ffa6c7c85f8f',
           seven='bb97a93d65992fdb53e642dde1584dd7d5b811d7',
           h8='681ec23c93fdd1090d325908f4b155a79115afdf',
           closed='862d732fba0a5f353c522a9219dec78ae2f21a4b',
           pauli='b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b',
           h4search='d3fdfad880902b7ff35b710d22e1c9759f7087c2')
INPUTS = {}
CACHE = {}
DOMAINS = {}
WINNERS = []
GAPS = []


def guard(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in ('numpy','scipy','qiskit','openfermion','pyscf','trotterlib'):
        raise RuntimeError('Only saved rows and static source may be read')


def load(ref, path):
    key = ref, path
    if key in CACHE:
        return CACHE[key]
    data = subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)
    INPUTS[ref + ':' + path] = hashlib.sha256(data).hexdigest()
    if path.endswith('.gz'):
        data = gzip.decompress(data)
    value = (list(csv.DictReader(io.StringIO(data.decode()))) if path.endswith('.csv') else
             data.decode() if path.endswith(('.md','.py')) else json.loads(data))
    CACHE[key] = value
    return value


def exact(v):
    if isinstance(v, dict):
        return str(v['numerator']) + '/' + str(v['denominator'])
    return '' if v is None else str(v)


def num(v):
    return None if v in ('', None, 'null') else int(v)


def record(candidate, epsilon, G='', SE='', n='', cohort='', eligible=True, priced=True, source_row=''):
    r = dict(candidate)
    for key in ('q','prefix','R_outer','r','K','s'):
        if key in r:
            r[key] = num(r[key])
    if 'r' in r and r.get('q') is not None and r['r'] is not None:
        r['s'] = r['q'] * r['r']  # Event count identity, not a cost proxy.
    r.update(epsilon=float(epsilon), eligible=eligible, priced=priced, G=exact(G), G_SE=SE,
             n=n, cohort=cohort, source_row=source_row)
    return r


def winner(dataset, row, scope, source, comparator='', comparator_G='', note=''):
    WINNERS.append(dict(dataset=dataset, row=row, formula_scope=scope, source=source,
                        comparator_id=comparator, comparator_G=exact(comparator_G), note=note))


def scientific_source(dataset):
    if dataset=='DF_H46_ORIGINAL':
        return json.dumps(load(REF['long'],'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json')['source_commits'],sort_keys=True)
    if dataset=='DF_H46_PLUS7_SAVED_ALL':return 'cab73373593ba7c113c1af018b1dc3077c8e78fa for seven added prices; inherited original prices retain original provenance'
    if dataset=='DF_H8_RESOURCE_MAP':return 'c31ef424175c02379eafebe8fc6c7548c1d9aabe; seed source cb8a56c6b2a772827275f9200b02638030f33f72'
    if dataset=='DF_H8_LATER_FIXED_CONFIRMATION':return '745a674546f2cb6258526fbfde1d9a00601689f8; see candidate-specific prior binding'
    if dataset=='DF_H246_COMMON_AND_OUTER32':return 'PF H2=1b35f9ac2efab4c15bc00ee2d3c81643aca422f9; H4=e2179f68f6979f3ad26392bff877fb422ddfb744; H6=bdf6772aade7063c65d69f24dfdc5e303a2819a2; PR original SOURCE in saved evaluation'
    return 'Mixed saved family sources; preserve candidate/cohort-specific references in latest15 and source_references.json'


def from_h46():
    base = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
    rows = load(REF['long'], base + 'candidate_precision_accounting.csv')
    columns = ('id','system','method','formula','prefix','q','R_outer','K','T')
    domain = [record({k:r[k] for k in columns}, r['epsilon'], r.get('G'), r.get('G_SE'),
                     r.get('n'), r.get('cohort'), r['status']=='ELIGIBLE', bool(r.get('G')),
                     'csv_line=' + str(i+2)) for i,r in enumerate(rows)]
    DOMAINS['DF_H46_ORIGINAL'] = domain
    saved = load(REF['long'], base + 'best_configurations.json')
    for i, value in enumerate(saved):
        b = value['best']
        if not b:
            continue
        r = record(b, value['epsilon'], b['G_exact'], b.get('G_SE'), b.get('n'), b.get('cohort'))
        r['family'] = value['family']
        other = next(v['best'] for v in saved if (v['system'],v['T'],v['epsilon'],v['formula_group'],v['family']) ==
                     (value['system'],value['T'],value['epsilon'],value['formula_group'],'PF' if value['family']=='PR' else 'PR'))
        winner('DF_H46_ORIGINAL',r,value['formula_group'],REF['long']+':'+base+'best_configurations.json#/'+str(i),
               other['id'] if other else '', other['G_exact'] if other else '',
               'Saved priced-subset best; PF combines Full/B0/B0I; no new family reranking')
    # The seven-candidate study saved extended ALL-formula PR winners only.
    pre = 'analysis/h46_seven_cost_readiness_20261027/results/'
    additions = load(REF['seven'], pre + 'candidate_costs.csv')
    added_ids = {r['id'] for r in additions}
    overlay = []
    for row in domain:
        a = next((a for a in additions if a['id']==row['id'] and float(a['epsilon'])==row['epsilon'] and a['cohort'].endswith('EXPLORATION')), None)
        overlay.append(dict(row,priced=True,G=a['G_exact'],G_SE=a['G_SE'],n=a['n'],cohort=a['cohort']) if a else dict(row))
    DOMAINS['DF_H46_PLUS7_SAVED_ALL'] = overlay
    cells = load(REF['seven'], pre + 'extended_cells.csv')
    for i, cell in enumerate(cells):
        identity = cell['extended_PR_id']
        r = next(dict(r) for r in overlay if r['id']==identity and r['epsilon']==float(cell['epsilon']))
        r['G'] = cell['extended_PR_G_exact'];r['family']='PR'
        winner('DF_H46_PLUS7_SAVED_ALL',r,'ALL',REF['seven']+':'+pre+'extended_cells.csv#csv_line='+str(i+2),
               cell['old_PF_id'],cell['old_PF_G'],'Saved extended exploration point winner; confirmation is separate')
        # A saved ALL winner is necessarily the winner within its own formula.
        # Other formula-specific updated rankings were not saved: do not invent them.
        if identity in added_ids:
            winner('DF_H46_PLUS7_SAVED_ALL',dict(r),'WITHIN_WINNER_FORMULA_DERIVED_FROM_SAVED_ALL',
                   REF['seven']+':'+pre+'extended_cells.csv#csv_line='+str(i+2),cell['old_PF_id'],cell['old_PF_G'],
                   'Logical restriction of saved ALL winner; no cost reranking')
    GAPS.append('PLUS7: no saved updated minima for other formulas or separate Full/B0/B0I families; original 320 rows retained as separate snapshot.')


def from_common():
    base='artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/'
    domain=[]
    for system in ('H2','H4','H6'):
        ep=f'artifacts/resource_applicability/track_a_pf_common_{system}_results/2026-10-22/PF_endpoint_costs.json.gz'
        for i,v in enumerate(load(REF['common'],ep)):
            c=dict(v['candidate'],T=.8);c['id']='_'.join(str(c[k]) for k in ('system','method','prefix','q','formula'))
            domain.append(record(c,v['epsilon'],v.get('total_RZ_exact'),None,1,'DETERMINISTIC',v['status']=='ELIGIBLE',bool(v.get('price')),ep+'#/'+str(i)))
    ev='artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json'
    evaluation=load(REF['common'],ev)
    by_id={g['candidate']['id']:g['candidate'] for g in evaluation['groups']}
    for i,g in enumerate(evaluation['groups']):
        c=g['candidate']
        if c['method']!='PR':continue
        for v in g['rows']:
            if v['mode']!='USUAL_SIGNED':continue
            domain.append(record(dict(c,T=.8,R_outer=c['r_value']),v['epsilon'],v.get('total_RZ_exact'),v.get('total_RZ_SE'),v.get('n'),
                                 'EXPLORATION',v['status']=='ELIGIBLE',True,ev+'#/groups/'+str(i)))
    DOMAINS['DF_H246_COMMON_AND_OUTER32']=domain
    for i,v in enumerate(load(REF['common'],base+'comparisons_60.json')):
        pf,pr=v['PF']['best'],v['PR_saved'];scope='ALL' if v['formula']=='ALL_FORMULAS' else v['formula']
        c=dict(pf['candidate'],T=.8);c['id']='_'.join(str(c[k]) for k in ('system','method','prefix','q','formula'))
        r=record(c,v['epsilon'],pf['total_RZ_exact'],None,1,'DETERMINISTIC');r['family']='PF'
        winner('DF_H246_COMMON_AND_OUTER32',r,scope,REF['common']+':'+base+'comparisons_60.json#/'+str(i)+'/PF',pr['candidate_id'],pr['total_RZ'])
        c=dict(by_id[pr['candidate_id']],T=.8,R_outer=32)
        r=record(c,v['epsilon'],pr.get('total_RZ_numerator')+'/'+pr.get('total_RZ_denominator'),pr.get('total_RZ_SE'),pr.get('n'),'EXPLORATION');r['family']='PR'
        winner('DF_H246_COMMON_AND_OUTER32',r,scope,REF['common']+':'+base+'comparisons_60.json#/'+str(i)+'/PR_saved','_'.join(str(pf['candidate'][k]) for k in ('system','method','prefix','q','formula')),pf['total_RZ_exact'])


def parse_id(identity, system, pauli=False):
    p=re.search(r'_?p(\d+)_q(\d+)',identity)
    assert p, identity
    r=re.search(r'_r(\d+)_K(\d+)',identity)
    method='PR' if r else 'B0I' if identity.startswith('B0I') else 'B0' if identity.startswith('B0') else 'FULL'
    formula='4th' if '_4th' in identity else '2nd'
    c=dict(id=identity,system=system,T=.8,method=method,formula=formula,prefix=int(p[1]),q=int(p[2]))
    if r:c.update(r=int(r[1]),K=int(r[2]))
    if pauli and r:c['method']='RTE' if c['prefix']==0 else 'HYBRID'
    return c


def from_h8():
    base='artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/'
    status=load(REF['h8'],base+'candidate_status_by_precision.json')['rows']
    domain=[record(parse_id(v['cell_id'],'H8'),v['epsilon'],eligible=v['status']=='ELIGIBLE',priced=v['status']=='ELIGIBLE',source_row='rows/'+str(i)) for i,v in enumerate(status)]
    DOMAINS['DF_H8_RESOURCE_MAP']=domain
    for i,v in enumerate(load(REF['h8'],base+'precision_comparison.json')['rows']):
        for phase in ('exploration','confirmation'):
            for family,b in v[phase].items():
                r=record(parse_id(b['cell_id'],'H8'),v['epsilon'],b['G_point'],b['G_standard_error'],b['axes']['cosine']['n'],phase)
                r['family']={'B2':'PR','B1':'FULL','B0':'B0'}[family]
                other=v[phase]['B1' if family=='B2' else 'B2']
                winner('DF_H8_RESOURCE_MAP',r,'ALL_AVAILABLE_FORMULAS',REF['h8']+':'+base+'precision_comparison.json#/rows/'+str(i)+'/'+phase+'/'+family,other['cell_id'],other['G_point'])
    # Later q16/p8 results are fixed confirmations, not a fresh exhaustive grid.
    later='artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv'
    rows=load(REF['closed'],later)
    DOMAINS['DF_H8_LATER_FIXED_CONFIRMATION']=[record(parse_id(v['candidate'],'H8'),v['epsilon'],v['mean_exact'],v['paired_SE'],v['n'],'confirmation') for v in rows if v['mode']=='primary']
    for i,v in enumerate(rows):
        if v['mode']!='primary':continue
        r=record(parse_id(v['candidate'],'H8'),v['epsilon'],v['mean_exact'],v['paired_SE'],v['n'],'confirmation');r['family']='PR'
        winner('DF_H8_LATER_FIXED_CONFIRMATION',r,'FIXED_CANDIDATE_NOT_GLOBAL_MIN',REF['closed']+':'+later+'#csv_line='+str(i+2),v['fixed_PF_id'],v['fixed_PF_G_exact'],
               'Existing q16/p8 evidence; closed rare-RTE study; singleton confirmation domain is not the historical search boundary')
    GAPS.append('H8 later fixed confirmation contains q16/p8 but is not a complete registered grid. No q16 winner for epsilon .001 is inferred from epsilon .0005.')


def from_pauli():
    pre='artifacts/pauli_h2_h6_comparison/2026-10-25/'
    domains={s:[] for s in ('H2','H4','H6')}
    specs=[(REF['pauli'],pre+'registered_candidate_precision_table.csv'),
           (REF['pauli'],'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_precision_table.csv'),
           (REF['h4search'],'artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_precision_table.csv'),
           (REF['pauli'],'artifacts/pauli_rte_event_extension/2026-10-26/candidate_precision_table.csv'),
           (REF['pauli'],'artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv')]
    for ref,path in specs:
        for i,v in enumerate(load(ref,path)):
            s=v.get('molecule','H4');identity=v['candidate_id'];c=parse_id(identity,s,True)
            if not re.search(r'_r\d+_K',identity) and s=='H4':c['method']='FULL' if c['prefix']==184 else 'B0'
            good=v['eligibility']=='CONDITIONAL_GUARD_ELIGIBLE'
            domains[s].append(record(c,v['epsilon_sig'],v.get('G_exact'),v.get('G_SE'),v.get('price_n',v.get('n')),
                                     v.get('cohort','deterministic' if c['method'] not in ('HYBRID','RTE') else ''),good,v.get('G_exact') not in (None,'','null'),ref+':'+path+'#csv_line='+str(i+2)))
    path='artifacts/pauli_rte_event_extension/2026-10-26/candidate_status.csv'
    for i,v in enumerate(load(REF['pauli'],path)):
        if v['target_status']=='ACQUIRED':continue
        c=parse_id(v['candidate_id'],v['molecule'],True)
        for e in (.05,.01,.005,.001,.0005):
            r=record(c,e,eligible=None,priced=False,source_row=REF['pauli']+':'+path+'#csv_line='+str(i+2))
            r['signal_status']=v['target_status'];domains[v['molecule']].append(r)
    # H4 deterministic registry is separate from the PR search.
    path='artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_precision_table.csv'
    # Some publications provide only a saved deterministic winner table. The
    # caller must not fabricate a domain from those winner rows.
    GAPS.append('Pauli H4 deterministic registry (106 configurations) is separate from PR. Latest15 quotes minima and does not re-rank all four methods. B0/B0I are equivalent in this Pauli lane.')
    for s,dom in domains.items():DOMAINS['PAULI_'+s]=dom
    path='docs/research/data/pauli_jw_summary/latest15.csv'
    for i,v in enumerate(load(REF['pauli'],path)):
        for label,family in [('Hybrid','HYBRID'),('RTE','RTE'),('PF','FULL'),('B0','B0')]:
            identity=v.get(label+'_candidate_id')
            if not identity:continue
            c=parse_id(identity,v['molecule'],True);c['method']=family
            r=record(c,v['epsilon_sig'],v[label+'_G_exact'],v.get(label+'_G_SE'),v.get(label+'_n'),v.get(label+'_cohort'));r['family']=family
            r['underlying_result_commit']=v['result_commit'] if label in ('Hybrid','RTE') else v['deterministic_result_commit']
            r['underlying_result_file']=v['result_file'] if label in ('Hybrid','RTE') else v['deterministic_result_file']
            other='RTE' if label=='Hybrid' else 'Hybrid'
            winner('PAULI_'+v['molecule'],r,'S2_ONLY;SAVED_FAMILY_MINIMUM',REF['pauli']+':'+path+'#csv_line='+str(i+2)+';family='+label,
                   v[other+'_candidate_id'],v[other+'_G_exact'],'Central-rotation; rho=0 hypothetical; independent family minima quoted; no four-way reranking')


def family(row):
    return row.get('family') or ('PR' if row['method']=='PR' else row['method'])


def select_domain(win):
    r=win['row'];wanted=family(r)
    return [v for v in DOMAINS[win['dataset']] if v['system']==r['system'] and float(v['T'])==float(r['T']) and v['epsilon']==r['epsilon']
            and ((v['method']!='PR') if wanted=='PF' else v['method']==wanted)
            and v['formula']==r['formula']]


def feasibility(dataset, parameter, side, r):
    if parameter=='q' and side in ('LOWER','FIXED') and r['q']==1:return 'PHYSICAL_INTEGER_MINIMUM; q<1 invalid'
    if parameter=='prefix' and r['method']=='FULL':return 'FULL_MODEL_RANK_OR_TERM_COUNT; not a tunable split'
    if parameter=='prefix' and r.get('prefix')==0 and side in ('LOWER','FIXED'):return 'PHYSICAL_EMPTY_PREFIX; negative prefix invalid'
    if side=='FIXED':return 'FIXED_DESIGN_OR_SINGLETON; not evidence that both directions improve'
    if dataset.startswith('DF_H46') and parameter=='R_outer':
        return 'ABS_TIME_PER_OUTER integer >= piece_count (S2=1,Y4=3,m5=11); same scientific inputs; new signal and price required'
    if parameter=='K':return 'Next even cutoff supported by generic law/builder; new law/signal/correctness/price evidence required'
    if parameter=='prefix':return 'Check same saved Hamiltonian split and input/control compatibility; no transfer across rank policies'
    return 'Requires separate registered scope, signal qualification and compatible compiled price; no extrapolated G claim'


def inventory():
    output=[]
    all_winners={(w['dataset'],w['row']['system'],str(w['row']['T']),w['row']['epsilon'],family(w['row']),w['row'].get('cohort')):w
                 for w in WINNERS if w['formula_scope']=='ALL'}
    for w in WINNERS:
        r=w['row'];dom=select_domain(w)
        for parameter in ('q','R_outer','r','K','prefix','s'):
            value=r.get(parameter)
            if value is None:continue
            for tier in ('REGISTERED','ELIGIBLE','PRICED_REGISTERED','PRICED_ELIGIBLE'):
                candidates=[v for v in dom if (tier in ('REGISTERED','PRICED_REGISTERED') or v['eligible']) and
                            (not tier.startswith('PRICED') or v['priced'])]
                values=sorted({v[parameter] for v in candidates if v.get(parameter) is not None})
                if not values:continue
                side='FIXED' if len(values)==1 and value==values[0] else 'LOWER' if value==values[0] else 'UPPER' if value==values[-1] else None
                if not side:continue
                # Matched neighbors hold every other scientific parameter fixed.
                varying={'q','r','s'} if parameter=='s' else {parameter,'s'}
                match=[v for v in dom if v['formula']==r['formula'] and v['method']==r['method'] and
                       all(v.get(k)==r.get(k) for k in ('q','R_outer','r','K','prefix') if k not in varying)]
                near_values=sorted({v[parameter] for v in match if v.get(parameter) is not None and v[parameter]!=value})
                nearest=([max(v for v in near_values if v<value)] if any(v<value for v in near_values) else [])+([min(v for v in near_values if v>value)] if any(v>value for v in near_values) else [])
                neighbors=[{k:v.get(k) for k in ('id',parameter,'eligible','signal_status','priced','G','G_SE','cohort','source_row')} for v in match if v.get(parameter) in nearest]
                all_sets={t:sorted({v[parameter] for v in dom if v.get(parameter) is not None and
                          (t in ('REGISTERED','PRICED_REGISTERED') or v['eligible']) and (not t.startswith('PRICED') or v['priced'])})
                          for t in ('REGISTERED','ELIGIBLE','PRICED_REGISTERED','PRICED_ELIGIBLE')}
                overall=all_winners.get((w['dataset'],r['system'],str(r['T']),r['epsilon'],family(r),r.get('cohort')))
                if overall is None:
                    overall=next((a for a in WINNERS if a['dataset']==w['dataset'] and a['row']['system']==r['system'] and str(a['row']['T'])==str(r['T']) and a['row']['epsilon']==r['epsilon'] and family(a['row'])==family(r) and a['formula_scope']=='ALL'),None)
                source_evidence=('analysis/h46_seven_cost_readiness_20261027/results/execution_provenance.json' if w['dataset']=='DF_H46_PLUS7_SAVED_ALL'
                    else 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json' if w['dataset']=='DF_H46_ORIGINAL'
                    else 'artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/source_result_references.json' if w['dataset']=='DF_H246_COMMON_AND_OUTER32'
                    else 'docs/research/track_a_h8_resource_map_result_v1.md' if w['dataset']=='DF_H8_RESOURCE_MAP'
                    else 'docs/research/track_a_h8_new_pr_confirmation_results_20261019.md' if w['dataset']=='DF_H8_LATER_FIXED_CONFIRMATION'
                    else 'docs/research/data/pauli_jw_summary/source_references.json')
                candidate_key=json.dumps([r['system'],r['T'],r['formula'],r['method'],r.get('q'),r.get('prefix'),r.get('R_outer'),r.get('r'),r.get('K')],separators=(',',':'))
                output.append(dict(record_type='FIXED_CONTROL_REFERENCE' if w['formula_scope']=='FIXED_CANDIDATE_NOT_GLOBAL_MIN' else 'SAVED_BEST_BOUNDARY',
                    dataset=w['dataset'], representation='Pauli-JW' if w['dataset'].startswith('PAULI') else 'DF-native',
                    system=r['system'],T=r['T'],epsilon=r['epsilon'],family=family(r),actual_method=r['method'],
                    formula=r['formula'],formula_scope=w['formula_scope'],candidate_id=r['id'],
                    parameter_domain_condition='same system/T/epsilon/family/actual formula; selection scope is separate',
                    physical_candidate_key=('Pauli-JW' if w['dataset'].startswith('PAULI') else 'DF_H46' if w['dataset'].startswith('DF_H46') else w['dataset'])+':'+candidate_key,
                    q=r.get('q'),R_outer=r.get('R_outer'),r=r.get('r'),K=r.get('K'),prefix=r.get('prefix'),s=r.get('s'),
                    parameter=parameter,boundary_basis=tier,side=side,boundary_value=value,
                    registered_values=json.dumps(all_sets['REGISTERED']),eligible_values=json.dumps(all_sets['ELIGIBLE']),
                    priced_registered_values=json.dumps(all_sets['PRICED_REGISTERED']),priced_eligible_values=json.dumps(all_sets['PRICED_ELIGIBLE']),
                    G=r['G'],G_SE=r.get('G_SE'),n=r.get('n'),cohort=r.get('cohort'),comparison_id=w['comparator_id'],comparison_G=w['comparator_G'],
                    all_formula_same_family_id=overall['row']['id'] if overall else '',all_formula_same_family_G=overall['row']['G'] if overall else '',
                    matched_adjacent_rows=json.dumps(neighbors,separators=(',',':')),
                    matched_unpriced_eligible_ids=json.dumps(sorted({v['id'] for v in match if v['eligible'] and not v['priced']})),
                    unpriced_eligible_distinct=len({v['id'] for v in dom if v['eligible'] and not v['priced']}),
                    technical_feasibility=feasibility(w['dataset'],parameter,side,r),
                    uncertainty='FINITE_SAMPLE; no population tail or selection-adjusted guarantee' if r['method'] in ('PR','HYBRID','RTE') else 'DETERMINISTIC_SAVED_PRICE; signal policy remains conditional',
                    result_commit=w['source'].split(':',1)[0],underlying_result_commit=r.get('underlying_result_commit',''),underlying_result_file=r.get('underlying_result_file',''),
                    scientific_SOURCE=scientific_source(w['dataset']),scientific_SOURCE_evidence=source_evidence,
                    saved_winner_row=w['source'],scope_note=w['note']))
    return output


def missing_minimum_rows():
    """Represent absent saved selectors explicitly rather than rank costs anew."""
    out=[]
    for name in ('DF_H46_ORIGINAL','DF_H246_COMMON_AND_OUTER32'):
        groups={}
        for r in DOMAINS[name]:
            if r['method']=='PR':continue
            groups.setdefault((r['system'],r['T'],r['epsilon'],r['method'],r['formula']),[]).append(r)
        for (s,t,e,m,f), group in groups.items():
            sets={tier:{p:sorted({r[p] for r in group if r.get(p) is not None and
                      (tier in ('REGISTERED','PRICED_REGISTERED') or r['eligible']) and (not tier.startswith('PRICED') or r['priced'])})
                      for p in ('q','prefix')} for tier in ('REGISTERED','ELIGIBLE','PRICED_REGISTERED','PRICED_ELIGIBLE')}
            out.append(dict(record_type='NO_SAVED_SEPARATE_METHOD_MINIMUM',dataset=name,representation='DF-native',
                system=s,T=t,epsilon=e,family=m,formula=f,formula_scope='METHOD_SPECIFIC_NOT_RERANKED',
                parameter='UNRESOLVED_WINNER',side='UNKNOWN',registered_values=json.dumps(sets['REGISTERED']),
                eligible_values=json.dumps(sets['ELIGIBLE']),priced_registered_values=json.dumps(sets['PRICED_REGISTERED']),priced_eligible_values=json.dumps(sets['PRICED_ELIGIBLE']),
                scope_note='Saved best selector combines Full/B0/B0I. Domain metadata known; method-specific G minimum not newly computed.'))
    out.append(dict(record_type='NO_SAVED_STUDY_IN_FIXED_REFERENCES',dataset='PAULI_H8',representation='Pauli-JW',system='H8',
                    parameter='ALL',side='UNKNOWN',scope_note='No H8 Pauli-JW resource result found in specified Pauli synthesis/source indexes; not ineligible or zero cost.'))
    return out


def evidence_receipts():
    for key,path in [('screening','analysis/h46_unpriced_screening_20261026/summary.json'),
                     ('screening','analysis/h46_unpriced_screening_20261026/verification.json'),
                     ('seven','analysis/h46_seven_cost_readiness_20261027/results/END_AUDIT.json'),
                     ('long','artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json'),
                     ('long','artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/completion_provenance.json'),
                     ('common','artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/summary.json'),
                     ('h8','artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/completion_audit.json'),
                     ('closed','docs/research/h8_rare_rte_closure_policy_20261021.md'),
                     ('closed','docs/research/track_a_h8_new_pr_confirmation_results_20261019.md'),
                     ('closed','artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18/current_PR_shots_cost_saved_literals.csv'),
                     ('h8','docs/research/track_a_h8_resource_map_result_v1.md'),
                     ('common','docs/research/track_a_h246_saved_cost_results_20261021.md'),
                     ('common','scripts/h2_outer32_contract.py'),
                     ('long','scripts/h46_new2_core.py'),
                     ('long','src/trotterlib/h46_m5_plan.py'),
                     ('long','src/trotterlib/rte.py'),
                     ('long','src/trotterlib/df_rte_qiskit.py'),
                     ('pauli','docs/research/pauli_jw_resource_evaluation_summary.md'),
                     ('pauli','docs/research/data/pauli_jw_summary/source_references.json'),
                     ('pauli','docs/research/data/pauli_jw_summary/verification.json')]:
        load(REF[key],path)


def check(rows):
    seen=set()
    for r in rows:
        if r['record_type'] not in ('SAVED_BEST_BOUNDARY','FIXED_CONTROL_REFERENCE'):continue
        key=r['dataset'],r['saved_winner_row'],r['formula_scope'],r['parameter'],r['boundary_basis']
        assert key not in seen;seen.add(key)
        vals=json.loads(r[{'REGISTERED':'registered_values','ELIGIBLE':'eligible_values','PRICED_REGISTERED':'priced_registered_values','PRICED_ELIGIBLE':'priced_eligible_values'}[r['boundary_basis']]])
        assert r['boundary_value'] in vals
        assert (r['side']=='FIXED' and len(vals)==1) or (r['side']=='LOWER' and r['boundary_value']==vals[0]) or (r['side']=='UPPER' and r['boundary_value']==vals[-1])
    h6=[r for r in rows if r.get('dataset')=='DF_H46_ORIGINAL' and r.get('family')=='PR' and r.get('system')=='H6'
        and float(r.get('T',0))==2 and r.get('formula_scope')=='ALL' and r.get('epsilon')==.0005 and r.get('boundary_basis')=='REGISTERED']
    assert any(r['parameter']=='R_outer' and r['side']=='LOWER' and r['boundary_value']==16 for r in h6)
    assert any(r['parameter']=='K' and r['side']=='UPPER' and r['boundary_value']==6 for r in h6)
    h8=[r for r in rows if r.get('dataset')=='DF_H8_RESOURCE_MAP' and r.get('family')=='PR' and r.get('epsilon')==.001 and r.get('cohort')=='confirmation' and r.get('boundary_basis')=='REGISTERED']
    assert {(r['parameter'],r['side']) for r in h8} >= {('q','UPPER'),('r','UPPER'),('K','UPPER'),('prefix','LOWER')}


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    sys.addaudithook(guard)
    from_h46();from_common();from_h8();from_pauli()
    for domain in DOMAINS.values():
        physical_prices={v['id'] for v in domain if v['priced']}
        for v in domain:
            v['priced']=v['id'] in physical_prices
    evidence_receipts()
    rows=inventory()+missing_minimum_rows();check(rows);write_csv(HERE/'BOUNDARY_INVENTORY.csv',rows)
    verification=dict(scope='SAVED_WINNERS_AND_DISCRETE_PARAMETER_METADATA_ONLY',new_scientific_calls=0,new_cost_rankings=0,
        G_N_B_SE_recomputed=False,registered_grid_extended=False,mandatory_STOP=True,
        winner_rows=len(WINNERS),inventory_rows=len(rows),record_types={kind:sum(r['record_type']==kind for r in rows) for kind in {r['record_type'] for r in rows}},
        boundary_checks='PASS; discrete metadata and source selectors only',
        datasets={name:dict(endpoint_rows=len(v),distinct_candidates=len({r['id'] for r in v})) for name,v in DOMAINS.items()},
        unresolved_saved_ranking_gaps=GAPS,inputs=INPUTS)
    (HERE/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='inputs'},ensure_ascii=False))


if __name__=='__main__':main()
