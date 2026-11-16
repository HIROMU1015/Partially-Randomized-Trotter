"""Publish an exact PR candidate/Q3/price join using C's existing generic reader.

No provider, compiler, fit, allocation generator or risk evaluator is imported.
Three publication phases avoid self-referencing commit hashes: prices, inputs,
join. Each input locator is immutable; the caller commits the produced files.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
INCONCLUSIVE = 'INCONCLUSIVE'
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def frac(value):
    if isinstance(value, dict):
        return Fraction(int(value['numerator']), int(value['denominator']))
    return Fraction(value)


def exact(value):
    value = frac(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def locator(root, value):
    require(isinstance(value, dict) and value.get('commit') and value.get('path'), 'Fixed Git locator required')
    fixed = subprocess.check_output(['git', 'rev-parse', value['commit']+'^{commit}'], cwd=root, text=True).strip()
    return dict(value, commit=fixed)


def read(root, value):
    loc = locator(root, value)
    result = json.loads(subprocess.check_output(['git', 'show', loc['commit']+':'+loc['path']], cwd=root))
    for key in loc.get('selector', []):
        result = result[int(key)] if isinstance(result, list) else result[key]
    return result


def shot_vector(allocation):
    rows = sorted(allocation['shots'], key=lambda row: row['m'])
    require(len(rows) == 13 and [r['m'] for r in rows] == list(range(13)), '13 unique shot rounds required')
    require(all(type(r['N_'+a]) is int and r['N_'+a] >= 1 for r in rows for a in ('X', 'Y')), 'Invalid shot vector')
    return {a: [r['N_'+a] for r in rows] for a in ('X', 'Y')}


def validated(root, request):
    """Bind saved raw results, exact prices and their original configuration."""
    sources = {k: locator(root, request[k]) for k in ('configuration', 'case_inputs', 'result', 'outcome', 'quote')}
    config, inputs, result, outcome, quote = (read(root, sources[k]) for k in ('configuration', 'case_inputs', 'result', 'outcome', 'quote'))
    cases = inputs['cases']
    label = request.get('label')
    matches = [c for c in cases if label is None or c['label'] == label]
    require(len(matches) == 1, 'One selected saved Q3 case required')
    case = matches[0];record = result.get('record') or {};identity = outcome.get('identity') or {}
    cid = case.get('original_configuration_id') or case.get('configuration_id')
    require(cid == config['id'] == quote['configuration_id'] == identity.get('configuration_id'), 'Configuration identity mismatch')
    require(config['spec']['method'] == 'PR', 'Only PR candidates are accepted')
    config_case = case.get('configuration') or case.get('configuration_source')
    require(config_case is not None and read(root, config_case) == config, 'Q3 configuration SOURCE mismatch')
    require(quote.get('complete') is True, 'Incomplete price quote must remain unpriced')
    require(quote.get('policy') in ('MODEL', 'AUTO', 'DIRECT'), 'Explicit price policy required')
    vector = shot_vector(case['allocation'])
    require(shot_vector(result['allocation']) == vector == shot_vector(record['allocation']), 'Raw Q3 allocation mismatch')
    require(identity.get('allocation') == case['allocation']['shots'] and identity.get('contract_matches') is True,
            'Adapter allocation/evaluation-contract gate mismatch')
    require(record.get('configuration_id') == case.get('total_configuration_id', case.get('total_view_id')) == identity.get('total_configuration_id'),
            'Total-view identity mismatch')
    require(record.get('canonical_binding') == case.get('binding_id') == identity.get('binding_id'), 'Total binding mismatch')
    require(identity.get('signal_packet') == case.get('signal_packet'), 'Signal packet identity mismatch')
    require(identity.get('evaluation_target') == record.get('evaluation_target'), 'Evaluation target mismatch')
    status = outcome['status'];risk = record['diagnostic']['deterministic_risk']
    require(status in (PASS, FAIL, INCONCLUSIVE), 'A saved valid model assessment is required; numerical UNKNOWN stays separate')
    require(result['status'] == status == result['adopted_q3_assessment']['status'], 'Saved Q3 verdict mismatch')
    require(all(frac(outcome[k]) == frac(risk[k]) for k in ('MSE_lower', 'MSE_upper')), 'Saved exact MSE mismatch')
    adopted = inputs['adopted_evaluation']
    tau = frac(adopted['MSE_upper'])
    lo,hi=frac(outcome['MSE_lower']),frac(outcome['MSE_upper'])
    require(0<=lo<=hi and tau>0, 'Invalid saved MSE interval')
    assessment=(hi<=tau)if status==PASS else(lo>tau)if status==FAIL else(lo<=tau<hi)
    require(assessment, 'Saved model verdict does not match its adopted threshold')
    if request.get('api'):
        sources['api'] = locator(root, request['api']);api = read(root, sources['api'])
        require(api == [record], 'Original API record differs from saved worker result')
    prices = quote['rows']
    require(len(prices) == 26 and {(p['m'],p['axis']) for p in prices} == {(m,a) for m in range(13) for a in ('X','Y')},
            '26 unique price axes required')
    for p in prices:
        m=p['m'];row=config['rounds'][m]
        require(p['classification'] in ('DIRECT_FINITE_COHORT','MODEL_REFERENCE') and p['unit_price_RZ'] is not None,
                'Unacquired axes cannot become zero-priced comparison inputs')
        require(frac(p['unit_price_RZ']) > 0, 'Positive price required')
        require(p['physical_id'] in (row['candidate']['id'], (row.get('physical_candidate') or {}).get('id')), 'Quote physical candidate mismatch')
        require(p['q'] == row['q_PF'] and p['R'] == row.get('R_outer'), 'Quote schedule mismatch')
        if p.get('shots') is not None:
            require(p['shots'] == vector[p['axis']][m], 'Quote and Q3 shots mismatch')
    return dict(sources=sources,configuration=config,inputs=inputs,case=case,record=record,
                outcome=outcome,quote=quote,vector=vector,adopted_evaluation=adopted)


def prices(root, request):
    bundle = validated(root, request);config=bundle['configuration'];out=[]
    for i,p in enumerate(bundle['quote']['rows']):
        row=config['rounds'][p['m']]
        out.append(dict(deepcopy(p),original_configuration_id=config['id'],original_candidate_id=row['candidate']['id'],
            R_outer=row.get('R_outer'),law_binding=deepcopy(row['law_binding']),
            price_class=p['classification'],C=exact(p['unit_price_RZ']),
            original_quote_source=dict(bundle['sources']['quote'],selector=list(bundle['sources']['quote'].get('selector',[]))+['rows',i]),
            qualified_for_candidate=False,E_C=None,CI=None))
    return dict(schema='PR_AUTOMATED_CANDIDATE_AXIS_PRICES_V1',configuration_id=config['id'],
                rows=out,original_quote=bundle['sources']['quote'],price_policy=bundle['quote']['policy'],
                quantity='MEASURED_FULL_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT',E_C=None,CI=None,G=None)


def consumer_inputs(root, request, price_source):
    bundle=validated(root,request);published=read(root,price_source);price_source=locator(root,price_source)
    expected=prices(root,request)
    require(published==expected, 'Published enriched prices differ from their saved quote')
    case=deepcopy(bundle['case']);label=request.get('comparison_label') or case['label']
    terms=[];total=Fraction(0)
    for i,p in enumerate(published['rows']):
        n=bundle['vector'][p['axis']][p['m']];value=frac(p['C']);total+=n*value
        terms.append(dict(m=p['m'],axis=p['axis'],N=n,C_ref=exact(value),N_times_C_ref=exact(n*value),
                          price_source=dict(price_source,selector=list(price_source.get('selector',[]))+['rows',i])))
    case.update(label=label,base_label=label,configuration=bundle['sources']['configuration'],
                original_configuration_id=bundle['configuration']['id'],conditional_cost=exact(total),price_terms=terms,
                rows=deepcopy(bundle['configuration']['rounds']))
    # The consumer's existing make() recognizes PR from the canonical spec.
    return dict(schema='PR_AUTOMATED_CANDIDATE_COMPARISON_INPUTS_V1',cases=[case],
        adopted_evaluation=deepcopy(bundle['adopted_evaluation']),
        producer_evidence=dict(sources=bundle['sources'],status=bundle['outcome']['status'],
            MSE_lower=deepcopy(bundle['outcome']['MSE_lower']),MSE_upper=deepcopy(bundle['outcome']['MSE_upper']),
            conditional_cost=exact(total),price_policy=bundle['quote']['policy'],new_scientific_calls=0),
        qualification='CONDITIONAL_MODEL_COMPARISON_NOT_FORMAL_E_C_OR_G',E_C=None,CI=None,G=None,winner=None)


def join(root, input_source, output_directory):
    input_source=locator(root,input_source);inp=read(root,input_source);case=inp['cases'][0];e=inp['producer_evidence']
    outdir=Path(output_directory).resolve();relative=outdir.relative_to(Path(root).resolve())
    row=dict(label=case['label'],base_label=case['base_label'],rule='AUTOMATED_SEARCH',allocation=case['allocation'],
        status=e['status'],conditional_cost=e['conditional_cost'],MSE_lower=e['MSE_lower'],MSE_upper=e['MSE_upper'],
        Q3_source=e['sources']['outcome'],result_source=e['sources']['result'])
    summary=dict(schema='PR_AUTOMATED_CANDIDATE_COMPARISON_V1',
        consumer_adapter='C Catalog.pf: existing method-generic saved-input path',
        conditional_minima={},adopted_evaluation=inp['adopted_evaluation'],execution_source=input_source,rows=[row],
        qualification=inp['qualification'],E_C=None,CI=None,G=None,winner=None)
    index=dict(schema='PR_AUTOMATED_CANDIDATE_C_JOIN_V1',summary=str(relative/'SUMMARY.json'),
        candidates=[dict(label=case['label'],allocation=case['allocation'],model_status=e['status'])],
        producer_evidence=e,qualification=inp['qualification'])
    return summary,index


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists():
        require(path.read_text()==text,'Refuse to overwrite a different published export')
    else:path.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prices','inputs','join'))
    p.add_argument('--request',type=Path);p.add_argument('--prices',help='JSON fixed price locator')
    p.add_argument('--inputs',help='JSON fixed consumer INPUTS locator');p.add_argument('--output',type=Path)
    p.add_argument('--output-directory',type=Path);a=p.parse_args()
    request=json.loads(a.request.read_text())if a.request else None
    if a.mode=='prices':write(a.output,prices(ROOT,request))
    elif a.mode=='inputs':write(a.output,consumer_inputs(ROOT,request,json.loads(a.prices)))
    else:
        summary,index=join(ROOT,json.loads(a.inputs),a.output_directory)
        write(a.output_directory/'SUMMARY.json',summary);write(a.output_directory/'C_RESULT_JOIN.json',index)


if __name__=='__main__':main()
