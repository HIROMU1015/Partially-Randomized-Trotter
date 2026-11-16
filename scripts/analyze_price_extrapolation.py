"""Analyze fixed saved price points; no signal, fitting, sampling or compilation.

Use --inputs for immutable catalog/price locators, or --normalized for a future
B2/D point with explicit axis prices, evidence tiers and coefficient features.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.saved_rpe_cost_catalog import Reader, enc, require
from trottertracks.price_extrapolation_analysis import analyze, break_even, number


def selected(reader, request):
    catalog = reader.read(request['catalog'])
    p = next(v for v in catalog['candidates'] if v['point_id'] == request['point_id'])
    require(p['status'] == 'FEASIBLE', 'Point is not a saved feasible outcome')
    require(p['point_id'] in catalog['method_minima'][p['method']]['point_ids'],
            'Not the fixed catalog minimum')
    config = reader.read(p['configuration_source'])
    require(config['id'] == p['configuration_id'], 'Configuration source mismatch')
    schedule = {r['m']: r for r in p['schedule']}
    for row in config['rounds']:
        s = schedule[row['m']]
        require(row['candidate']['id'] == s['canonical_id'] and row['q_PF'] == s['q']
                and row.get('R_outer') == s['R']
                and number(row['actual_delta_pf']) == number(s['actual_delta']),
                'Canonical schedule mismatch')
    out = dict(point_id=p['point_id'], method=p['method'], label=p['label'], M=p['M'],
        configuration_id=p['configuration_id'], total_configuration_id=p['total_configuration_id'],
        binding_id=p['binding_id'], configuration_source=p['configuration_source'],
        allocation=[dict(m=m,N_X=p['N_X'][m],N_Y=p['N_Y'][m]) for m in range(p['M']+1)],
        saved_total_RZ=p['conditional_cost'], saved_Q3=p['q3_status'],
        catalog_source=request['catalog'], profile=p['profile'], axes=[],
        price_qualification='CONDITIONAL_MODEL_REFERENCE_NOT_E_C_CI_OR_G')
    replay = None
    if request['kind'] == 'SAVED_PF_AXES':
        price = reader.read(request['price_source'])
        require(price['configuration_id'] == p['configuration_id'] and price['allocation'] == p['allocation'],
                'Price configuration/allocation mismatch')
        require(number(price['conditional_cost']['conditional_total_RZ'])
                == number(p['conditional_cost']), 'Price total mismatch')
        for i, row in enumerate(price['axes']):
            s = schedule[row['m']]; a = row['axis']; m = row['m']
            require(row['original_candidate_id'] == s['canonical_id'] and row['q'] == s['q'],
                    'PF price physical identity differs')
            # Producer's native circuit uses the sourced binary64 step, while
            # the canonical schedule retains the exact rational T/q.
            require(float(number(s['actual_delta'])) == row['actual_delta'], 'Native step differs')
            require(row['N'] == p['N_'+a][m], 'PF price N differs')
            direct = row['kind'] == 'DIRECT_COMPILED_DETERMINISTIC'
            tier = ('DIRECT' if direct else 'EXTRAPOLATION' if
                    row['region'] == 'UNVALIDATED_EXTRAPOLATION_ABOVE_Q80' else 'SUPPORTED_MODEL_RANGE')
            coefficients = price['coefficients'][a]
            components = [] if direct else [dict(id='p7.'+a+'.'+k, value=v,
                feature=(1 if k == 'one' else s['q'])) for k,v in coefficients.items()]
            out['axes'].append(dict(m=m,axis=a,N=row['N'],configuration_id=p['configuration_id'],
                physical_id=s['canonical_id'],q=s['q'],R=s['R'],actual_delta=s['actual_delta'],
                unit_price_RZ=row['unit_price_RZ'],saved_N_times_price_RZ=row['N_times_price_RZ'],
                source=dict(request['price_source'],selector=['axes',i]),
                observation_source=deepcopy(row['price_evidence']),tier=tier,components=components))
        out['holdout'] = price['holdout']
    elif request['kind'] == 'PR_FROZEN_RULE_REPLAY':
        # Exactly B2's published passive policy: the selected point did not
        # persist its temporary quote; never reuse baseline round prices.
        code = request['quote_code']
        raw = subprocess.check_output(['git','show',code['commit']+':'+code['path']],cwd=ROOT)
        scope = dict(__name__='fixed_price_quote',__file__=str(ROOT/code['path']))
        exec(compile(raw,code['commit']+':'+code['path'],'exec'),scope)
        rule = deepcopy(p['price_evidence']['rule'])
        rules = scope['PriceRules'](reader,dict(entries=[rule]))
        replay = rules.quote(dict(configuration=config,implementation=rule['implementation'],
            policy='MODEL',shots={a:p['N_'+a] for a in ('X','Y')}))
        require(replay['complete'] and number(replay['total_RZ']) == number(p['conditional_cost']),
                'Frozen rule does not reproduce the selected total')
        coefficients = reader.read(rule['model_source'])['coefficients']
        for i, row in enumerate(replay['rows']):
            m=row['m'];a=row['axis'];s=schedule[m]
            require(row['physical_id'] == s['canonical_id'] and row['q']==s['q'] and row['R']==s['R'],
                    'PR quote physical identity mismatch')
            require(row['classification']=='MODEL_REFERENCE' and not row['direct_alternatives'],
                    'Update evidence tiers: matching direct evidence is available')
            extrap = row['model_alternative']['extrapolation']
            require(bool(extrap), 'Update coverage: candidate is now inside calibration range')
            features=dict(A5=1,H5=s['q'],E5=s['q']*s['R'])
            out['axes'].append(dict(m=m,axis=a,N=p['N_'+a][m],configuration_id=p['configuration_id'],
                physical_id=s['canonical_id'],q=s['q'],R=s['R'],actual_delta=s['actual_delta'],
                unit_price_RZ=row['unit_price_RZ'],saved_N_times_price_RZ=row['weighted_RZ'],
                source=row['source'],tier='EXTRAPOLATION',extrapolated_dimensions=extrap,
                value_origin='REPLAY_OF_FROZEN_RULE_AT_SELECTED_SCHEDULE_NOT_PERSISTED_POINT_QUOTE',
                quote_code=code,components=[dict(id='LD5.'+k,value=coefficients[k],feature=f)
                                           for k,f in features.items()]))
        out['original_point_unit_prices_persisted']=p['price_axis_unit_values_persisted']
        out['kappa']=p['kappa_absolute']
    else:
        raise ValueError('Unsupported saved-point adapter; use --normalized')
    return out, replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inputs',type=Path);mode.add_argument('--normalized',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reader=Reader(ROOT);quotes=[]
    if args.inputs:
        manifest=json.loads(args.inputs.read_text());points=[]
        for item in manifest['points']:
            point,quote=selected(reader,item);points.append(point)
            if quote:quotes.append(quote)
    else:points=json.loads(args.normalized.read_text())['points']
    results=[analyze(p) for p in points]
    require(len(results)==2,'Choose exactly two fixed points')
    comparison=break_even(*results)
    result=dict(schema='SAVED_PRICE_EXTRAPOLATION_ANALYSIS_V1',
        analysis_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        points=results,break_even=comparison,new_fit=0,new_compile=0,new_Q3=0,new_signal=0,
        winner=None,global_RMSE_bound=None,CI=None)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in [('ANALYSIS',result),('NORMALIZED_INPUT',dict(points=points)),('PRICE_REPLAY',quotes)]:
        (args.output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'AXES.csv').open('w') as file:
        fields=['point_id','m','axis','q','R','N','price_RZ_exact','cost_RZ_exact','tier','d_total_per_unit_RZ']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for point in results:
            for row in point['axes']:
                writer.writerow(dict(point_id=point['point_id'],m=row['m'],axis=row['axis'],q=row['q'],R=row['R'],
                    N=row['N'],price_RZ_exact=str(number(row['unit_price_RZ'])),cost_RZ_exact=str(number(row['cost_RZ'])),
                    tier=row['tier'],d_total_per_unit_RZ=row['d_total_per_unit_RZ']))
    print(json.dumps(dict(totals=[p['total_RZ']for p in results],gap=comparison['gap_RZ'])))


if __name__=='__main__':main()
