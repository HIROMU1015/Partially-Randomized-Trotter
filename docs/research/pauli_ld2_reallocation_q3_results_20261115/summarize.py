"""Summarize the saved LD2 re-allocation Q3 results (no new evaluation).

Reads INPUTS.json and run01/<label>/RESULT.json; writes SUMMARY.json and
SUMMARY.csv (create-only). Cost comparisons use the actual model cost T(N),
not the budget. The uniform m10-m12 price change that reverses each PF
relationship is x = (T_PF - T(N)) / T_m10_12(N), a sensitivity guide only.
"""
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
HERE=Path(__file__).resolve().parent
PF={'FULL_PF_S2_B_PRICE':615304502,'B0I_P7_S2_B_PRICE':435481090}


def frac(v):
    return Fraction(v['numerator'],v['denominator'])if isinstance(v,dict)else Fraction(v)


def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)


def show(x,digits=6):
    x=Fraction(x);sign='-'if x<0 else'';x=abs(x);s=10**digits;q=(2*x*s).__floor__();q=(q+1)//2
    return '%s%d.%0*d'%(sign,q//s,digits,q%s)


def sci(x):
    return '%.16e'%float(Fraction(x))


def main():
    inputs=json.loads((HERE/'INPUTS.json').read_text());rows=[]
    tau=frac(inputs['adopted_evaluation']['MSE_upper'])
    for case in inputs['cases']:
        result=json.loads((HERE/'run01'/case['label']/'RESULT.json').read_text())
        risk=result['record']['diagnostic']['deterministic_risk']
        T=frac(case['conditional_cost']);early=frac(case['cost_m0_m9']);late=frac(case['cost_m10_m12'])
        if early+late!=T:raise ValueError('Cost split differs')
        comparisons=[]
        for name,value in PF.items():
            D=Fraction(value)-T;x=D/late
            comparisons.append(dict(benchmark=name,T_PF=value,D=enc(D),D_display=show(D),
                LD2_lower_cost=D>0,uniform_m10_m12_change_to_reverse=enc(x),
                uniform_m10_m12_change_to_reverse_percent=show(100*x,4),
                reversal_possible_with_nonnegative_prices=x>-1))
        rows.append(dict(proposal=case['proposal'],label=case['label'],
            allocation=[r['N_X']for r in case['allocation']['shots']],
            allocation_Y_equals_X=[r['N_Y']for r in case['allocation']['shots']]==[r['N_X']for r in case['allocation']['shots']],
            status=result['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
            MSE_lower_display=sci(frac(risk['MSE_lower'])),MSE_upper_display=sci(frac(risk['MSE_upper'])),
            tau_display=sci(tau),upper_over_tau_display='%.6f'%float(frac(risk['MSE_upper'])/tau),
            budget=case['budget'],T=case['conditional_cost'],T_display=show(T),unused_budget=case['unused_budget'],
            T_m0_m9=case['cost_m0_m9'],T_m0_m9_display=show(early),T_m10_m12=case['cost_m10_m12'],T_m10_m12_display=show(late),
            m10_m12_share_display='%.4f'%float(late/T),comparisons=comparisons,
            scientific_evaluation_calls=result['scientific_evaluation_calls'],result_source='run01/'+case['label']+'/RESULT.json'))
    value=dict(kind='CONDITIONAL_FIXED_ALLOCATION_COST_WITH_MODEL_REFERENCE_PRICES',rows=rows,PF_benchmarks=PF,
        note='x is a uniform relative change of all m10-m12 model-reference prices; sensitivity guide only',
        sufficient_N=None,optimal_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    with (HERE/'SUMMARY.json').open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    with (HERE/'SUMMARY.csv').open('x',newline='')as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['proposal','allocation','status','MSE_lower','MSE_upper','T','T_m0_m9','T_m10_m12',
                    'D_FullPF_S2','x_FullPF_percent','D_p7_S2','x_p7_percent'])
        for r in rows:
            c={x['benchmark']:x for x in r['comparisons']}
            w.writerow([r['proposal'],' '.join(map(str,r['allocation'])),r['status'],r['MSE_lower_display'],r['MSE_upper_display'],
                r['T_display'],r['T_m0_m9_display'],r['T_m10_m12_display'],
                c['FULL_PF_S2_B_PRICE']['D_display'],c['FULL_PF_S2_B_PRICE']['uniform_m10_m12_change_to_reverse_percent'],
                c['B0I_P7_S2_B_PRICE']['D_display'],c['B0I_P7_S2_B_PRICE']['uniform_m10_m12_change_to_reverse_percent']])
    for r in rows:print(r['proposal'],r['status'],r['MSE_upper_display'],r['T_display'],[(c['benchmark'],c['D_display'],c['uniform_m10_m12_change_to_reverse_percent'])for c in r['comparisons']])


if __name__=='__main__':main()
