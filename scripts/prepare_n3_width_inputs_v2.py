#!/usr/bin/env python3
"""Coefficient literals only. No U/c, eigensolver, solver, or acceptance evaluation."""
from fractions import Fraction as F
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.representation_exploration.n3_width_contract_v2 import decode_case,enc,digest
OUT='artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json'


def make_pack():
    cases=[]
    for family_index,family in enumerate(('banded','dense_density','two_external_stress')):
        for size_index,n in enumerate((5,6)):
            seed=20261012031+10*family_index+size_index;r=random.Random(seed);nu=2 if n==5 else 3
            h=[[F(0) for _ in range(n)] for _ in range(n)]
            for i in range(n):h[i][i]=F(-24+5*i+r.randint(-2,2),32)
            for i in range(n):
                for j in range(i+1,n):
                    if family=='banded' and j>i+1:continue
                    h[i][j]=h[j][i]=F(r.choice((-3,-2,-1,1,2,3)),64)
            h[0][0]=F((16 if family=='dense_density' else 28),32)
            h[-1][-1]=F((20 if family=='two_external_stress' else 44),32)
            J=[[i,j,F(r.randint(1,3),32 if family=='dense_density' else 96)] for i in range(n) for j in range(i+1,n)]
            case=enc({'id':f'V2_{family}_{n}','family':family,'seed':seed,'modes':n,'nu':nu,
                'h':h,'density':J,'candidate_modes':[0,n-1],'epsilon_proj':F(1,100)})
            decode_case(case);cases.append(case)
    return {'schema_version':2,'recipe':{'families':['banded','dense_density','two_external_stress'],
        'sizes':[5,6],'particles':[2,3],'base_seed':20261012031,'seed_rule':'base+10*family_index+size_index',
        'epsilon_proj':[1,100],'candidate_modes_rule':'[0,n-1]','kind':'preregistered coefficient stress collection; not an IID distribution benchmark',
        'result_screening_or_NORM_filtering':False,'old_12_8_S1_are_development_only':True},'cases':cases}


def main():
    pack=make_pack();path=ROOT/OUT;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(pack,f,indent=2);f.write('\n')
    print(json.dumps({'status':'UNUSED_COEFFICIENTS_SAVED_NO_NUMERICAL_EVALUATION',
        'cases':{c['id']:digest(c) for c in pack['cases']},'new_eigensolver_or_solver_calls':0},indent=2))


if __name__=='__main__':main()
