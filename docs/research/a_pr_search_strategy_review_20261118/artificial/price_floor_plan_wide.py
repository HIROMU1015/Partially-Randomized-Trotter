"""Artificial: SCHEDULE_WIDE_V1 plan_wide (5c8ced4b2) for one family, without / with price_floor (as WidthDriver vs round05 Driver).

Unit price per round and axis: a + b*q_m + c*q_m*R_m (c > 0, the shape of D2 MODEL rules); floor(R) = sum over rounds and both
axes of the one-shot price.  A saved feasible point at kappa = 2 sets the method best M.  Proposals are counted per decision
until the rule stops (each proposal would be one preparation + Q3 work in the real run; here nothing is computed).
"""
import json, sys
from fractions import Fraction
sys.path.insert(0, sys.argv[1] + '/src')
from trottertracks import rpe_schedule_search as W

rows = W.family_rows(Fraction(3, 2), Fraction(1, 10), Fraction(1, 10))
fam = W.Family('ART|LD3|2nd|delta=0.1|K2', rows)
a, b, c = Fraction(300), Fraction(2700), Fraction(7)
def floor(R):
    return sum(2 * (a + b * r['q'] + c * r['q'] * x) for r, x in zip(fam.rows, R))
M = floor(fam.R(Fraction(2))) * 30          # the saved feasible allocation at kappa=2 costs 30x its one-shot floor
def run(with_floor, decisions=12):
    points = [dict(family=fam.key, kappa=Fraction(2), status='FEASIBLE', best=M, min_lower_over_tau=None)]
    proposed = []
    for d in range(decisions):
        pf = (lambda key, R: floor(R)) if with_floor else None
        props, why, rep = W.plan_wide({fam.key: fam}, points, dict(per_decision=8), pf)
        if not props: break
        for p in props:
            proposed.append(p)
            # artificial outcome: every new schedule is evaluated; its best is its floor x 30 (never below M unless cheaper)
            points.append(dict(family=fam.key, kappa=p['kappa'], status='FEASIBLE', best=floor(fam.R(p['kappa'])) * 30,
                               min_lower_over_tau=None))
    r = rep['families'][fam.key]
    over = [p for p in proposed if floor(fam.R(p['kappa'])) >= M]
    return dict(price_floor=with_floor, proposals=len(proposed), proposals_with_floor_ge_M=len(over),
                max_kappa=str(max(p['kappa'] for p in proposed)) if proposed else None,
                upper_basis=r['upper_basis'], upper_limit=str(r['upper_limit']), stop=rep['stop'])
print(json.dumps([run(False), run(True)], indent=1))
