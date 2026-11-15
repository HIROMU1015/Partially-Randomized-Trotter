"""Small exact synthetic arithmetic: no H4 prices, fit or Q3 run."""
from fractions import Fraction as F
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import freeze_pr_df_ld_split as R


def test_fixed_square_design_identifies_full_wrapper_coefficients():
    cells=[dict(m=m,q=q,R=r,mean_C=R.enc(3+5*q+7*q*r))for m,q,r in ((0,1,1),(3,8,5),(6,64,40))]
    assert R.fit_means(cells)==[3,5,7]
    with pytest.raises(ValueError):R.fit_means(cells[:2])

@pytest.mark.parametrize('rule',['A_SHAPE','B_PRICE'])
def test_reused_rational_generator_largest_breakpoint(rule):
    caps=[7,5,4];prices=[F(3,2),F(11,3),F(1,7)];budget=F(3,4)*sum(n*c for n,c in zip(caps,prices))
    r=R.generate(caps,prices,budget,rule)
    assert r['conditional_cost']==sum(n*c for n,c in zip(r['shots'],prices))<=budget
    assert r['unused_budget']==budget-r['conditional_cost']
    assert r['next_breakpoint_cost']is None or r['next_breakpoint_cost']>budget
    assert all(1<=n<=cap for n,cap in zip(r['shots'],caps))


def test_nonpositive_prices_and_infeasible_floor():
    with pytest.raises(ValueError):R.generate([2],[F(0)],F(4))
    assert R.generate([2],[F(3,2)],F(1))is None
