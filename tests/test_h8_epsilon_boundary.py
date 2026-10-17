"""Artificial C endpoint arithmetic and authorization only."""
import copy
from fractions import Fraction as F
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/trotterlib'),str(ROOT/'scripts')]
import h8_epsilon_boundary as n
import run_h8_epsilon_boundary as run


def record(method='B0'):
    return dict(cell=dict(id='synthetic',method=method),log_B=0.,
        error_decomposition=dict(total_signed=dict(real=.001,imag=.002)),
        allowance=dict(real=1e-12,imag=1e-12,log_B_margin=0.,kind='EMPIRICAL',certified=False))


def test_frozen_rule_and_saved_price_weighting():
    r=record();b=dict(real=.001,imag=.002)
    a=n.accounting(r,b,.025)
    assert a==n.old.d0.accounting(r,.025,1.)
    prices={a:dict(RZ=str(v),measured=True,status='SYNTHETIC')
            for a,v in zip(n.old.CHANNEL,(11,17))}
    row=n.cost_row(r,b,.025,n.MODES[0],unit=prices)
    assert F(row['G_exact'])==11*row['shots']['real']+17*row['shots']['imag']
    assert row['cost_status']=='SAVED_MEASURED_POINT'


def test_nonnegative_bias_and_unresolved_preserved():
    assert n.accounting(record(),None,.025) is None
    with pytest.raises(ValueError,match='NEGATIVE_BIAS'):
        n.accounting(record(),dict(real=-1.,imag=0.),.025)
    row=n.cost_row(record(),None,.025,n.MODES[1])
    assert row['eligible'] is None and row['G_exact'] is None


def test_missing_price_stops_instead_of_zero_fill():
    with pytest.raises(ValueError,match='SAVED_UNIT_PRICE'):
        n.cost_row(record(),dict(real=0.,imag=0.),.025,n.MODES[0])


def test_paired_n32_new_shot_weighting():
    pairs=[{a:dict(RZ=i+v) for a,v in zip(n.old.CHANNEL,(11,17))} for i in range(32)]
    row=n.cost_row(record('B2'),dict(real=0.,imag=0.),.025,n.MODES[0],pairs=pairs)
    totals=[sum(row['shots'][ch]*p[a]['RZ'] for a,ch in n.old.CHANNEL.items()) for p in pairs]
    assert F(row['G_exact'])==F(sum(totals),32)
    assert row['G_standard_error']>0 and row['paired_samples']==32
    with pytest.raises(ValueError,match='N32'):
        n.cost_row(record('B2'),dict(real=0.,imag=0.),.025,n.MODES[0],pairs=pairs[:8])


def test_unresolved_domain_blocks_minimum_and_ties_stable():
    def row(cid,g,eligible=True):
        return dict(cell_id=cid,eligible=eligible,G_exact=g,cost_status='SYNTHETIC')
    rows=[row('D2','10'),row('D1','10'),row('PR','5')]
    c=n.compare(rows,['D1','D2'],['PR'],.025,n.MODES[0])
    assert c['deterministic']['best']['cell_id']=='D1'
    assert F(c['reduction_percent_exact'])==50
    rows[0]=row('D2',None,None)
    assert n.compare(rows,['D1','D2'],['PR'],.025,n.MODES[0])['reduction_percent_exact'] is None


def test_draft_rejected_before_any_input_read():
    with pytest.raises(ValueError,match='APPROVAL_REQUIRED'):
        run.authorize({},dict(authorized=False),root=Path('/path/does/not/exist'))


def test_approved_bit_cannot_change_scope():
    with pytest.raises(ValueError,match='SCOPE_SOURCE_CONFIG_BINDING'):
        run.authorize(dict(authorized=True,scope={}),dict(authorized=True))
