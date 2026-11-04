"""Artificial numerical inputs only; H4 is touched solely by metadata tests."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

import pytest
from flint import ctx

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_success as run


@pytest.fixture
def alpha():
    with ctx.workprec(160):
        yield run.alpha_module()


def row(**kwargs):
    return dict(dict(m=0,axis='cosine',N='1000',epsilon='1/2',mu0='0',alpha='0.001',
                     U_cat_upper='1/10',mean_extra_upper='1/100'),**kwargs)


def bundle():
    return dict(mode='ARTIFICIAL',axes=[row(m=m,axis=a) for m in range(18) for a in run.AXES],
                proof=dict(artificial=True))


def test_exact_mean_addition_once(alpha):
    result=run.controlled_axis(row(),alpha)
    assert result['U_controlled_upper']=='11/100'
    assert Q(result['t']['lower'])<=Q(39,100)<=Q(result['t']['upper'])
    assert not result['pi64_added_again']


@pytest.mark.parametrize('error',['1/2','3/4'])
def test_t_zero_negative_stops(alpha,error):
    with pytest.raises(ValueError,match='t not proven'):
        run.controlled_axis(row(U_cat_upper=error,mean_extra_upper='0'),alpha)


@pytest.mark.parametrize('key',['U_cat_upper','mean_extra_upper'])
def test_negative_bound_stops(alpha,key):
    with pytest.raises(ValueError,match='nonnegative'):
        run.controlled_axis(row(**{key:'-1/100'}),alpha)


def test_arb_exponential_endpoint_outward(alpha):
    # Independent high-precision Decimal exponential, for artificial rational t.
    result=alpha.failure_upper(1000,'1/2','11/100')
    hi=Q(result['probability_upper']['upper'])
    with localcontext() as c:
        c.prec=100
        reference=2*(-Decimal(1000)*Decimal('0.39')**2/2).exp()
        endpoint=Decimal(hi.numerator)/Decimal(hi.denominator)
        assert endpoint>=reference and endpoint-reference<Decimal('1e-60')


@pytest.mark.parametrize('total,decision',[('1/20','DIRECT_HOEFFDING_95_SUFFICIENT'),
    ('50000000000000000001/1000000000000000000000','UNKNOWN/STOP')])
def test_exact_aggregate_boundary(alpha,total,decision):
    q=Q(total)/36
    values=[dict(probability_upper=dict(lower=str(q),upper=str(q))) for _ in range(36)]
    result=alpha.aggregate(values,target='1/20')
    assert result['failure_upper_exact']==str(Q(total)) and result['decision']==decision


def test_missing_tail_unknown(alpha):
    assert alpha.aggregate([dict(probability_upper=None)])['decision']=='UNKNOWN/STOP'


def test_synthetic_end_to_end(tmp_path):
    result=run.evaluate(tmp_path/'synthetic',bundle())
    assert result['status']=='COMPLETED_MANDATORY_STOP'
    assert result['counts']==dict(tail_cells_started=36,exact_aggregations_started=1)
    assert result['decision'].startswith('CONDITIONAL_')
    assert result['mode']=='ARTIFICIAL' and result['U_exec_upper'] is None


@pytest.mark.parametrize('change',['missing','duplicate','nonpositive'])
def test_partial_stop_no_aggregate(tmp_path,change):
    b=bundle()
    if change=='missing': b['axes'].pop()
    elif change=='duplicate': b['axes'][-1]=dict(b['axes'][0])
    else: b['axes'][2]['epsilon']='1/1000'
    result=run.evaluate(tmp_path/'synthetic',b)
    assert result['status']=='UNKNOWN_MANDATORY_STOP' and result['aggregate'] is None
    assert 'exact_aggregations_started' not in result['counts']
    assert json.loads((tmp_path/'synthetic/RESULT.json').read_text())==result
    with pytest.raises(FileExistsError):run.evaluate(tmp_path/'synthetic',b)


def test_default_never_evaluates(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('scientific operation in metadata mode')
    monkeypatch.setattr(run,'evaluate',forbidden)
    monkeypatch.setattr(run,'alpha_module',forbidden)
    import numpy as np
    monkeypatch.setattr(np,'load',forbidden)
    result=run.describe()
    assert result['fixed_axes']==36 and result['new_H4_scientific_calls']==0
    assert result['controlled_success_probability'] is None
    assert result['controlled_probability_decision']=='NOT_EXECUTED_UNDETERMINED'
    assert not result['proof']['delta_mean_added_to_energy']


@pytest.fixture(scope='module')
def literals():
    # This fixture reads literals only, never calls controlled_axis/evaluate.
    spec=json.loads((run.HERE/'FIXED_BINDINGS.json').read_text())
    raw={k:run.base.verify_bound(v,current=v['current']) for k,v in spec['files'].items()}
    import csv,io
    original=run.base.static_inputs()
    table=lambda k:list(csv.DictReader(io.StringIO(raw[k].decode()),delimiter='\t'))
    return (original['rows'],original['saved_axes'],json.loads(raw['MEAN_RESULT'])['rounds'],
            table('CAT_AXES'),table('MEAN_AXES'),json.loads(raw['A_RESULT'])['axes'],raw)


@pytest.mark.parametrize('change',['center','N','epsilon','Ucat','delta','pi','missing','duplicate'])
def test_identity_mismatch_stops(literals,change):
    rows,cat,w,old,mean,a,_=copy.deepcopy(literals)
    if change=='center':old[0]['saved_mu0']='0'
    elif change in ('N','epsilon'):a[0][change]='1'
    elif change=='Ucat':w[0]['axes']['cosine']['U_cat_upper']='0'
    elif change=='delta':mean[0]['mean_extra_upper']='0'
    elif change=='pi':w[0]['axes']['cosine']['pi64_added_again']=True
    elif change=='missing':mean.pop()
    else:mean[-1]=dict(mean[0])
    with pytest.raises(ValueError):run.validate_join(rows,cat,w,old,mean,a)


@pytest.mark.parametrize('change',['branch','alias','reference'])
def test_proof_gap_stops(literals,change):
    raw=literals[-1]
    d,c=json.loads(raw['D_RESULT']),json.loads(raw['C_RESULT'])
    if change=='branch':d['branch']['ideal_margins'].pop()
    elif change=='alias':d['energy']['representative_lift_decision']='UNKNOWN/STOP'
    else:c['eta_ref_upper']=None
    with pytest.raises(ValueError):run.proof_connection(d,c)


def test_corrupt_hash_and_frozen_reuse(literals):
    spec=json.loads((run.HERE/'FIXED_BINDINGS.json').read_text())
    item=dict(spec['files']['ALPHA'],sha256='0'*64)
    with pytest.raises(ValueError):run.base.verify_bound(item,current=True)
    for key in ('ALPHA','COMMON','D_ADAPTER'):
        run.base.verify_bound(spec['files'][key],current=True)


@pytest.mark.parametrize('change',['sine','support_policy'])
def test_scope_change_rejected(monkeypatch,change):
    original=Path.read_text
    def modified(path,*a,**k):
        text=original(path,*a,**k)
        if path==run.HERE/'FIXED_BINDINGS.json':
            spec=json.loads(text)
            if change=='sine':spec['signal_convention']['sine']='-Im'
            else:spec['support_policy']='unrestricted'
            return json.dumps(spec)
        return text
    monkeypatch.setattr(Path,'read_text',modified)
    with pytest.raises(ValueError,match='fixed scope'):run.describe()


@pytest.mark.parametrize('argv',[['--run'],['--output','unused']])
def test_switch_pair_required(argv):
    with pytest.raises(ValueError):run.main(argv)
