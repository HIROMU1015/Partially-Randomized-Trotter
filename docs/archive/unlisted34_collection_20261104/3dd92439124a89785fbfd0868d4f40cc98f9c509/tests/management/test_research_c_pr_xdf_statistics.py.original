"""New numerical tests use invented integer prices only; saved metadata is passive."""
import json
from pathlib import Path

import pytest

from trottertracks import c_pr_xdf_statistics as m

ROOT = Path(__file__).resolve().parents[2]


def rows(values):
    from trottertracks.c_r32_price_statistics import ret
    return [{k:v for k in ret.m4.METRICS} for v in values]


def artificial(mean=900):
    return rows([mean-10,mean+10]*16)


def policy(**kw):
    return dict(m.POLICY, **kw)


def primary(data=None,**kw):
    return m.primary_statistics(artificial() if data is None else data,weight=10,shell=100,xdf=10000,**kw)


@pytest.mark.parametrize('mean,label',[(900,'NOMINAL_MODEL_PR_LOWER'),(1000,'NOMINAL_MODEL_PR_HIGHER'),(990,'UNRESOLVED_SIGN_NOT_EQUIVALENCE')])
def test_affine_t_difference_ratio_and_posthoc(mean,label):
    from scipy.stats import t,ttest_1samp
    from trottertracks.c_r32_price_statistics import price_statistics
    data=artificial(mean); out=primary(data)
    old=price_statistics(data,dict(W=10,shell={'rz_count':100},D={'rz_count':10000}),expected_n=32,statistics=m.POLICY)
    assert out['decision']=='POSTHOC_EXPLORATORY_'+label
    for key in ('kernel_mean','kernel_SE','total_point','total_SE','total_interval','difference_interval','ratio_interval'):
        assert out[key]==old[key]
    assert out['kernel_mean']==mean
    assert out['kernel_SE']==pytest.approx(10/(31**.5))
    assert out['total_point']==10*mean+100
    half=t.ppf(.975,31)*out['total_SE']
    assert out['total_interval']==pytest.approx([10*mean+100-half,10*mean+100+half])
    assert out['one_sided_test']['pvalue']==pytest.approx(ttest_1samp([x['rz_count'] for x in data],990,alternative='less').pvalue)
    assert out['test_interval_consistent'] is True
    assert out['one_sided_upper_bound']==out['total_interval'][1]
    assert out['confirmatory'] is False and 'post-hoc' in out['selection_scope']
    assert out['actual_CA_achieved'] is None and not out['ideal_law_cost_transfer_qualified']


def test_secondary_uses_same_interval_without_engine(monkeypatch):
    out=primary()
    import scipy.stats
    from trottertracks import c_r32_price_statistics as old
    def forbidden(*a,**kw):raise AssertionError('second statistics engine forbidden')
    monkeypatch.setattr(old,'price_statistics',forbidden)
    monkeypatch.setattr(scipy.stats,'ttest_1samp',forbidden)
    v=m.secondary_statistics(out,xdf=10000,cdf=11000)
    assert v['difference_interval']==[x-11000 for x in out['total_interval']]
    assert v['ratio_interval']==[x/11000 for x in out['total_interval']]
    assert v['additional_test_count']==0 and v['pvalue'] is None
    assert v['primary_lower_implies_secondary_lower'] is True
    assert v['decision']=='POSTHOC_EXPLORATORY_SHARED_BOUND_PR_LOWER'


@pytest.mark.parametrize('kind',['zero','precision','diagnostic'])
def test_unresolved_labels_keep_all_observations(kind):
    data=artificial(); p=policy()
    if kind=='zero':data=rows([900]*32);label='ZERO_OBSERVED_VARIANCE'
    elif kind=='precision':p['precision_halfwidth_fraction_of_D']=1e-8;label='PRECISION_TARGET'
    else:data=rows([900]*31+[3000]);label='DISTRIBUTION_DIAGNOSTIC'
    out=primary(data,policy=p)
    assert out['decision']=='POSTHOC_EXPLORATORY_UNRESOLVED_'+label
    assert out['samples']==32
    if kind=='zero':assert out['one_sided_test'] is None and out['total_interval'] is None
    sec=m.secondary_statistics(out,xdf=10000,cdf=11000)
    assert sec['decision']=='POSTHOC_EXPLORATORY_UNRESOLVED_PRIMARY_QUALITY_OR_SIGN'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,1.5,True])
def test_bad_prices(value):
    data=artificial();data[0]['rz_count']=value
    with pytest.raises(ValueError,match='INVALID_NATIVE_METRIC'):primary(data)


@pytest.mark.parametrize('count',[0,1,31,33])
def test_N_short_or_extra(count):
    with pytest.raises(ValueError):primary(rows([900]*count))
    if count==1:
        with pytest.raises(ValueError,match='AT_LEAST_TWO'):primary(rows([900]),policy=policy(samples=1))


@pytest.mark.parametrize('kw',[dict(weight=0),dict(shell=-1),dict(xdf=0),dict(weight=float('inf'))])
def test_bad_affine(kw):
    params=dict(weight=10,shell=100,xdf=10000);params.update(kw)
    with pytest.raises(ValueError):m.primary_statistics(artificial(),**params)


def test_levels_and_nested_thresholds():
    with pytest.raises(ValueError,match='LEVEL'):primary(policy=policy(one_sided_alpha=.05))
    with pytest.raises(ValueError,match='NESTED'):m.secondary_statistics(primary(),xdf=10000,cdf=9999)


def test_inconsistent_t_endpoint_is_unresolved(monkeypatch):
    from types import SimpleNamespace
    import scipy.stats
    monkeypatch.setattr(scipy.stats,'ttest_1samp',lambda *a,**k:SimpleNamespace(pvalue=1.,statistic=0.))
    out=primary()
    assert out['test_interval_consistent'] is False
    assert out['decision']=='POSTHOC_EXPLORATORY_UNRESOLVED_TEST_INTERVAL_NUMERICAL_BOUNDARY'
    assert m.secondary_statistics(out,xdf=10000,cdf=11000)['decision'].endswith('UNRESOLVED_PRIMARY_QUALITY_OR_SIGN')


def saved_metadata():
    # Scalars/identities only: no saved native prices are read by these tests.
    return m.metadata(ROOT,m.fixed())


@pytest.mark.parametrize('mutation', ['candidate','W','shell','cohort_source','compiler','lane','XDF_CDF','STOP'])
def test_scope_rejects_wrong_saved_binding(mutation):
    d=saved_metadata()
    if mutation=='candidate':d['PR_result']['arms']['PR']['candidate']='wrong'
    elif mutation=='W':d['PR_result']['arms']['PR']['bindings']['ideal_exact_native']['W']*=32
    elif mutation=='shell':d['PR_result']['arms']['PR']['bindings']['ideal_exact_native']['shell']['rz_count']*=2
    elif mutation=='cohort_source':d['PR_contract']['source_manifest_sha256']='wrong'
    elif mutation=='compiler':d['PR_contract']['compiler']['seed_transpiler']=42
    elif mutation=='lane':d['ideal_result']['candidates'][m.IDS['XDF']]['lane']='evidence'
    elif mutation=='XDF_CDF':d['ideal_result']['candidates'][m.IDS['XDF']]['id']=m.IDS['CDF']
    elif mutation=='STOP':d['PR_STOP']['retry']=1
    with pytest.raises(ValueError):m.scope_gate(d)


def native_fixture():
    from trottertracks import c_near_practical_r32 as old
    d=saved_metadata();g=dict(max_entry=0.,tolerance=1e-10,rtol=0)
    packets=[];journal=[]
    for i in range(32):
        packets.append(dict(metrics=artificial()[i],events=dict(sample=i,sampling=old.sampling('PR'),
            SOURCE=m.OLD_SOURCE_SHA,JU=m.CANDIDATE,events=[dict(taylor_order=0,artificial=i)]*32,
            coherent_occurrences_independent=True,quantum_shots=0,old_cohort_pooled=False),
            raw=dict(reference=g,unitarity=g),compiled=dict(reference=g,raw=g,unitarity=g),
            adjoint={k:dict(reference=g,returned=g) for k in ('raw','compiled')},
            XY=dict(status='PASS',attenuation_applied=False,state='saved own-DF ground oracle; preparation excluded',
                signals={k:dict(error=0.,tolerance=2e-10) for k in ('raw','compiled')},
                norm_errors={k:[0.,0.] for k in ('raw','compiled')})))
        for kind in ('trajectory_sample','raw_operator_gate','compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics'):
            for event in ('attempt','COMPLETE'):
                journal.append(json.dumps(dict(kind='PR.C.'+kind,event=event,label=f'sample{i:02d}')))
    return d,packets,'\n'.join(journal)


@pytest.mark.parametrize('mutation',['none','cohort','candidate','SOURCE','duplicate','missing','correctness','microevent','journal','seed'])
def test_native_gate_artificial_only(mutation):
    d,packets,journal=native_fixture()
    if mutation=='cohort':packets[0]['events']['sampling']['cohort']='old'
    elif mutation=='candidate':packets[0]['events']['JU']='wrong'
    elif mutation=='SOURCE':packets[0]['events']['SOURCE']='wrong'
    elif mutation=='duplicate':packets[1]=packets[0]
    elif mutation=='missing':packets.pop()
    elif mutation=='correctness':packets[0]['XY']['signals']['raw']['error']=1.
    elif mutation=='microevent':packets[0]['events']['events'].pop()
    elif mutation=='journal':journal=journal.split('\n',1)[1]
    elif mutation=='seed':d['PR_seed_manifest']['rows'][0]['seed']=0
    if mutation=='none':assert m.native_gate(d,packets,journal)['samples']==32
    else:
        with pytest.raises(ValueError):m.native_gate(d,packets,journal)


def test_realized_duplicate_events_retained():
    d,p,j=native_fixture();p[1]['events']['events']=p[0]['events']['events']
    audit=m.native_gate(d,p,j)
    assert audit['duplicate_realized_event_sequences']==1 and audit['dropped']==0


def test_permission_precedes_any_input(monkeypatch,tmp_path):
    def forbidden(*a):raise AssertionError('input touched before permission')
    monkeypatch.setattr(m,'preflight',forbidden)
    with pytest.raises(PermissionError):m.run(tmp_path,tmp_path/'missing')
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('fail',['none','native','statistics'])
def test_mock_run_calls_partial_evidence_STOP(tmp_path,monkeypatch,fail):
    c=m.fixed();c.update(source_manifest='source.json',source_manifest_sha256='artificial');m.io.save(tmp_path/'source.json',{})
    d,p,j=native_fixture()
    monkeypatch.setattr(m,'preflight',lambda *args:(c,{}))
    monkeypatch.setattr(m,'metadata',lambda *args:d)
    monkeypatch.setattr(m,'load_native',lambda *args:p)
    jp=tmp_path/c['records']['PR_journal'];jp.parent.mkdir(parents=True);jp.write_text(j)
    if fail=='native':p[0]['events']['JU']='wrong'
    if fail=='statistics':
        def broken(*args):raise ValueError('ARTIFICIAL_FAILURE')
        monkeypatch.setattr(m,'primary_statistics',broken)
    # Numerical path receives invented prices; use its invented affine mapping.
    elif fail=='none':
        actual=m.primary_statistics
        monkeypatch.setattr(m,'primary_statistics',lambda data:actual(data,weight=10,shell=100,xdf=10000))
        secondary=m.secondary_statistics
        monkeypatch.setattr(m,'secondary_statistics',lambda value:secondary(value,xdf=10000,cdf=11000))
    if fail=='none':m.run(tmp_path,'unused',execute_science=True)
    else:
        with pytest.raises(ValueError):m.run(tmp_path,'unused',execute_science=True)
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert stop['failed']==(fail!='none')
    assert stop['scientific_units']==dict(none=7,native=4,statistics=5)[fail]
    assert m.io.read(tmp_path/'source.json')=={}
    if fail!='none':assert (out/'failure.json').exists() and not (out/'result.json').exists()


def test_existing_call_caps(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    call=Calls(tmp_path,{'x':1},1);assert call('x','artificial',lambda:1)==1
    with pytest.raises(ValueError,match='TOTAL_CALL_LIMIT'):call('x','extra',lambda:2)


@pytest.mark.parametrize('mutation',['none','source','input','missing','policy','fresh'])
def test_passive_preflight_drift_and_fresh(tmp_path,monkeypatch,mutation):
    c=m.fixed();c.update(source_manifest='source.json',source_manifest_sha256='',packages={})
    names=set(m.roles().values())|{p for row in m.native_paths() for p in row.values()}
    for name in names|{m.ENTRY,'impl.py'}:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    old=dict(endpoints={'PR':{'candidate':'candidate.npz'}},input='input.npz')
    (tmp_path/m.roles()['PR_contract']).write_text(json.dumps(old))
    # This fixture tests preflight rejection only; no arrays are created/decoded.
    names.update(('candidate.npz','input.npz'))
    for name in ('candidate.npz','input.npz'):(tmp_path/name).write_text('artificial')
    src={'files':{name:m.io.sha(tmp_path/name) for name in (m.ENTRY,'impl.py')}}
    m.io.save(tmp_path/'source.json',src);c['source_manifest_sha256']=m.io.sha(tmp_path/'source.json')
    c['saved_inputs']={name:m.io.sha(tmp_path/name) for name in names}
    c['saved_inputs']['candidate.npz']=m.CANDIDATE
    c['saved_inputs']['input.npz']='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
    cp=tmp_path/'contract.json'
    if mutation=='source':(tmp_path/'impl.py').write_text('drift')
    elif mutation=='input':(tmp_path/m.roles()['PR_STOP']).write_text('drift')
    elif mutation=='missing':(tmp_path/m.roles()['PR_STOP']).unlink()
    elif mutation=='policy':c['statistics']=dict(m.POLICY,one_sided_alpha=.05)
    elif mutation=='fresh':(tmp_path/m.FRESH).mkdir(parents=True)
    m.io.save(cp,c)
    monkeypatch.setattr(m,'metadata',lambda *a:{})
    monkeypatch.setattr(m,'scope_gate',lambda *a:{})
    monkeypatch.setattr(m,'load_native',lambda *a:[])
    monkeypatch.setattr(m,'native_gate',lambda *a:{})
    sha=m.io.sha
    monkeypatch.setattr(m.io,'sha',lambda p:dict({'candidate.npz':m.CANDIDATE,'input.npz':
        '1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'}).get(Path(p).name) or sha(p))
    if mutation=='none':assert m.preflight(tmp_path,cp)[1]['status']=='PASSIVE_READY'
    else:
        with pytest.raises((ValueError,FileNotFoundError)):m.preflight(tmp_path,cp)
