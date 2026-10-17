"""Artificial-only mathematical, statistical and authorization regressions."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/trotterlib'),str(ROOT/'scripts')]
import h8_saved_analysis as s
import h8_saved_analysis_adapter as adapter
import run_h8_saved_analysis as run
from h8_rare_tail_selection import digest, in_stratum


def law():
    return dict(R=2,saved_pi0='.8',saved_pi2='.15',saved_pi4='.05',
                conditional_N2_counts=[1,2],conditional_N2_probabilities=['.75','.25'])


def rare_ep():
    return dict(candidate=s.RARE[0],id='synthetic',bias_kind=s.MODES[0],epsilon='.01',factor='1.0',
                eligible=True,shots=dict(cosine=2,sine=3),saved_PR_winner=s.RARE[0],
                comparator=dict(cost_exact='100',cost_kind='SYNTHETIC',candidate='D'))


def rare(rows0=None,rows1=None,ep=None,assumptions=None):
    return s.rare_endpoint(law(),rows0 or [dict(cosine=2,sine=4)],
        rows1 or [dict(cosine=4,sine=6)],ep or rare_ep(),assumptions or {})


def cfg():
    return dict(candidate_set=list(s.CANDIDATES),epsilon_scope=list(s.EPS[:4]),
        bias_mode=[s.MODES[0]],allowance_scope=['1.0'],numeric_precision='FRACTION_EXACT_DECIMAL80',
        analysis_family='UNION6_X_ORIGINAL_FOUR_EPSILON',estimand=['FIXED_CONFIRMATION','POST_POINT_MIN'],
        eligibility_mask='SAVED_FIXED_NO_POSTHOC_CANDIDATE_REMOVAL',
        statistical_method='EMPIRICAL_PRODUCT_BOOTSTRAP',cross_candidate_pairing=False,
        population_guarantee=False,family_alpha=None,bootstrap_replicates=20,
        interval_kind='EXPLORATORY_EMPIRICAL_PERCENTILES_NOT_CONFIDENCE_INTERVAL',
        dependence_assumption='PRODUCT_OF_CANDIDATE_EMPIRICAL_DISTRIBUTIONS_NOT_MEASURED_INDEPENDENCE',
        diagnostic_tail_fraction='0.025',resampling_seed_policy='SYNTHETIC_TEST_SEED',
        new_sampling=0,new_compile=0,new_shot_calculation=0)


def selection_data():
    by={c:[dict(candidate=c,replica=i,seed=j*100+i,phase='confirmation',law='unconditional',
        cosine=10+j*5+i,sine=20+j*7+2*i,event_digest_cosine=str(i),event_digest_sine=str(i))
        for i in range(4)] for j,c in enumerate(s.CANDIDATES)}
    endpoints=[]
    for i,e in enumerate(s.EPS[:4]):
        endpoints.append(dict(id=e,epsilon=e,bias_kind=s.MODES[0],factor='1.0',
            candidates={c:dict(eligible=True,shots=dict(cosine=2*(i+1),sine=3*(i+1)),
                saved_G_exact=None) for c in by}))
    return by,endpoints


def test_layer_partition():
    for n2 in range(8):
        for n4 in range(8):
            assert sum(in_stratum(n2,n4,x) for x in ('s0','s2_single','s2_multi','s4'))==1


def test_probability_and_boundary_arithmetic():
    out=rare();p=out['probabilities'];b=out['boundary']
    assert Decimal(p['multi_mass'])==Decimal('.0375')
    assert Decimal(out['known_baseline_G'])==Decimal('.8')*16+Decimal('.1125')*26
    assert Decimal(b['remaining_G'])==Decimal('84.275')
    assert out['mu4_threshold']['value'] is None
    assert out['unobserved_s4_mean'] is None
    assert not out['boundary']['unknown_axis_costs_assumed_equal']


@pytest.mark.parametrize('a,b',[(0,0),(100,200),(2000,1000)])
def test_boundary_equation_and_conditional_thresholds(a,b):
    out=rare(assumptions=dict(mu2_multi=a,mu4=b));boundary=out['boundary']
    m=Decimal(boundary['multi_coefficient']);f=Decimal(boundary['s4_coefficient'])
    right=Decimal(boundary['remaining_G'])
    with localcontext() as ctx:
        ctx.prec=80
        assert abs(m*a+f*Decimal(out['mu4_threshold']['value'])-right)<Decimal('1e-70')
        assert abs(m*Decimal(out['mu2_multi_threshold']['value'])+f*b-right)<Decimal('1e-70')


def test_negative_threshold_domain():
    e=rare_ep();e['comparator']['cost_exact']='1';out=rare(ep=e,assumptions=dict(mu2_multi=10))
    assert out['boundary']['interpretation']=='BASELINE_ALREADY_EXCEEDS_COMPARATOR'
    assert out['mu4_threshold']['status'].startswith('NEGATIVE_THRESHOLD')


@pytest.mark.parametrize('field',['saved_pi0','saved_pi2','saved_pi4'])
def test_nonfinite_probabilities(field):
    l=law();l[field]='NaN'
    with pytest.raises(ValueError):s.probabilities(l)


@pytest.mark.parametrize('change',[dict(saved_pi0='.7'),dict(conditional_N2_probabilities=['.5','.4']),
                                  dict(conditional_N2_counts=[1,1])])
def test_normalization_failure(change):
    l=law();l.update(change)
    with pytest.raises(ValueError):s.probabilities(l)


@pytest.mark.parametrize('coefficient',['0','.1'])
def test_threshold_zero_and_unknown(coefficient):
    r=s.threshold(Decimal(10),Decimal(coefficient),Decimal('.2'),None)
    assert r['value'] is None


def test_zero_w_zero_pi4():
    l=law();l.update(saved_pi0='.8',saved_pi2='.2',saved_pi4='0',
                    conditional_N2_probabilities=['1','0'])
    r=s.rare_endpoint(l,[dict(cosine=1,sine=2)],[dict(cosine=3,sine=4)],rare_ep(),{})
    assert r['mu4_threshold']['status']=='ZERO_COEFFICIENT_NO_DIVISION'
    assert r['mu2_multi_threshold']['status']=='ZERO_COEFFICIENT_NO_DIVISION'


@pytest.mark.parametrize('field,value,status',[('eligible',False,'INELIGIBLE_SAVED_ENDPOINT'),
    ('shots',None,'MISSING_SAVED_SHOTS'),('comparator',dict(cost_exact=None),'COMPARATOR_UNAVAILABLE')])
def test_unavailable_endpoint(field,value,status):
    e=rare_ep();e[field]=value;assert rare(ep=e)['status']==status


def test_no_mutation_and_axis_pair_weighting():
    e=rare_ep();before=copy.deepcopy(e);out=rare(ep=e)
    assert e==before and out['s0']['mean_G_exact']=='16'
    assert s.weighted_mean([dict(cosine=7,sine=1)],dict(cosine=2,sine=3))==17


def test_exact_summary():
    r=s.summary([1,3]);assert r['mean_exact']=='2' and r['sample_variance_exact']=='2'
    assert Decimal(r['sample_SE'])==1


@pytest.mark.parametrize('bad',[None,-1,float('nan')])
def test_layer_cost_missing_not_zero(bad):
    with pytest.raises(ValueError,match='LAYER_COST'):
        s.weighted_mean([dict(cosine=bad,sine=1)],dict(cosine=1,sine=1))


@pytest.mark.parametrize('values',[[1],[1,-1],[1,float('nan')],[1,True]])
def test_invalid_samples(values):
    with pytest.raises(ValueError):s.summary(values)


def test_fixed_vs_post_selected():
    by,e=selection_data();r=s.selection_analysis(by,e,cfg(),s.FIXED)
    o=r['endpoints'][1]
    assert o['exploration_fixed_candidate']==s.CANDIDATES[2]
    assert o['post_confirmation_point_selected']==s.CANDIDATES[0]
    assert o['selected_candidate_population_mean'] is None
    assert r['simultaneous_coverage_certified'] is False
    assert Fraction(o['fixed_minus_postselected_point_exact'])>0
    assert 'NOT_TRUE_POPULATION_OPTIMISM' in o['bootstrap']['empirical_selection_optimism_estimand']


def test_shared_indices_across_endpoints_and_pairing():
    by,e=selection_data();a=s.selection_analysis(by,e,cfg(),s.FIXED)
    b=s.selection_analysis(by,e,cfg(),s.FIXED);assert a==b
    for c in by:
        lo=a['endpoints'][0]['bootstrap']['candidate_empirical_intervals'][c]['lower_exact']
        hi=a['endpoints'][3]['bootstrap']['candidate_empirical_intervals'][c]['lower_exact']
        assert Fraction(hi)==4*Fraction(lo)
    assert a['covariance_measured_zero'] is False


@pytest.mark.parametrize('key,value',[('cross_candidate_pairing',True),('population_guarantee',True),
    ('family_alpha',.05),('dependence_assumption','PAIRED_REPLICA'),
    ('statistical_method','HOEFFDING_WITH_OBSERVED_MAX'),('bootstrap_replicates',-1)])
def test_invalid_statistics_contract(key,value):
    c=cfg();c[key]=value
    with pytest.raises(ValueError):s.validate_stat_contract(c)


def test_fixed_eligibility_not_removed_for_unfavorable_result():
    by,e=selection_data();e[0]['candidates'][s.CANDIDATES[0]]['eligible']=False
    r=s.selection_analysis(by,e,cfg(),s.FIXED)['endpoints'][0]
    assert r['post_confirmation_point_selected']==s.CANDIDATES[1]
    assert r['exploration_fixed_status']=='SAVED_FIXED_CANDIDATE_INELIGIBLE'
    assert r['bootstrap']['winner_counts'][s.CANDIDATES[1]]==20


def test_saved_mean_drift():
    by,e=selection_data();e[0]['candidates'][s.CANDIDATES[0]]['saved_G_exact']='1'
    with pytest.raises(ValueError,match='SAVED_MEAN_REPRODUCTION'):s.selection_analysis(by,e,cfg(),s.FIXED)


def test_retrospective_selection_rejected():
    by,e=selection_data();e[0]['epsilon']='0.0005'
    fixed=dict(s.FIXED,**{'0.0005':s.CANDIDATES[0]})
    with pytest.raises(ValueError,match='RETROACTIVE'):s.selection_analysis(by,e,cfg(),fixed)


@pytest.mark.parametrize('mutation',['phase','law','seed','axis','count','cost'])
def test_saved_trajectory_integrity(mutation):
    by,e=selection_data();r=by[s.CANDIDATES[0]][0]
    if mutation=='phase':r['phase']='exploration'
    if mutation=='law':r['law']='conditional_s2'
    if mutation=='seed':r['seed']=100
    if mutation=='axis':r['event_digest_sine']='different'
    if mutation=='count':by[s.CANDIDATES[0]].pop()
    if mutation=='cost':r['sine']=None
    with pytest.raises(ValueError):s.validate_trajectories(by,s.CANDIDATES,expected_n=4,exploration_seeds=set())


def artificial_binding(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    for f in run.FILES:
        path=root/f;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source '+f)
    monkeypatch.setattr(run.subprocess,'check_output',lambda args,**kw:(root/args[-1].split(':',1)[1]).read_bytes())
    source='a'*40
    closure=dict(SOURCE=source,files={f:run.sha((root/f).read_bytes()) for f in run.FILES})
    config=dict(authorization_status='APPROVED',runtime_authorization=True,stage='SELECTION_SAVED',
        SOURCE=source,one_shot_id='SYNTHETIC_SELECTION_RUN01',analysis=cfg(),
        closure_sha256=digest(closure),input_manifest_sha256=run.sha((root/run.MANIFEST).read_bytes()),
        runtime=run.runtime())
    approval=dict(authorized=True,stage=config['stage'],user_instruction='実際に実行するところまで行ってよい',
                  scope_sha256=digest(config['analysis']),scope_confirmation='SYNTHETIC TEST ONLY')
    grant=dict(valid=True,stage=config['stage'],one_shot=True,one_shot_id=config['one_shot_id'],
        config_sha256=digest(config),approval_sha256=digest(approval),SOURCE=source)
    return root,config,grant,approval,closure


def test_draft_refused_before_input_open():
    with pytest.raises(ValueError,match='EXPLICIT_SCIENTIFIC'):
        run.run(dict(authorization_status='UNAPPROVED'),dict(valid=False),{}, {},
                reader=lambda:pytest.fail('scientific input opened'))


@pytest.mark.parametrize('key',['stage','config_sha256','SOURCE','approval_sha256','one_shot_id'])
def test_grant_binding(tmp_path,monkeypatch,key):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch);g[key]='DRIFT'
    with pytest.raises(ValueError):run.authorization(c,g,a,cl,root)


def test_source_drift(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch)
    (root/run.FILES[0]).write_text('drift')
    with pytest.raises(ValueError,match='SOURCE_DRIFT'):run.authorization(c,g,a,cl,root)


def test_end_to_end_synthetic_and_reuse_rejection(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch);by,e=selection_data()
    data=dict(by=by,endpoints=e,fixed=s.FIXED,provenance=dict(input_kind='SYNTHETIC'))
    kwargs=dict(root=root,reader=lambda:{'fixture':b'unchanged'},projector=lambda raw:data,
                execution_root=tmp_path/'execution')
    out=run.run(c,g,a,cl,**kwargs)
    assert json.loads((out/'final_gate.json').read_text())['status']=='PASS_MANDATORY_STOP'
    with pytest.raises(ValueError,match='OUTPUT_NAMESPACE_COLLISION'):run.run(c,g,a,cl,**kwargs)
    out.rename(out.with_name('moved'))
    with pytest.raises(ValueError,match='ONE_SHOT_ALREADY_CONSUMED'):run.run(c,g,a,cl,**kwargs)


def test_first_STOP_preserved(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch)
    def fail(raw):raise ValueError('SYNTHETIC_ORIGINAL_REASON')
    with pytest.raises(RuntimeError,match='SYNTHETIC_ORIGINAL_REASON'):
        run.run(c,g,a,cl,root=root,reader=lambda:{'fixture':b'bytes'},projector=fail,
                execution_root=tmp_path/'execution')
    out=tmp_path/'execution'/c['stage'].lower()/c['one_shot_id']
    assert 'SYNTHETIC_ORIGINAL_REASON' in (out/'first_STOP.json').read_text()
    assert json.loads((out/'terminal.json').read_text())['science_dispatch_count']==0


def test_authorization_file_drift_causes_STOP(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch);by,e=selection_data()
    auth=tmp_path/'grant.json';auth.write_text('original synthetic grant')
    def project(raw):
        auth.write_text('drift');return dict(by=by,endpoints=e,fixed=s.FIXED,provenance={})
    with pytest.raises(RuntimeError,match='AUTHORIZATION_FILE_DRIFT'):
        run.run(c,g,a,cl,root=root,reader=lambda:{'fixture':b'bytes'},projector=project,
                execution_root=tmp_path/'execution',authorization_paths=[auth])
    out=tmp_path/'execution'/c['stage'].lower()/c['one_shot_id']
    assert not (out/'final_gate.json').exists()


def test_post_input_drift_causes_STOP(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch);by,e=selection_data();calls=[]
    def reader():
        calls.append(1);return {'fixture':bytes(len(calls))}
    with pytest.raises(RuntimeError,match='POST_INPUT_DRIFT'):
        run.run(c,g,a,cl,root=root,reader=reader,
                projector=lambda raw:dict(by=by,endpoints=e,fixed=s.FIXED,provenance={}),
                execution_root=tmp_path/'execution')


def test_rare_runner_synthetic(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch)
    c['stage']='RARE_SENSITIVITY';c['analysis']=dict(candidate_set=list(s.RARE),epsilon_scope=list(s.EPS),
        bias_mode=list(s.MODES),allowance_scope=['1.0','10.0'],mode='TWO_VARIABLE_BOUNDARY',
        unknown_cost_assumptions={},comparator_scope='strengthened184',
        numeric_precision='FRACTION_EXACT_DECIMAL80',new_sampling=0,new_compile=0,new_shot_calculation=0)
    a['stage']=g['stage']=c['stage'];a['scope_sha256']=digest(c['analysis'])
    g.update(config_sha256=digest(c),approval_sha256=digest(a))
    ep=rare_ep();ep['candidates']={k:dict(eligible=True,shots=ep['shots']) for k in s.RARE}
    by={k:[dict(cosine=2,sine=3)] for k in s.RARE}
    data=dict(laws={k:law() for k in s.RARE},by=by,pilot=by,endpoints=[ep],provenance={})
    out=run.run(c,g,a,cl,root=root,reader=lambda:{'fixture':b'bytes'},projector=lambda raw:data,
                execution_root=tmp_path/'execution')
    result=json.loads((out/'result.json').read_text());assert len(result['endpoints'])==2
    assert result['unknown_costs_imputed'] is False


def test_manifest_schema_and_hash_failure(tmp_path,monkeypatch):
    p=tmp_path/adapter.MANIFEST;p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(schema='WRONG',inputs=[])))
    with pytest.raises(ValueError,match='MANIFEST_SCHEMA'):adapter.read_fixed(tmp_path)
    p.write_text(json.dumps(dict(schema='H8_SAVED_ANALYSIS_INPUTS_V1',inputs=[dict(
        role='SYNTHETIC',commit='a'*40,path='fixture.csv',bytes=3,sha256='b'*64,git_blob='c'*40)])))
    monkeypatch.setattr(adapter.subprocess,'check_output',lambda *a,**k:b'bad')
    with pytest.raises(ValueError,match='FIXED_INPUT_DRIFT'):adapter.read_fixed(tmp_path)


def test_registered_eligibility_mask_drift(tmp_path,monkeypatch):
    root,c,g,a,cl=artificial_binding(tmp_path,monkeypatch);by,e=selection_data()
    c['analysis']['saved_eligibility_mask_sha256']='0'*64
    a['scope_sha256']=digest(c['analysis']);g.update(config_sha256=digest(c),approval_sha256=digest(a))
    with pytest.raises(RuntimeError,match='SAVED_ELIGIBILITY_MASK_DRIFT'):
        run.run(c,g,a,cl,root=root,reader=lambda:{'fixture':b'bytes'},
            projector=lambda raw:dict(by=by,endpoints=e,fixed=s.FIXED,provenance={}),
            execution_root=tmp_path/'execution')


@pytest.mark.parametrize('q',[Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(1)])
def test_quantile(q):
    assert s.quantile([Fraction(0),Fraction(4)],q)==4*q
