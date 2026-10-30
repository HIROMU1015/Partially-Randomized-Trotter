"""Artificial scalars/phase data only. Never analyze saved H4 numerics."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_roundwise_det_pr as m
from trottertracks import c_roundwise_det_pr_saved as runner


def binding():
    b=dict(input_sha256='I',JU_sha256='J',ground_sha256='G',reference_sha256='R',physical_input_sha256='P',
        geometry=[4,1.,'ARTIFICIAL'],sector=[2,2,36],frame_order=list(range(8)),kappa_policy=True,
        scalar_policy='nuclear once',spin_policy='invented',state_model=m.STATE,
        phase_convention='X+iY=exp(-iEt)',delta='1/10')
    return dict(D=deepcopy(b),PR=deepcopy(b))


def fixture(lane='ideal_exact_native',mode='mixed'):
    target=m.accuracy.AccuracyTarget(**m.TARGET);identity=m.common_target(binding());plans={};prices={};raw={}
    for a in m.METHODS:
        raw[a]=[];plans[a]=[];prices[a]=[]
        kernel=10 if a=='D' else 5
        if mode=='D':kernel=10 if a=='D' else 20
        if mode=='PR':kernel=100 if a=='D' else 1
        if mode=='tie':kernel=10
        for i in m.ROUNDS:
            A=1. if a=='D' or mode in ('tie','PR') else .9 if i%2==0 else .3
            row=dict(m=i,elapsed_time=.1*(1<<i),state_model=m.STATE,phase_convention='X+iY=exp(-iEt)',
                iid_per_shot=True,attenuation=A,pf_operator_error=.001,rte_corrected_operator_error=0.,
                synthesis_signal_error=1e-8 if a=='D' else None,finite_law_signal_error=0. if a=='D' else None,
                source_target_id='old_'+a)
            raw[a].append(row);plans[a].append(m.assess(target,identity,a,lane,row,evidence_id=a+str(i),alias_verified=True))
            prices[a].extend(m.costs(identity,a,i,{k:kernel for k in m.accuracy.METRICS},
                {name:{k:1 for k in m.accuracy.METRICS} for name in ('head','X','Y')}))
    return target,identity,plans,prices,raw


def test_passive_import_no_science():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_roundwise_det_pr_saved;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim'])"],cwd=ROOT,check=True)


def artificial_bundle():
    from trottertracks.c_pr_aware_df_m2 import CONDITIONS
    from trottertracks.c_pr_aware_df_m4 import POLICY
    _,_,_,_,rows=fixture();hashes=dict(input='I',JU='J',ground='G',reference='R')
    base=dict(input='input',exports=dict(proposed='JU'),grounds=dict(proposed='ground'),
        references=dict(proposed='reference'),original_input_sha256='P',target=m.TARGET,
        compiler='ARTIFICIAL_P0',gaussian='ARTIFICIAL_G',compiler_phase='ARTIFICIAL_PHASE',
        shells='ARTIFICIAL_SHELL_PATHS',saved_inputs=hashes,maximum_logical_units=1,records=dict(B1='B1'))
    d=dict(base,conditions=deepcopy(CONDITIONS),allowances=dict(alias_window_Ha=[-10.,10.]))
    p=dict(base,policy=deepcopy(POLICY));n=deepcopy(p);n['policy']['native']='native-specific metadata'
    bundle={}
    for a,old in [('D',d),('PR',p),('N',n)]:
        bundle[a+'_contract']=old;bundle[a+'_SOURCE']=dict(execution_commit=runner.EXECUTIONS[a])
        bundle[a+'_STOP']=dict(failed=False,mandatory_stop=True,retry=0,resume=0,scientific_units=1)
    D=dict(id='D_proposed',export_sha256='J',rounds=[dict(m=i,target_id='D_proposed') for i in m.ROUNDS],
        PF_rounds=rows['D'],synthesis_rounds=[dict(m=i,complex_signal_bound_pilot=1e-8) for i in m.ROUNDS],
        metrics={k:10 for k in m.accuracy.METRICS})
    bundle['D']=dict(status=runner.STATUSES['D'],candidates=dict(proposed=D))
    bundle['PR']=dict(status=runner.STATUSES['PR'],A=dict(saved_DF_bound_Ha=1e-5,
        rounds=[dict(m=i,reference=row) for i,row in enumerate(rows['PR'])]),
        B={lane:dict(rounds=[dict(m=i,status='UNRESOLVED_SIGNAL_ERROR') if lane=='evidence'
            else dict(m=i,target_id='proposed:J') for i in m.ROUNDS]) for lane in m.LANES})
    bundle['N']=dict(status=runner.STATUSES['N'],calibration=dict(samples=32,
        sample_mean={k:5 for k in m.accuracy.METRICS},sample_mean_standard_error={k:.2 for k in m.accuracy.METRICS}))
    bundle['ground']=dict(status='PASS',alias_margin=1.,energy_Ha=-2.7)
    bundle['physical_ground']=dict(status='PASS',alias_margin=1.)
    bundle['B1']=dict(arms=dict(proposed=dict(saved_DF_bound_Ha=1e-5)))
    bundle['shells']={a:dict(metrics={k:1 for k in m.accuracy.METRICS}) for a in ('head','X','Y')}
    return bundle,dict(saved_inputs=hashes,records=dict(D='artificial_D.json',PR='artificial_PR.json'))


@pytest.mark.parametrize('fault',[None,'native_hash','old_id','ground','source','DF','time_policy'])
def test_actual_saved_schema_bridge_with_artificial_metadata(fault):
    b,c=artificial_bundle()
    if fault=='native_hash':b['N_contract']['saved_inputs']=dict(c['saved_inputs'],JU='OTHER')
    if fault=='old_id':b['D']['candidates']['proposed']['rounds'][0]['target_id']='proposed:J'
    if fault=='ground':b['ground']['alias_margin']=-1.
    if fault=='source':b['D_SOURCE']['execution_commit']='different'
    if fault=='DF':b['PR']['A']['saved_DF_bound_Ha']=.002
    if fault=='time_policy':b['N_contract']['policy']['s']=4
    if fault:
        with pytest.raises(ValueError):runner.bridge(b,c)
    else:
        value=runner.bridge(b,c);assert value['target_id'].startswith('same_saved_DF_energy:')
        assert value['bindings']['D']==value['bindings']['PR']
        assert value['rows']['D'][0]['source_target_id']=='D_proposed'
        assert value['rows']['PR'][0]['source_target_id']=='proposed:J'


def test_passive_preflight_hash_only_fresh_and_metadata(tmp_path):
    (tmp_path/'opaque.npz').write_bytes(b'OPAQUE_ARTIFICIAL_NO_ARRAY_DECODE')
    source=dict(files={},reuse_files={});runner.io.save(tmp_path/'source.json',source)
    c=dict(scope=runner.SCOPE,base_commit=runner.BASE,implementation=m.VERSION,target=m.TARGET,
        lanes=list(m.LANES),call_limits=runner.LIMITS,maximum_logical_units=189,
        fresh_output=runner.FRESH,retry=0,resume=0,source_manifest='source.json',
        source_manifest_sha256=runner.io.sha(tmp_path/'source.json'),
        saved_inputs={'opaque.npz':runner.io.sha(tmp_path/'opaque.npz')},records={},packages={})
    runner.io.save(tmp_path/'contract.json',c)
    assert runner.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'opaque.npz').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):runner.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault',['JU_sha256','ground_sha256','sector','phase_convention','delta'])
def test_common_energy_binding_refuses_different_target(fault):
    b=binding();b['PR'][fault]='OTHER'
    with pytest.raises(ValueError,match='BINDING'):m.common_target(b)


@pytest.mark.parametrize('mode,expected',[('D','MIXED_ROUND_DOMINATED'),('PR','MIXED_ROUND_POTENTIALLY_USEFUL'),('mixed','MIXED_ROUND_POTENTIALLY_USEFUL'),('tie','MIXED_ROUND_DOMINATED')])
def test_separable_assignment_dominance_tie_and_same_schedule_six_metrics(mode,expected):
    target,identity,plans,prices,_=fixture(mode=mode);choice=m.select(plans,prices)
    assert choice['status']==expected
    if mode in ('D','tie'):assert set(choice['schedule'])=={'D'}
    if mode=='PR':assert set(choice['schedule'])=={'PR'} and not choice['strictly_mixed']
    if mode=='mixed':assert choice['strictly_mixed']
    value=m.evaluate_schedule(target,1e-5,plans,prices,choice['schedule'],alias_verified=True)
    assert value['energy']['same_accuracy'] and value['energy']['statistical_failure_upper']<=.01
    assert value['energy']['energy_error_bound_ha']==pytest.approx(1e-5+target.beta_total/(.1*4096))
    # Check separability directly, without enumerating 8192 schedules.
    expected_cost=math.fsum(min(r['conditional_RZ'].values()) for r in choice['comparisons'])
    assert value['resources']['totals']['rz_count']==expected_cost
    assert len({p['target_id'] for p in value['plans']})==1
    assert len({p['source_target_id'] for p in value['plans']})==(2 if mode=='mixed' else 1)


def test_cost_shell_once_serial_depth_and_shared_calibration_SE():
    target,identity,plans,prices,_=fixture();choice=m.select(plans,prices)
    first=prices['D'][0];assert first['mean']['rz_count']==12
    last=prices['D'][-1];assert last['mean']['rz_count']==4096*10+2
    assert last['depth_semantics']=='scheduled_serial_sum'
    values={name:m.evaluate_schedule(target,1e-5,plans,prices,s,alias_verified=True) for name,s in
        dict(all_D=['D']*13,all_PR=['PR']*13,mixed=choice['schedule']).items()}
    se=m.shared_uncertainty(values,{k:2. for k in m.accuracy.METRICS})
    W=sum(2*p['shots']*(1<<p['m']) for p in values['mixed']['plans'] if p['method_id']=='PR')
    assert se['schedules']['mixed']['weighted_SE']['rz_count']==2*W
    assert se['mixed_minus_all_D_SE']['rz_count']==2*W
    assert se['selection_on_same_data'] and se['formal_CI']is None


def test_unknown_errors_not_zero_and_same_conditional_scenario_for_D_PR():
    target,identity,plans,prices,raw=fixture(lane='evidence')
    assert plans['D'][0]['status']=='CONDITIONAL_ROUND_PASS'
    assert plans['PR'][0]['status']=='UNRESOLVED_SIGNAL_ERROR'
    assert m.select(plans,prices)['status']=='UNQUALIFIED'
    p=m.assess(target,identity,'D','assumed_raw_signal_budget',raw['D'][0],evidence_id='D',alias_verified=True)
    assert p['synthesis_error_used']==.001 and p['law_error_used']==0.
    p=m.assess(target,identity,'PR','assumed_raw_signal_budget',raw['PR'][0],evidence_id='PR',alias_verified=True)
    assert p['synthesis_error_used']==p['law_error_used']==.001*raw['PR'][0]['attenuation']


@pytest.mark.parametrize('fault',['unknown_price','nonfinite_price','nonfinite_error','wrong_time','wrong_method','wrong_target','wrong_lane','wrong_scope','missing_round','alias','failure_sum','DF_budget'])
def test_adversarial_failures(fault):
    target,identity,plans,prices,raw=fixture()
    if fault=='unknown_price':
        prices['PR'][0]['mean']=None;assert m.select(plans,prices)['status']=='UNQUALIFIED';return
    if fault=='failure_sum':
        bad=deepcopy(plans)
        for a in m.METHODS:
            for p in bad[a]:p['alpha_axis']=.01
        value=m.evaluate_schedule(target,1e-5,bad,prices,['D']*13,alias_verified=True)
        assert value['energy']['status']=='FAILURE_PROBABILITY_EXCEEDED';return
    if fault=='DF_budget':
        value=m.evaluate_schedule(target,.002,plans,prices,['D']*13,alias_verified=True)
        assert value['energy']['status']=='DF_RESERVE_EXCEEDED';return
    with pytest.raises(ValueError):
        if fault=='nonfinite_price':prices['PR'][0]['mean']['rz_count']=math.inf;m.select(plans,prices)
        elif fault=='nonfinite_error':
            row=dict(raw['PR'][0],pf_operator_error=math.nan);m.assess(target,identity,'PR','evidence',row,evidence_id='x',alias_verified=True)
        elif fault=='wrong_time':m.assess(target,identity,'D','ideal_exact_native',dict(raw['D'][0],elapsed_time=.2),evidence_id='x',alias_verified=True)
        elif fault=='wrong_method':plans['D'][0]['method_id']='PR';m.select(plans,prices)
        elif fault=='wrong_target':plans['PR'][0]['target_id']='another_DF';m.select(plans,prices)
        elif fault=='wrong_lane':plans['PR'][0]['lane']='assumed_raw_signal_budget';m.select(plans,prices)
        elif fault=='wrong_scope':prices['PR'][0]['scope']='whole_round_global_compile';m.select(plans,prices)
        elif fault=='missing_round':plans['PR'].pop();m.select(plans,prices)
        elif fault=='alias':m.assess(target,identity,'D','ideal_exact_native',raw['D'][0],evidence_id='x',alias_verified=False)


def test_phase_room_exclusion_and_unknown_are_different():
    target,identity,plans,prices,raw=fixture()
    p=m.assess(target,identity,'PR','ideal_exact_native',dict(raw['PR'][0],pf_operator_error=.5),evidence_id='x',alias_verified=True)
    assert p['status']=='INFEASIBLE_PHASE';plans['PR'][0]=p
    assert m.select(plans,prices)['schedule'][0]=='D'
    p=m.assess(target,identity,'D','ideal_exact_native',dict(raw['D'][0],pf_operator_error=1.),evidence_id='x',alias_verified=True)
    plans['D'][0]=p;assert m.select(plans,prices)['status']=='INFEASIBLE_REGISTERED_ROUND'


@pytest.mark.parametrize('energy',[-9.9,-2.7,0.,7.2,9.9])
def test_actual_RPE_unwrap_all13_different_methods_and_positive_attenuation(energy):
    decoder=m.accuracy._legacy('rpe_four_round_phase_validation').reconstruct_rpe_phase
    theta=-energy*.1;beta=math.pi/6
    phases=[]
    for i in m.ROUNDS:
        A=1. if i%2==0 else .3
        error=(-1 if i%2 else 1)*beta*.99
        z=A*complex(math.cos((1<<i)*theta+error),math.sin((1<<i)*theta+error))
        phases.append(math.atan2(z.imag,z.real))
    result=decoder(phases,[1<<i for i in m.ROUNDS]);estimate=result['final_phase_estimate']
    assert abs(estimate-theta)<=beta/4096+1e-12
    assert abs(-estimate/.1-energy)<=beta/(4096*.1)+1e-11


@pytest.mark.parametrize('fault',[None,'binding','phase','saved_shots'])
def test_artificial_saved_only_end_to_end189_and_partial_STOP(tmp_path,monkeypatch,fault):
    target,identity,plans,prices,raw=fixture()
    packet=dict(target_id=identity,bindings=binding(),scope='artificial',rows=raw,df=1e-5,alias_verified=True,
        kernel_means={a:{k:10 if a=='D' else 5 for k in m.accuracy.METRICS} for a in m.METHODS},
        shells={name:{k:1 for k in m.accuracy.METRICS} for name in ('head','X','Y')},
        PR_SE={k:.2 for k in m.accuracy.METRICS},evidence_sources=dict(D='artificial_D',PR='artificial_PR'))
    packet['saved_PR_rounds']={lane:[m.assess(target,identity,'PR',lane,row,evidence_id='artificial',alias_verified=True)
        for row in raw['PR']] for lane in m.LANES}
    c=dict(source_manifest='source.json');runner.io.save(tmp_path/'source.json',dict(files={}))
    monkeypatch.setattr(runner,'preflight',lambda *args:c)
    monkeypatch.setattr(runner,'load_bundle',lambda *args:{})
    if fault=='binding':monkeypatch.setattr(runner,'bridge',lambda *args:(_ for _ in ()).throw(ValueError('ARTIFICIAL_BINDING')))
    else:monkeypatch.setattr(runner,'bridge',lambda *args:packet)
    if fault=='phase':packet['rows']['PR'][0]['phase_convention']='X-iY'
    if fault=='saved_shots':packet['saved_PR_rounds']['ideal_exact_native'][0]['shots']+=1
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *args,**kw:'ARTIFICIAL_SOURCE\n')
    if fault:
        with pytest.raises(ValueError):runner.run(tmp_path,'unused')
        stop=runner.io.read(tmp_path/runner.FRESH/'STOP.json');assert stop['failed'] and stop['scientific_units']<189
    else:
        runner.run(tmp_path,'unused');stop=runner.io.read(tmp_path/runner.FRESH/'STOP.json')
        assert stop['scientific_units']==runner.MAX_CALLS==189 and stop['call_counts']==runner.LIMITS
        result=runner.io.read(tmp_path/runner.FRESH/'result.json')
        assert result['lanes']['evidence']['allocation']['status']=='UNQUALIFIED'
        assert result['lanes']['ideal_exact_native']['allocation']['status']=='MIXED_ROUND_POTENTIALLY_USEFUL'
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    with pytest.raises(FileExistsError):runner.run(tmp_path,'unused')
