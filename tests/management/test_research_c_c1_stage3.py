"""Artificial scalars/matrices and mocked circuits only; no molecular decode."""
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch, Mock, call

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from trottertracks import c1_stage3_accuracy as a, c1_stage3_connection as c, c1_stage3_reference as ref
from trottertracks import c1_stage2_native as native
from test_research_c_c1_stage2_native import bridges,CircuitDouble


def target(**updates):
    kw=dict(energy_ha=1/8,df_reserve_fraction=1/4,base_time=1.,beta_total=.4,
            beta_stat=.2,alpha_axis=.001,alpha_total=.05)
    return a.AccuracyTarget(**(kw|updates))


def evidence(m=0,**updates):
    kw=dict(m=m,attenuation=.8,pf_operator_error=.01,rte_corrected_operator_error=.005,
        synthesis_signal_error=.001,finite_law_signal_error=.002,target_id='ARTIFICIAL',
        state_model='own_DF_ground_state_oracle',alias_branch_verified=True,iid_per_shot=True)
    return a.RoundEvidence(**(kw|updates))


def plans(t=None):
    t=t or target()
    return [a.assess_round(t,evidence(m)) for m in range(t.horizon()['maximum_round_index_M']+1)]


def costs(ps,**updates):
    return [dict(m=p['m'],axis=axis,target_id='ARTIFICIAL',
        scope='serial_native_kernel_composition_without_state_preparation',long_round_coverage=True,
        mean={k:2. for k in a.METRICS},lower={k:1. for k in a.METRICS},upper={k:3. for k in a.METRICS},
        simultaneous_interval_qualified=True,classical_cost_samples=9999,**updates)
        for p in ps for axis in ['X','Y']]


def test_passive_import_and_no_scientific_cli_default():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_accuracy,c1_stage3_connection,c1_stage3_reference;assert not any(k in sys.modules for k in ['numpy','qiskit','scipy','ffsim'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    r=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_stage3.py'),
        '--contract','/NEVER_READ','--evaluate-evidence','/NEVER_READ'],capture_output=True,text=True)
    assert r.returncode==2 and 'fresh output' in r.stderr


def test_error_units_attenuation_geometry_and_existing_shot_helper():
    t=target();e=evidence();r=a.assess_round(t,e)
    delta=.8*(.01+.005)+.001+.002
    assert r['raw_signal_error']==pytest.approx(delta)
    assert r['radius_lower']==pytest.approx(.8-delta)
    assert r['beta_deterministic']==pytest.approx(math.asin(delta/.8))
    eps=(.8-delta)*math.sin(.2)/math.sqrt(2)
    assert r['shots']==math.ceil(2*math.log(2/.001)/eps**2)
    assert r['quantum_shots_drawn']==0


@pytest.mark.parametrize('updates',[dict(synthesis_signal_error=None),dict(finite_law_signal_error=None)])
def test_unknown_errors_are_not_zero_filled(updates):
    r=a.assess_round(target(),evidence(**updates));assert r['shots'] is None and r['status']=='UNRESOLVED_SIGNAL_ERROR'


@pytest.mark.parametrize('updates',[dict(state_model='HF'),dict(alias_branch_verified=False),dict(iid_per_shot=False)])
def test_survival_signal_does_not_imply_ground_energy(updates):
    with pytest.raises(ValueError):a.assess_round(target(),evidence(**updates))


def test_ideal_law_is_an_explicit_conditional_model_not_actual_sampler_certificate():
    r=a.assess_round(target(),evidence(law_model='ideal_finite_RTE_law',finite_law_signal_error=0.))
    assert r['law_model']=='ideal_finite_RTE_law'
    with pytest.raises(ValueError):a.assess_round(target(),evidence(law_model='ideal_finite_RTE_law'))


def test_radius_and_phase_fail_before_shot_evaluation():
    assert a.assess_round(target(),evidence(synthesis_signal_error=1.))['status']=='INFEASIBLE_RADIUS'
    assert a.assess_round(target(),evidence(pf_operator_error=.4))['status']=='INFEASIBLE_PHASE'


def test_energy_uses_weyl_once_after_phase_to_energy_conversion():
    t=target();ps=plans(t);r=a.assess_energy(t,.01,ps,state_model='own_DF_ground_state_oracle',alias_branch_verified=True)
    assert r['same_accuracy'] and r['energy_error_bound_ha']==pytest.approx(.01+.4/(1<<ps[-1]['m']))
    assert r['statistical_failure_upper']==pytest.approx(2*len(ps)*.001)
    assert r['numerical_pilot'] and r['state_preparation']=='ORACLE_COST_EXCLUDED'


def test_full_horizon_df_cap_and_failure_budget_are_required():
    t=target();ps=plans(t);kw=dict(state_model='own_DF_ground_state_oracle',alias_branch_verified=True)
    assert a.assess_energy(t,.01,ps[:-1],**kw)['status']=='INCOMPLETE_ROUNDS'
    assert a.assess_energy(t,.1,ps,**kw)['status']=='DF_RESERVE_EXCEEDED'
    for row in ps:row['alpha_axis']=.1
    assert a.assess_energy(t,.01,ps,**kw)['status']=='FAILURE_PROBABILITY_EXCEEDED'


def test_total_cost_counts_shots_and_axes_but_not_classical_calibration_samples():
    ps=plans();cs=costs(ps);r=a.aggregate_totals({'same_accuracy':True},ps,cs,cost_scope=cs[0]['scope'])
    assert r['shots']==r['executions']==2*sum(p['shots'] for p in ps)
    assert r['totals']['rz_count']==2*r['shots']
    assert r['state_preparation_cost'] is None and not r['unconditional_total_resource_claim']
    assert r['intervals']['rz_count']==[r['shots'],3*r['shots']]


def test_missing_scope_coverage_or_intervals_never_produces_false_advantage():
    ps=plans();cs=costs(ps)
    assert a.aggregate_totals({'same_accuracy':False},ps,cs,cost_scope=cs[0]['scope'])['totals'] is None
    assert a.aggregate_totals({'same_accuracy':True},ps,cs[:-1],cost_scope=cs[0]['scope'])['totals'] is None
    cs[0]['long_round_coverage']=False
    assert a.aggregate_totals({'same_accuracy':True},ps,cs,cost_scope=cs[0]['scope'])['totals'] is None
    with pytest.raises(ValueError):a.aggregate_totals({'same_accuracy':True},ps,cs,cost_scope='time_evolution_subcircuit')
    cs=costs(ps)
    for row in cs:row['simultaneous_interval_qualified']=False
    r=a.aggregate_totals({'same_accuracy':True},ps,cs,cost_scope=cs[0]['scope'])
    assert a.compare_totals(r,r)['verdict']=='POINT_ESTIMATES_ONLY'


def test_serial_native_kernel_depth_and_reused_interval_scale_linearly():
    kernel=dict(target_id='A',policy='native_kernel_repeat_with_fences',mean={k:3. for k in a.METRICS},
        lower={k:2. for k in a.METRICS},upper={k:4. for k in a.METRICS},simultaneous_interval_qualified=True,classical_cost_samples=32)
    shell=dict(axis='Y',policy='separate_native_readout',mean={k:1. for k in a.METRICS},
        lower={k:1. for k in a.METRICS},upper={k:1. for k in a.METRICS},simultaneous_interval_qualified=True)
    r=c.compose_serial_cost(kernel,shell,m=3,substeps=4,target_id='A')
    assert r['kernel_occurrences']==32 and r['mean']['total_depth']==97
    assert r['lower']['rz_count']==65 and r['upper']['rz_count']==129
    assert r['depth_semantics']=='scheduled_serial_sum'
    kernel['policy']='unqualified_global_optimized_q1_proxy'
    with pytest.raises(ValueError):c.compose_serial_cost(kernel,shell,m=3,substeps=4,target_id='A')


def toy_model(ops):
    U=((1.+0j,0j),(0j,1.+0j))
    body=ops.DiagonalEvolutionPrimitives(-.2,((0,-.3),),())
    frame=ops.DiagonalEvolutionPrimitives(-.7,((0,.4),(1,-.2)),((0,1,.6),))
    return native.NativeModel('ARTIFICIAL',(native.NativeBlock('one_body',U,body),native.NativeBlock('F0',U,frame)),.375,'S','I')


def test_generalK_tail_split_retains_sign_support_and_extracts_identity_once():
    with bridges() as (ops,_):
        m=toy_model(ops);p=c.split_native_model(m,0)
        assert [a.coefficient for a in p.atoms]==[Fraction.from_float(.2),Fraction.from_float(-.1),Fraction.from_float(.3)]
        assert p.extracted_identity==.7 and len(p.atoms)==3
        assert [a.support for a in p.atoms]==[(0,),(1,),(0,1)]
        assert not c.split_native_model(m,1).atoms


def test_reuse_real_RTE_configuration_sampler_and_fullframe_provider_with_artificial_data():
    with bridges() as (ops,_):
        p=c.split_native_model(toy_model(ops),0);tail,config,distribution=c.configure_rte(p,'1/10',2,2)
        assert config.rte_steps==2 and config.finite_taylor_order==2
        events=c.sample_occurrence(tail,distribution,2,17);assert len(events)==2
        assert c.trajectory_seed(1,'A','C',0,'X',0,0)!=c.trajectory_seed(1,'A','C',0,'X',1,0)
        provider=c.FullFrameProvider(p)
        with patch('trottertracks.pr_dictionary_validation.circuits.BasisProvider.basis',return_value='FULL') as method:
            assert provider.basis(p.atoms[0],True)=='FULL'
            method.assert_called_once_with(p.atoms[0],singleton=False)


def test_partial_kernel_order_actual_phase_and_wrong_event_time_rejection():
    with bridges() as (ops,_):
        p=c.split_native_model(toy_model(ops),0);tail,config,d=c.configure_rte(p,'1/10',2,0)
        events=c.sample_occurrence(tail,d,2,3);builder=Mock()
        builder.sequence.side_effect=[CircuitDouble(3,name='D_forward'),CircuitDouble(3,name='D_reverse')]
        with patch('trottertracks.pr_dictionary_validation.circuits.build_event_sequence',return_value=CircuitDouble(3,name='RTE')) as emit:
            qc=c.partial_kernel(p,'1/10',events,rte_config=config,builder=builder,provider='FULL')
            assert [r[0] for r in qc.records]==['compose:D_forward','compose:RTE','compose:D_reverse','p']
            assert qc.records[-1][1]==pytest.approx(-(.375+.7)/10)
            assert all(v.kwargs['scalar_time']==0 for v in builder.sequence.call_args_list)
            emit.assert_called_once_with(events,p.atoms,'FULL',optimized=True)
            with pytest.raises(ValueError):c.partial_kernel(p,'1/5',events,rte_config=config,builder=builder,provider='FULL')


def test_artificial_reference_reuses_Taylor_and_distinguishes_pf_rte_and_attenuation():
    X=np.array([[0.,1.],[1.,0.]]);Z=np.diag([1.,-1.]);I=np.eye(2)
    rte=a._legacy('rte')
    # A plain original-tail record exercises the existing canonical config.
    component=rte.RTEComponent(component_id='Z',probability=1.,coefficient_abs=.3,coefficient_sign=1)
    tail=SimpleNamespace(tail_id='TOY',tail_hash='TOY',lambda_r=.3,components=(component,))
    config,_=rte.make_rte_config(tail,evolution_time=.2,rte_steps=2,truncation_tolerance=1.,finite_taylor_order=2,seed=0)
    cache={};kernels=ref.reference_kernels((.1*X+.375*I,.3*Z+.7*I),'1/5',ld=0,identity_tail=.7,rte_config=config,exponential_cache=cache)
    assert len(cache)==1  # reused forward/reverse exponential
    r=ref.reference_round(kernels,(.1*X+.3*Z+1.075*I,.4),repetitions=2)
    assert r['pf_operator_error']>0 and r['rte_corrected_operator_error']>0 and 0<r['attenuation']<1
    deterministic=ref.reference_kernels((.1*X,.3*Z),'1/5',ld=1)
    assert ref.reference_round(deterministic,(.1*X+.3*Z,.2),repetitions=1)['rte_corrected_operator_error']==0
    with pytest.raises(ValueError):ref.reference_round(kernels,(X,.3),repetitions=2)


def test_evidence_packet_full_artificial_pipeline():
    t=target();ps=plans(t)
    packet=dict(target=vars(t),df_error_bound_ha=.01,state_model='own_DF_ground_state_oracle',
        alias_branch_verified=True,rounds=[vars(evidence(p['m'])) for p in ps],costs=costs(ps),
        cost_scope='serial_native_kernel_composition_without_state_preparation')
    r=a.evaluate_evidence_packet(packet)
    assert r['accuracy']['same_accuracy'] and r['resources']['totals'] is not None and r['mandatory_stop']


def test_serial_native_policy_refreshes_every_occurrence_and_fences_boundaries():
    with bridges(),patch.object(CircuitDouble,'barrier',lambda self,*qs:self._record('barrier',qubits=qs),create=True):
        factory=Mock(side_effect=lambda j:CircuitDouble(3,name='occurrence'+str(j)))
        qc=c.serial_native_sequence(factory,occurrences=3,modes=2)
        assert factory.call_args_list==[call(j) for j in range(3)]
        assert [r[0] for r in qc.records]==['barrier','compose:occurrence0','barrier',
            'compose:occurrence1','barrier','compose:occurrence2','barrier']
        head,tail=c.readout_segments(2,'Y')
        assert [r[0] for r in head.records]==['h']
        assert [r[0] for r in tail.records]==['sdg','h','measure']


def test_cost_interval_comparison_requires_disjoint_intervals():
    ps=plans();cs=costs(ps)
    total=a.aggregate_totals({'same_accuracy':True},ps,cs,cost_scope=cs[0]['scope'])
    assert a.compare_totals(total,total)['verdict']=='INDETERMINATE_OVERLAP'
    larger={**total,'intervals':{k:[10*v[0],10*v[1]] for k,v in total['intervals'].items()}}
    assert a.compare_totals(total,larger)['verdict']=='CDF_LOWER_CONDITIONAL_INTERROGATION_COST'


def test_cost_bounds_need_declared_support_not_observed_range():
    rows=[{k:2. for k in a.METRICS},{k:4. for k in a.METRICS}]
    assert not a.bounded_cost_intervals(rows)['simultaneous_interval_qualified']
    r=a.bounded_cost_intervals(rows,support_bounds={k:[0.,10.] for k in a.METRICS},
        alpha_per_metric=.001,support_provenance='ARTIFICIAL_KNOWN_RANGE')
    assert r['mean']['rz_count']==3. and r['lower']['rz_count']==0. and r['upper']['rz_count']==10.
    with pytest.raises(ValueError):a.bounded_cost_intervals(rows,support_bounds={k:[0.,1.] for k in a.METRICS},
        alpha_per_metric=.001,support_provenance='INVALID_RANGE')


def test_virtual_rpe_reuses_arbitrary_round_reconstruction_on_artificial_signals():
    signals=[np.exp(1j*.2*(1<<m)) for m in range(3)]
    r=ref.sample_rpe_phases(signals,[1000]*3,trials=8,seed=7)
    assert len(r['phase_estimates'])==8 and r['actual_quantum_shots_executed']==0
    assert max(abs(v-.2) for v in r['phase_estimates'])<.1


def test_zero_observed_radius_omits_corrected_matrix_power_and_shots():
    assert a.assess_round(target(),evidence(attenuation=0.))['status']=='INFEASIBLE_RADIUS'
    kernels=dict(config=object(),step_time=.1)
    with patch.object(ref,'_legacy',return_value=SimpleNamespace(finite_rte_attenuation=lambda *args,**kw:0.)),patch.object(np.linalg,'matrix_power',side_effect=AssertionError('Unneeded matrix power')):
        r=ref.reference_round(kernels,(np.eye(2),.2),repetitions=2)
    assert r['status']=='INFEASIBLE_OBSERVED_RADIUS'


def test_virtual_shots_beyond_numpy_integer_range_are_not_silently_clipped():
    r=ref.sample_rpe_phases([1.],[1<<64],trials=8,seed=0)
    assert r['status']=='VIRTUAL_BINOMIAL_INTEGER_RANGE_EXCEEDED' and r['virtual_trials']==0
