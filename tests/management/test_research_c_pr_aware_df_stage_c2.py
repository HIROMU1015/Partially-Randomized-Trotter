"""Artificial categorical laws, proof inputs and circuits; no saved H4 decode."""
from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_stage_c2 as m


def test_passive_import():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_stage_c2;assert not any(x in sys.modules for x in ('numpy','qiskit','ffsim'))"],cwd=ROOT,check=True)


@pytest.mark.parametrize('p',[[.25,.25,.5],[0,.3,0,.7],[1e-200,.5,.5-1e-200],[.1,.2,.3,.4]])
def test_grid_counts_match_exhaustive_small_grid_and_boundary_side(p):
    model=m.choice_grid(p,bits=8)
    actual=np.searchsorted(model['cdf'],np.arange(256)/256,side='right')
    assert model['counts']==np.bincount(actual,minlength=len(p)).tolist()
    assert sum(model['counts'])==256 and model['zero_random_draws']


def test_choice_real_entry_boundary_with_injected_uniforms_and_no_PCG_claim():
    class BoundaryGenerator(np.random.Generator):
        def random(self,size=None,**kw):
            # Cover exact ties, zero, predecessor and last grid value.
            u=np.array([0.,np.nextafter(.25,0),.25,.5,1.-2.**-53])
            return u if size is not None else u[0]
    g=BoundaryGenerator(np.random.PCG64(5));p=np.array([.25,.25,.5])
    assert g.choice(3,size=5,p=p).tolist()==[0,0,1,2,2]
    assert np.searchsorted(m.choice_grid(p)['cdf'],g.random(5),side='right').tolist()==[0,0,1,2,2]


def test_tiny_zero_underflow_mass_never_silently_deleted():
    tiny=F(1,2**2000)
    a=m.analyze_table([tiny,1],[0.,1.])
    assert F(a['eta_exact'])==tiny/(1+tiny) and a['eta_upper_float']>0
    assert a['lost_positive_ideal_atoms']==[0]
    b=m.choice_grid([1e-200,1.]);assert b['counts'][0]==1 # grid point0
    c=m.choice_grid([1.,1e-200]);assert c['counts'][1]==0
    assert m.tv([0,1],[0,1])==0


@pytest.mark.parametrize('p',[[.1,.1],[-.1,1.1],[math.nan,1],[0,0],[[1.]]])
def test_invalid_choice_inputs(p):
    with pytest.raises((ValueError,TypeError)):m.choice_grid(p)


def test_exact_TV_normalization_stages_and_no_double_count():
    a=m.analyze_table([1,2,7],[.1,.2,.7])
    assert F(a['eta_exact'])==m.tv([F(1,10),F(1,5),F(7,10)],a['grid']['probabilities_exact'])
    assert F(a['eta_exact'])<=sum(map(F,a['stages_TV'].values()))
    assert 0<=F(a['eta_exact'])<=1 and a['actual_implemented_law_error'] is None
    assert m.tv([1,0],[0,1])==1
    with pytest.raises(ValueError):m.tv([.1,.2],[.2,.1])


def test_round_event_multiplicity_raw_units_no_attenuation_and_unknown_PCG():
    a=dict(status='EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL',eta_exact='1/1000000000000')
    r=m.law_round(a,12)
    assert r['events']==16*4*4096
    assert F(r['conditional_raw_complex_bound_exact'])==2*r['events']*F(a['eta_exact'])
    assert r['actual_implemented_law_error'] is None
    assert m.law_round(a,12,random_model='PCG64_FIXED_SEEDS')['conditional_raw_complex_bound_upper'] is None
    a['eta_exact']='1';assert m.law_round(a,0)['conditional_raw_complex_bound_exact']=='2'


def proof(e='1/1000000',**kw):
    return dict(kind='analytic_unitary_bound',norm='full_controlled_operator_2',
        phase_sensitive=True,epsilon=e,reference_error='0',compiler_path='whole_kernel_P0',
        coverage='all_final_whole_kernel_outputs',provenance='invented exact proof only',**kw)


def test_local_telescoping_with_reference_error_encloses_toy_phase_and_legacy():
    theta=1e-6;raw=np.diag([1.,np.exp(1j*theta)])
    assert abs(raw[1,1]-1)<=theta
    p=proof();shells={a:proof('0') for a in ('head','X','Y')}
    v=m.synthesis_preview([p],path='whole_kernel_P0',occurrences=7,shells=shells)
    assert F(v['candidate_raw_complex_bound'])==F(28,10**6)
    p['reference_error']='1/1000000'
    assert F(m.synthesis_preview([p],path='whole_kernel_P0',occurrences=7,shells=shells)['candidate_raw_complex_bound'])==F(56,10**6)
    from trottertracks.c1_stage3_deterministic_accuracy import round_synthesis
    legacy=round_synthesis(dict(unitary_distance_pilot=1e-6),{k:dict(unitary_distance_pilot=0.) for k in shells},7)
    assert float(F(v['candidate_raw_complex_bound']))>=legacy['complex_signal_error_pilot']


@pytest.mark.parametrize('change',[dict(kind='sample_max_residual'),dict(phase_sensitive=False),dict(reference_error=None),dict(compiler_path='event_P0')])
def test_proof_scope_phase_reference_rejection(change):
    p=proof();p.update(change)
    with pytest.raises(ValueError):m.synthesis_preview([p],path='whole_kernel_P0',occurrences=1,shells={a:proof('0') for a in ('head','X','Y')})


def test_eventwise_compilation_does_not_qualify_whole_kernel():
    p=proof();p['coverage']='individual_events_only'
    with pytest.raises(ValueError,match='WHOLE_KERNEL'):m.synthesis_preview([p],path='whole_kernel_P0',occurrences=1,shells={a:proof('0') for a in ('head','X','Y')})
    assert m.synthesis_preview([None],path='whole_kernel_P0',occurrences=1,shells={a:None for a in ('head','X','Y')})['candidate_raw_complex_bound'] is None


def test_cost_support_provenance_and_existing_CI_bridge_on_artificial_data():
    from trottertracks import c1_stage3_accuracy as ac
    assert m.support_preview(None)['support'] is None
    for kind in ('observed_max','sample_min_max','event_compiled_price_sum'):
        with pytest.raises(ValueError):m.support_preview(dict(kind=kind))
    p=dict(kind='structural_whole_kernel_P0_range',all_words=True,pass_iteration_proof='invented finite identity compiler',provenance='artificial proof',bounds={k:[0,10] for k in ac.METRICS})
    s=m.support_preview(p)
    out=ac.bounded_cost_intervals([{k:5 for k in ac.METRICS}]*32,support_bounds=s['support'],support_provenance=p['provenance'],alpha_per_metric=.01/12)
    assert out['simultaneous_interval_qualified']


def test_C1_preview_never_promotes_table_only_or_runs_shots():
    rows=[m.law_round(dict(status='EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL',eta_exact='0'),i) for i in range(13)]
    p=m.c1_preview(rows)
    assert p['error_evidence'] is p['guaranteed_shots'] is p['guaranteed_total'] is None
    assert all(x['synthesis'] is x['law'] is None for x in p['round_errors'])


def artificial_saved():
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom,make_tail,atom_record
    atoms=tuple(MixedAtom(str(i),'PHYSICAL_PAULI',F(i+1),('mock',),('Z',),'mock') for i in range(216))
    tail=make_tail(atoms,'fake')
    return dict(atoms_catalog=[atom_record(a) for a in atoms],components=[{k:getattr(c,k) for k in ('component_id','probability','coefficient_abs','coefficient_sign','basis_id','df_fragment_id')} for c in tail.components],native_policy='common_precision_spin_kappa_relative_head_full_frame_tail_v1',configuration=dict(r=16,K=0),finite_distribution=dict(orders=[0],order_probabilities=[1.]))


def test_artificial_E2E_exact35calls_and_failure_stops(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    root=tmp_path;out=root/'out';out.mkdir()
    records=dict(C1_result=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE'),C1_SOURCE=dict(execution_commit='b1c735da394d4b0a444cbc618ead9781ef4d9134'),baseline_tail=artificial_saved(),proposed_tail=artificial_saved())
    c=dict(records={})
    for k,v in records.items():m.io.save(root/(k+'.json'),v);c['records'][k]=k+'.json'
    call=Calls(out,m.LIMITS,m.MAX_CALLS);r=m.execute(root,c,out,call)
    assert m.MAX_CALLS==35 and dict(call.counts)==m.LIMITS and r['scientific_units']==35
    assert r['guaranteed_shots'] is r['guaranteed_total'] is r['actual_implemented_law_error'] is None
    assert len(r['arms'])==2
    bad=deepcopy(records['baseline_tail']);bad['components'][0]['probability']*=2
    m.io.save(root/'bad_table.json',bad);c['records']['baseline_tail']='bad_table.json'
    out2=root/'bad';out2.mkdir();call=Calls(out2,m.LIMITS,m.MAX_CALLS)
    with pytest.raises(ValueError,match='REPLAY'):m.execute(root,c,out2,call)
    assert call.stop.is_set() and call.counts['output_summary']==0


def test_static_audit_and_artificial_P0_phase_under_existing_mutex():
    from trottertracks.c_pr_aware_df_stage_c2_audit import compiler_audit
    audit=compiler_audit()
    assert not audit['pass_manager_executed'] and audit['useful_global_synthesis_bound'] is None
    assert any(x['name']=='DoWhileController' for x in audit['stages']['optimization'])
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    shared=SharedQiskitCalls(lambda k,l,fn:fn())
    circuit=QuantumCircuit(2);circuit.h(1);circuit.cp(.15,1,0);circuit.p(.3,1)
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    compiled=shared('compile','artificial',lambda:compiled_wrapper(circuit,CompilerProfile()))
    assert np.max(abs(Operator(compiled).data-Operator(circuit).data))<1e-10
    with shared.adjoint_builds():
        from trottertracks.c1_stage2_native import actual_adjoint
        adjoint=actual_adjoint(compiled)
    assert np.max(abs(Operator(adjoint).data@Operator(compiled).data-np.eye(4)))<1e-10


def test_full_artificial_run_STOP_fresh_and_hash_failure(tmp_path,monkeypatch):
    import subprocess
    root=tmp_path;records=dict(C1_result=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE'),
        C1_SOURCE=dict(execution_commit='b1c735da394d4b0a444cbc618ead9781ef4d9134'),
        C1_STOP=dict(failed=False,scientific_units=164,retry=0,resume=0,mandatory_stop=True),
        baseline_tail=artificial_saved(),proposed_tail=artificial_saved())
    paths={}
    for k,v in records.items():m.io.save(root/(k+'.json'),v);paths[k]=k+'.json'
    m.io.save(root/'source.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,status='UNAPPROVED_SCIENCE_CONTRACT_DRAFT',
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,
        retry=0,resume=0,random_model=m.IID,physical=m.POLICY,source_manifest='source.json',
        source_manifest_sha256=m.io.sha(root/'source.json'),saved_inputs={p:m.io.sha(root/p) for p in paths.values()},
        packages={'numpy':'1.26.4'},records=paths)
    m.io.save(root/'contract.json',c)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'mock artificial SOURCE\n')
    m.run(root,root/'contract.json',root/m.FRESH)
    stop=m.io.read(root/m.FRESH/'STOP.json')
    assert stop['scientific_units']==35 and not stop['failed'] and stop['mandatory_stop']
    with pytest.raises(ValueError,match='FRESH'):m.run(root,root/'contract.json',root/m.FRESH)
    (root/'baseline_tail.json').write_text('{}') # exclusively this artificial fixture
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(root,root/'contract.json')


def test_existing_make_tail_rejects_positive_underflow():
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom,make_tail
    with pytest.raises(ValueError,match='loss'):
        make_tail([MixedAtom('tiny','PHYSICAL_PAULI',F(1,2**2000),('mock',),('Z',),'fake')],'fake')
