"""TESTS_DEFINED; TESTS_NOT_EXECUTED in Phase 2D-I.

These are scientific tests, not a preflight shortcut. Future execution needs an
explicit authorization and mapping to call inventory; pytest is not run in I.
No fixture/circuit is constructed during collection of this module.
"""
from __future__ import annotations
from fractions import Fraction
import pytest


def prepared():
    from trottertracks.pr_dictionary_validation.fixture import p0_candidate, EvolutionContext
    from trottertracks.pr_dictionary_validation.algebra import materialize_alternatives
    from trottertracks.pr_dictionary_validation.prototype import make_fragment_alternatives
    spec=p0_candidate(); spec.validate()
    data=materialize_alternatives(spec)
    return spec,EvolutionContext('1/5',2,2,2),data,make_fragment_alternatives(data)


def arm_data(arm):
    from trottertracks.pr_dictionary_validation.prototype import canonicalize_arm,make_tail,configure
    from trottertracks.pr_dictionary_validation.circuits import BasisProvider
    spec,context,data,alternatives=prepared()
    atoms=canonicalize_arm(alternatives,arm); tail=make_tail(atoms,arm)
    # A future approved final packet must bind the API tolerance; tests specify
    # a local diagnostic value rather than claim a signal-bias certificate.
    config,law=configure(tail,context,truncation_tolerance=1.0)
    provider=BasisProvider(4,{f.fragment_id:v for f,v,_s,_p in data})
    return spec,context,atoms,tail,config,law,provider


def test_exact_serializer_roundtrip_and_noncanonical_fraction_rejection():
    from trottertracks.pr_dictionary_validation.fixture import (
        p0_candidate,specification_record,specification_from_record,fraction)
    spec=p0_candidate()
    assert specification_from_record(specification_record(spec))==spec
    for invalid in ('2/4','0.2',' 1/5','1/0'):
        with pytest.raises((ValueError,ZeroDivisionError)): fraction(invalid)


def test_context_time_meaning_is_not_q_r_swap():
    from trottertracks.pr_dictionary_validation.fixture import EvolutionContext
    f1=EvolutionContext('1/5',2,2,2); f2=EvolutionContext('1/5',1,2,0)
    assert f1.delta==Fraction(1,10) and f1.micro_time==Fraction(1,20)
    assert f2.delta==Fraction(1,5) and f2.micro_time==Fraction(1,10)
    with pytest.raises(ValueError): EvolutionContext('1/5',2,2,4).validate()


def test_exact_jw_multiplication_matches_independent_car():
    from trottertracks.pr_dictionary_validation.algebra import annihilator
    from trottertracks.pr_dictionary_validation import reference as r
    from trottertracks.pr_dictionary_validation.gates import matrix_gate
    ctx=r.precision_context(90)
    for p in range(4):
        terms=annihilator(4,p); actual=ctx.matrix(16,16)
        for w,c in terms.items():
            actual+=(r.exact(ctx,str(c.real))+ctx.j*r.exact(ctx,str(c.imag)))*r.pauli_operator(ctx,w)
        matrix_gate(r.to_numpy(actual),r.to_numpy(r.car_annihilator(ctx,4,p)),1e-11,label='JW')


def test_all_four_dictionaries_reconstruct_original_h_and_sector():
    from trottertracks.pr_dictionary_validation.prototype import canonicalize_arm
    from trottertracks.pr_dictionary_validation import reference as r
    from trottertracks.pr_dictionary_validation.gates import matrix_gate
    spec,_context,_data,alternatives=prepared(); ctx=r.precision_context(90)
    h,d,_res,_fs,frames=r.original_operators(ctx,spec)
    for arm in ('SS','SP','PS','PP'):
        atoms=canonicalize_arm(alternatives,arm)
        got=r.dictionary_operator(ctx,atoms,frames)+d+r.exact(ctx,spec.scalar)*ctx.eye(16)
        matrix_gate(r.to_numpy(got),r.to_numpy(h),1e-11,label='H_'+arm)


def test_pauli_collect_exact_cancellation_without_screening():
    from trottertracks.pr_dictionary_validation.prototype import (
        MixedAtom,FragmentAlternatives,canonicalize_arm)
    small=Fraction(1,10**30); word=('Z','I','I','I')
    a=MixedAtom('a','PHYSICAL_PAULI',Fraction(1),('F0',),word,'local:ZZZZ')
    b=MixedAtom('b','PHYSICAL_PAULI',-Fraction(1)+small,('F1',),word,'local:ZZZZ')
    atoms=canonicalize_arm((FragmentAlternatives('F0',(),(a,)),FragmentAlternatives('F1',(),(b,))),'PP')
    assert len(atoms)==1 and atoms[0].coefficient==small
    assert atoms[0].origins==('F0','F1')


def test_zero_tail_is_exactly_certified_only():
    from trottertracks.pr_dictionary_validation.prototype import make_tail,configure
    from trottertracks.pr_dictionary_validation.fixture import EvolutionContext
    empty=make_tail((),'exact_empty')
    assert empty.deterministic_only_certified
    assert configure(empty,EvolutionContext('1/5',2,2,2),truncation_tolerance=1.0)==(None,None)


def test_physical_pauli_not_disguised_as_df_support():
    from trottertracks.pr_dictionary_validation.prototype import make_tail
    from trottertracks.pr_dictionary_validation.cases import catalog_atoms
    atoms=catalog_atoms('test_frame_identity')
    tail=make_tail(atoms,'catalog')
    c=next(x for x in tail.components if x.component_id=='C')
    assert c.diagonal_pauli_support is None and c.df_fragment_id is None
    assert atoms[2].support==(0,2,3)


def test_scalar_interval_binary_endpoint_decoding_and_ceil_refusal():
    from trottertracks.pr_dictionary_validation.intervals import binary_mpf_fraction
    from trottertracks.pr_dictionary_validation.gates import shot_count_from_enclosure
    assert binary_mpf_fraction((0,3,-2,2))==Fraction(3,4)
    assert shot_count_from_enclosure(Fraction(11,10),Fraction(19,10))==2
    with pytest.raises(ValueError): shot_count_from_enclosure(Fraction(19,10),Fraction(21,10))


def test_actual_component_product_rotation_control_blocks():
    from trottertracks.pr_dictionary_validation import reference as r
    from trottertracks.pr_dictionary_validation.circuits import component_probe
    from trottertracks.pr_dictionary_validation.gates import controlled_gate
    spec,context,atoms,_tail,_config,law,provider=arm_data('SP')
    ctx=r.precision_context(90); *_head,frames=r.original_operators(ctx,spec)
    for atom in atoms:
        q=r.unsigned_atom_operator(ctx,atom,frames); sign=1 if atom.coefficient>0 else -1
        controlled_gate(component_probe(atom,provider),r.to_numpy(sign*q))
        for n in (0,2):
            import math
            theta=math.atan(law.dimensionless_step_time/(n+1))
            angle=sign*ctx.atan(ctx.mpf(law.dimensionless_step_time)/(n+1))
            expected=ctx.cos(angle)*ctx.eye(16)-ctx.j*ctx.sin(angle)*q
            controlled_gate(component_probe(atom,provider,angle=theta),r.to_numpy(expected))


def test_support_completion_preserves_columns_and_conjugation():
    import numpy as np
    from trottertracks.pr_dictionary_validation.circuits import support_completion
    from trottertracks.pr_dictionary_validation import reference as r
    spec,_context,data,_alts=prepared(); ctx=r.precision_context(90)
    for _f,v,_s,_p in data:
        full=np.asarray(v,dtype=complex)
        for support in ((0,),(0,2)):
            got=support_completion(full,support)
            assert np.max(np.abs(full[:,support]-got[:,support]))<=1e-10
            assert np.max(np.abs(got.conj().T@got-np.eye(4)))<=1e-10


def test_n0_n2_sign_order_identity_scalar_fusion_and_actual_adjoint():
    from trottertracks.pr_dictionary_validation.cases import motif_events
    from trottertracks.pr_dictionary_validation.circuits import build_evolution,actual_adjoint
    from trottertracks.pr_dictionary_validation.gates import controlled_gate
    from trottertracks.pr_dictionary_validation import reference as r
    for arm in ('SS','SP','PS','PP'):
        spec,context,atoms,tail,_config,law,provider=arm_data(arm)
        ctx=r.precision_context(90)
        for index in range(8):
            macros=motif_events(tail,law,index)
            expected=r.to_numpy(r.trajectory_operator(ctx,spec,context,macros,atoms))
            for optimized in (False,True):
                result=build_evolution(spec,context,macros,atoms,provider,
                    input_identity='unbound_test_input',source_identity='unbound_test_source',optimized=optimized)
                controlled_gate(result.circuit,expected)
                if optimized: controlled_gate(actual_adjoint(result),expected.conj().T)


def test_hadamard_xy_readout_sign_and_state_prep():
    from trottertracks.pr_dictionary_validation.cases import motif_events
    from trottertracks.pr_dictionary_validation.circuits import build_evolution,hadamard_wrapper
    from trottertracks.pr_dictionary_validation.gates import wrapper_signal,signal_gate
    from trottertracks.pr_dictionary_validation import reference as r
    spec,context,atoms,tail,_config,law,provider=arm_data('PS')
    macros=motif_events(tail,law,3)
    evolution=build_evolution(spec,context,macros,atoms,provider,
        input_identity='unbound_test_input',source_identity='unbound_test_source')
    ctx=r.precision_context(90); expected=r.signal(ctx,r.trajectory_operator(ctx,spec,context,macros,atoms),spec.occupied)
    signal_gate(*(wrapper_signal(hadamard_wrapper(evolution,spec,axis),axis) for axis in ('X','Y')),
                complex(expected))


def test_component_derived_corrected_mean_uses_actual_blocks():
    from qiskit.quantum_info import Operator
    import math
    from trottertracks.pr_dictionary_validation.circuits import component_probe
    from trottertracks.pr_dictionary_validation.gates import component_derived_mean,compose_corrected_mean,matrix_gate
    from trottertracks.pr_dictionary_validation import reference as r
    spec,context,atoms,tail,_config,law,provider=arm_data('SP'); blocks={}
    for atom in atoms:
        blocks[(atom.atom_id,'product')]=Operator(component_probe(atom,provider)).data[16:,16:]
        for n in (0,2):
            angle=math.atan(law.dimensionless_step_time/(n+1))
            blocks[(atom.atom_id,'rotation',n)]=Operator(component_probe(atom,provider,angle=angle)).data[16:,16:]
    actual=compose_corrected_mean(component_derived_mean(tail,law,blocks),spec,context)
    ctx=r.precision_context(90)
    matrix_gate(actual,r.to_numpy(r.finite_target(ctx,spec,context)),2e-10,label='FINITE_MEAN')


def test_literal_dp_matches_independent_finite_event_scan():
    from trottertracks.pr_dictionary_validation.pricing import AtomPrice,direct_literal_price,expected_literal_price
    import itertools
    prices={'A':AtomPrice('A','F',False,False,8,4,4,2),
            'B':AtomPrice('B','F',False,True,8,6,4,2),
            'C':AtomPrice('C','L',False,False,4,4,4,2),
            'I':AtomPrice('I',None,True,False,None,None,0,1)}
    probabilities={k:Fraction(1,4) for k in prices}; orders={0:Fraction(2,3),2:Fraction(1,3)}
    expected=Fraction(0)
    for n,pn in orders.items():
        for indices in itertools.product(prices,repeat=n+1):
            apps=[(x,'product') for x in indices[:-1]]+[(indices[-1],'rotation')]
            expected+=pn*Fraction(1,4)**(n+1)*direct_literal_price(apps,prices,event_count=1,scalar_reserves=1)
    mass,actual=expected_literal_price(prices,probabilities,orders,q=1,r=1,fixed_price=0)
    assert mass==1 and actual==expected


def test_unknown_recipe_prices_do_not_become_zero():
    from trottertracks.pr_dictionary_validation.pricing import AtomPrice,UnpricedError
    with pytest.raises(UnpricedError): AtomPrice('A','F',False,False,None,None,4,2).validate()


def test_selector_and_exhaustive_share_information_and_tie_policy():
    from trottertracks.pr_dictionary_validation.pricing import ScoreInterval,select_dictionary,exhaustive_four
    from hashlib import sha256
    provider=sha256(b'test_provider_not_a_scientific_binding').hexdigest()
    values={'SS':9,'SP':7,'PS':8,'PP':10}; cache={}; requests=[]
    def evaluate(arm):
        requests.append(arm)
        if arm not in cache: cache[arm]=ScoreInterval(Fraction(values[arm]),Fraction(values[arm]),provider,True)
        return cache[arm]
    selected=select_dictionary(evaluate); baseline=exhaustive_four(evaluate,cache=cache)
    assert selected['arm']==baseline['arm']=='SP'
    assert len(selected['requests'])<=6 and len(cache)==4
    assert select_dictionary(lambda arm:ScoreInterval(Fraction(1),Fraction(2),provider,True))['arm']=='SS'


def test_selector_unqualified_interval_returns_explicit_fallback():
    from trottertracks.pr_dictionary_validation.pricing import ScoreInterval,select_dictionary
    out=select_dictionary(lambda arm:ScoreInterval(Fraction(0),Fraction(1),'unbound',False))
    assert out['arm']=='SS' and out['status']=='UNPRICED_FALLBACK'


def test_two_independent_precision_paths_are_registered_without_retry():
    from trottertracks.pr_dictionary_validation import reference as r
    from trottertracks.pr_dictionary_validation.gates import high_precision_agreement
    spec,context,_data,_alts=prepared()
    high_precision_agreement(r.finite_target(r.precision_context(90),spec,context),
                             r.finite_target(r.precision_context(128),spec,context))


def test_compiled_full_wrapper_retains_operator_and_signal():
    from qiskit.quantum_info import Operator
    from trottertracks.pr_dictionary_validation.cases import motif_events
    from trottertracks.pr_dictionary_validation.circuits import build_evolution,hadamard_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.gates import compiled_wrapper,matrix_gate
    from trottertracks.pr_dictionary_validation import reference as r
    spec,context,atoms,tail,_config,law,provider=arm_data('PP'); macros=motif_events(tail,law,2)
    evolution=build_evolution(spec,context,macros,atoms,provider,
        input_identity='unbound_test_input',source_identity='unbound_test_source')
    ctx=r.precision_context(90); u=r.trajectory_operator(ctx,spec,context,macros,atoms)
    for axis in ('X','Y'):
        got=compiled_wrapper(hadamard_wrapper(evolution,spec,axis),CompilerProfile())
        got=got.remove_final_measurements(inplace=False)
        matrix_gate(Operator(got).data,r.to_numpy(r.wrapper_operator(ctx,u,spec.occupied,axis)),
                    5e-10,label='COMPILED')


def test_f1_authorization_rejects_p0_or_i_before_scientific_import(tmp_path):
    import json
    from trottertracks.pr_dictionary_validation.authorization import verify_packet
    (tmp_path/'authorization.json').write_text(json.dumps(dict(stage='I',authority='USER',status='DRAFT')))
    with pytest.raises(PermissionError): verify_packet(tmp_path,tmp_path,tmp_path/'fresh_output')
