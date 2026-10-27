"""Invented families only. Never read molecular arrays or saved QPY."""
import math
from fractions import Fraction

from . import c_pr_aware_df_zsxx_uniform as u


def evidence():
    from flint import ctx,acb
    from qiskit import QuantumCircuit
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f())
    cuts=[('multi_branch','0','4'),('near_zero','-1/1000000000000','1/1000000000000'),
          ('halfpi',str(u.PI/2-u.EPS),str(u.PI/2+u.EPS)),
          ('pi',str(u.PI-u.EPS),str(u.PI+u.EPS))]
    phase_examples=[]
    for value in [-math.ulp(0.),0.,2*math.pi,2*math.pi+math.ulp(2*math.pi),1e5,-1e5]:
        def setter():
            q=QuantumCircuit(1);q.global_phase=value
            return float(q.global_phase)
        returned=lock('kernel_build','artificial phase setter',setter)
        with u.z._BALL_LOCK,ctx.workprec(u.z.PRECISION):
            gap=(acb(0,1)*u.z._real(value)).exp()-(acb(0,1)*u.z._real(returned)).exp()
            distance=u.z._norm(u.z._scale(u.z.gate_matrix('id'),gap))
        phase_examples.append(dict(input_hex=value.hex(),returned_hex=returned.hex(),
            distance=distance,status='VALIDATED_CAPTURED_PHASE_SETTER_ONLY',uniform_remainder_accuracy_verified=False))
    structural=u.grammar_node_bound(official_topology_coverage=True)
    return dict(status='ARTIFICIAL_DOMAIN_AND_CONDITIONAL_BOUNDS_COMPLETE',
        precision_bits=u.z.PRECISION,structural_node_bound=structural,
        run_models=[u.run_bound(L,node_matrix_error='0',model=u.ieee_model()) for L in [1,4,16,256,1024,structural['initial_nodes_upper']]],
        run_model_domain='hypothetical exact-node IEEE input model;includes SX/X-only words;not all RZ evaluations or actual H4',
        unknown_actual_run_model=u.run_bound(structural['initial_nodes_upper'],model=u.ieee_model()),
        theta_branch_supersets=[dict(label=label,result=u.theta_branch_cover(lo,hi,model=u.ieee_model())) for label,lo,hi in cuts],
        ideal_template_image=u.ideal_template_image([('rz',('-1','1')),('sx',None),('ry',('0','4'))],phase_interval=('-1/5','1/5')),
        template_image_is_Rust_image=False,
        translation_rules=[u.translation_rule(k) for k in ['CRZ_definition','RZZ_definition','RZ_inverse']],
        phase_setter_captures=phase_examples,
        true_period_gap_exact=str(u.period_gap()),
        conditional_phase_model=u.conditional_phase_model(),
        conditional_phase_bound=u.phase_bound(1000,'100',remainder_model=u.conditional_phase_model()),
        unknown_actual_whole_P0=u.uniform_whole_bound(structural['initial_nodes_upper'],
            conditions=u.CONDITIONS,exact_unitary_targets=True),
        classification=dict(A='ideal-template interval images and analytic component rules under stated IEEE conditions',
            B='uniform run/branch/phase/whole-P0 formulas with explicit unverified model/proof inputs',
            C='actual Rust output remains captured-pair only',D='useful all-reachable Rust image is unresolved'),
        whole_law_bound=None,actual_H4_bound=None,new_H4_scientific_calls=0)
