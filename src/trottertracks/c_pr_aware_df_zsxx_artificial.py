"""Fixed invented2x2 cases. No molecular paths or saved-array inputs."""
import math
from fractions import Fraction

from . import c_pr_aware_df_zsxx_enclosure as v


def angle_cases():
    eps=1e-12;ulp=math.ulp(math.pi)
    rows=[('theta_identity',0.,0.,0.,0.),('theta_pi',math.pi,.3,.2,.1),
          ('generic',.7,.3,-.7,.2),('signed_zero',-0.,-0.,0.,-0.),
          ('phi_flip_exact',.7,0.,.3,.2),
          ('mod_endpoint_clamp',0.,0.,math.pi-5e-13,0.)]
    for delta,tag in ((-ulp,'below'),(0.,'center'),(ulp,'above')):
        rows.extend([('theta_zero_cut_'+tag,eps+delta,.3,-.7,.2),
            ('theta_halfpi_cut_'+tag,math.pi/2+eps+delta,.3,-.7,.2),
            ('theta_pi_cut_'+tag,math.pi-eps+delta,.3,.2,.1),
            ('lambda_plus_pi_cut_'+tag,.7,.3,-math.pi+eps+delta,.2),
            ('phi_zero_cut_'+tag,.7,eps+delta,.3,.2),
            ('RZ_drop_'+tag,0.,-.13+eps+delta,.13,0.),
            ('mod_endpoint_'+tag,0.,0.,math.pi-eps+delta,0.)])
    rows.append(('tiny_subnormal_angle',math.ulp(0.),.3,-.7,.2))
    return rows


def matrix_cases():
    import numpy as np
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
    SX=np.array([[1+1j,1-1j],[1-1j,1+1j]],complex)/2
    signed=I.copy();signed[0,1]=complex(-0.,-0.)
    subnormal=I.copy();subnormal[1,0]=complex(math.ulp(0.),-math.ulp(0.))
    return [('identity',I),('PauliX',X),('PauliZ',np.diag([1,-1]).astype(complex)),
            ('SX',SX),('phase_i_identity',1j*I),('signed_zero_matrix',signed),
            ('subnormal_entry',subnormal),('near_unitary',np.diag([1+2**-45,1-2**-45]).astype(complex)),
            ('element_perturbation',SX+np.array([[2**-48,0],[0,-1j*2**-48]]))]


def evidence():
    from copy import deepcopy
    from flint import ctx
    rows=[]
    for label,t,p,l,b in angle_cases():
        rows.append(dict(label=label,kind='angle_emitter',enclosure=v.captured_angles(t,p,l,b)))
    for label,M in matrix_cases():
        rows.append(dict(label=label,kind='matrix_to_ZSXX',enclosure=v.captured_matrix(M)))
    # Non-singleton reference regions;the returned compiler output is held fixed.
    import numpy as np
    rows.append(dict(label='matrix_reference_box',kind='held_output_reference_region',
        enclosure=v.captured_matrix(np.eye(2,dtype=complex),reference_radius='1/1000000000000000')))
    rows.append(dict(label='theta_reference_interval',kind='held_output_reference_region',
        enclosure=v.captured_angles(.7,.3,-.7,.2,theta_reference_radius='1/1000000000000000')))
    for label,gates in [('near_cancel_run',[('rz',.31),('rz',-.31)]*512),
                        ('long_run',[('rz',.13),('sx',None),('rz',-.09)]*256)]:
        rows.append(dict(label=label,kind='artificial_run',enclosure=v.captured_run(gates)))
    pair=v.captured_matrix(1j*np.eye(2));bad=deepcopy(pair['output']);bad['phase_hex']=(0.).hex()
    bad_distance=v.verify_fixed_record(1j*np.eye(2),bad)
    # Actual whole-P0,including initial translation,for two artificial1q inputs.
    from qiskit import QuantumCircuit
    from .c_pr_aware_df_c2a import trace_artificial
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from . import c1_stage2_native as native
    mutex=SharedQiskitCalls(lambda k,l,fn:fn())
    whole=[]
    for label,gs,phase in [('translation_phase',[('ry',.31),('p',-.17),('sx',None),('rz',.23)],-.19),
                           ('cross_boundary_run',[('rz',.31),('rz',-.11),('ry',.27)],.13)]:
        def build():
            q=QuantumCircuit(1);q.global_phase=phase
            for name,angle in gs:
                if angle is None:getattr(q,name)(0)
                else:getattr(q,name)(angle,0)
            return q
        q=mutex('kernel_build','artificial:'+label,build)
        compiled,trace=trace_artificial(q)
        enclosure=v.captured_circuit(gs,compiled,input_phase=phase)
        with mutex.adjoint_builds():adj=native.actual_adjoint(compiled)
        whole.append(dict(label=label,enclosure=enclosure,adjoint=v.captured_inverse(enclosure['output'],adj),
                          passes=[r['pass_name'] for r in trace['passes']]))
    return dict(status='ARTIFICIAL_CAPTURED_PAIR_ENCLOSURES_COMPLETE',cases=rows,
        captured_whole_P0=whole,
        counterexample=dict(status='CORRECTNESS_FAIL',intentional=True,
            defect='captured phase deliberately deleted',distance=bad_distance),
        domain_qualification='fixed captured pairs andheld-output reference regions only',
        validated_cases=len(rows),all_branch_domain_guaranteed=False,
        global_libm_error=None,whole_law_bound=None,actual_H4_bound=None,
        new_H4_scientific_calls=0,precision_bits=v.PRECISION)
