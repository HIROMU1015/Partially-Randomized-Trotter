"""Two saved CDF arms through the existing PR/reference/native implementations.

Passive import. Molecular arrays, operators, laws and circuits may only enter
these functions after separate scientific permission. Tests use invented arrays.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace

from . import c1_general_k_adapter as general
from . import c1_stage1 as stage1
from . import c1_stage2_native as native
from . import c1_stage3_connection as connection
from . import c1_stage3_reference as reference
from . import c_pr_aware_df as objective
from .c1_stage3_molecular import component_check

VERSION='paired_CDF_original_prefix_PR_v1'
NATIVE_POLICY='common_precision_spin_kappa_relative_head_full_frame_tail_v1'
ARMS=('baseline','proposed')
ATOL=1e-10


@dataclass(frozen=True)
class Arm:
    role: str
    model: object
    general: object
    plan: object
    J: object
    U: object


def prepare_arm(h,g,constant,J,U,*,role,source_identity,input_identity,ld):
    """No XDF conversion; original-g kappa and one shared spin-basis policy."""
    import numpy as np
    from .c1_h4_strong_baseline import spin_adapted_eigh
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives
    if role not in ARMS:raise ValueError('EXPLICIT_PAIRED_CDF_ROLE')
    kappa=stage1.corrected_kappa(h,g)
    model=general.from_ffsim_number_arrays(one_body_tensor=kappa,
        diag_coulomb_mats=J,orbital_rotations=U,constant=float(constant),
        z_representation=False,frame_ids=[f'F{i}' for i in range(len(J))],
        group_ids=[f'G{i}' for i in range(len(J))],source_identity=source_identity,
        input_identity=input_identity,one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    u,eps,_=spin_adapted_eigh(model.one_body_tensor)
    blocks=[native.NativeBlock('one_body',native._rotation(u),one_body_diagonal_primitives(eps,1.))]
    blocks.extend(native.NativeBlock(f.frame_id,f.orbital_rotation,general.diagonal_primitives(f,1.)) for f in model.frames)
    nm=native.NativeModel('CDF_'+role,tuple(blocks),float(constant),source_identity,input_identity)
    return Arm(role,nm,model,connection.split_native_model(nm,ld),np.asarray(J),np.asarray(U))


def candidate_factor(ctx,arm,index):
    import numpy as np
    view=SimpleNamespace(spatial_one_body_tensor=arm.general.spatial_one_body_tensor if index==0 else np.zeros_like(arm.J[0],dtype=complex),
        constant=arm.model.constant if index==0 else 0.,frames=() if index==0 else (arm.general.frames[index-1],))
    return ctx.candidate(view)


def atom_operators(ctx,arm,*,arithmetic=None):
    """Independent signed rotated-Z products from raw sector CAR bilinears."""
    import numpy as np
    identity=np.eye(ctx.d,dtype=complex) if arithmetic is None else arithmetic.eye(ctx.d)
    densities={};out={}
    for atom in arm.plan.atoms:
        if atom.frame_id not in densities:
            V=(np.asarray(arm.plan.frames[atom.frame_id]) if arithmetic is None
               else arithmetic.array(arm.plan.frames[atom.frame_id]));zs=[]
            for a in range(arm.model.modes):
                d=sum((V[p,a]*V[q,a].conjugate()*E for (p,q),E in ctx.bilinears.items())).toarray()
                zs.append(identity-2*d)
            densities[atom.frame_id]=zs
        p=identity
        for a in atom.support:p=p@densities[atom.frame_id][a]
        out[atom.atom_id]=(1 if atom.coefficient>0 else -1)*p
    return out


def configure(arm,h,r,K):
    """Regenerate probabilities/normalization for this immutable arm only."""
    return connection.configure_rte(arm.plan,h,r,K)


def tail_gate(ctx,arm,generators,tail):
    import numpy as np
    operators=atom_operators(ctx,arm);identity=np.eye(ctx.d,dtype=complex)
    total=arm.plan.extracted_identity*identity
    for atom in arm.plan.atoms:
        p=operators[atom.atom_id]
        component_check(p,p.conj().T,ATOL,label='signed atom Hermitian')
        component_check(p@p,identity,ATOL,label='signed atom involution')
        total=total+abs(float(atom.coefficient))*p
    raw=sum(generators[arm.plan.ld+1:],np.zeros_like(identity))
    report=component_check(total,raw,ATOL,label='original CAR tail vs law atoms + identity')
    # Native primitive and JAX objective conventions are independently matched.
    mass=float(np.sum(abs(objective.coefficients(arm.J[arm.plan.ld:],np))))
    if not math.isclose(mass,tail.lambda_r,rel_tol=1e-12,abs_tol=1e-14):raise ValueError('STAGE_A_LAMBDA_DEFINITION')
    return dict(status='PASS',representation=report,lambda_R=tail.lambda_r,
        objective_exact_mass=mass,identity=arm.plan.extracted_identity,
        atoms=len(arm.plan.atoms),probability_sum=math.fsum(c.probability for c in tail.components),
        ideal_law='finite_RTE_IID_reference',implemented_finite_bit_law_error=None)


def kernels(arm,generators,h,config):
    # Each invocation creates an arm-local exponential cache; no CDF/XDF alias.
    return reference.reference_kernels(generators,h,
        ld=arm.plan.ld if config is not None else len(generators)-1,
        identity_tail=arm.plan.extracted_identity,rte_config=config,exponential_cache={})


def independent_occurrence(arm,tail,distribution,config,*,master,round_index,axis,shot,occurrence):
    if config is None:return ()
    seed=connection.trajectory_seed(master,arm.role,arm.model.source_identity,round_index,axis,shot,occurrence)
    return connection.sample_occurrence(tail,distribution,config.rte_steps,seed)


def build_kernel(arm,h,events,config,*,builder=None):
    """Common precise bases; existing relative fusion inside each head half.

Tail uses full-frame bases from the same precision builder. Cross-head/tail
relative fusion is deliberately absent, so no unobserved saving is assumed.
The corrected existing signed-event bridge retains every phase/actual order.
"""
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_stage3_native_bridge import partial_kernel
    from .pr_dictionary_validation.circuits import BasisProvider
    builder=builder or StrongBuilder(relative_fusion=True)
    if config is None:return builder.ordered(arm.model,h,optimized=True)
    class Provider(BasisProvider):
        def basis(self,atom,singleton=False):
            for block in arm.model.blocks:
                if block.rotation==arm.plan.frames[atom.frame_id]:return builder._basis(arm.model,block).copy()
            raise ValueError('UNKNOWN_ARM_BASIS')
    qc=partial_kernel(arm.plan,h,events,rte_config=config,builder=builder,
        provider=Provider(arm.model.modes,arm.plan.frames))
    qc.metadata=dict(native_policy=NATIVE_POLICY,arm=arm.role,source=arm.model.source_identity,
        relative_head=True,cross_head_tail_fusion=False,state_preparation_included=False)
    return qc


def trajectory_reference(arm,generators,operators,h,events,config):
    """Reuse CAR exponentials and existing event_unitary, not Gaussian gates."""
    import numpy as np
    if config is None:return kernels(arm,generators,h,None)['pf']
    from trotterlib.rte import event_unitary
    half=tuple((i,Fraction(h)/2) for i in range(arm.plan.ld+1))
    left=reference._ordered_exponentials(generators,half)
    right=reference._ordered_exponentials(generators,tuple(reversed(half)))
    middle=np.eye(len(generators[0]),dtype=complex)
    for event in events:middle=event_unitary(event,operators)@middle
    return np.exp(-1j*float(h)*arm.plan.extracted_identity)*right@middle@left


def assess_native_probe(raw,compiled,system_reference):
    """Full controlled numerical check for one trajectory, not the entire law."""
    import numpy as np
    from qiskit.quantum_info import Operator
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    if raw.num_clbits or compiled.num_clbits or any(x.operation.name not in {'rz','sx','x','cx','barrier'} for x in compiled.data):
        raise ValueError('UNITARY_P0_KERNEL_REQUIRED')
    ref=np.asarray(system_reference);d=len(ref);zero=np.zeros_like(ref)
    target=np.block([[np.eye(d),zero],[zero,ref]])
    R=Operator(raw).data;C=Operator(compiled).data
    reports=dict(raw=component_check(R,target,ATOL,label='raw full controlled/reference'),
        compiled=component_check(C,target,ATOL,label='compiled full controlled/reference'),
        compiled_raw=component_check(C,R,ATOL,label='compiled/raw'),
        adjoint=component_check(Operator(native.actual_adjoint(raw)).data,target.conj().T,ATOL,label='actual adjoint'),
        unitarity=component_check(C.conj().T@C,np.eye(2*d),ATOL,label='compiled unitarity'))
    return dict(status='SINGLE_TRAJECTORY_FULL_OPERATOR_NUMERICAL_PASS',diagnostics=reports,
        metrics=circuit_metrics(compiled),global_phase_fit=False,whole_law_synthesis_bound=None,
        rigorous_enclosure=None,cost_support=None)


def cost_point(samples,*,scope,independent_trajectories):
    """Finite-bit calibration diagnostics, never an ideal mean/support bound."""
    from .c1_stage3_accuracy import METRICS
    if scope!='complete_controlled_partial_S2_kernel' or not independent_trajectories or not samples:
        raise ValueError('COMPLETE_KERNEL_INDEPENDENT_CALIBRATION_REQUIRED')
    for row in samples:
        if not set(METRICS)<=row.keys() or any(type(row[k])is not int or row[k]<0 for k in METRICS):raise ValueError('NATIVE_METRICS_SCHEMA')
    n=len(samples);mean={k:math.fsum(row[k] for row in samples)/n for k in METRICS}
    se=None if n<2 else {k:math.sqrt(math.fsum((row[k]-mean[k])**2 for row in samples)/(n*(n-1))) for k in METRICS}
    return dict(status='FINITE_BIT_POINT_ESTIMATE_ONLY',samples=n,sample_mean=mean,sample_mean_standard_error=se,
        ideal_law_expectation=None,cost_support=None,lower=None,upper=None,
        simultaneous_interval_qualified=False,guaranteed_resource_superiority=None)
