"""Linear-Z relocation glue. Imports are passive; no saved arrays are loaded.

Only invented arrays are used during preparation. Scientific callers require
separate user permission. The original number model remains provenance only;
the relocated Arm has no general-K number view (its tail is now I+ZZ).
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import math

VERSION = 'tail_linear_Z_to_existing_kappa_keep_frame_identity_v1'
R_VALUES = (8, 16, 32, 64, 128)
TIE_ORDER = (32, 16, 64, 8, 128)


@dataclass(frozen=True)
class Relocation:
    original: object
    arm: object
    kappa: object
    delta_kappa: object
    scalar_shift: float
    rows: object


def prepare(kappa, constant, J, U, *, role, source_identity, input_identity, ld):
    """Accept EXISTING corrected spatial kappa; never accept h/g or refit J/U.

    Keep each frame's original identity. Move sum(s*n)-S*I to the
    one-body/scalar head; remove only RZ primitives from original tail frames.
    D must reuse this model, not relocate again with a different LD.
    """
    import numpy as np
    from . import c_pr_aware_df_stage_b as b
    from . import c1_general_k_adapter as general
    from .c1_h4_strong_baseline import spin_adapted_eigh
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives
    kappa, J, U = (np.array(x, copy=True) for x in (kappa, J, U))
    if (role not in b.ARMS or kappa.dtype != np.float64 or J.dtype != np.float64
            or U.dtype != np.float64 or J.ndim != 3 or U.shape != J.shape
            or J.shape[1:] != kappa.shape or kappa.shape != (J.shape[1], J.shape[1])
            or type(ld) is not int or not 0 <= ld < len(J)
            or not all(np.isfinite(x).all() for x in (kappa,J,U))
            or not math.isfinite(float(constant))
            or not np.array_equal(J,J.transpose(0,2,1))
            or np.max(abs(kappa-kappa.T)) > 1e-10
            or np.max(abs(U.transpose(0,2,1)@U-np.eye(len(kappa)))) > 1e-10):
        raise ValueError('FINITE_NUMBER_J_U_KAPPA_ORIGINAL_PREFIX')
    model = general.from_ffsim_number_arrays(one_body_tensor=kappa,
        diag_coulomb_mats=J, orbital_rotations=U, constant=constant,
        z_representation=False, frame_ids=[f'F{i}' for i in range(len(J))],
        group_ids=[f'G{i}' for i in range(len(J))], source_identity=source_identity,
        input_identity=input_identity, one_body_role='ffsim_corrected_kappa', enable_general_k=True)
    def head(one):
        v, eps, _ = spin_adapted_eigh(np.kron(one,np.eye(2)))
        return b.native.NativeBlock('one_body',b.native._rotation(v),one_body_diagonal_primitives(eps,1.))
    blocks = [head(kappa)]+[b.native.NativeBlock(f.frame_id,f.orbital_rotation,
        general.diagonal_primitives(f,1.)) for f in model.frames]
    before = b.native.NativeModel('CDF_'+role,tuple(blocks),float(constant),source_identity,input_identity)
    original = b.Arm(role,before,model,b.connection.split_native_model(before,ld),J,U)
    rows = J.sum(axis=2)
    delta = sum((U[t]@np.diag(rows[t])@U[t].T for t in range(ld,len(J))),np.zeros_like(kappa))
    shift = -math.fsum(float(x) for x in rows[ld:].flat)
    blocks[0] = head(kappa+delta)
    for t in range(ld,len(J)):
        block=blocks[t+1]
        blocks[t+1]=replace(block,diagonal=replace(block.diagonal,rz=()))
    identity=f'{source_identity}:{VERSION}:LD{ld}'
    after = replace(before,blocks=tuple(blocks),constant=float(constant)+shift,source_identity=identity)
    # Fail closed if an old number-frame helper is accidentally used downstream.
    arm=b.Arm(role,after,None,b.connection.split_native_model(after,ld),J,U)
    for x in (kappa,J,U,delta,rows):x.setflags(write=False)
    return Relocation(original,arm,kappa,delta,shift,rows)


def raw_generators(rel, *, ctx=None, car=None):
    """Independent original number-CAR factors minus L, including constants.

    SectorCAR and full-Fock raw_frame are the existing implementations. This
    route never derives its reference from emitted native Z/ZZ primitives.
    """
    import numpy as np
    from . import c1_stage1 as stage
    if (ctx is None)==(car is None):raise ValueError('ONE_CAR_CONTEXT_REQUIRED')
    if ctx is not None:
        eye=np.eye(ctx.d,dtype=complex)
        one=ctx.one
        frame=ctx.raw_frame
    else:
        eye=np.eye(car.shape[-1],dtype=complex)
        one=lambda k:stage.raw_base(k,0.,car)
        frame=lambda j,u:stage.raw_frame(j,u,car)[0]
    original=[one(rel.kappa)+rel.original.model.constant*eye]
    original.extend(frame(j,u) for j,u in zip(rel.arm.J,rel.arm.U,strict=True))
    after=[one(rel.kappa+rel.delta_kappa)+rel.arm.model.constant*eye]
    for t,raw in enumerate(original[1:]):
        if t>=rel.arm.plan.ld:
            linear=one(rel.arm.U[t]@np.diag(rel.rows[t])@rel.arm.U[t].T)-float(rel.rows[t].sum())*eye
            raw=raw-linear
        after.append(raw)
    from .c1_stage3_molecular import component_check
    component_check(sum(original),sum(after),1e-10,label='relocation full Hamiltonian including scalar')
    return original,after


def tail_gate(ctx, arm, generators, tail):
    """Reuse the signed-law/CAR equality gate; use the relocated mass formula."""
    import numpy as np
    from . import c_pr_aware_df_stage_b as b, c_pr_aware_df_stage_b2 as b2
    from .c_h4_z_relocation import diagnose
    if arm.general is not None or any(len(a.support)!=2 for a in arm.plan.atoms):
        raise ValueError('RELOCATED_ZZ_ONLY_TAIL_REQUIRED')
    ops=b.atom_operators(ctx,arm)
    identity=np.eye(ctx.d,dtype=complex)
    for p in ops.values():
        b.component_check(p,p.conj().T,1e-10,label='relocated signed atom Hermitian')
        b.component_check(p@p,identity,1e-10,label='relocated signed atom involution')
    saved=dict(lambda_R=tail.lambda_r,identity=arm.plan.extracted_identity,
        atoms=len(arm.plan.atoms),components=[dict(probability=x.probability) for x in tail.components])
    check=b2.law_gate(arm,tail,dict(left=np.eye(ctx.d),atoms=ops,generators=generators),saved)
    mass=diagnose(arm.J,ld=arm.plan.ld)['M_ZZ']
    if not math.isclose(mass,tail.lambda_r,rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError('RELOCATED_MASS_MISMATCH')
    return dict(status='PASS',representation=check,lambda_R=mass,identity=arm.plan.extracted_identity,
        atoms=len(arm.plan.atoms),version=VERSION)


def reference_r(a,r,out,call,*,pf_cache=None):
    from .c_integrated_retuning_adapter import reference_r as reused
    return reused(a,r,out,call,pf_cache=pf_cache,r_values=R_VALUES,tail_gate=tail_gate)


def native_context(a, car, full_generators, shells, *, call):
    """Explicit prepared context prevents the old native runner rebuilding h/g.

    All nine full/sector factors and the new law are checked before sampling.
    Existing native() then performs all full controlled matrices per sample.
    """
    import numpy as np
    from . import c_pr_aware_df_stage_b2 as b2
    from .c_pr_aware_df_m2 import sector_match
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request
    arm=a['arm']
    ix,report=call('full_sector_gate','relocated all factors',lambda:sector_match(full_generators,a['sector']))
    psi=np.zeros(len(full_generators[0]),complex);psi[ix]=a['ground']
    ctx=call('reference_context','relocated CAR/head',lambda:b2.reference_context(arm,full_generators,car,Fraction(1,10)))
    saved=dict(lambda_R=a['tail'].lambda_r,identity=arm.plan.extracted_identity,
        atoms=len(arm.plan.atoms),components=[dict(probability=x.probability) for x in a['tail'].components])
    call('full_law_gate','relocated law',lambda:b2.law_gate(arm,a['tail'],ctx,saved))
    builder=StrongBuilder(relative_fusion=True,call=call)
    for block in arm.model.blocks:
        call('gaussian_build',block.block_id,lambda block=block:basis_request(builder,arm.model,block))
    for order in (tuple(range(arm.plan.ld+1)),tuple(reversed(range(arm.plan.ld+1)))):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    return dict(arm=arm,ctx=ctx,psi=psi,builder=builder,shells=shells,sector_report=report)


def strong_d(root,contract,out,call,a,full_generators):
    """Reuse the SAME relocated model and the entire registered D evaluator."""
    from .c_update064_strong_d import evaluate
    from .c1_stage3_connection import split_native_model
    from .c_h4_relocated_handoff import restore_d
    restored=restore_d(contract,out,a)
    if restored is not None:return restored
    if a.get('restored_ground_only'):raise ValueError('D_REQUIRES_FULL_EIGENPAIR_OR_COMPLETE_SAVED_D')
    arm=a['arm'];d=replace(arm,plan=split_native_model(arm.model,len(arm.model.blocks)-1))
    return evaluate(root,contract,out,call,a,prepared_arm=d,full_generators=full_generators)


def select_r(rows,candidate):
    from .c_integrated_retuning_policy import select_r as reused
    return reused(rows,candidate,r_values=R_VALUES,tie_order=TIE_ORDER)


def cohort_key(candidate):
    return f'relocated:{VERSION}:{candidate}'


def seed(candidate,split,r,batch,sample,*,source_digest='synthetic-source',role='proposed'):
    from .c1_stage3_connection import trajectory_seed
    if split not in ('selection','final') or r not in R_VALUES or batch not in (0,1) or not 0<=sample<32:
        raise ValueError('COHORT_DOMAIN')
    cohort=f'integrated:{cohort_key(candidate)}:{split}:r{r}:batch{batch:02d}'
    return trajectory_seed((2026110501,cohort,source_digest),role,candidate,-1,
        f'INTEGRATED_{split.upper()}_PRICE_ONLY',sample,0)


def logical_work(final_pairs=5):
    """Work estimates only; no molecular numbers or statistical estimates."""
    if type(final_pairs) is not int or not 0<=final_pairs<=5:raise ValueError('FINAL_PAIR_BOUND')
    n=3*len(R_VALUES)*32+final_pairs*64
    return dict(candidates=3,PR_policies=15,D_policies=3,
        selection_native_max=480,final_native_max=final_pairs*64,
        native_PR_max=n,D_native_max=3,full_controlled_decodes_max=4*(n+3),
        PR_reference_rounds=3*5*13,D_reference_rounds=3*13,
        PR_PF_kernels=3,D_PF_kernels=3,exact_evolutions=3*13,
        RTE_kernels=15,round_lane_assessments=15*13*3+3*13*2)


def prepare_candidate(kappa,constant,J,U,saved_sector,physical,*,candidate,input_identity,
                      role,out,call,shell_reports,saved_ground=None):
    """Connect already byte-bound saved data to the existing reference A packet.

    No original h/g correction, DF norm recomputation, fit or optimization.
    Scientific input loading/identity checks must precede this call.
    """
    import numpy as np
    from . import c1_molecule_extension as molecule
    from . import c_pr_aware_df_stage_b_runner as loader
    from .c1_stage3_accuracy import AccuracyTarget
    from .c_pr_aware_df_m4 import TARGET
    from .c1_stage3_molecular import component_check
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    df=physical
    if (df['candidate_sha256']!=candidate or df['input_sha256']!=input_identity
            or not df['budget_PASS'] or df['actual_rank']!=8
            or df['budget_Ha']!=.00019875 or df['kappa']!='original_g_corrected'
            or df['scalar']!='nuclear_once'
            or df['guard_policy']!='sector_operation_scale_guard8_proposal_v1'
            or min(df['sector_error'],df['numerical_guard'])<0
            or not math.isfinite(df['sector_error']+df['numerical_guard'])
            or df['sector_error']+df['numerical_guard']>.00019875):
        raise ValueError('SAVED_DF_INPUT_IDENTITY_AND_BUDGET')
    if J.shape!=(8,4,4) or U.shape!=J.shape:raise ValueError('H4_CDF8_ONLY')
    if not all(np.array_equal(x,saved_sector[k]) for k,x in (('J',J),('U',U))):
        raise ValueError('SAVED_JU_REFERENCE_IDENTITY')
    rel=call('relocated_model',candidate,lambda:prepare(kappa,constant,J,U,role=role,
        source_identity=candidate,input_identity=input_identity,ld=2))
    ctx=call('sector_context',candidate,lambda:molecule.SectorCAR(4))
    before,G=call('relocated_sector',candidate,lambda:raw_generators(rel,ctx=ctx))
    for i,(x,y) in enumerate(zip(before,saved_sector['generators'],strict=True)):
        call('original_factor_gate',str(i),lambda x=x,y=y:component_check(x,y,1e-10,label='saved original factor'))
    call('Hamiltonian_gate',candidate,lambda:component_check(sum(G),saved_sector['H'],1e-10,label='same saved H after relocation'))
    if saved_ground is None or not Path(saved_ground).exists():
        pair=call('candidate_ground',candidate,lambda:loader.ground(saved_sector['H'],out,'candidate'))
    else:
        from .c_h4_relocated_handoff import copy_file
        saved=call('ground_restore',candidate,lambda:loader.load(saved_ground,('energy','sector_state')))
        pair=(np.asarray([saved['energy'].item()]),saved['sector_state'][:,None])
        copy_file(Path(saved_ground),out/'ground_candidate.npz')
    try:report=call('ground_gate',candidate,lambda:loader.ground_gate(saved_sector['H'],pair))
    except ValueError as exc:
        if str(exc)!='UNRESOLVED_GROUND_ALIAS':raise
        return rel,None,dict(status='UNRESOLVED_GROUND_ALIAS',eligible=False)
    sector=dict(generators=np.asarray(G),H=saved_sector['H'],
        full_input_indices=molecule.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices])
    return rel,dict(candidate_identity=rel.arm.model.source_identity,candidate_label=candidate,
        arm=rel.arm,ctx=ctx,sector=sector,ground=pair[1][:,0],pair=pair,d=dict(J=J,U=U),p={},
        target=AccuracyTarget(**TARGET),saved=dict(physical=df,shells=shell_reports),
        restored_ground_only=saved_ground is not None and Path(saved_ground).exists()),report


def final_plan(selection,deterministic):
    """Freeze common/tuned PR tasks and PR-or-D policies using selection only.

    Five distinct PR pairs at most. Overlaps share one final cohort and are
    explicitly dependent evidence. All candidates keep their identities.
    """
    if set(selection)!={'P','E1','E2'} or set(deterministic)!={'P','E1','E2'}:
        raise ValueError('THREE_FIXED_CANDIDATES')
    chosen={};pairs=set();policy={}
    for label in ('P','E1','E2'):
        rows=selection[label]
        candidate=rows[0]['candidate_sha256']
        pick=select_r(rows,candidate)
        chosen[label]=pick['selected_r']
        if pick['selected_r'] is not None:pairs.add((label,pick['selected_r']))
        if label in ('P','E1'):
            fixed=next(x for x in rows if x['r']==32)
            if fixed['eligible']:pairs.add((label,32))
        d=deterministic[label]
        if d['candidate_sha256']!=candidate:raise ValueError('SAME_JU_D_IDENTITY')
        dv=d['total_rz'] if d['eligible'] else None
        if dv is not None and (isinstance(dv,bool) or not math.isfinite(dv) or dv<=0):
            raise ValueError('D_ACCOUNTED_TOTAL')
        pv=pick.get('selection_total_rz')
        policy[label]=('D' if dv is not None and (pv is None or dv<=pv) else
            'PR' if pv is not None else None)
    return dict(tuned_r=chosen,PR_final_pairs=sorted(pairs),best_execution=policy,
        selected_using='selection_only',overlap_is_independent_replication=False)


def native_cohort(root,contract,out,call,a,result,packet,*,split,prepared):
    """Only the new CAR context may be supplied to the reused native32/64 path."""
    from .c_integrated_retuning_adapter import native_cohort as reused
    if not prepared or prepared['arm'] is not a['arm'] or a['arm'].general is not None:
        raise ValueError('RELOCATED_NATIVE_CONTEXT_REQUIRED')
    return reused(root,contract,out,call,a,result,packet,split=split,
        cohort_key=cohort_key(a['arm'].model.source_identity),prepared=prepared)


def candidate_selection(root,contract,out,call,rel,a,*,car,shells):
    """All five fixed r opportunities; D is independent of PR eligibility."""
    from . import c1_stage1_execution as io
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _,G=call('relocated_full_CAR',a['candidate_label'],lambda:raw_generators(rel,car=car))
    dpath=out/'D';dpath.mkdir()
    D=strong_d(root,contract,dpath,call,a,G)
    states={};rows=[];cache=None;prepared=None
    for r in (32,8,16,64,128):
        from .c_h4_relocated_handoff import restore_reference
        restored=restore_reference(contract,a,r,out/f'r{r}'/'reference',call,cache)
        if restored is None:
            ar,result,packet,cache=reference_r(a,r,out/f'r{r}'/'reference',call,pf_cache=cache)
        else:ar,result,packet,cache=restored
        io.save(out/f'r{r}'/'reference_result.json',result)
        value=None
        if result['native_admitted']:
            if prepared is None:prepared=native_context(ar,car,G,shells,call=call)
            result,packet=native_cohort(root,contract,out/f'r{r}'/'selection',call,
                ar,result,packet,split='selection',prepared=prepared)
            value=result['lanes']['ideal_exact_native']['conditional_totals']['totals']['rz_count']
        states[r]=dict(a=ar,result=result,packet=packet)
        rows.append(dict(candidate_sha256=a['candidate_identity'],r=r,split='selection',
            samples=32,actual_samples=32 if value is not None else 0,correctness='PASS',
            eligible=value is not None,total_rz=value))
    io.save(out/'selection.json',rows)
    return dict(rows=rows,policies=states,D=D,prepared=prepared,a=a)


def final_pair(root,contract,out,call,state,r):
    """Fresh final64, existing affine W/shell accounting and individual PR/D CI."""
    from . import c_near_r32_statistics as stats
    from . import c1_stage1_execution as io
    policy=state['policies'][r]
    result,packet=native_cohort(root,contract,out,call,policy['a'],policy['result'],policy['packet'],
        split='final',prepared=state['prepared'])
    binding=stats.affine(packet,result)
    summary=stats.arm_summary(packet['kernel_samples'],binding,state['D'],expected_n=64,
        calibration=result['calibration'],statistics=dict(stats.POLICY,samples=64,confidence=.95))
    value=dict(result=result,rows=packet['kernel_samples'],bindings=binding,statistics=summary)
    io.save(Path(out)/'summary.json',value)
    return value


def run_fresh(output,tasks,worker):
    """Minimal one-shot saver for bound tasks; no approval proof or retry.

    Task labels must be unique and local. A correctness exception preserves
    earlier result files and always writes STOP. It never fills a failed arm.
    """
    from . import c1_stage1_execution as io
    output=Path(output)
    labels=[x['id'] for x in tasks]
    if len(set(labels))!=len(labels) or any(Path(x).name!=x or x in ('','.', '..') for x in labels):
        raise ValueError('UNIQUE_LOCAL_TASK_LABELS')
    output.mkdir(parents=True,exist_ok=False)
    completed=[];failure=None
    try:
        for task in tasks:
            value=worker(task,output)
            io.save(output/(task['id']+'.json'),value);completed.append(task['id'])
    except BaseException as exc:
        failure=repr(exc);io.save(output/'failure.json',dict(error=failure,completed=completed));raise
    finally:
        io.save(output/'STOP.json',dict(failed=failure is not None,completed=completed,
            reason=failure or 'REGISTERED_TASKS_COMPLETE',retry=0,resume=0,mandatory_stop=True))
    return completed
