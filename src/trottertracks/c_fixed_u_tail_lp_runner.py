"""Dormant PR-only LP scientific connection; preflight is bytes/metadata only."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time
from . import c_fixed_u_tail_lp as lp
from . import c_rc2_pr_stage_b as old

io,save,b=old.io,old.save,old.b
BASE='611a618674aba7146b0103cc34f156c9bc89bc16'
SCOPE='C_FIXED_U_PR_TAIL_PRESERVING_LP_V1'
FAMILY='artifacts/research_c_fixed_u_tail_lp/2026-11-01'
FRESH=FAMILY+'/run01'
POLICY=dict(norb=4,rank=8,ld=2,head=[0,1],tail=list(range(2,8)),
    Nalpha=2,Nbeta=2,sector_dimension=36,h='1/10',s=1,r=128,K=0,
    CA_Ha=.00159,DF_budget_Ha=.00019875,U='exact original stored bytes; no polar/codec',
    variable='upper triangular tail J only, unconstrained sign',tensor='all 256 original entries',
    solver='highs-ds',LP_calls=1,new_fit=0,optimizer_iterations=0,native=0,shots=0,
    outer_U_updates=0,RC2_LP=0,retry=0,resume=0)
LIMITS=dict(evidence_load=1,input_load=1,export_load=1,saved_reference_load=1,
    fixed_input_gate=1,LP_assemble=1,LP_solve=1,LP_diagnose=1,candidate_export=1,
    tensor_gate=1,arm_prepare=2,law_prepare=2,coefficient_gate=2,sector_context=1,
    sector_raw_frame=6,sector_generalK_gate=6,full_CAR_context=1,full_raw_frame=12,
    full_tail_gate=1,fixed_head_phase_gate=1,DF_preservation=1,kernel_reference=1,
    PF_RTE_gate=1,attenuation_profile=1,candidate_qualification=1,comparison=1)
MAX_CALLS=sum(LIMITS.values())


def evidence(root,c):return {k:io.read(Path(root)/p) for k,p in c['records'].items()}


def saved_gate(saved,c):
    r,stop,source,contract=(saved[k] for k in ('B_result','B_STOP','B_SOURCE','B_contract'))
    arm=r['arms']['pr']
    if (r['status']!='SAVED_ENDPOINT_STAGE_B_COMPLETE' or stop['failed']
            or stop['scientific_units']!=143 or stop['call_counts']!=old.LIMITS
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or source['execution_commit']!='d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6'
            or source['files']!=saved['B_manifest']['files']
            or contract['policy']!=old.POLICY or contract['fit_source']!=old.FIT_SOURCE
            or contract['identities']['pr']!=c['identity'] or arm['identity']!=c['identity']
            or arm['DF_budget']!='PASS' or arm['optimizer_qualification']!='UNQUALIFIED'
            or saved['pr_tail']['status']!='PASS'):
        raise ValueError('SAVED_PR_SOURCE_OR_QUALIFICATION')
    bound=arm['saved_DF_bound_Ha'];mass=saved['pr_tail']['lambda_R']
    if not (0<=bound<=POLICY['DF_budget_Ha'] and 0<=mass<float('inf')):raise ValueError('SAVED_DF_OR_MASS_METADATA')
    return dict(saved_DF_guard_inclusive_bound=bound,saved_exact_lambda=mass,
        optimizer_qualification='UNQUALIFIED',new_optimizer=False)


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,w in dict(scope=SCOPE,implementation=lp.VERSION,base_commit=BASE,policy=POLICY,
            solver_options=lp.OPTIONS,allowances=lp.ALLOWANCES,call_limits=LIMITS,
            maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0).items():
        if c[k]!=w:raise ValueError('FIXED_LP_SCOPE:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    required={c['input'],c['export'],c['reference'],*c['records'].values()}
    if not required<=c['saved_inputs'].keys():raise ValueError('UNPINNED_LP_INPUT')
    if c['identity']!=dict(role='pr',export=c['export'],sha256=c['saved_inputs'][c['export']],
            fit_source=old.FIT_SOURCE,physical_input=c['saved_inputs'][c['input']]):raise ValueError('PR_IDENTITY')
    old.old.export_headers(root/c['export']);saved_gate(evidence(root,c),c)
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT:'+name)
    if (root/FRESH).exists():raise ValueError('FRESH_LP_NO_RETRY_RESUME')
    return c


def tensor_gate(J,U,newJ,newU):
    import numpy as np
    if (U.dtype!=newU.dtype or U.shape!=newU.shape or U.tobytes()!=newU.tobytes()
            or J[:2].tobytes()!=newJ[:2].tobytes()):raise ValueError('U_OR_HEAD_CHANGED')
    before=lp.reconstruct(J[2:],U[2:]);after=lp.reconstruct(newJ[2:],newU[2:])
    report=b.component_check(after.reshape(-1,after.shape[-1]**2),before.reshape(-1,before.shape[-1]**2),
        lp.ALLOWANCES['tensor_absolute'],label='independent complete tail tensor (pair-index matrix)')
    return dict(status='PASS',tail=report,head_J_exact=True,U_exact=True,
        full_tensor='same head tensor plus checked complete tail tensor',rigorous_enclosure=None)


def prepare(packet,J,U,identity,*,physical_identity):
    arm=b.prepare_arm(**packet,J=J,U=U,role='proposed',source_identity=identity,
        input_identity=physical_identity,ld=2)
    model=replace(arm.model,label='FIXED_U_TAIL_LP:'+identity)
    return replace(arm,model=model,plan=replace(arm.plan,model=model))


def coefficient_gate(arm,tail,expected):
    import math
    import numpy as np
    mass=float(np.sum(abs(lp.objective.coefficients(arm.J[2:],np))))
    if mass==0 and np.any(arm.J[2:]!=0):raise ValueError('COEFFICIENT_UNDERFLOW_NOT_ZERO_TAIL')
    if not math.isclose(mass,tail.lambda_r,rel_tol=1e-12,abs_tol=1e-14) or not math.isclose(mass,expected,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('EXACT_PAULI_MAP_MODEL_MISMATCH')
    return dict(status='PASS',exact_mass=mass,law_mass=tail.lambda_r,identity=arm.plan.extracted_identity,
        deterministic_only=not arm.plan.atoms)


def DF_preservation(generators,new_generators,saved_bound):
    import numpy as np
    delta=sum(new_generators)-sum(generators)
    b.component_check(delta,np.zeros_like(delta),lp.ALLOWANCES['operator_component'],label='sector Hamiltonian preservation')
    drift=float(np.linalg.norm(delta,2))
    scale=sum(float(np.linalg.norm(x,'fro')) for x in [*generators,*new_generators])
    n=POLICY['norb'];rank=POLICY['rank'];d=len(delta)
    guard=8*np.finfo(float).eps*2*(n**4+rank*n**3)*(d+1)*max(1.,scale)
    bound=float(saved_bound+drift+guard)
    if bound>POLICY['DF_budget_Ha']:raise ValueError('DF_BUDGET_NOT_MET')
    return dict(status='PASS',sector_drift_norm=drift,drift_guard=guard,
        saved_guard_inclusive_bound=saved_bound,new_numerical_triangle_bound=bound,budget=POLICY['DF_budget_Ha'],
        policy='saved_DF_plus_sector_drift_guard8_pilot_v1',rigorous_enclosure=None,
        physical_operator_or_original_DF_norm_recomputed=False)


def attenuation(old_config,new_config):
    from .c1_stage3_accuracy import _legacy
    rte=_legacy('rte');rows=[]
    for m in range(13):
        value=lambda config:1. if config is None else rte.finite_rte_attenuation(config,tail_evolutions=1<<m)
        a,z=value(old_config),value(new_config)
        if z+lp.ALLOWANCES['attenuation_absolute']<a:raise ValueError('ATTENUATION_MONOTONICITY_FAIL')
        rows.append(dict(m=m,old=a,new=z))
    return dict(ideal_K0_IID_only=True,rows=rows,shots=None,actual_CA=None,total_resources=None)


def execute(root,c,out,call):
    import numpy as np
    from threadpoolctl import threadpool_limits
    out=Path(out)
    saved=call('evidence_load','fixed old PR evidence',lambda:evidence(root,c));qualification=saved_gate(saved,c)
    packet=call('input_load','original h/g/scalar',lambda:old.old.load(Path(root)/c['input'],('h','g','constant')))
    arrays=call('export_load','old PR J/U only',lambda:old.old.load(Path(root)/c['export'],('J','U')))
    reference=call('saved_reference_load','saved sector G/H/PF/corrected',lambda:old.old.load(Path(root)/c['reference'],('generators','H','pf','corrected')))
    J,U=arrays['J'],arrays['U']
    def fixed():
        if (J.shape!=(8,4,4) or U.shape!=J.shape or packet['h'].shape!=(4,4)
                or packet['g'].shape!=(4,)*4 or packet['constant'].shape!=()):raise ValueError('FIXED_RANK_PARTITION')
        lp.pack(J);lp.FixedU(U);b.stage1.finite_arrays(*packet.values(),*reference.values())
        if reference['generators'].shape!=(9,36,36) or any(reference[k].shape!=(36,36) for k in ('H','pf','corrected')):raise ValueError('SAVED_REFERENCE_SCHEMA')
        b.component_check(sum(reference['generators']),reference['H'],1e-10,label='saved generator sum')
        return dict(status='PASS',rank=8,ld=2,U_corrected=False)
    save(out/'fixed_input.json',call('fixed_input_gate','shape/order/kappa/scalar policy',fixed))
    p=call('LP_assemble','full tensor rows; exact Pauli T',lambda:lp.problem(J[2:],U[2:]))
    old.old.save_arrays(out/'LP_problem.npz',A=p.A,T=p.T.toarray(),b=p.b,j0=p.j0)
    raw=call('LP_solve','one highs-ds, no retry',lambda:lp.solve_once(p))
    raw_arrays={}
    for name in ('x','slack','con'):
        if getattr(raw,name,None)is not None:raw_arrays[name]=np.asarray(getattr(raw,name))
    for group in ('eqlin','ineqlin','lower','upper'):
        for key in ('marginals','residual'):
            value=getattr(getattr(raw,group,None),key,None)
            if value is not None:raw_arrays[group+'_'+key]=np.asarray(value)
    old.old.save_arrays(out/'LP_raw.npz',**raw_arrays)
    save(out/'LP_solver.json',dict(success=bool(raw.success),status=int(raw.status),message=str(raw.message),
        fun=None if raw.fun is None else float(raw.fun),nit=raw.nit,method='highs-ds',options=lp.OPTIONS))
    diagnosis=call('LP_diagnose','primal/dual/full-row gap',lambda:lp.diagnose(p,raw));save(out/'LP_diagnosis.json',diagnosis)
    if not diagnosis['numerical_optimality'] or diagnosis['status'].startswith('UNRESOLVED'):
        result=call('comparison','unqualified LP: no downstream adoption',lambda:dict(status='UNRESOLVED_LP',LP=diagnosis,
            candidate=None,old_optimizer='UNQUALIFIED',mandatory_stop=True))
        save(out/'result.json',result);return result
    newJ,newU=lp.replace_tail(J,U,lp.unpack(raw.x[:p.mapping.q],6,4),2)
    def export():
        old.old.save_arrays(out/'candidate.npz',J=newJ,U=newU)
        return dict(export=str(Path(out/'candidate.npz').relative_to(root)),sha256=io.sha(out/'candidate.npz'),
            parent=c['identity'],kind='coefficient-only LP postprocessing, not a new CDF fit')
    candidate=call('candidate_export','unqualified until all preservation gates',export);save(out/'candidate.json',candidate)
    save(out/'tensor.json',call('tensor_gate','unmodified U/head and independent contraction',lambda:tensor_gate(J,U,newJ,newU)))
    arms=[];configs=[]
    for name,a,u,identity in [('old',J,U,c['identity']['sha256']),('new',newJ,newU,candidate['sha256'])]:
        arm=call('arm_prepare',name,lambda a=a,u=u,identity=identity:prepare(packet,a,u,identity,physical_identity=c['identity']['physical_input']))
        tail,config,dist=call('law_prepare',name,lambda:b.configure(arm,Fraction(1,10),128,0))
        expected=qualification['saved_exact_lambda'] if name=='old' else diagnosis['new_lambda']
        save(out/(name+'_coefficients.json'),call('coefficient_gate',name,lambda:coefficient_gate(arm,tail,expected)))
        arms.append(arm);configs.append(config)
    def head_phase():
        if abs(arms[0].plan.extracted_identity-arms[1].plan.extracted_identity)>lp.ALLOWANCES['identity']:raise ValueError('IDENTITY_PHASE_CHANGED')
        if (not np.array_equal(arms[0].general.spatial_one_body_tensor,arms[1].general.spatial_one_body_tensor)
                or arms[0].model.constant!=arms[1].model.constant
                or [x.block_id for x in arms[1].model.blocks[:3]]!=['one_body','F0','F1']):raise ValueError('HEAD_KAPPA_SCALAR_CHANGED')
        return dict(status='PASS',head_exact=True,old_identity=arms[0].plan.extracted_identity,new_identity=arms[1].plan.extracted_identity)
    save(out/'head_phase.json',call('fixed_head_phase_gate','kappa/scalar/head/order/identity',head_phase))
    ctx=call('sector_context','existing SectorCAR36',lambda:old.old.molecule.SectorCAR(4))
    newG=list(reference['generators'][:3])
    for i in range(6):
        G=call('sector_raw_frame',str(i+2),lambda i=i:ctx.raw_frame(newJ[i+2],newU[i+2]));newG.append(G)
        report=call('sector_generalK_gate',str(i+2),lambda i=i,G=G:b.component_check(
            b.candidate_factor(ctx,arms[1],i+3),G,lp.ALLOWANCES['operator_component'],label='new general-K/raw CAR'))
        save(out/f'generalK_{i+2}.json',report)
    plan=old.old.integrated.resource_plan(task_count=12,full_dimension=128);save(out/'resource_plan.json',plan)
    car=call('full_CAR_context','independent full256 CAR',lambda:b.stage1.car_bilinears(8))
    def frame(name,i):
        a,u=(J,U) if name=='old' else (newJ,newU)
        return call('full_raw_frame',f'{name}:{i}',lambda:b.stage1.raw_frame(a[i],u[i],car)[0])
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            jobs=[pool.submit(frame,name,i) for name in ('old','new') for i in range(2,8)]
            try:full=[f.result() for f in jobs]
            except BaseException:
                call.stop.set()
                for f in jobs:f.cancel()
                raise
    def tail_gate():
        oldfull,newfull=sum(full[:6]),sum(full[6:])
        ix=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices]
        return dict(status='PASS',full=b.component_check(newfull,oldfull,1e-10,label='full tail CAR'),
            old_sector=b.component_check(oldfull[np.ix_(ix,ix)],sum(reference['generators'][3:]),1e-10,label='old full/saved sector'),
            new_sector=b.component_check(newfull[np.ix_(ix,ix)],sum(newG[3:]),1e-10,label='new full/sector'))
    save(out/'tail_CAR.json',call('full_tail_gate','full old/new and sector mapping',tail_gate))
    df=call('DF_preservation','saved bound plus new sector drift/guard',lambda:DF_preservation(reference['generators'],newG,qualification['saved_DF_guard_inclusive_bound']));save(out/'DF_preservation.json',df)
    kernel=call('kernel_reference','new partial PF and corrected K0, one h-step',lambda:b.kernels(arms[1],newG,Fraction(1,10),configs[1]))
    old.old.save_arrays(out/'new_reference.npz',generators=np.array(newG),pf=kernel['pf'],corrected=kernel['corrected'])
    gates=call('PF_RTE_gate','compare to immutable saved old kernels',lambda:{k:b.component_check(kernel[k],reference[k],lp.ALLOWANCES['kernel_component'],label=k+' preserved') for k in ('pf','corrected')});save(out/'PF_RTE.json',gates)
    save(out/'attenuation.json',call('attenuation_profile','ideal K0 13-round diagnostic only',lambda:attenuation(*configs)))
    qualification=call('candidate_qualification','all registered preservation gates completed',lambda:dict(status='NUMERICAL_PRESERVATION_PASS',candidate=candidate,DF=df,
        PF_RTE_scope='one h-step numerical gate; no new 13-round PF/RTE errors',rigorous_enclosure=None,original_optimizer='UNQUALIFIED'))
    save(out/'candidate_qualification.json',qualification)
    result=call('comparison','LP result class, no downstream resource claim',lambda:dict(status=diagnosis['status'],LP=diagnosis,
        preservation=qualification,candidate=candidate,actual_CA=None,shots=None,native_cost=None,total_resources=None,independent_novelty=None,mandatory_stop=True))
    save(out/'result.json',result);return result


def run(root,path,output):
    root=Path(root).resolve();c=preflight(root,path);output=Path(output).resolve()
    if output!=root/FRESH or output.exists():raise ValueError('FRESH_LP_NO_RETRY_RESUME')
    output.mkdir(parents=True);calls=old.old.integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(output/'contract.json',c);save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,output,calls)
    except BaseException as error:
        calls.stop.set();save(output/'failure.json',dict(status='FAILED_STOP',error=str(error),first_failure=calls.first_failure));raise
    finally:
        save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),retry=0,resume=0,mandatory_stop=True))
        save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
