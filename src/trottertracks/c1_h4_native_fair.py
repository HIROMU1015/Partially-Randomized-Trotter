"""Dormant CDF8/XDF8 fair precision/full512 pilot. No science on import.

Reuses NativeBuilder (including square primitives), H6 precision/P0 fixes,
saved CAR/PF evidence, full Operator, existing residuals and call journal.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from fractions import Fraction
import importlib.metadata
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1 as stage
from . import c1_stage1_execution as io
from . import c1_stage2_native as native
from . import c1_stage3_reference as reference
from . import c1_stage3_connection as connection
from . import c1_stage3_integrated as integrated
from . import c1_h4_full_synthesis as synthesis
from .c1_stage3_molecular import component_check
from .c1_stage3_cost_compat import CompileSerialCalls
from .c1_h6_native_precision import PrecisionBuilder, VERSION as GAUSSIAN
from .c1_h6_compiler_precision import compiled_wrapper, VERSION as PHASE
from .pr_dictionary_validation.fixture import CompilerProfile

SCOPE='C1_H4_CDF8_XDF8_FAIR_NATIVE_FULL512_V1'
FAMILY='artifacts/research_c_c1_h4_native_fair'
ARMS=('CDF','XDF'); JOBS=tuple((a,s) for a in ARMS for s in (1,4)); SHELLS=('head','X','Y')
CONDITIONS=dict(atoms=4,distance_angstrom=1.,basis='STO-3G',Nalpha=2,Nbeta=2,
    sector_dimension=36,system_dimension=256,controlled_dimension=512,CDF_rank=8,XDF_rank=8,
    XDF_order='original saved OpenFermion1.6.1 eigh prefix8',number_representation=True,
    original_g_kappa=True,scalar='nuclear once',frame_order=list(range(8)),
    PF='all-frame ordered symmetric S2',substeps=[1,4],base_time='1/10',rounds=list(range(13)),
    route='existing adjacent identical-basis sharing and fsum angles',
    serial_policy='separate head/kernel/readout with all-wire fences; no long-round compile',
    oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',CA_Ha=.00159,
    DF_reserve_Ha=.00019875,beta_total=math.pi/6,beta_stat=math.pi/12,
    candidate_failure=.01,axis_failure=.01/26,PR_split=None)
ALLOWANCES=dict(operator_component_abs=1e-10,unitarity_component_abs=1e-10,
    signal_complex=2e-10,normalization=1e-12,rtol=0,local_time='1/20',
    global_phase_fit=False,full_SVD=False,qualification='full512 Frobenius numerical pilot; no rounding enclosure')
LIMITS=dict(input_load=1,export_load=1,evidence_load=10,cdf_operator_load=9,
    sector_generator_load=1,cdf_sector_load=1,ground_load=2,saved_kernel_load=2,
    identity_gate=1,native_model=2,car_context=1,xdf_generator=9,sector_identity=2,
    gaussian_build=18,local_reference=18,local_build=18,local_operator=18,local_gate=18,
    shell_build=3,shell_structure=6,kernel_reference=4,kernel_sector_gate=2,kernel_build=4,compile=7,
    operator_decode=14,segment_assessment=7,adjoint_build=8,adjoint_operator=8,
    adjoint_assessment=8,signal_reference=4,signal_prepare=8,signal_axis=16,signal_gate=12,
    metrics=7,round_synthesis=52,cost_comparison=2,fairness_gate=1)
MAX_CALLS=sum(LIMITS.values())
FORBIDDEN=('new_fit','optimizer','new_DF_decomposition','DF_error_recalculation',
    'PF_error_recalculation','literal_shared_comparison','CA_evaluation','shots_evaluation',
    'total_cost_evaluation','retry','resume')


def profile_dict():
    p=asdict(CompilerProfile());p['basis_gates']=list(p['basis_gates']);return p


def preflight(root,path,input_path):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions']!=CONDITIONS or c['allowances']!=ALLOWANCES
            or c['compiler']!=profile_dict() or c['gaussian']!=GAUSSIAN
            or c['compiler_phase']!=PHASE or c['synthesis']!=synthesis.VERSION
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('FAIR_NATIVE_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group,label in [(io.read(root/c['source_manifest'])['files'],'SOURCE'),(c['saved_inputs'],'INPUT')]:
        for f,want in group.items():
            p=(root/f).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError(label+': '+f)
    if io.sha(input_path)!=io.P_H4_SHA256 or c['original_input']['sha256']!=io.P_H4_SHA256:
        raise ValueError('ORIGINAL_PACKET')
    required=[c['export'],c['fit_SOURCE'],c['fit_metadata'],c['CDF_sector'],c['XDF_generators'],
        *c['CDF_operators'],*c['records'].values(),*c['ground_states'].values(),*c['XDF_kernels'].values()]
    if (len(c['records'])!=10 or len(c['CDF_operators'])!=9 or set(c['ground_states'])!=set(ARMS)
            or set(c['XDF_kernels'])!={'1','4'} or not set(required)<=c['saved_inputs'].keys()):
        raise ValueError('UNPINNED_EVIDENCE')
    if c['fresh_output']!=FAMILY+'/<execution-date>/run01':raise ValueError('OUTPUT_SCOPE')
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT: '+name)
    return c


def fresh_output(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    parts=output.relative_to(root/FAMILY).parts
    if len(parts)!=2 or parts[1]!='run01' or date.fromisoformat(parts[0]).isoformat()!=parts[0] or output.exists():
        raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    return output


def matrix_load(path,member,shape):
    import numpy as np
    with np.load(path,allow_pickle=False) as a:value=a[member]
    if value.shape!=shape or value.dtype!=np.complex128 or not np.isfinite(value).all():
        raise ValueError('SAVED_MATRIX_SCHEMA: '+str(path))
    value.setflags(write=False);return value


def load_data(root,c,input_path,call):
    import numpy as np
    root=Path(root)
    packet=call('input_load','original h/g/c and saved XDF whitelist',lambda:io.load_input(input_path,evaluation=True))
    exported,metadata=call('export_load','saved CDF8; no fitter',lambda:io.load_export(
        (root/c['export']).parent,dict(input=input_path,source_path=root/c['fit_SOURCE'])))
    evidence={k:call('evidence_load',k,lambda p=p:io.read(root/p)) for k,p in c['records'].items()}
    cdf=[call('cdf_operator_load',str(i),lambda p=p:matrix_load(root/p,'matrix',(256,256))) for i,p in enumerate(c['CDF_operators'])]
    xgens=call('sector_generator_load','saved XDF8',lambda:matrix_load(root/c['XDF_generators'],'generators',(9,36,36)))
    csector=call('cdf_sector_load','saved CDF8',lambda:matrix_load(root/c['CDF_sector'],'matrix',(36,36)))
    grounds={a:call('ground_load',a,lambda p=p:matrix_load(root/p,'sector_state',(36,))) for a,p in c['ground_states'].items()}
    kernels={int(s):call('saved_kernel_load','XDF8 s'+s,lambda p=p:matrix_load(root/p,'pf',(36,36))) for s,p in c['XDF_kernels'].items()}
    return dict(packet=packet,exported=exported,metadata=metadata,evidence=evidence,
        CDF=cdf,XDF_sector=xgens,CDF_sector=csector,grounds=grounds,XDF_kernels=kernels,
        modes=8,indices=stage.sector_indices(4,2,2))


def identity(data,c):
    import numpy as np
    p,e,md,r=data['packet'],data['exported'],data['metadata'],data['evidence']
    if md['actual_rank']!=8 or md['frame_ids']!=[f'F{i}' for i in range(8)]:raise ValueError('CDF_RANK_ORDER')
    if (float(e['constant'])!=float(p['nuclear_repulsion'])
            or not np.array_equal(data['indices'],stage.sector_indices(4,2,2))):raise ValueError('SCALAR_OR_BASIS')
    component_check(np.kron(e['kappa'],np.eye(2)),p['df_one_body'],1e-10,label='saved original-g kappa')
    if (r['XDF_result']['status']!='XDF8_STAGE1_COMPLETE' or r['XDF_result']['representation']!='PASS'
            or r['XDF_SOURCE']['execution_commit']!=c['lineage']['XDF_PF_SOURCE']
            or r['XDF_STOP']['failed'] or r['XDF_STOP']['scientific_units']!=93
            or not r['XDF_STOP']['mandatory_stop'] or r['CDF_STOP']['failed']
            or not r['CDF_STOP']['mandatory_stop'] or r['CDF_prepared']['state_preparation']!='ORACLE_EXCLUDED'
            or r['CDF_stage1']['status']!='STAGE1_REPRESENTATION_PASS'
            or r['CDF_provenance']['fit_export_sha256']!=c['saved_inputs'][c['export']]
            or r['CDF_provenance']['input_sha256']!=c['original_input']['sha256']):raise ValueError('SAVED_TARGET_PROVENANCE')
    rows={};df={}
    for arm in ARMS:
        if arm=='CDF':df[arm]=r['CDF_prepared']['df_error_bounds_ha']['CDF']
        else:
            if r['XDF_result']['saved_DF']['status']!='PASS':raise ValueError('SAVED_DF_STATUS')
            df[arm]=r['XDF_result']['saved_DF']['saved_norm_plus_allowance_Ha']
        if not math.isfinite(df[arm]) or not 0<=df[arm]<=.00019875:raise ValueError('SAVED_DF_RESERVE')
        psi=data['grounds'][arm]
        if abs(np.vdot(psi,psi)-1)>1e-12:raise ValueError('SAVED_GROUND_NORMALIZATION')
        for s in (1,4):
            v=r[f'CDF_PF_s{s}'] if arm=='CDF' else r['XDF_result']['candidates'][str(s)]
            status=v['status'] if arm=='CDF' else v['summary']['status']
            if status!='POTENTIALLY_FEASIBLE' or [x['m'] for x in v['rounds']]!=list(range(13)):
                raise ValueError('SAVED_PF_COVERAGE')
            if arm=='CDF' and v['candidate']!=dict(arm='CDF',family='D',s=s,r=0,K=0):raise ValueError('CDF_PF_TARGET')
            for row in v['rounds']:
                eps=row['pf_operator_error']
                if (eps is not None and (not math.isfinite(eps) or eps<0)
                        or row['attenuation']!=1 or row['rte_corrected_operator_error']!=0):raise ValueError('SAVED_DETERMINISTIC_PF')
            rows[arm,s]=v['rounds']
    if r['XDF_result']['ground']['status']!='PASS':raise ValueError('SAVED_XDF_ALIAS')
    return dict(status='PASS',DF_bounds_Ha=df,rows=rows,
        alias='saved H4 own-ground/[-10,10] base branch checks; no new eigensolve')


def prepare_models(data,c,call):
    from .c1_general_k_adapter import from_ffsim_number_arrays
    e,p,md=data['exported'],data['packet'],data['metadata']
    def cdf():
        general=from_ffsim_number_arrays(one_body_tensor=e['kappa'],diag_coulomb_mats=e['diag_coulomb_mats'],
            orbital_rotations=e['orbital_rotations'],constant=float(e['constant']),z_representation=False,
            frame_ids=md['frame_ids'],group_ids=md['group_ids'],source_identity=c['saved_inputs'][c['export']],
            input_identity=c['original_input']['sha256'],one_body_role='ffsim_corrected_kappa',enable_general_k=True)
        return native.prepare_cdf(general)
    models={'CDF':call('native_model','CDF8 general-K',cdf),
        'XDF':call('native_model','XDF8 signed square primitives',lambda:native.prepare_xdf(
            p['df_one_body'],p['df_lambdas'][:8],p['df_g_matrices'][:8],float(p['nuclear_repulsion']),
            source_identity=c['original_input']['sha256']+':prefix8',input_identity=c['original_input']['sha256']))}
    return models


def controlled(system):
    import numpy as np
    d=len(system);out=np.zeros((2*d,2*d),complex);out[:d,:d]=np.eye(d);out[d:,d:]=system;return out


def decode(circuit):
    from qiskit.quantum_info import Operator
    from .c1_h6_state_action import measured_tail
    clean=measured_tail(circuit) if circuit.num_clbits else circuit
    return Operator(clean).data


def compile_checked(circuit,profile):
    value=compiled_wrapper(circuit,profile)
    if (value.num_qubits!=circuit.num_qubits or value.num_clbits!=circuit.num_clbits
            or value.parameters or any(x.operation.name not in {'rz','sx','x','cx','measure','barrier'} for x in value.data)):
        raise ValueError('P0_NATIVE_BASIS_OR_REGISTER_MISMATCH')
    return value


def local_gate(actual,target,label):
    import numpy as np
    report=component_check(actual,controlled(target),1e-10,label='local controlled:'+label)
    report['unitarity']=component_check(actual.conj().T@actual,np.eye(len(actual)),1e-10,label='local unitary:'+label)
    return report


def fairness(models,builder):
    if not isinstance(builder,PrecisionBuilder) or set(models)!=set(ARMS):raise ValueError('COMMON_PRECISION_REQUIRED')
    for arm,model in models.items():
        if (model.label!=arm or [b.block_id for b in model.blocks]!=['one_body']+[f'F{i}' for i in range(8)]):
            raise ValueError('MODEL_RANK_ORDER')
        for block in model.blocks:
            report=builder.basis_reports[arm+':'+block.block_id]
            if report['version']!=GAUSSIAN or report['single_particle_max_entry']>1e-10:
                raise ValueError('GAUSSIAN_PRECISION_COVERAGE')
    return dict(status='PASS',gaussian=GAUSSIAN,compiler_phase=PHASE,compiler=profile_dict(),
        route=CONDITIONS['route'],serial_policy=CONDITIONS['serial_policy'],
        coverage='all CDF general-K and XDF square-primitive basis requests use the same PrecisionBuilder',
        additional_relative_Gaussian_fusion=False,old_CDF_metrics_reused=False)


def basis_request(builder,model,block):
    """Keep exact existing cache/sharing; report coverage also on cache hits."""
    value=builder._basis(model,block);label=model.label+':'+block.block_id
    if label not in builder.basis_reports:
        previous=next(b for b in model.blocks if b.rotation==block.rotation
            and model.label+':'+b.block_id in builder.basis_reports)
        builder.basis_reports[label]=dict(builder.basis_reports[model.label+':'+previous.block_id],
            reused_basis_from=previous.block_id)
    return value


def execute_pipeline(data,models,verified,call,output,plan):
    import numpy as np
    from scipy.linalg import expm
    from qiskit import qpy
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    from .pr_dictionary_validation.gates import signal_gate
    from .c1_h6_native import compare_metrics, shell_unitary
    from .c1_h6_state_action import ancilla_z
    output=Path(output);serial=CompileSerialCalls(call);builder=PrecisionBuilder()
    d=1<<data['modes'];generators={'CDF':data['CDF']};cache={a:{} for a in ARMS}
    profile=CompilerProfile();profile.validate()
    def qsave(label,kind,qc):
        with (output/'circuits'/f'{label}_{kind}.qpy').open('xb') as f:qpy.dump(qc,f)
    def msave(label,value):
        with (output/'operators'/f'{label}.npz').open('xb') as f:np.savez_compressed(f,matrix=value)
    def parallel(function,jobs,workers=None):
        values=[]
        with ThreadPoolExecutor(max_workers=workers or plan['workers']) as pool:
            futures={pool.submit(function,j):j for j in jobs}
            for f in as_completed(futures):
                try:values.append(f.result())
                except BaseException:
                    call.stop.set()
        if call.stop.is_set():raise ValueError('FAIR_NATIVE_DEPENDENT_STOP')
        return values
    car=call('car_context','full CAR only; no circuit projection',lambda:stage.car_bilinears(data['modes']))
    p=data['packet']
    generators['XDF']=[call('xdf_generator','base',lambda:float(p['nuclear_repulsion'])*np.eye(d)+native.xdf_car_generator(p['df_one_body'],None,car))]
    for i in range(8):
        generators['XDF'].append(call('xdf_generator',f'F{i}',lambda i=i:native.xdf_car_generator(p['df_g_matrices'][i],p['df_lambdas'][i],car)))
    idx=data['indices'];outside=np.array([i for i in range(d) if i not in set(idx)])
    for arm in ARMS:
        def sector_gate(arm=arm):
            total=sum(generators[arm]);target=data['CDF_sector'] if arm=='CDF' else sum(data['XDF_sector'])
            gate=component_check(total[np.ix_(idx,idx)],target,1e-10,label=arm+':same saved DF target')
            for g in generators[arm]:
                component_check(g[np.ix_(outside,idx)],np.zeros((len(outside),len(idx))),1e-10,label=arm+':complete-factor invariance')
            return gate
        io.save(output/(arm+'_sector_identity.json'),call('sector_identity',arm,sector_gate))
        for block in models[arm].blocks:
            call('gaussian_build',arm+':'+block.block_id,lambda arm=arm,block=block:basis_request(builder,models[arm],block))
    io.save(output/'gaussian_precision.json',builder.basis_reports)
    io.save(output/'fairness.json',call('fairness_gate','both arms before local checks',lambda:fairness(models,builder)))
    locals=[]
    for arm in ARMS:
        for i in range(9):
            label=f'{arm}_F{i}';qc=call('local_build',label,lambda arm=arm,i=i:builder.local(models[arm],i,Fraction(1,20)))
            qsave(label,'raw',qc);locals.append((arm,i,label,qc))
    def check_local(job):
        arm,i,label,qc=job
        target=call('local_reference',label,lambda:expm(-.05j*generators[arm][i]))
        cache[arm][i,Fraction(1,20)]=target
        actual=call('local_operator',label,lambda:decode(qc))
        report=call('local_gate',label,lambda:local_gate(actual,target,label))
        io.save(output/'local'/(label+'.json'),report);return report
    local_reports=parallel(check_local,locals,plan.get('local_factor_workers',plan['workers']))
    shells={};shell_metrics={};H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
    for name in SHELLS:
        def build(name=name):
            head,tail=connection.readout_segments(data['modes'],'Y' if name=='Y' else 'X')
            return head if name=='head' else tail
        raw=call('shell_build',name,build);compiled=serial('compile','shell:'+name,lambda:compile_checked(raw,profile))
        qsave('shell_'+name,'raw',raw);qsave('shell_'+name,'compiled',compiled)
        # Ancilla-only structure is a separate structural check, not a sector projection.
        for kind,qc in [('raw',raw),('compiled',compiled)]:
            call('shell_structure',name+':'+kind,lambda qc=qc:shell_unitary(qc))
        matrices={k:call('operator_decode','shell:'+name+':'+k,lambda qc=qc:decode(qc)) for k,qc in [('raw',raw),('compiled',compiled)]}
        for kind,value in matrices.items():msave('shell_'+name+'_'+kind,value)
        ideal=np.kron(H@np.diag([1,-1j]) if name=='Y' else H,np.eye(d))
        report=call('segment_assessment','shell:'+name,lambda:synthesis.segment(matrices['raw'],matrices['compiled'],ideal,'shell:'+name))
        shells[name]=dict(**matrices,report=report);io.save(output/('shell_'+name+'.json'),report)
        shell_metrics[name]=call('metrics','shell:'+name,lambda:circuit_metrics(compiled))
    raw_kernels={}
    for arm,s in JOBS:
        label=f'{arm}_s{s}';raw_kernels[arm,s]=call('kernel_build',label,lambda arm=arm,s=s:builder.ordered(models[arm],Fraction(1,10*s),optimized=True))
        qsave(label,'raw',raw_kernels[arm,s])
    def candidate(job):
        arm,s=job;label=f'{arm}_s{s}';h=Fraction(1,10*s)
        # Own cache per candidate avoids concurrent writes; reuse exact local exponentials.
        target=call('kernel_reference',label,lambda:reference._ordered_exponentials(
            generators[arm],native.ordered_schedule(9,h),dict(cache[arm])))
        msave(label+'_reference',target)
        if arm=='XDF':
            io.save(output/(label+'_saved_kernel.json'),call('kernel_sector_gate',label,
                lambda:component_check(target[np.ix_(idx,idx)],data['XDF_kernels'][s],1e-10,label='XDF8 saved PF kernel')))
        raw=raw_kernels[job];compiled=serial('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
        matrices={k:call('operator_decode',label+':'+k,lambda qc=qc:decode(qc)) for k,qc in [('raw',raw),('compiled',compiled)]}
        for kind,value in matrices.items():msave(label+'_'+kind,value)
        ideal=controlled(target)
        report=call('segment_assessment',label,lambda:synthesis.segment(matrices['raw'],matrices['compiled'],ideal,label))
        io.save(output/'candidates'/(label+'_segment.json'),report)
        for kind,qc in [('raw',raw),('compiled',compiled)]:
            def inverse(qc=qc):
                # Qiskit inverse creation shares translator internals with P0.
                with serial.lock:
                    if call.stop.is_set():raise integrated.Stopped('Dependent inverse stopped')
                    return native.actual_adjoint(qc)
            adj=call('adjoint_build',label+':'+kind,inverse);qsave(label,kind+'_adjoint',adj)
            mat=call('adjoint_operator',label+':'+kind,lambda:decode(adj))
            def check(mat=mat,kind=kind):
                return dict(actual=component_check(mat,matrices[kind].conj().T,1e-10,label='actual adjoint:'+label),
                    reference=component_check(mat,ideal.conj().T,1e-10,label='adjoint/reference:'+label),
                    recovery=component_check(mat@matrices[kind],np.eye(2*d),1e-10,label='adjoint recovery:'+label))
            io.save(output/'candidates'/(label+'_'+kind+'_adjoint.json'),call('adjoint_assessment',label+':'+kind,check))
        psi=np.zeros(d,complex);psi[idx]=data['grounds'][arm]
        def expected_signal():
            v=psi.copy()
            for _ in range(s):v=target@v
            return complex(np.vdot(psi,v))
        expected=call('signal_reference',label,expected_signal);signals={}
        for kind in ('raw','compiled'):
            def prepare_signal(kind=kind):
                v=shells['head'][kind]@np.r_[psi,np.zeros(d)]
                for _ in range(s):v=matrices[kind]@v
                return v
            state=call('signal_prepare',label+':'+kind,prepare_signal)
            axes={axis:call('signal_axis',label+':'+kind+':'+axis,
                lambda axis=axis:float(ancilla_z((shells[axis][kind]@state)[None,:])[0])) for axis in ('X','Y')}
            signals[kind]=complex(axes['X'],axes['Y'])
        XY={}
        for name,a,b in [('raw_reference',signals['raw'],expected),('compiled_raw',signals['compiled'],signals['raw']),('compiled_reference',signals['compiled'],expected)]:
            XY[name]=call('signal_gate',label+':'+name,lambda a=a,b=b:signal_gate(a.real,a.imag,b,2e-10))
        metrics=call('metrics',label,lambda:circuit_metrics(compiled));rows=[]
        for m in range(13):
            row=call('round_synthesis',label+':m'+str(m),lambda m=m:synthesis.propagate(report,
                {k:v['report'] for k,v in shells.items()},s=s,m=m,saved_pf_error=verified['rows'][arm,s][m]['pf_operator_error']))
            rows.append(row)
        value=dict(arm=arm,s=s,segment=report,XY=XY,metrics=metrics,rounds=rows,
            CA_achieved=None,shots=None,total_cost=None)
        io.save(output/'candidates'/(label+'.json'),value);return label,value
    results=dict(parallel(candidate,JOBS));comparisons=[]
    for s in (1,4):
        comparisons.append(call('cost_comparison','s'+str(s),lambda s=s:compare_metrics(
            results[f'CDF_s{s}']['metrics'],results[f'XDF_s{s}']['metrics'],'CDF8/XDF8 same s')))
    return dict(status='H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE',local_reports=local_reports,
        shells={k:v['report'] for k,v in shells.items()},shell_metrics=shell_metrics,
        candidates=results,comparisons=comparisons,CA_achieved=None,shots=None,total_cost=None,
        rigorous_enclosure=None,mandatory_stop=True)


def run(root,path,input_path,output):
    from threadpoolctl import threadpool_limits
    root=Path(root).resolve();c=preflight(root,path,input_path);output=fresh_output(root,output)
    output.mkdir(parents=True,exist_ok=False)
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        cache=root/'.cache/h4-native-fair'/leaf;cache.mkdir(parents=True,exist_ok=True);os.environ[key]=str(cache)
    for name in ('circuits','operators','local','candidates'):(output/name).mkdir()
    call=integrated.Calls(output,LIMITS,MAX_CALLS);plan=integrated.resource_plan(task_count=4,full_dimension=256)
    plan['local_factor_workers']=integrated.resource_plan(task_count=18,full_dimension=256)['workers']
    start,cpu=time.monotonic(),time.process_time()
    try:
        io.save(output/'contract.json',c)
        io.save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),execution_commit=
            subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        io.save(output/'resource_plan.json',plan)
        data=load_data(root,c,input_path,call);verified=call('identity_gate','saved DF/PF/ground identity',lambda:identity(data,c))
        io.save(output/'saved_identity.json',{k:v for k,v in verified.items() if k!='rows'})
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            models=prepare_models(data,c,call)
            result=execute_pipeline(data,models,verified,call,output,plan)
        io.save(output/'result.json',result);return result
    except BaseException as e:
        call.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=call.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=call.stop.is_set(),retry=0,resume=0,
            scientific_units=sum(call.counts.values()),call_counts=dict(call.counts)))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
