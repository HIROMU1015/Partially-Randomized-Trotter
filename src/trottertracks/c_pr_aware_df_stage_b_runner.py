"""Dormant B1-only paired-CDF mechanism pilot; no native, sampling or shots."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
import ast
import importlib.metadata
import math
from pathlib import Path
import resource
import struct
import subprocess
import time
from types import SimpleNamespace
import zipfile

from . import c_pr_aware_df_stage_b as b
from . import c1_stage1_execution as io
from . import c1_molecule_extension as molecule
from . import c1_stage3_integrated as integrated
from . import c1_stage3_feasibility as feasibility

SCOPE='C_PR_AWARE_DF_STAGE_B1_MECHANISM_V1'
FAMILY='artifacts/research_c_pr_aware_df_stage_b'
FRESH=FAMILY+'/2026-10-27/B1-run01'
STAGE_A='artifacts/research_c_pr_aware_df/2026-10-27/run01'
POLICY=dict(norb=4,rank=8,Nalpha=2,Nbeta=2,sector_dimension=36,ld=2,delta='1/10',s=4,r=16,K=0,
    rounds=list(range(13)),CA_Ha=.00159,DF_reserve_Ha=.00019875,beta_total=math.pi/6,
    beta_stat=math.pi/12,alpha_total=.01,alpha_axis=.01/26,oracle='own_DF_ground_state_oracle',
    state_preparation='EXCLUDED',law='ideal_finite_RTE_IID',frame_order=list(range(8)),
    new_fit=0,optimizer=0,sampling_samples=0,native_build=0,compile=0,shots=0,resources=0,retry=0,resume=0)
ALLOWANCES=dict(component=1e-10,ground_residual=1e-10,normalization=1e-12,mass_relative=1e-12,
    mass_absolute=1e-14,alias_window_Ha=[-10.,10.],rtol=0,rigorous_enclosure=None,
    interpretation='binary64 numerical pilot; reuse Stage A DF norm and engineering guard')
LIMITS=dict(input_load=1,stage_A_evidence=1,sector_context=1,physical_operator=1,fit_load=2,
    native_model=2,raw_factor=18,factor_representation=18,arm_sum=2,tail_law=2,tail_gate=2,
    mass_gate=2,ground_eigenpair=3,ground_alias=3,kernel_reference=2,exact_evolution=26,
    round_reference=26,round_classification=26,arm_summary=2,comparison=1)
MAX_CALLS=sum(LIMITS.values())


def export_headers(path):
    """NPY metadata only: never read the saved J/U/x/jac array payload."""
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                magic=f.read(8)
                if magic[:6]!=b'\x93NUMPY' or magic[6:] not in (b'\x01\x00',b'\x02\x00'):raise ValueError('NPY_HEADER_VERSION')
                width=2 if magic[6]==1 else 4
                length=struct.unpack('<H' if width==2 else '<I',f.read(width))[0]
                header=ast.literal_eval(f.read(length).decode('latin1'))
            result[name]=dict(shape=list(header['shape']),dtype=header['descr'],fortran_order=header['fortran_order'])
    expected={k:dict(shape=v,dtype='<f8',fortran_order=False) for k,v in dict(J=[8,4,4],U=[8,4,4],x=[128],jac=[128]).items()}
    if result!={k+'.npy':v for k,v in expected.items()}:raise ValueError('SAVED_FINAL_EXPORT_SCHEMA')
    return result


def saved_gate(c,root):
    result,stop,source=(io.read(Path(root)/c['stage_A_records'][k]) for k in ('result','STOP','SOURCE'))
    if (result['status']!='STAGE_A_PROXY_AND_DF_PASS' or stop['failed'] or stop['scientific_units']!=30
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or source['execution_commit']!='e1085c95f5c01b68566e17d543ace5832cb94061'):
        raise ValueError('STAGE_A_PROVENANCE')
    bounds={}
    for role in b.ARMS:
        row=result['sector'][role];d=result['fit_diagnostics'][role]
        if (not d['optimizer']['success'] or row['actual_rank']!=8 or row['guard_policy']!=molecule.GUARD_POLICY):raise ValueError('STAGE_A_ROLE_OR_DF')
        value=row['sector_error']+row['numerical_guard']
        if not math.isfinite(value) or not 0<=value<=POLICY['DF_reserve_Ha']:raise ValueError('SAVED_DF_BUDGET')
        bounds[role]=value
    return result,bounds


def preflight(root,path,input_path):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_PROPOSAL' or c['policy']!=POLICY
            or c['allowances']!=ALLOWANCES or c['call_limits']!=LIMITS
            or c['maximum_logical_units']!=MAX_CALLS or c['fresh_output']!=FRESH
            or c['adapter']!=b.VERSION or c['native_policy']!=b.NATIVE_POLICY):raise ValueError('STAGE_B1_SCOPE')
    if io.sha(input_path)!=c['original_input']['sha256']:raise ValueError('ORIGINAL_INPUT')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_OR_INPUT: '+name)
    required={c['stage_A_input'],*c['final_exports'].values(),*c['stage_A_records'].values()}
    if (set(c['final_exports'])!=set(b.ARMS) or set(c['stage_A_records'])!={'result','STOP','SOURCE'}
            or not required<=c['saved_inputs'].keys()):raise ValueError('UNPINNED_STAGE_A_INPUT')
    for role in b.ARMS:export_headers(root/c['final_exports'][role])
    saved_gate(c,root)
    for name,want in c['evaluation_packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT: '+name)
    return c


def load(path,keys):
    import numpy as np
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in keys}


def save_arrays(path,**values):
    import numpy as np
    with Path(path).open('xb') as f:np.savez(f,**values)


def ground(H,output,label):
    from .c1_stage3_molecular import MolecularBackend
    helper=SimpleNamespace(output=output,tol={'operator_component_abs':ALLOWANCES['component']})
    return MolecularBackend._ground(helper,H,label)


def ground_gate(H,pair):
    import numpy as np
    values,vectors=pair;psi=vectors[:,0]
    residual=float(np.linalg.norm(H@psi-values[0]*psi));normalization=float(abs(np.vdot(psi,psi)-1))
    lo,hi=ALLOWANCES['alias_window_Ha'];margin=math.pi-(max(abs(lo),abs(hi))*float(Fraction(POLICY['delta']))+POLICY['beta_total'])
    if (not np.isfinite(values).all() or not np.isfinite(vectors).all() or residual>ALLOWANCES['ground_residual']
            or normalization>ALLOWANCES['normalization']):raise ValueError('GROUND_CORRECTNESS')
    if not lo<=values[0]<=hi or margin<=0:raise ValueError('UNRESOLVED_GROUND_ALIAS')
    return dict(status='PASS',energy_Ha=float(values[0]),residual=residual,normalization_error=normalization,
        alias_margin=margin,state='own_DF_ground_state_oracle',physical_ground_overlap=None)


def run(root,path,input_path,output):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from .c1_stage3_accuracy import AccuracyTarget
    root=Path(root).resolve();c=preflight(root,path,input_path);output=Path(output).resolve()
    if output!=root/FRESH or output.exists():raise ValueError('FRESH_B1_NO_RETRY_RESUME')
    output.mkdir(parents=True);call=integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        io.save(output/'contract.json',c);io.save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        p=call('input_load','original h/g/constant saved by Stage A',lambda:load(root/c['stage_A_input'],('h','g','constant')))
        if (p['h'].shape!=(4,4) or p['g'].shape!=(4,)*4 or p['constant'].shape!=()):raise ValueError('SAVED_INPUT_SHAPE')
        b.stage1.finite_arrays(*p.values())
        evidence,bounds=call('stage_A_evidence','same saved final J/U and DF bounds',lambda:saved_gate(c,root))
        ctx=call('sector_context','spin sector36',lambda:molecule.SectorCAR(4))
        physical=call('physical_operator','original physical sector, no new DF norm',lambda:ctx.physical(p['h'],p['g'],float(p['constant'])))
        pair=call('ground_eigenpair','physical',lambda:ground(physical,output,'physical'))
        io.save(output/'physical_ground.json',call('ground_alias','physical',lambda:ground_gate(physical,pair)))
        plan=integrated.resource_plan(task_count=2,full_dimension=36);io.save(output/'resource_plan.json',plan)
        target=AccuracyTarget(POLICY['CA_Ha'],1/8,.1,POLICY['beta_total'],POLICY['beta_stat'],POLICY['alpha_axis'],POLICY['alpha_total'])
        prepared={}
        # Correctness gates complete before independent round evaluation starts.
        for role in b.ARMS:
            data=call('fit_load',role,lambda role=role:load(root/c['final_exports'][role],('J','U')))
            if data['J'].shape!=(8,4,4) or data['U'].shape!=(8,4,4):raise ValueError('SAVED_CDF_RANK_SHAPE')
            arm=call('native_model',role,lambda:b.prepare_arm(p['h'],p['g'],p['constant'],**data,
                role=role,source_identity=c['saved_inputs'][c['final_exports'][role]],input_identity=c['original_input']['sha256'],ld=2))
            G=[];C=[]
            for i in range(9):
                G.append(call('raw_factor',f'{role}:{i}',lambda i=i:ctx.one(b.stage1.corrected_kappa(p['h'],p['g']))+float(p['constant'])*np.eye(ctx.d) if i==0 else ctx.raw_frame(data['J'][i-1],data['U'][i-1])))
                def check(i=i):
                    candidate=b.candidate_factor(ctx,arm,i)
                    b.component_check(G[i],G[i].conj().T,ALLOWANCES['component'],label='factor Hermitian')
                    report=b.component_check(candidate,G[i],ALLOWANCES['component'],label='general-K / raw CAR factor')
                    return candidate,report
                matrix,report=call('factor_representation',f'{role}:{i}',check);C.append(matrix)
                io.save(output/f'{role}_factor_{i}.json',report)
            def total():
                H=sum(G);report=b.component_check(sum(C),H,ALLOWANCES['component'],label='sum general-K/raw')
                return H,report
            H,report=call('arm_sum',role,total);io.save(output/(role+'_sum.json'),report)
            tail,config,distribution=call('tail_law',role,lambda:b.configure(arm,Fraction(1,40),16,0))
            tail_report=call('tail_gate',role,lambda:b.tail_gate(ctx,arm,G,tail))
            def mass_gate():
                saved=evidence['fit_diagnostics'][role]['diagnostic']['exact_tail_mass_Ha']
                if not math.isclose(tail.lambda_r,saved,rel_tol=ALLOWANCES['mass_relative'],abs_tol=ALLOWANCES['mass_absolute']):raise ValueError('SAVED_MASS_LAW_MISMATCH')
                return dict(status='PASS',saved_exact_mass=saved,law_lambda_R=tail.lambda_r)
            io.save(output/(role+'_mass.json'),call('mass_gate',role,mass_gate))
            from .pr_dictionary_validation.prototype import atom_record
            catalog=[dict(component_id=x.component_id,probability=x.probability,coefficient_abs=x.coefficient_abs,
                coefficient_sign=x.coefficient_sign,basis_id=x.basis_id,df_fragment_id=x.df_fragment_id) for x in tail.components]
            io.save(output/(role+'_tail.json'),dict(tail_report,atoms_catalog=[atom_record(x) for x in arm.plan.atoms],
                components=catalog,finite_distribution=None if distribution is None else dict(orders=distribution.orders,
                    order_probabilities=distribution.order_probabilities,normalization=distribution.exact_finite_distribution),
                configuration=None if config is None else dict(h=config.evolution_time,r=config.rte_steps,K=0,
                    tau=config.dimensionless_step_time),native_policy=b.NATIVE_POLICY))
            pair=call('ground_eigenpair',role,lambda:ground(H,output,role))
            io.save(output/(role+'_ground.json'),call('ground_alias',role,lambda:ground_gate(H,pair)))
            kernel=call('kernel_reference',role,lambda:b.kernels(arm,G,Fraction(1,40),config))
            save_arrays(output/(role+'_reference.npz'),generators=np.array(G),H=H,pf=kernel['pf'],corrected=kernel['corrected'],
                full_input_indices=b.stage1.bit_reverse_indices(2*ctx.n)[ctx.sector.basis_indices])
            prepared[role]=(H,pair[1][:,0],kernel)
        def task(role):
            H,psi,kernel=prepared[role];rows=[];decisions=[]
            for m in POLICY['rounds']:
                time=Fraction(1,10)*(1<<m)
                exact=call('exact_evolution',f'{role}:m{m}',lambda:expm(-1j*float(time)*H))
                row=call('round_reference',f'{role}:m{m}',lambda:b.reference.reference_round(kernel,(H,time),repetitions=4*(1<<m),state=psi,exact_operator=exact))
                signal=row.pop('raw_signal',None);row['raw_signal_parts']=None if signal is None else [float(signal.real),float(signal.imag)]
                row['evidence_kind']='binary64_sector_operator_numerical_pilot'
                decision=call('round_classification',f'{role}:m{m}',lambda:feasibility.classify_round(row,target))
                value=dict(m=m,reference=row,feasibility=decision,synthesis_error=None,implemented_law_error=None,shots=None,total_cost=None,CA_achieved=None)
                io.save(output/f'{role}_m{m:02d}.json',value);rows.append(value);decisions.append(decision)
            classification=call('arm_summary',role,lambda:feasibility.summarize(decisions,bounds[role],target))
            result=dict(arm=role,classification=classification,rounds=rows,saved_DF_bound_Ha=bounds[role],
                synthesis_bound=None,implemented_law_bound=None,cost_support=None,shots=None,total_cost=None,CA_achieved=None)
            io.save(output/(role+'_result.json'),result);return result
        results={}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures={pool.submit(task,role):role for role in b.ARMS}
                for future in as_completed(futures):results[futures[future]]=future.result()
        def compare():return dict(status='B1_MECHANISM_EVALUATION_COMPLETE',arms={k:results[k] for k in b.ARMS},
            native_build=0,compile=0,sampling_samples=0,shots=None,total_resources=None,guaranteed_advantage=None,mandatory_stop=True)
        io.save(output/'result.json',call('comparison','two registered CDF arms only',compare))
    except BaseException as error:
        call.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
