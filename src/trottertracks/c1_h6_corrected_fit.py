"""Dormant H6 rank14 single-fit pilot. Separate scientific permission required.

Saved initialization -> codec/tensor/sector gates -> one fixed ffsim fit ->
independent final sector assessment. No rank search or automatic fallback.
"""
from pathlib import Path
import importlib.metadata
import resource
import subprocess
import time

from . import c1_h6_initialization as old
from . import c1_real_orthogonal_codec as codec
from . import c1_ffsim_corrected_codec as bridge
from .c1_stage3_molecular import component_check

io=old.io
SCOPE='C1_H6_RANK14_CORRECTED_CODEC_ONE_FIT_V1'
FAMILY='artifacts/research_c_c1_h6_corrected_fit'
PREPARE_LIMITS=dict(saved_fit_load=1,saved_initial_identity=1,corrected_codec=1,tensor_assessment=2)
FIT_LIMITS=dict(fit_input_load=1,optimized_fitter=1,initial_value_gradient=1,
                final_tensor_assessment=1,final_value_gradient=1,optimizer_objective_check=1)
SECTOR_LIMITS=dict(snapshots_load=2,sector_context=1,physical_sector=1,sector_assessment=3,
                   initial_equivalence_budget_gate=1,final_decision=1)
MAX_CALLS=sum(PREPARE_LIMITS.values())+sum(FIT_LIMITS.values())+sum(SECTOR_LIMITS.values())
ALLOWANCES=dict(component_maxabs=old.ATOL,tensor_frobenius='1e-10*max(1,||g_before||F)',
    objective='1e-12+1e-8*abs(expected)',qualified_x0='exact array equality')


def preflight(root,path,*,fit=False):
    """Only bytes/JSON/version metadata; never decode molecular arrays."""
    root=Path(root);c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['rank']!=14 or c['max_new_fit']!=1 or c['retry'] or c['resume']
            or c['codec']!=codec.VERSION or c['bridge']!=bridge.VERSION
            or c['optimizer_settings']!=dict(bridge.SETTINGS,max_vecs=14)
            or c['DF_reserve_Ha']!=.00159/8 or c['allowances']!=ALLOWANCES
            or c['guard_policy']!=old.molecule.GUARD_POLICY
            or c['prepare_calls']!=PREPARE_LIMITS or c['fit_calls']!=FIT_LIMITS
            or c['sector_calls']!=SECTOR_LIMITS or c['maximum_logical_units']!=MAX_CALLS):
        raise ValueError('CORRECTED_ONE_FIT_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for f,w in io.read(root/c['source_manifest'])['files'].items():
        if io.sha(root/f)!=w:raise ValueError('SOURCE: '+f)
    for f,w in c['saved_inputs'].items():
        if io.sha(root/f)!=w:raise ValueError('SAVED_INPUT: '+f)
    # Bind every numerical input path to the verified input inventory.
    for key in ('original_input','XDF_full','initial_arrays'):
        if c[key] not in c['saved_inputs']:raise ValueError('UNPINNED_INPUT: '+key)
    for k,w in c['fit_packages' if fit else 'evaluation_packages'].items():
        if importlib.metadata.version(k)!=w:raise ValueError('ENVIRONMENT: '+k)
    if fit:
        import ffsim
        for f,w in c['ffsim_sources'].items():
            if io.sha(Path(ffsim.__file__).parent/f)!=w:raise ValueError('FFSIM_SOURCE: '+f)
    return c


def finish(output,calls,start):
    counts=dict(calls.counts);failed=calls.stop.is_set()
    for name in ('prepare','fit'):
        child=output/name/'STOP.json'
        if child.exists():
            value=io.read(child);failed|=value['failed']
            for k,v in value['call_counts'].items():counts[k]=counts.get(k,0)+v
    io.save(output/'STOP.json',dict(mandatory_stop=True,failed=failed,retry=0,resume=0,
        scientific_units=sum(counts.values()),call_counts=counts))
    io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,
        process_lifetime_CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


def load_npz(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def prepare_worker(root,output,c):
    """Saved-only qualification exports. Optimizer invocation forbidden."""
    import scipy.optimize
    from unittest.mock import patch
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False)
    calls=old.Calls(output,PREPARE_LIMITS,sum(PREPARE_LIMITS.values()));start=time.monotonic()
    try:
        def load():return load_npz(root/c['original_input']),load_npz(root/c['initial_arrays']),load_npz(root/c['XDF_full'])
        packet,initial,full=calls('saved_fit_load','saved g/XDF14/full',load)
        J,U=initial['J'],initial['U']
        if len(J)!=c['rank']:raise ValueError('INITIAL_ACTUAL_RANK')
        identity=calls('saved_initial_identity','XDF14/full prefix',lambda:old.initialization_identity(J,U,full['J'][:c['rank']],full['U'][:c['rank']]))
        io.save(output/'identity.json',identity)
        with patch.object(scipy.optimize,'minimize',side_effect=AssertionError('PREPARE_OPTIMIZER_FORBIDDEN')):
            J2,U2,x,report,raw=calls('corrected_codec','saved XDF14',lambda:codec.roundtrip(J,U))
        io.save(output/'codec.json',report)
        old._save_arrays(output/'parameters.npz',x=x,**raw)
        for state,a,b in [('before',J,U),('initial',J2,U2)]:
            old._save_arrays(output/(state+'.npz'),J=a,U=b)
            io.save(output/(state+'_tensor.json'),calls('tensor_assessment',state,lambda a=a,b=b:old.tensor_metrics(packet['g'],a,b)))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:finish(output,calls,start)


def fit_worker(root,output,c):
    """Dormant scientific worker: exactly one fixed fitter, no retry."""
    import numpy as np
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False)
    calls=old.Calls(output,FIT_LIMITS,sum(FIT_LIMITS.values()));start=time.monotonic()
    try:
        def load():
            return load_npz(root/c['original_input']),load_npz(root/c['initial_arrays']),load_npz(output.parent/'prepare/parameters.npz')['x']
        packet,initial,x=calls('fit_input_load','fixed g/J/U/qualified x0',load)
        # Sector gate is a numerical prerequisite, not proof of user permission.
        if io.read(output.parent/'initial_sector_gate.json')['status']!='PASS':raise ValueError('INITIAL_SECTOR_GATE')
        initial_metric=io.read(output.parent/'prepare/initial_tensor.json');observed={}
        def before(fun,x0,state):
            def assess():
                value,gradient=fun(x0)
                old.molecule.stage1.finite_arrays(value,gradient)
                old.check_objective(value,initial_metric['objective_from_tensor'])
                old._save_arrays(output/'initial_parameters.npz',x=x0,gradient=gradient)
                io.save(output/'initial_value_gradient.json',dict(objective=float(value),gradient_l2=float(np.linalg.norm(gradient)),gradient_infinity=float(np.max(abs(gradient)))))
                io.save(output/'fitter_codec.json',state['codec'])
            calls('initial_value_gradient','actual scipy_func at qualified x0',assess)
        def after(fun,result):
            # Preserve actual optimizer x/jac/status before independent diagnostics.
            old._save_arrays(output/'optimizer_parameters.npz',x=result.x,jac=result.jac)
            observed.update(success=bool(result.success),status=int(result.status),message=str(result.message),
                fun=float(result.fun),nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev))
            io.save(output/'optimizer.json',observed)
            def assess():
                value,gradient=fun(result.x);old.molecule.stage1.finite_arrays(value,gradient,result.jac)
                old.check_objective(value,result.fun)
                old._save_arrays(output/'final_parameters.npz',x=result.x,gradient=gradient)
                io.save(output/'final_value_gradient.json',dict(objective=float(value),gradient_l2=float(np.linalg.norm(gradient)),gradient_infinity=float(np.max(abs(gradient))),
                    optimizer_jac_maxabs_difference=float(np.max(abs(gradient-result.jac)))))
            calls('final_value_gradient','actual scipy_func at returned x',assess)
        J,U,result=calls('optimized_fitter','one rank14 ffsim call',lambda:bridge.fit(
            packet['g'],initial['J'],initial['U'],x,before_optimizer=before,after_optimizer=after))
        old._save_arrays(output/'final.npz',J=J,U=U)
        def tensor():
            if (J.shape!=initial['J'].shape or U.shape!=J.shape or not np.array_equal(J,J.transpose(0,2,1))
                    or np.max(abs(U.transpose(0,2,1)@U-np.eye(len(U[0]))))>old.ATOL):
                raise ValueError('FINAL_DF_SCHEMA')
            return old.tensor_metrics(packet['g'],J,U)
        metrics=calls('final_tensor_assessment','returned ffsim J/U',tensor)
        io.save(output/'final_tensor.json',metrics)
        io.save(output/'objective_check.json',calls('optimizer_objective_check','returned tensor/optimizer fun',lambda:old.check_objective(result.fun,metrics['objective_from_tensor'])))
        io.save(output/'result.json',dict(optimizer=observed,fit_invocations=1,retry=0,resume=0))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:finish(output,calls,start)


def decision(before,initial,final,optimizer,before_tensor,initial_tensor,final_tensor,budget):
    """Keep optimizer success, tensor improvement and physical budget separate."""
    budget_pass=final['sector_error']+final['numerical_guard']<=budget
    tensor_change=final_tensor['objective_from_tensor']-initial_tensor['objective_from_tensor']
    sector_change=final['sector_error']-initial['sector_error']
    def direction(value):return 'DECREASED' if value<0 else 'INCREASED' if value>0 else 'UNCHANGED'
    return dict(status=('OPTIMIZER_NOT_SUCCESS' if not optimizer['success'] else
        'FIT_NUMERICAL_PILOT_PASS' if budget_pass else 'DF_BUDGET_NOT_MET'),
        optimizer_success=optimizer['success'],DF_budget_status='PASS' if budget_pass else 'NOT_MET',
        before=before,initial=initial,final=final,optimizer=optimizer,
        tensor_objective_change=tensor_change,tensor_change=direction(tensor_change),
        sector_error_change_Ha=sector_change,sector_change=direction(sector_change),
        changes_are_numerical_point_comparisons=True,
        tensor_before=before_tensor,tensor_initial=initial_tensor,tensor_final=final_tensor,
        selected_initial_as_fallback=False,rigorous_enclosure=None,CA_achieved=None,
        fit_invocations=1,retry=0,resume=0,mandatory_stop=True)


def run(root,contract_path,output,fit_python):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root,output=Path(root),Path(output);c=preflight(root,contract_path)
    output.mkdir(parents=True,exist_ok=False);calls=old.Calls(output,SECTOR_LIMITS,sum(SECTOR_LIMITS.values()));start=time.monotonic()
    io.save(output/'contract.json',c)
    def worker(mode,name):
        subprocess.run([str(Path(fit_python).absolute()),str(root/c['entry']),'--contract',
            str(Path(contract_path).absolute()),mode,'--output',str(output/name)],check=True)
    try:
        worker('--prepare-worker','prepare')
        def load():return load_npz(root/c['original_input']),{s:load_npz(output/'prepare'/(s+'.npz')) for s in ('before','initial')}
        packet,states=calls('snapshots_load','original and qualified initial arrays',load)
        ctx=calls('sector_context','sector400',lambda:old.molecule.SectorCAR(len(packet['h'])))
        physical=calls('physical_sector','original-g Hamiltonian',lambda:ctx.physical(packet['h'],packet['g'],packet['constant']))
        plan=old.resource_plan(task_count=2,full_dimension=ctx.d);io.save(output/'resource_plan.json',plan)
        def assess(item):
            name,data=item
            row,G=calls('sector_assessment',name,lambda:old.molecule.rank_assessment(ctx,packet,data['J'],data['U'],physical,c['rank']))
            row['status']='SECTOR_EVALUATED'  # not a claim about optimizer success
            io.save(output/(name+'_sector.json'),row);return name,(row,sum(G))
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:rows=dict(pool.map(assess,states.items()))
        def gate():
            a,b=rows['before'],rows['initial']
            check=component_check(a[1],b[1],old.ATOL,label='initial codec sector equivalence')
            if any(row['sector_error']+row['numerical_guard']>c['DF_reserve_Ha'] for row in (a[0],b[0])):
                raise ValueError('INITIAL_DF_BUDGET_NOT_PRESERVED')
            return dict(status='PASS',operator=check,operator_difference_norm=float(np.linalg.norm(a[1]-b[1],2)))
        io.save(output/'initial_sector_gate.json',calls('initial_equivalence_budget_gate','before optimizer',gate))
        worker('--fit-worker','fit')
        final=calls('snapshots_load','fit final J/U',lambda:load_npz(output/'fit/final.npz'))
        # One sector matrix only; reuse the same context and original Hamiltonian.
        with threadpool_limits(limits=1 if plan['workers']>1 else None):row,_=assess(('final',final))[1]
        result=calls('final_decision','optimizer and DF budget independent',lambda:decision(
            rows['before'][0],rows['initial'][0],row,io.read(output/'fit/optimizer.json'),
            io.read(output/'prepare/before_tensor.json'),io.read(output/'prepare/initial_tensor.json'),
            io.read(output/'fit/final_tensor.json'),c['DF_reserve_Ha']))
        io.save(output/'result.json',result)
        if result['status']!='FIT_NUMERICAL_PILOT_PASS':calls.stop.set()
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:finish(output,calls,start)
