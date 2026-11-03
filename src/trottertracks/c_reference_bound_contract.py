"""B2 state-independent, reference-only bound contract; passive on import.

The exact saved-input atom map, mathematical CAR map and binary64 CAR replay
are three distinct targets. No ground, native, PF-error or accuracy solver.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import multiprocessing
import os
import re
import subprocess
import time

from . import c1_stage1_execution as io
from . import c_reference_numerical_uncertainty as old

BASE = '2ac4a38dced97406d15fcd157e9d7a0ec7aba982'
FAMILY = 'artifacts/research_c_reference_bound_contract/2026-11-03'
CONTRACT = 'docs/research/research_c_reference_bound_contract_unapproved_20261103.json'
ENTRY = 'scripts/run_research_c_reference_bound.py'
PRECISION = 384
LIMITS = dict(metadata_load=1, identity_gate=1, reference_arrays=1, sector_context=1,
              atom_context=1, atom_mean=1, head_exponential=3, head_products=1,
              car_tail=1, kernel_pair=1, power_chains=1, round_bound=13, summary=1)
MAX_CALLS = sum(LIMITS.values())
TARGETS = dict(
    atom='exact saved atom coefficients,dyadic saved U,saved angle bits;exact p;literal CAR Z/ZZ',
    CAR='exact dyadic saved sector generators and lambda/tau/identity;h=1/10;K0 polynomial;A=(1+tau^2)^(-16*q)',
    numeric='binary64 saved corrected matrix;shared NumPy power chain;each saved binary64 attenuation',
    state='all normalized sector36 vectors;no saved-state expectation or oracle-precision assertion')
METADATA = dict(
    atoms='artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/atoms.json',
    table='artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/categorical_table.json',
    mean='artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/ideal_event_mean.json',
    law='artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json',
    normalization='artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/normalization.json')
ARRAYS = dict(candidate='artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz',
    factors='artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz',
    CAR='artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/sector_reference.npz')


def fixed():
    return dict(scope='C_B2_STATE_INDEPENDENT_REFERENCE_BOUNDS_V1', base_commit=BASE,
                status='UNAPPROVED_EXECUTABLE_REFERENCE_ONLY', executable=True,
                execution_authorized=False, candidate_sha256=old.integration.identity()['candidate'],
                stage_A_scientific_SOURCE=old.integration.STAGE_A_SOURCE,
                stage_A_result=old.integration.STAGE_A, model=deepcopy(old.integration.MODEL),
                conditions=old.integration.identity()['conditions'], targets=TARGETS,
                metadata=METADATA, arrays=ARRAYS,
                dimensions=dict(spatial=4, modes=8, sector=36, rank=8, atoms=216, ld=2, r=32),
                precision_bits=PRECISION, FLINT='3.3.1', call_limits=LIMITS,
                maximum_logical_units=MAX_CALLS, rounds=list(range(13)),
                fresh_output=FAMILY+'/run01', retry=0, resume=0, mandatory_stop=True,
                entry=ENTRY, Stage_C='NOT_READY_B1_AND_TRAJECTORY_PREMISES',
                scientific_input_uncertainty_covered=False, saved_ground_precision_qualified=False,
                generator_construction_uncertainty_covered=False,
                physical_CA_qualified=False, new_ground_solve=0, new_native_compile=0,
                new_PF_error_evaluation=0, new_sampling=0, new_shots=0,
                precision_fallback=False, resource_caps=dict(CPU=None,RAM=None,workers=None,wall=None))


def metadata(root, c):
    root = Path(root)
    return {key:io.read(root/path) for key,path in c['metadata'].items()}


def bind(d, c):
    """Saved JSON and exact rational metadata identity, not a new fit/law."""
    atoms = d['atoms']['atoms']; table = d['table']; dim = c['dimensions']
    coeff = [F(a['coefficient']) for a in atoms]
    probabilities = [F(x) for x in table['ideal_probabilities']]
    mass = sum(map(abs,coeff))
    norm = d['normalization']; law = d['law']; means = d['mean']['report']
    if (len(atoms) != dim['atoms'] or len({a['atom_id'] for a in atoms}) != len(atoms)
            or any(x == 0 for x in coeff)
            or mass <= 0 or probabilities != [abs(x)/mass for x in coeff]
            or norm['r'] != dim['r'] or norm['tau'] <= 0 or norm['lambda_R'] <= 0
            or law['lambda_R'] != norm['lambda_R'] or law['atoms'] != len(atoms)
            or means['angle'] <= 0 or not means['scalar_phase_excluded']
            or [x['m'] for x in norm['rows']] != list(range(13))):
        raise ValueError('SAVED_ATOM_CONFIG_IDENTITY')
    for a in atoms:
        frame = int(a['frame_id'].rsplit('F',1)[1])
        if (not dim['ld'] <= frame < dim['rank'] or len(a['support']) not in (1,2)
                or len(set(a['support'])) != len(a['support'])
                or any(type(i) is not int or not 0 <= i < dim['modes'] for i in a['support'])):
            raise ValueError('FIXED_ATOM_SUPPORT_FRAME')
    # No atan/normalization recalculation: the atom target has these frozen bits.
    return dict(status='SAVED_INPUT_TARGETS_BOUND', atoms=len(atoms), targets=TARGETS)


def arrays(root, c):
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load
    root = Path(root); p = c['arrays']; dim = c['dimensions']
    candidate = load(root/p['candidate'],('J','U'))
    saved = load(root/p['factors'],('J','U','generators'))
    saved['corrected'] = load(root/p['CAR'],('corrected',))['corrected']
    n,s,r = dim['spatial'],dim['sector'],dim['rank']
    if (saved['J'].shape != (r,n,n) or saved['U'].shape != (r,n,n)
            or saved['generators'].shape != (r+1,s,s) or saved['corrected'].shape != (s,s)
            or not all(np.isfinite(v).all() for v in saved.values())
            or not all(np.array_equal(candidate[k],saved[k]) for k in ('J','U'))):
        raise ValueError('FIXED_REFERENCE_ARRAYS')
    return saved


def atom_context(ctx, saved, d, c):
    import numpy as np
    from .c_pr_aware_df_fixed_car import Arithmetic, Matrix
    from .c_pr_aware_df_stage_b import atom_operators
    from . import c_pr_aware_df_zsxx_enclosure as z
    atoms = tuple(NS(atom_id=x['atom_id'],frame_id=x['frame_id'],
                     support=tuple(x['support']),coefficient=F(x['coefficient'])) for x in d['atoms']['atoms'])
    frames = {a.frame_id:np.kron(saved['U'][int(a.frame_id.rsplit('F',1)[1])],np.eye(2)) for a in atoms}
    car = NS(d=ctx.d,bilinears={k:Matrix(z._matrix(v.toarray())) for k,v in ctx.bilinears.items()})
    arm = NS(model=NS(modes=c['dimensions']['modes']),plan=NS(atoms=atoms,frames=frames))
    return atom_operators(car,arm,arithmetic=Arithmetic())


def atom_mean(operators, d):
    events = [NS(taylor_order=0,phase=1,product_component_ids=(),
                 rotation_component_id=x['atom_id'],rotation_angle=d['mean']['report']['angle'])
              for x in d['atoms']['atoms']]
    return old.weighted_events(events,operators,d['table']['ideal_probabilities']).ball


def _head_task(task):
    from flint import ctx
    from .c_pr_aware_df_fixed_car import Arithmetic, Matrix, pack
    from . import c_pr_aware_df_zsxx_enclosure as z
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        return pack(Arithmetic.exponential(F(1,20),Matrix(z._matrix(task))).ball)


def parallel_calls(call, kind, tasks, function, *, parallel):
    """Register in the parent before dispatch; independent process-local Arb."""
    from .c1_stage3_integrated import resource_plan
    plan = resource_plan(len(tasks),full_dimension=36)
    if not parallel or plan['workers'] <= 1:
        return [call(kind,str(i),lambda task=task:function(task)) for i,task in enumerate(tasks)]
    with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn')) as processes:
        def one(pair):
            i,task = pair
            return call(kind,str(i),lambda:processes.submit(function,task).result())
        with ThreadPoolExecutor(max_workers=plan['workers']) as threads:
            return list(threads.map(one,enumerate(tasks)))


def head_products(packed, size):
    from .c_pr_aware_df_fixed_car import Arithmetic, Matrix, unpack
    from .c1_stage3_reference import _ordered_exponentials
    factors = [Matrix(unpack(p,size)) for p in packed]
    cache = {(i,F(1,20)):a for i,a in enumerate(factors)}
    schedule = tuple((i,F(1,20)) for i in range(len(factors)))
    # Existing order convention and cache, with no second exponentiation.
    return (_ordered_exponentials(factors,schedule,cache,arithmetic=Arithmetic()).ball,
            _ordered_exponentials(factors,tuple(reversed(schedule)),cache,arithmetic=Arithmetic()).ball)


def power(a, n):
    from flint import acb_mat
    if type(n) is not int or n < 1: raise ValueError('POSITIVE_POWER')
    result = acb_mat([[int(i==j) for j in range(a.nrows())] for i in range(a.nrows())])
    while n:
        if n&1: result = a*result
        n //= 2
        if n: a = a*a
    return result


def car_tail(saved, d, c):
    from flint import acb
    from .c_pr_aware_df_fixed_car import Arithmetic
    from . import c_pr_aware_df_zsxx_enclosure as z
    eye = Arithmetic.eye(c['dimensions']['sector']).ball
    total = sum((z._matrix(x) for x in saved['generators'][c['dimensions']['ld']+1:]),0*eye)
    total -= z._scale(eye,z._real(d['law']['identity']))
    ratio = z._real(d['normalization']['tau'])/z._real(d['normalization']['lambda_R'])
    return power(eye-z._scale(total,acb(0,1)*ratio),c['dimensions']['r'])


def kernel_pair(mean, tail, heads, identity, r):
    from flint import acb
    from . import c_pr_aware_df_zsxx_enclosure as z
    phase = (-acb(0,1)*z._real(F(1,10))*z._real(identity)).exp()
    left,right = heads
    return dict(atom=z._scale(right*power(mean,r)*left,phase),
                CAR_corrected=z._scale(right*tail*left,phase))


def power_tasks(kernels, corrected, d, r):
    """Two Arb chains and one NumPy chain; 12 dependent squares each."""
    import numpy as np
    from .c_pr_aware_df_fixed_car import pack
    a,b,n = kernels['atom'],kernels['CAR_corrected'],np.array(corrected,copy=True)
    tasks = []
    for m,row in enumerate(d['normalization']['rows']):
        # eta at each round includes the actual binary64 scaling operation.
        replay = row['attenuation']*n
        if not np.isfinite(replay).all(): raise ValueError('NUMERIC_REFERENCE_OVERFLOW')
        tasks.append(dict(m=m,atom=pack(a),CAR=pack(b),numeric=replay,
                          tau=d['normalization']['tau'],r=r,size=len(n)))
        if m < 12:
            a,b,n = a*a,b*b,np.linalg.matrix_power(n,2)
    return tasks


def _round_task(task):
    from flint import ctx
    from .c_pr_aware_df_fixed_car import unpack
    from . import c_pr_aware_df_zsxx_enclosure as z
    m = task['m']; size = task['size']
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        a,b = unpack(task['atom'],size),unpack(task['CAR'],size)
        # Real mathematical normalization; no log/exp evaluation assumptions.
        attenuation = (1+z._real(task['tau'])**2)**(-task['r']*(1<<m)//2)
        car = z._scale(b,attenuation); numeric = z._matrix(task['numeric'])
        bridge = z._norm(a-car); uncertainty = z._norm(car-numeric)
        direct = z._norm(a-numeric)
        return dict(m=m,bridge=bridge,reference_uncertainty=uncertainty,
                    total_upper_exact=str(F(bridge['upper_exact'])+F(uncertainty['upper_exact'])),
                    direct_total_diagnostic=direct,
                    norm_conversion='operator2<=outward_Frobenius;|psi*D psi|<=operator2 for every normalized psi',
                    rounding_enclosed=True,saved_state_used=False)


def records(rows):
    if [r['m'] for r in rows] != list(range(13)): raise ValueError('THIRTEEN_BOUND_ROWS')
    result = {}
    for quantity,key in [('reference_bridge','bridge'),('reference_uncertainty','reference_uncertainty')]:
        # The old state-specific converter requires an oracle-vector distance.
        # Construct the distinct universal-state mode explicitly, without
        # setting that premise true or relabelling an unknown old record.
        record = dict(identity=old.integration.identity(),quantity=quantity,norm='raw_complex_signal',
            space='sector36_reference',phase_sensitive=True,coherent_model='independent_IID_kernels',
            scope='model_reference',reference_target='saved_CAR_raw_finite_RTE',
            qualification='bound',numerical_qualification='rigorous_enclosure',
            included_error_terms=['atom_to_mathematical_CAR'] if key=='bridge' else ['mathematical_CAR_to_binary64_reference'],
            raw_conversion='uniform over all normalized sector36 states;no saved-state approximation',
            error_model=dict(targets=TARGETS,precision_bits=PRECISION,rounding_enclosed=True,
                saved_state_used=False,ground_oracle_precision_claimed=False,
                physical_input_error_included=False,generator_construction_error_included=False),
            unresolved=[],proofs_verified_by_schema=False,rows=[],
            provenance='B2 state-independent Arb384 reference contract',
            actual_CA_achieved=None,PCG64_joint_stream_qualified=False,mandatory_stop=True)
        for row in rows:
            if not row.get('rounding_enclosed') or row.get('saved_state_used') is not False:
                raise ValueError('UNIFORM_STATE_ROUNDING_COVERAGE')
            bound = F(row[key]['upper_exact'])
            if bound < 0: raise ValueError('NONNEGATIVE_BOUND')
            record['rows'].append(dict(m=row['m'],quantity=quantity,norm='raw_complex_signal',
                value=str(bound),bound=str(bound),kind='rigorous_bound',rounding_enclosed=True,
                provenance=record['provenance'],
                **({'reference_uncertainty_included':False} if key=='bridge' else {})))
        result[quantity] = record
    return result


def execute(root,c,out,call,*,parallel=True):
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    from .c1_molecule_extension import SectorCAR
    from .c1_stage3_integrated import resource_plan
    from threadpoolctl import threadpool_limits
    d = call('metadata_load','fixed saved JSON',lambda:metadata(root,c))
    call('identity_gate','same frozen literal targets',lambda:bind(d,c))
    saved = call('reference_arrays','candidate/factors/corrected only;no ground',lambda:arrays(root,c))
    car = call('sector_context','existing integer excitation table',lambda:SectorCAR(c['dimensions']['spatial']))
    if car.d != c['dimensions']['sector']: raise ValueError('SECTOR_DIMENSION')
    z.runtime_model()
    with z._BALL_LOCK,ctx.workprec(PRECISION),threadpool_limits(limits=1,user_api='blas'):
        operators = call('atom_context','existing CAR signed Z/ZZ with Arb',lambda:atom_context(car,saved,d,c))
        mean = call('atom_mean','all saved K0 atoms;exact p',lambda:atom_mean(operators,d))
    packed = parallel_calls(call,'head_exponential',list(saved['generators'][:c['dimensions']['ld']+1]),_head_task,parallel=parallel)
    with z._BALL_LOCK,ctx.workprec(PRECISION),threadpool_limits(limits=1,user_api='blas'):
        heads = call('head_products','same forward/reverse source order',lambda:head_products(packed,car.d))
        tail = call('car_tail','K0 corrected polynomial on saved G',lambda:car_tail(saved,d,c))
        kernels = call('kernel_pair','same scalar phase;two exact maps',lambda:kernel_pair(mean,tail,heads,d['law']['identity'],c['dimensions']['r']))
        tasks = call('power_chains','two Arb plus one binary64 chain',lambda:power_tasks(kernels,saved['corrected'],d,c['dimensions']['r']))
    rows = parallel_calls(call,'round_bound',tasks,_round_task,parallel=parallel)
    result = call('summary','B2 only;no accuracy call',lambda:dict(status='B2_REFERENCE_OPERATOR_BOUNDS_COMPLETE',
        rounds=rows,records=records(rows),targets=TARGETS,model=c['model'],
        resource_plan=dict(head=resource_plan(3,36),rounds=resource_plan(13,36)),
        Stage_C_ready=False,categorical_unitary_trajectory_premise_qualified=False,
        scientific_input_uncertainty_covered=False,saved_ground_precision_qualified=False,
        generator_construction_uncertainty_covered=False,
        actual_CA_achieved=None,PCG64_joint_stream_qualified=False,mandatory_stop=True))
    io.save(Path(out)/'result.json',result)
    return result


def preflight(root,path):
    root = Path(root).resolve(); c = io.read(path)
    if any(c.get(k) != v for k,v in fixed().items()): raise ValueError('FIXED_REFERENCE_BOUND_CONTRACT')
    inherited = old.preflight(root,root/old.CONTRACT)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    impl = source['implementation_commit']
    required = {ENTRY,'src/trottertracks/c_reference_bound_contract.py',
                'tests/management/test_research_c_reference_bound_contract.py'}
    if (not isinstance(impl,str) or re.fullmatch('[0-9a-f]{40}',impl) is None
            or not required <= source['files'].keys()
            or source['base_commit'] != BASE
            or source['implementation_commit'] != c['implementation_source_commit']
            or not set([*METADATA.values(),*ARRAYS.values()]) <= c['saved_inputs'].keys()):
        raise ValueError('FIXED_SOURCE_INPUT_CLOSURE')
    for name,want in {**source['files'],**c['saved_inputs']}.items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or io.sha(p) != want: raise ValueError('SOURCE_INPUT:'+name)
    for name in required:
        frozen = subprocess.check_output(['git','show',impl+':'+name],cwd=root,
            env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_OPTIONAL_LOCKS='0'))
        import hashlib
        if hashlib.sha256(frozen).hexdigest() != source['files'][name]:
            raise ValueError('FIXED_IMPLEMENTATION_SNAPSHOT:'+name)
    if (root/c['fresh_output']).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c,dict(status='PASSIVE_UNAPPROVED_B2_REFERENCE_READY',scientific_calls=0,
        maximum_logical_units=MAX_CALLS,reference_only=True,Stage_C_ready=False,
        H4_bound_obtained=False,source_files=len(source['files']),
        inherited_saved_inputs=inherited['inherited_saved_inputs'],mandatory_stop=True)


def run(root,path,*,execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_FIXED_B2_SCIENTIFIC_PERMISSION_REQUIRED')
    from .c1_stage3_integrated import Calls
    root = Path(root).resolve(); c,_ = preflight(root,path); out = root/c['fresh_output']
    out.mkdir(parents=True,exist_ok=False); call = Calls(out,LIMITS,MAX_CALLS)
    tick = time.monotonic(); complete = False
    cache_env = dict(MPLCONFIGDIR=str(out/'runtime-cache/matplotlib'),XDG_CACHE_HOME=str(out/'runtime-cache'))
    previous_env = {k:os.environ.get(k) for k in cache_env}
    os.environ.update(cache_env)
    try:
        io.save(out/'contract.json',c)
        io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        result = execute(root,c,out,call); complete = True
        return result
    except BaseException as exc:
        call.stop.set(); io.save(out/'failure.json',dict(error=repr(exc),first_failure=call.first_failure)); raise
    finally:
        for key,value in previous_env.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key] = value
        io.save(out/'STOP.json',dict(failed=not complete or call.stop.is_set(),
            scientific_units=sum(call.counts.values()),maximum_logical_units=MAX_CALLS,
            call_counts=dict(call.counts),first_failure=call.first_failure,
            retry=0,resume=0,wall_seconds=time.monotonic()-tick,mandatory_stop=True))
