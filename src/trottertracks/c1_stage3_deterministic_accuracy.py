"""Fixed-shot numerical qualification of four saved deterministic S2 circuits.

Import and preflight are passive. Molecular NPZ/QPY decoding and saved-value
assessment are science and require separate chat permission. Floating-point
residuals remain numerical pilots, never certified reference enclosures.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
import importlib.metadata
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from . import c1_stage1 as stage1
from . import c1_stage1_execution as io
from . import c1_stage2_native as native
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_cost_only as cost
from . import c1_stage3_integrated as integrated
from . import c1_stage3_reference as reference
from .c1_stage3_molecular import MolecularBackend, component_check

SCOPE = 'C1_STAGE3_DETERMINISTIC_FIXED_SHOT_ACCURACY_V1'
FAMILY = 'artifacts/research_c_c1_stage3_deterministic_accuracy_science'
NUMERIC = 'binary64_frobenius_telescoping_pilot_v1'


def candidates():
    # Retain arm/registration order; s1 is the primary and s4 the control.
    return tuple(c for c in cost.candidates() if c.family == 'D')


def call_limits():
    return dict(saved_bundle_load=1, input_load=1, saved_operator_load=9,
        car_context=1, xdf_full_generator=11, local_kernel_reference=2,
        qpy_load=8, operator_decode=8, kernel_assessment=4,
        shell_assessment=3, round_accuracy_assessment=52, energy_assessment=4,
        cost_attachment=4, cost_comparison=2)


def required_saved_paths(c):
    paths=set()
    for kind in ('feasibility','shots','cost'):
        folder=Path(c[kind+'_run'])
        paths.update(str(folder/name) for name in ('result.json','STOP.json'))
        paths.update(str(folder/'candidates'/(x.id+'.json')) for x in candidates())
    paths.add(str(Path(c['feasibility_run'])/'prepared.json'))
    paths.update(str(Path(c['stage2_run'])/name) for name in ('result.json','STOP.json'))
    paths.update(str(Path(c['stage1_operator_directory'])/(name+'.npz'))
        for name in ['raw_base']+[f'raw_frame_{i}' for i in range(8)])
    for group in ('raw_s4_circuits','shell_circuits'):
        for pair in c[group].values(): paths.update(pair.values())
    return paths


def preflight(root, contract_path, input_path):
    """Check bytes/versions/conditions, without parsing scientific results."""
    root, path, inp = (Path(x).resolve() for x in (root, contract_path, input_path))
    c = io.read(path)
    if (c['scope'] != SCOPE or c['candidate_ids'] != [x.id for x in candidates()]
            or c['call_limits'] != call_limits()
            or c['maximum_logical_scientific_units'] != sum(call_limits().values())
            or c['numerical_policy'] != NUMERIC
            or any(c[k] != 0 for k in ('new_fit','optimizer','retry','resume'))):
        raise ValueError('DETERMINISTIC_SCOPE')
    parent = io.read(root/c['parent_contract'])
    if (io.sha(root/c['parent_contract']) != c['parent_contract_sha256']
            or c['proposal'] != parent['proposal']
            or c['physical_conditions'] != parent['physical_conditions']):
        raise ValueError('CHANGED_DETERMINISTIC_CONDITIONS')
    if io.sha(inp) != c['input_sha256']: raise ValueError('ORIGINAL_INPUT')
    if (set(c['raw_s4_circuits']) != {x.id for x in candidates() if x.s==4}
            or set(c['shell_circuits']) != {'head','X','Y'}
            or not required_saved_paths(c) <= c['saved_inputs'].keys()):
        raise ValueError('UNFIXED_SAVED_DEPENDENCY')
    source_path = root/c['source_manifest']
    if io.sha(source_path) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    source = io.read(source_path)
    for name, want in {**source['files'], **c['saved_inputs']}.items():
        if io.sha(root/name) != want: raise ValueError('FIXED_FILE: '+name)
    versions = {p:importlib.metadata.version(p) for p in c['runtime_required_packages']}
    if versions != c['runtime_required_packages'] or sys.version_info[:2] != (3,11):
        raise ValueError('NUMERICAL_ENVIRONMENT')
    return dict(root=root, contract=c, contract_path=path, input=inp,
                source=source, runtime=versions)


def load_saved(v):
    """Only permitted execution parses four PF/shot/cost records; no PR fit."""
    root, c = v['root'], v['contract']
    folders = {k:root/c[k+'_run'] for k in ('feasibility','shots','cost')}
    expected = dict(feasibility=('COMPLETE_FEASIBILITY_ONLY',639),
        shots=('COMPLETE_OPTIMISTIC_FORMULA_SCREEN',282),cost=('COMPLETE_COST_ONLY',936))
    for kind, folder in folders.items():
        result, stop = io.read(folder/'result.json'), io.read(folder/'STOP.json')
        if (result['status'] != expected[kind][0] or result['scientific_units'] != expected[kind][1]
                or stop['failed'] or not stop['mandatory_stop']):
            raise ValueError('SAVED_SUCCESSFUL_RUN: '+kind)
    prepared = io.read(folders['feasibility']/'prepared.json')
    fresult = io.read(folders['feasibility']/'result.json')
    if (prepared['state_preparation'] != 'ORACLE_EXCLUDED'
            or fresult['call_counts']['state_alias_sector_checks'] != 2):
        raise ValueError('SAVED_ORACLE_ALIAS_QUALIFICATION')
    stage2=io.read(root/c['stage2_run']/'result.json')
    if (stage2['status']!='STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS'
            or not io.read(root/c['stage2_run']/'STOP.json')['mandatory_stop']):
        raise ValueError('SAVED_STAGE2')
    records = {}
    for candidate in candidates():
        cid = candidate.id
        f, s, k = (io.read(folders[key]/'candidates'/(cid+'.json'))
                   for key in ('feasibility','shots','cost'))
        for record in (f,s,k):
            if record['id'] != cid or record['candidate'] != asdict(candidate):
                raise ValueError('SAVED_CANDIDATE_IDENTITY')
        if (f['status'] != 'POTENTIALLY_FEASIBLE'
                or s['status'] != 'OPTIMISTIC_FORMULA_SCREENED'
                or k['status'] != 'CONDITIONAL_OPTIMISTIC_COST'
                or not k['whole_law_cost_support_qualified']
                or not k['calibration']['whole_support_covered']
                or f['df_error_bound_ha'] != prepared['df_error_bounds_ha'][candidate.arm]):
            raise ValueError('SAVED_DETERMINISTIC_QUALIFICATION')
        if [x['m'] for x in f['rounds']] != list(range(13)) or [x['m'] for x in s['rounds']] != list(range(13)):
            raise ValueError('SAVED_ROUND_ORDER')
        for a,b in zip(f['rounds'],s['rounds'],strict=True):
            if b['saved_reference'] != a or a['attenuation'] != 1 or a['rte_corrected_operator_error'] != 0:
                raise ValueError('SAVED_DETERMINISTIC_PF_IDENTITY')
            if (b['actual_synthesis_signal_error'] is not None or b['actual_law_signal_error'] is not None
                    or b['CA_achieved'] is not None):
                raise ValueError('FALSE_SAVED_SHOT_QUALIFICATION')
        if k['CA_achieved'] is not None or k['synthesis_signal_error'] is not None:
            raise ValueError('FALSE_SAVED_COST_QUALIFICATION')
        records[cid] = dict(feasibility=f, shots=s, cost=k)
    return dict(records=records, prepared=prepared, alias_verified=True,
                stage2_gates={x['id']:x for x in stage2['gates']})


def numerical_segment(raw, compiled, ideal, tolerance, label):
    """Norm inequalities are analytic; these computed residuals are not proofs.

    raw-to-independent-CAR and compiled-to-raw are both included. A maximum
    entry tolerance is a correctness gate, never an operator-norm allowance.
    """
    import numpy as np
    a = component_check(raw,ideal,tolerance['operator_component_abs'],label='raw:'+label)
    b = component_check(compiled,raw,tolerance['operator_component_abs'],label='P0:'+label)
    eye = np.eye(len(raw))
    u = [component_check(x.conj().T@x,eye,tolerance['unitarity_component_abs'],label=name+':'+label)
         for name,x in [('raw-unitarity',raw),('P0-unitarity',compiled)]]
    return dict(label=label,raw_reference=a,compiler=b,unitarity=u,
        unitary_distance_pilot=min(2.,math.fsum([a['frobenius'],b['frobenius']])),
        evidence_kind='numerically_qualified_pilot',rigorous_enclosure=None,
        missing_enclosures=['CAR/expm reference rounding','Operator reconstruction rounding','norm evaluation rounding'])


def reused_report(report,tolerance):
    """Preserve the qualification level of the stored numerical residual."""
    for key in ('frobenius','max_entry'): accuracy._finite(report[key],'saved '+key)
    accuracy._finite(report['unitarity']['max_entry'],'saved unitarity')
    if (report['max_entry']>tolerance['operator_component_abs']
            or report['unitarity']['max_entry']>tolerance['unitarity_component_abs']):
        raise ValueError('SAVED_NUMERICAL_GATE_FAIL')
    return dict(unitary_distance_pilot=min(2.,report['frobenius']),
                reused_numerical_report=report,evidence_kind='numerically_qualified_pilot',
                rigorous_enclosure=None,missing_enclosures=['saved CAR/expm/Operator/norm rounding'])


def round_synthesis(kernel, shells, occurrences):
    """Telescoping unitary distance; axis expectation Lipschitz <=2*d.

    For ideal unitary matrices ||C^n-U^n||<=n||C-U||. Every X/Y
    expectation changes by <=2*d_axis. Thus complex signal discrepancy
    <=2*hypot(d_X,d_Y), including head and both separate axis readouts.
    This is a numerical surrogate until all inputs have rigorous enclosures.
    """
    if type(occurrences) is not int or occurrences < 1: raise ValueError('OCCURRENCES')
    values = [kernel['unitary_distance_pilot']] + [shells[x]['unitary_distance_pilot'] for x in ('head','X','Y')]
    if any(x is None for x in values):
        return dict(occurrences=occurrences,complex_signal_error_pilot=None,
                    rigorous_signal_error_bound=None)
    for x in values: accuracy._finite(x,'segment distance')
    head, x, y = values[1:]
    dx = min(2.,math.fsum([occurrences*values[0],head,x]))
    dy = min(2.,math.fsum([occurrences*values[0],head,y]))
    return dict(occurrences=occurrences,axis_unitary_distance_pilot=dict(X=dx,Y=dy),
        complex_signal_error_pilot=2*math.hypot(dx,dy),
        rigorous_signal_error_bound=None,unitary_telescoping_is_analytic=True,
        floating_point_inputs_are_certificates=False)


def assess_fixed_round(target, candidate, row, saved, synthesis, *, alias_verified):
    """Invert Hoeffding at FIXED saved N; never solve for a replacement N."""
    if not alias_verified: raise ValueError('SAVED_ALIAS_REQUIRED')
    m = row['m']
    if row['attenuation'] != 1 or row['rte_corrected_operator_error'] != 0:
        raise ValueError('DETERMINISTIC_LAW')
    if saved['m'] != m or synthesis['occurrences'] != candidate.s*(1<<m):
        raise ValueError('ROUND_OCCURRENCE_IDENTITY')
    values = [saved['axes'][a]['optimistic_shots_decimal'] for a in ('X','Y')]
    if any(not isinstance(x,str) or not x.isascii() or not x.isdigit() or int(x)<1 for x in values):
        raise ValueError('SAVED_POSITIVE_INTEGER_SHOTS')
    if values[0] != values[1]: raise ValueError('REGISTERED_SYMMETRIC_SHOTS')
    base = dict(m=m,target_id=candidate.id,shots=int(values[0]),
        axes_shots_decimal=dict(zip(('X','Y'),values)),alpha_axis=target.alpha_axis,
        evidence_kind='numerically_qualified_pilot',law_model='deterministic',
        deterministic_law_error=0.,synthesis=synthesis,quantum_shots_drawn=0,
        rigorous_CA_achieved=None)
    syn = synthesis['complex_signal_error_pilot']
    pf = row['pf_operator_error']
    if syn is None or pf is None: return dict(**base,status='UNRESOLVED_SIGNAL_ERROR')
    delta = math.fsum([accuracy._finite(pf,'saved PF error'),accuracy._finite(syn,'numerical synthesis')])
    rho = 1-delta
    if rho <= 0: return dict(**base,status='INFEASIBLE_RADIUS',raw_signal_error_pilot=delta)
    beta = math.asin(delta)
    if beta+target.beta_stat > target.beta_total:
        return dict(**base,status='INFEASIBLE_PHASE',beta_deterministic_pilot=beta)
    # Exact integer N is retained; Fraction avoids a float integer overflow.
    square = float(Fraction.from_float(2*math.log(2/target.alpha_axis))/int(values[0]))
    if square == 0: return dict(**base,status='UNRESOLVED_SHOT_RESOLUTION')
    coordinate = math.sqrt(square)
    noise = math.sqrt(2.)*coordinate
    margin = rho*math.sin(target.beta_stat)-noise
    return dict(**base,status='CONDITIONAL_ROUND_PASS' if margin >= 0 else 'INFEASIBLE_FIXED_SHOTS',
        raw_signal_error_pilot=delta,radius_lower_pilot=rho,
        beta_deterministic_pilot=beta,hoeffding_axis_coordinate=coordinate,
        fixed_statistical_margin=margin,shot_count_recomputed=False,
        interpretation='Numerical sufficient-condition assessment, not physical impossibility')


class SavedBackend(MolecularBackend):
    """Existing CAR/expm/Operator paths, with no build/compile/PF-round rerun."""
    def prepare(self,call):
        import numpy as np
        self.call=call; self.modes=self.c['physical_conditions']['system_qubits']; self.dimension=1<<self.modes
        self.tol=self.p['correctness_tolerance']
        self.bundle=call('saved_bundle_load','four saved candidates',lambda:load_saved(self.v))
        directory=self.v['root']/self.c['stage1_operator_directory']
        self.generators['CDF']=[call('saved_operator_load',name,lambda name=name:
            self._load_matrix(directory/(name+'.npz'))) for name in ['raw_base']+[f'raw_frame_{i}' for i in range(8)]]
        packet=call('input_load','saved XDF whitelist',self._load_xdf)
        car=call('car_context','original CAR',lambda:stage1.car_bilinears(self.modes))
        self.generators['XDF']=[call('xdf_full_generator','base',lambda:
            float(packet['nuclear_repulsion'])*np.eye(self.dimension)+native.xdf_car_generator(packet['df_one_body'],None,car))]
        self.preparation_plan=integrated.resource_plan(task_count=10)
        def frame(i):
            return call('xdf_full_generator',f'F{i}',lambda:
                native.xdf_car_generator(packet['df_g_matrices'][i],packet['df_lambdas'][i],car))
        with ThreadPoolExecutor(max_workers=self.preparation_plan['workers']) as pool:
            # map keeps the original frame order even when calls finish out of order.
            self.generators['XDF'].extend(pool.map(frame,range(10)))
        self.shell_reports={name:call('shell_assessment',name,lambda name=name:self.shell(name))
                            for name in ('head','X','Y')}

    def matrix(self,path):
        from qiskit import qpy
        from qiskit.quantum_info import Operator
        def load():
            with (self.v['root']/path).open('rb') as f: circuits=qpy.load(f)
            if len(circuits)!=1 or circuits[0].num_qubits!=self.modes+1:
                raise ValueError('SAVED_QPY_SCHEMA')
            q=circuits[0]
            # Only the registered final ancilla measurement is removable.
            if any(x.operation.name in ('reset','initialize','if_else') for x in q.data):
                raise ValueError('NONUNITARY_SAVED_QPY')
            if q.num_clbits:
                measures=[x for x in q.data if x.operation.name=='measure']
                if (len(measures)!=1 or q.find_bit(measures[0].qubits[0]).index!=self.modes
                        or q.data[-1].operation.name!='measure'):
                    raise ValueError('FINAL_ANCILLA_MEASUREMENT_ONLY')
                q=q.remove_final_measurements(inplace=False)
            return q
        q=self.call('qpy_load',path,load)
        return self.call('operator_decode',path,lambda:Operator(q).data)

    def shell(self,name):
        import numpy as np
        paths=self.c['shell_circuits'][name]
        a,b=(self.matrix(paths[k]) for k in ('raw','compiled'))
        H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
        target=np.kron(H@np.diag([1,-1j]) if name=='Y' else H,np.eye(self.dimension))
        return numerical_segment(a,b,target,self.tol,'shell:'+name)

    def kernel(self,candidate):
        import numpy as np
        if candidate.s==1:
            return reused_report(self.bundle['stage2_gates']['compiled:'+candidate.arm+'::shared'],self.tol)
        generators=self.generators[candidate.arm]
        pf=self.call('local_kernel_reference',candidate.id,lambda:
            reference.reference_kernels(generators,Fraction(self.p['base_time'])/candidate.s,
                ld=len(generators)-1,exponential_cache={})['pf'])
        expected=np.block([[np.eye(self.dimension),np.zeros_like(pf)],[np.zeros_like(pf),pf]])
        a=self.matrix(self.c['raw_s4_circuits'][candidate.id]['raw'])
        raw=component_check(a,expected,self.tol['operator_component_abs'],label='raw:'+candidate.id)
        compiler=self.bundle['records'][candidate.id]['cost']['calibration']['diagnostics'][0]['diagnostics']['compiled']
        old=reused_report(compiler,self.tol)
        return dict(old,raw_reference=raw,
            unitary_distance_pilot=min(2.,math.fsum([raw['frobenius'],compiler['frobenius']])))


def attach_cost(record, plans, energy):
    """Qualify existing fixed-N cost only after the numerical energy gates."""
    total=accuracy.aggregate_totals(energy,plans,record['cost']['rounds'],
                                  cost_scope=record['cost']['cost_scope'])
    if total.get('totals') is not None:
        for key,value in total['totals'].items():
            if value != record['cost']['point_totals'][key]:
                raise ValueError('SAVED_COST_WEIGHT_IDENTITY')
    return dict(**total,qualification='numerical_pilot_only',rigorous_same_accuracy=None,
                guaranteed_total_resource_superiority=False,shot_counts_changed=False)


def run(backend,proposal,output,*,workers=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'candidates').mkdir()
    calls=integrated.Calls(output,call_limits(),sum(call_limits().values()))
    plan=integrated.resource_plan(task_count=4)
    if workers is not None: plan['workers']=workers
    tick,cpu=time.monotonic(),time.process_time()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):backend.prepare(calls)
        target=accuracy.AccuracyTarget(**proposal['accuracy_target'])
        io.save(output/'shells.json',backend.shell_reports)
        def task(candidate):
            record=backend.bundle['records'][candidate.id]
            kernel=calls('kernel_assessment',candidate.id,lambda:backend.kernel(candidate))
            io.save(output/'candidates'/(candidate.id+'.kernel.json'),kernel)
            rows=[]
            for row,saved in zip(record['feasibility']['rounds'],record['shots']['rounds'],strict=True):
                def assess(row=row,saved=saved):
                    synth=round_synthesis(kernel,backend.shell_reports,candidate.s*(1<<row['m']))
                    return assess_fixed_round(target,candidate,row,saved,synth,alias_verified=backend.bundle['alias_verified'])
                value=calls('round_accuracy_assessment',f'{candidate.id}:m{row["m"]}',assess)
                rows.append(value);integrated.progress(output,candidate.id,'round_accuracy',value)
            energy=calls('energy_assessment',candidate.id,lambda:accuracy.assess_energy(target,
                record['feasibility']['df_error_bound_ha'],rows,state_model='own_DF_ground_state_oracle',
                alias_branch_verified=backend.bundle['alias_verified']))
            costs=calls('cost_attachment',candidate.id,lambda:attach_cost(record,rows,energy))
            status=('NUMERICAL_PILOT_CA_PASS' if energy.get('same_accuracy') else
                'UNRESOLVED' if any(r['status'].startswith('UNRESOLVED') for r in rows) else
                'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION')
            value=dict(id=candidate.id,candidate=asdict(candidate),primary=candidate.s==1,
                status=status,rounds=rows,numerical_energy_assessment=energy,cost=costs,
                rigorous_qualification='UNRESOLVED_NUMERICAL_REFERENCE_ENCLOSURES',
                CA_achieved=None,guaranteed_actual_shots=None,
                guaranteed_total_resource_superiority=False,DF_evidence='saved numerical sector norm plus original engineering allowance',
                deterministic_law_error=0.,PR_qualification='OUT_OF_SCOPE')
            io.save(output/'candidates'/(candidate.id+'.json'),value);return value
        completed={}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures={pool.submit(task,c):c for c in candidates()}
                try:
                    for f in as_completed(futures): completed[futures[f].id]=f.result()
                except BaseException:
                    calls.stop.set()
                    for f in futures:f.cancel()
                    raise
        comparisons=[]
        for s in (1,4):
            a,b=(completed[next(c.id for c in candidates() if c.arm==arm and c.s==s)] for arm in ('CDF','XDF'))
            comparisons.append(calls('cost_comparison',f's{s}',lambda a=a,b=b:dict(
                substeps=s,qualification='numerical_pilot_only',comparison=accuracy.compare_totals(a['cost'],b['cost']),
                rigorous_same_accuracy=None,guaranteed_total_resource_superiority=False)))
        result=dict(status='COMPLETE_DETERMINISTIC_ACCURACY_PILOT',scope=SCOPE,
            candidates=[completed[c.id] for c in candidates()],comparisons=comparisons,
            candidate_status_counts=dict(Counter(x['status'] for x in completed.values())),
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),
            new_fit=0,optimizer=0,retry=0,resume=0,circuit_build=0,compile=0,
            PF_rounds_recomputed=0,DF_norm_recomputed=0,shot_count_solved=0,
            CA_achieved=None,guaranteed_total_resource_superiority=False,mandatory_stop=True)
        io.save(output/'result.json',result);return result
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),first_failure=calls.first_failure))
        raise
    finally:
        io.save(output/'resources.json',dict(plan=plan,wall_seconds=time.monotonic()-tick,
            preparation_plan=getattr(backend,'preparation_plan',None),
            process_cpu_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),retry=0,resume=0,
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts)))


def execute(root,contract_path,input_path,output):
    v=preflight(root,contract_path,input_path);output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/FAMILY):
        raise ValueError('FRESH_DETERMINISTIC_OUTPUT_REQUIRED')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=v['root']/'.cache/stage3-deterministic-accuracy'/leaf
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    sys.pycache_prefix=str(v['root']/'.cache/stage3-deterministic-accuracy/pycache')
    try: return run(SavedBackend(v,output),v['contract']['proposal'],output)
    finally:
        if output.exists():
            io.save(output/'SOURCE.json',v['source']);io.save(output/'contract.json',v['contract'])
            io.save(output/'runtime.json',v['runtime'])
