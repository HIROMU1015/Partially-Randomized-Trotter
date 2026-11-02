"""Feasibility-only version of Stage3. No science runs on import/preflight.

Preserves the integrated SOURCE. Reuses its grid, sufficient-bound predicate,
call journal and resource plan, plus the existing numerical backend helpers.
No circuit, synthesis qualification, shots, sampling or resource ranking path
is called by this mode. Separate chat permission is required for real inputs.
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

from . import c1_stage1_execution as io
from . import c1_stage1 as stage1
from . import c1_stage2_native as native
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection
from . import c1_stage3_reference as reference
from . import c1_stage3_integrated as integrated
from .c1_stage3_molecular import MolecularBackend

MODE = 'feasibility-only'
SCOPE = 'C1_STAGE3_FEASIBILITY_ONLY_V1'
FAMILY = 'artifacts/research_c_c1_stage3_feasibility_science'
PARAMETER_KEYS = ('accuracy_target','base_time','rounds','PR_LD','correctness_tolerance',
    'alias_energy_window_ha','target_state','state_preparation_included','law_model',
    'substeps','rte_steps','finite_taylor_orders','original_frame_order')


def inherited_parameters(parent):
    return {key:parent['proposal'][key] for key in PARAMETER_KEYS}


def call_limits():
    return dict(input_load=1, fit_export_load=1, saved_operator_load=13,
        ground_eigenpair=3, model_prepare=2, car_context=1, xdf_full_generator=11,
        xdf_full_sum=1, state_alias_sector_checks=2, split_native_model=2,
        rte_configuration=16, exact_model_evolution=26, kernel_reference=20,
        round_reference=260, feasibility_assessment=260, candidate_feasibility_summary=20)


def preflight(root, contract_path, input_path):
    """Byte/schema/version preflight, with no NPZ or QPY decode."""
    root, contract_path, input_path = (Path(p).resolve() for p in (root, contract_path, input_path))
    c = io.read(contract_path)
    if (c['scope'] != SCOPE or c['mode'] != MODE or c['new_fit_calls'] != 0
            or c['optimizer_calls'] != 0 or c['retry'] != 0 or c['resume'] != 0
            or c['call_limits'] != call_limits()
            or c['maximum_logical_scientific_units'] != sum(call_limits().values())
            or c['candidate_ids'] != [x.id for x in integrated.candidates()]):
        raise ValueError('FEASIBILITY_SCOPE_GRID_CALLS')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    for name, want in source['files'].items():
        if io.sha(root/name) != want: raise ValueError('SOURCE: '+name)
    # Carry the old scientific parameters unchanged, including beta and tolerances.
    parent = io.read(root/c['parent_contract'])
    if (io.sha(root/c['parent_contract']) != c['parent_contract_sha256']
            or c['proposal'] != inherited_parameters(parent)
            or c['physical_conditions'] != parent['physical_conditions']):
        raise ValueError('CHANGED_SCIENTIFIC_CONDITIONS')
    if io.sha(input_path) != c['input_sha256']: raise ValueError('ORIGINAL_INPUT')
    for name, want in c['saved_inputs'].items():
        if io.sha(root/name) != want: raise ValueError('SAVED_INPUT: '+name)
    packages = {name: importlib.metadata.version(name) for name in c['runtime_required_packages']}
    if packages != c['runtime_required_packages'] or sys.version_info[:2] != (3, 11):
        raise ValueError('NUMERICAL_ENVIRONMENT')
    return dict(root=root, input=input_path, contract_path=contract_path,
                contract=c, source=source, runtime=packages)


def classify_round(row, target):
    """Keep the old bound and its binary64 zero-radius policy, no zero-fill.

    A missing value is UNRESOLVED. Invalid/nonfinite numerical output is a
    correctness failure and stops the run, as in the integrated contract.
    """
    a = row['attenuation']
    if a is not None and (not math.isfinite(a) or not 0 <= a <= 1):
        raise ValueError('INVALID_ATTENUATION')
    values = [row['pf_operator_error'], row['rte_corrected_operator_error']]
    if any(x is not None and (not math.isfinite(x) or x < 0) for x in values):
        raise ValueError('INVALID_REFERENCE_ERROR')
    if a is None: return dict(status='UNRESOLVED', reason='attenuation unavailable')
    if a == 0: return integrated._feasibility(row, target)
    if any(x is None for x in values):
        # A measured part alone can already violate the sufficient bound.
        # This rejects a candidate; it never treats an unknown part as zero
        # to certify potential feasibility.
        if any(x is not None and x > math.sin(target.beta_total-target.beta_stat) for x in values):
            return dict(status='INFEASIBLE', reason='known PF/RTE part alone exceeds phase room')
        return dict(status='UNRESOLVED', reason='PF/RTE error unavailable')
    return integrated._feasibility(row, target)


def summarize(rows, df_error, target):
    if df_error is not None and (not math.isfinite(df_error) or df_error < 0):
        raise ValueError('INVALID_SAVED_DF_ERROR')
    if df_error is not None and df_error > target.df_cap:
        return dict(status='INFEASIBLE', reason='registered DF reserve exceeded')
    statuses = [row['status'] for row in rows]
    if 'INFEASIBLE' in statuses:
        return dict(status='INFEASIBLE', reason='registered sufficient conditions not satisfied')
    if df_error is None or 'UNRESOLVED' in statuses:
        return dict(status='UNRESOLVED', reason='required feasibility evidence incomplete')
    return dict(status='POTENTIALLY_FEASIBLE', reason='PF/RTE and saved DF reserve fit; synthesis unknown')


class FeasibilityBackend(MolecularBackend):
    def prepare(self, call):
        """Numerical subset of the frozen backend's preparation.

        This explicit subset avoids its QPY decode and native-only atom catalog.
        IO, eigenpair/alias checks, CAR and model conversion helpers are reused;
        no original algorithm or fixed SOURCE is edited.
        """
        import numpy as np
        self.call = call
        root, c, p = self.v['root'], self.c, self.c['physical_conditions']
        self.modes = p['system_qubits']; self.dimension = 1 << self.modes
        self.tol = self.p['correctness_tolerance']
        packet = call('input_load', 'saved XDF whitelist', self._load_xdf)
        parent = (root/c['saved_fit_export']).parent
        exported, metadata = call('fit_export_load', 'saved converged fit', lambda:
            io.load_export(parent, dict(input=self.v['input'], source_path=parent/'SOURCE.json')))
        if metadata['actual_rank'] != p['CDF_rank']: raise ValueError('SAVED_CDF_RANK')
        if float(exported['constant']) != float(packet['nuclear_repulsion']): raise ValueError('SCALAR_OWNERSHIP')
        from .c1_general_k_adapter import from_ffsim_number_arrays
        model = from_ffsim_number_arrays(one_body_tensor=exported['kappa'],
            diag_coulomb_mats=exported['diag_coulomb_mats'], orbital_rotations=exported['orbital_rotations'],
            constant=float(exported['constant']), z_representation=False,
            frame_ids=metadata['frame_ids'], group_ids=metadata['group_ids'],
            source_identity=c['saved_fit_export_sha256'], input_identity=c['input_sha256'],
            one_body_role='ffsim_corrected_kappa', enable_general_k=True)
        self.models['CDF'] = call('model_prepare', 'CDF', lambda: native.prepare_cdf(model))
        self.models['XDF'] = call('model_prepare', 'XDF', lambda: native.prepare_xdf(packet['df_one_body'],
            packet['df_lambdas'], packet['df_g_matrices'], float(exported['constant']),
            source_identity=c['input_sha256']+':XDF', input_identity=c['input_sha256']))
        if any(m.modes != self.modes or len(m.blocks)-1 != p[arm+'_rank'] for arm,m in self.models.items()):
            raise ValueError('MODEL_DIMENSION_RANK')
        directory = root/c['stage1_operator_directory']
        matrices = {name:call('saved_operator_load', name, lambda name=name:
            self._load_matrix(directory/(name+'.npz'), sector=name.endswith('_sector')))
            for name in ['physical_sector','CDF_sector','XDF_sector','raw_base','raw_total']+
                        [f'raw_frame_{i}' for i in range(p['CDF_rank'])]}
        self.generators['CDF'] = [matrices['raw_base']]+[matrices[f'raw_frame_{i}'] for i in range(p['CDF_rank'])]
        self.full['CDF'] = matrices['raw_total']
        self.car = call('car_context', str(self.modes)+' modes', lambda:stage1.car_bilinears(self.modes))
        self.generators['XDF'] = [call('xdf_full_generator', 'one-body+c', lambda:
            float(exported['constant'])*np.eye(self.dimension)+native.xdf_car_generator(packet['df_one_body'],None,self.car))]
        for i in range(p['XDF_rank']):
            self.generators['XDF'].append(call('xdf_full_generator', f'F{i}', lambda i=i:
                native.xdf_car_generator(packet['df_g_matrices'][i],packet['df_lambdas'][i],self.car)))
        self.full['XDF'] = call('xdf_full_sum', 'XDF', lambda:sum(self.generators['XDF']))
        ground = {arm:call('ground_eigenpair', arm, lambda arm=arm:
            self._ground(matrices[arm+'_sector'],arm)) for arm in ('physical','CDF','XDF')}
        self.energies = {arm:float(pair[0][0]) for arm,pair in ground.items()}
        records = {r['id']:r for r in io.read(root/c['stage1_result'])['records']}
        self.df_errors = {arm:records['sector_DF_'+arm]['error']+records['sector_DF_'+arm]['allowance']
                          for arm in ('CDF','XDF')}
        for arm in ('CDF','XDF'):
            self.states[arm] = call('state_alias_sector_checks', arm,
                lambda arm=arm:self._state(arm,ground[arm],matrices[arm+'_sector']))
            self.plans[arm] = call('split_native_model', arm,
                lambda arm=arm:connection.split_native_model(self.models[arm],self.p['PR_LD']))
        from scipy.linalg import expm
        from threadpoolctl import threadpool_limits
        workers = self.execution_plan['workers']
        with threadpool_limits(limits=1 if workers>1 else None):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(call, 'exact_model_evolution', f'{arm}:m{m}', lambda arm=arm,m=m:
                    expm(-1j*float(Fraction(self.p['base_time']))*(1<<m)*self.full[arm])):(arm,m)
                    for arm in ('CDF','XDF') for m in self.p['rounds']}
                for future in as_completed(futures): self.exact[futures[future]] = future.result()

    def kernel(self, candidate):
        arm = candidate.arm; h = Fraction(self.p['base_time'])/candidate.s
        plan = self.plans[arm]; config = None
        if candidate.family == 'PR':
            _,config,_ = self.call('rte_configuration', candidate.id,
                lambda:connection.configure_rte(plan,h,candidate.r,candidate.K))
        with self.cache_locks[arm]:
            return reference.reference_kernels(self.generators[arm], h,
                ld=self.p['PR_LD'] if config is not None else len(self.generators[arm])-1,
                identity_tail=plan.extracted_identity, rte_config=config, exponential_cache=self.cache[arm])

    def reference(self, candidate, kernel, m):
        # Ground/alias are checked, but no signal, Hadamard or RPE evaluation.
        return reference.reference_round(kernel,
            (self.full[candidate.arm],Fraction(self.p['base_time'])*(1<<m)),
            repetitions=(1<<m)*candidate.s, exact_operator=self.exact[candidate.arm,m])


def run(backend, proposal, output, *, workers=None):
    """Limited one-shot pipeline; no dispatch to the integrated full runner."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'candidates').mkdir()
    calls = integrated.Calls(output, call_limits(), sum(call_limits().values()))
    start, cpu = time.monotonic(), time.process_time()
    completed = {}; plan = integrated.resource_plan()
    if workers is not None: plan['workers'] = workers
    target = accuracy.AccuracyTarget(**proposal['accuracy_target'])
    try:
        backend.execution_plan = plan; backend.prepare(calls)
        io.save(output/'prepared.json',backend.prepared_summary())
        preparation_workers = plan['workers']; plan = integrated.resource_plan()
        if workers is not None: plan['workers'] = workers
        plan['preparation_workers'] = preparation_workers
        backend.execution_plan = plan
        def task(candidate):
            label = candidate.id
            try:
                kernel = calls('kernel_reference',label,lambda:backend.kernel(candidate))
                rows, decisions = [], []
                for m in proposal['rounds']:
                    row = calls('round_reference',f'{label}:m{m}',lambda m=m:backend.reference(candidate,kernel,m))
                    decision = calls('feasibility_assessment',f'{label}:m{m}',lambda row=row:classify_round(row,target))
                    integrated.progress(output,label,'reference',dict(m=m,reference=row,feasibility=decision))
                    rows.append(dict(m=m,**row)); decisions.append(decision)
                df_error = backend.df_errors[candidate.arm]
                summary = calls('candidate_feasibility_summary',label,lambda:summarize(decisions,df_error,target))
                result = dict(id=label,candidate=asdict(candidate),**summary,rounds=rows,
                    feasibility=decisions,df_error_bound_ha=df_error,CA_achieved=None,
                    synthesis_signal_error=None,shots=None,total_resources=None,final_resource_ranking=None)
                io.save(output/'candidates'/(label+'.json'),result)
                return result
            except BaseException as error:
                calls.stop.set()
                io.save(output/'candidates'/(label+'.json'),dict(id=label,
                    status='STOPPED_DEPENDENCY' if isinstance(error,integrated.Stopped) else 'FAIL',
                    error_type=type(error).__name__,error=str(error)))
                raise
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures = {pool.submit(task,c):c for c in integrated.candidates()}
                try:
                    for future in as_completed(futures):
                        c = futures[future]; completed[c.id] = future.result()
                except BaseException:
                    calls.stop.set()
                    for f in futures: f.cancel()
                    raise
        ordered = [completed[c.id] for c in integrated.candidates()]
        result = dict(status='COMPLETE_FEASIBILITY_ONLY',mode=MODE,
            candidate_status_counts=dict(Counter(r['status'] for r in ordered)),
            candidate_ids=[c.id for c in integrated.candidates()],
            candidate_statuses=[dict(id=r['id'],status=r['status']) for r in ordered],
            CA_achieved=None,shots=None,total_resources=None,final_resource_ranking=None,
            native_build=0,compile=0,synthesis_qualification=0,new_fit=0,optimizer=0,retry=0,resume=0,
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),mandatory_stop=True)
        io.save(output/'result.json',result)
        return result
    except BaseException as error:
        calls.stop.set()
        io.save(output/'failure.json',dict(first_failure=calls.first_failure,error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),completed_candidate_ids=list(completed),
            call_counts=dict(calls.counts)))
        raise
    finally:
        io.save(output/'resources.json',dict(plan=plan,wall_seconds=time.monotonic()-start,
            process_cpu_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        io.save(output/'STOP.json',dict(mandatory_stop=True,mode=MODE,failed=calls.stop.is_set(),
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),new_fit=0,retry=0,resume=0))


def execute(verified, output):
    """Real-input science requires separate explicit chat permission."""
    v = preflight(verified['root'],verified['contract_path'],verified['input'])
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/FAMILY):
        raise ValueError('FRESH_FEASIBILITY_OUTPUT_REQUIRED')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path = v['root']/'.cache/stage3-feasibility'/leaf; path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    sys.pycache_prefix = str(v['root']/'.cache/stage3-feasibility/pycache')
    backend = FeasibilityBackend(v,output)
    try: return run(backend,v['contract']['proposal'],output)
    finally:
        if output.exists():
            io.save(output/'contract.json',v['contract']);io.save(output/'SOURCE.json',v['source'])
            io.save(output/'runtime.json',v['runtime'])
