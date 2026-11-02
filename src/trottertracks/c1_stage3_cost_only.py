"""Eight-candidate native cost pilot; no science on import or preflight.

Saved optimistic shots are inputs, never an accuracy certificate. Numerical
work/circuit compilation requires separate explicit permission. Reuse the old
calibrator but bypass its molecular CAR/PF/RTE/ground/signal evaluation paths.
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
from . import c1_stage2_native as native
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection
from . import c1_stage3_integrated as integrated
from . import c1_stage3_qualification as qualification
from .c1_stage3_molecular import MolecularBackend, component_check

SCOPE = 'C1_STAGE3_EIGHT_CANDIDATE_COST_ONLY_V1'
FAMILY = 'artifacts/research_c_c1_stage3_cost_science'
PROPOSAL_KEYS = ('accuracy_target','base_time','rounds','PR_LD','compiler','basis_policy',
    'cost_scope','cost_policy','long_round_depth','readout','cost_samples_per_PR_candidate',
    'cost_seed_master','correctness_tolerance','target_state','state_preparation_included',
    'law_model','original_frame_order')


def candidates():
    return tuple(c for c in integrated.candidates() if c.family=='D' or
                 (c.s,c.r,c.K) in ((4,16,0),(4,16,2)))


def call_limits():
    # Four PR candidates x32 words, two new D(s4), three shared shells.
    segments = 4*32+2+3
    return dict(input_load=1,fit_export_load=1,model_prepare=2,saved_shots_load=1,
        saved_stage2_load=1,split_native_model=2,gaussian_build=20,
        rte_configuration=4,cost_support_probe=4,cost_kernel_sample=128,
        circuit_build=segments,compile=segments,compile_equivalence_check=segments,
        metrics=segments,native_calibration=8,cost_interval_assessment=4,
        serial_cost_expansion=208,optimistic_total_aggregate=8,cost_comparison=12)


def comparison_pairs():
    cs = candidates();pairs = []
    for c in cs[:4]:
        other = next(x for x in cs if x.arm=='XDF' and (x.family,x.s,x.r,x.K)==(c.family,c.s,c.r,c.K))
        pairs.append((c.id,other.id))
    for arm in ('CDF','XDF'):
        for pr in (c for c in cs if c.arm==arm and c.family=='PR'):
            for d in (c for c in cs if c.arm==arm and c.family=='D'):
                pairs.append((pr.id,d.id))
    return tuple(pairs)


def preflight(root, contract_path, input_path):
    """Byte identity, metadata and versions only; no saved-value decode."""
    root,contract_path,input_path = (Path(p).resolve() for p in (root,contract_path,input_path))
    c = io.read(contract_path)
    if (c['scope'] != SCOPE or c['candidate_ids'] != [x.id for x in candidates()]
            or c['call_limits'] != call_limits() or c['maximum_logical_scientific_units'] != sum(call_limits().values())
            or c['new_fit'] != 0 or c['retry'] != 0 or c['resume'] != 0):
        raise ValueError('COST_SCOPE_GRID_CALLS')
    parent = io.read(root/c['parent_contract'])
    if (io.sha(root/c['parent_contract']) != c['parent_contract_sha256']
            or c['proposal'] != {key:parent['proposal'][key] for key in PROPOSAL_KEYS}
            or c['physical_conditions'] != parent['physical_conditions']):
        raise ValueError('CHANGED_COST_SCIENTIFIC_CONDITIONS')
    if io.sha(input_path) != c['input_sha256']: raise ValueError('SAVED_H4_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    source = io.read(root/c['source_manifest'])
    for p,h in source['files'].items():
        if io.sha(root/p) != h: raise ValueError('SOURCE: '+p)
    for p,h in c['saved_inputs'].items():
        if io.sha(root/p) != h: raise ValueError('SAVED_INPUT: '+p)
    versions = {name:importlib.metadata.version(name) for name in c['runtime_required_packages']}
    if versions != c['runtime_required_packages'] or sys.version_info[:2] != (3,11):
        raise ValueError('NUMERICAL_ENVIRONMENT')
    return dict(root=root,contract_path=contract_path,input=input_path,contract=c,source=source,runtime=versions)


def load_shots(v):
    """Load just the eight saved optimistic records, without any shot formula."""
    folder = v['root']/v['contract']['saved_shots_run']
    result,stop = io.read(folder/'result.json'),io.read(folder/'STOP.json')
    if (result['status'] != 'COMPLETE_OPTIMISTIC_FORMULA_SCREEN' or stop['failed']
            or not stop['mandatory_stop'] or result['scientific_units'] != 282):
        raise ValueError('SAVED_SHOTS_RUN')
    lookup = {x['id']:x for x in result['candidates']}
    if list(lookup) != [x.id for x in integrated.candidates()]: raise ValueError('SAVED_TWENTY_CANDIDATES')
    records = {}
    for candidate in candidates():
        r = io.read(folder/'candidates'/(candidate.id+'.json'))
        if ({k:x for k,x in r.items() if k!='rounds'} != lookup[candidate.id]
                or r['candidate'] != asdict(candidate) or r['status'] != 'OPTIMISTIC_FORMULA_SCREENED'
                or [x['m'] for x in r['rounds']] != list(range(13))):
            raise ValueError('SAVED_SHOTS_IDENTITY')
        for x in r['rounds']:
            if (x['status'] != 'OPTIMISTIC_FORMULA_ESTIMATE' or x['CA_achieved'] is not None
                    or x['guaranteed_actual_shots'] is not None
                    or x['actual_synthesis_signal_error'] is not None or x['actual_law_signal_error'] is not None):
                raise ValueError('SAVED_SHOTS_QUALIFICATION')
            for axis in ('X','Y'):
                text = x['axes'][axis]['optimistic_shots_decimal']
                if not isinstance(text,str) or not text.isascii() or not text.isdigit() or int(text)<1:
                    raise ValueError('SAVED_SHOTS_DECIMAL_INTEGER')
        records[candidate.id] = r
    return records


class CostBackend(MolecularBackend):
    def prepare(self, call):
        """Only the value/native subset of the old saved-molecule preparation."""
        self.call = call;root,c = self.v['root'],self.c;p = c['physical_conditions']
        self.modes = p['system_qubits'];self.dimension = 1<<self.modes
        self.tol = self.p['correctness_tolerance']
        self.shots = call('saved_shots_load','eight optimistic records',lambda:load_shots(self.v))
        packet = call('input_load','saved XDF whitelist',self._load_xdf)
        parent = (root/c['saved_fit_export']).parent
        arrays,metadata = call('fit_export_load','saved CDF export',lambda:
            io.load_export(parent,dict(input=self.v['input'],source_path=parent/'SOURCE.json')))
        if metadata['actual_rank'] != p['CDF_rank'] or float(arrays['constant']) != float(packet['nuclear_repulsion']):
            raise ValueError('SAVED_RANK_SCALAR')
        from .c1_general_k_adapter import from_ffsim_number_arrays
        model = from_ffsim_number_arrays(one_body_tensor=arrays['kappa'],
            diag_coulomb_mats=arrays['diag_coulomb_mats'],orbital_rotations=arrays['orbital_rotations'],
            constant=float(arrays['constant']),z_representation=False,frame_ids=metadata['frame_ids'],
            group_ids=metadata['group_ids'],source_identity=c['saved_fit_export_sha256'],
            input_identity=c['input_sha256'],one_body_role='ffsim_corrected_kappa',enable_general_k=True)
        self.models['CDF'] = call('model_prepare','CDF',lambda:native.prepare_cdf(model))
        self.models['XDF'] = call('model_prepare','XDF',lambda:native.prepare_xdf(packet['df_one_body'],
            packet['df_lambdas'],packet['df_g_matrices'],float(arrays['constant']),
            source_identity=c['input_sha256']+':XDF',input_identity=c['input_sha256']))
        for arm,m in self.models.items():
            if m.modes != self.modes or len(m.blocks)-1 != p[arm+'_rank']: raise ValueError('NATIVE_MODEL_RANK')
            self.plans[arm] = call('split_native_model',arm,lambda arm=arm:
                connection.split_native_model(self.models[arm],self.p['PR_LD']))
        saved = call('saved_stage2_load','deterministic s1 metrics',lambda:io.read(root/c['stage2_result']))
        if saved['status'] != 'STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS': raise ValueError('STAGE2_VALIDATION_STATUS')
        self.saved_metrics = saved['metrics'];self.builder = native.NativeBuilder()
        # No CAR, ground, exact model evolution, PF/RTE reference or QPY decode.

    def cost_kernel(self, candidate):
        h = Fraction(self.p['base_time'])/candidate.s
        plan = self.plans[candidate.arm];tail=config=distribution=None;support=None
        if candidate.family == 'PR':
            tail,config,distribution = self.call('rte_configuration',candidate.id,lambda:
                connection.configure_rte(plan,h,candidate.r,candidate.K))
            if config is None: raise ValueError('REGISTERED_PR_TAIL_EMPTY')
            support = self.call('cost_support_probe',candidate.id,lambda:
                qualification.exhaustive_trajectories(tail,distribution,candidate.r,
                    budget=self.p['cost_samples_per_PR_candidate']))
        return dict(h=h,plan=plan,tail=tail,config=config,distribution=distribution,support=support)

    def _validate_compile(self,label,raw,target,*,controlled,arm=None):
        """Reuse P0 and metrics; local raw/compiled phase check only.

        The inherited calibrator supplies a lazy molecular reference. It is
        intentionally not called: no CAR/PF/RTE or X/Y signal re-evaluation.
        Prior Stage2 plus artificial tests qualify the raw emitter for this
        exploratory cost scope. These checks are not all-law error bounds.
        """
        import numpy as np
        from qiskit.quantum_info import Operator
        from .pr_dictionary_validation.gates import compiled_wrapper
        from .pr_dictionary_validation.fixture import CompilerProfile
        from .pr_dictionary_validation.f2_execution import circuit_metrics
        self._write_circuit(label,'raw',raw)
        profile = CompilerProfile(**{**self.p['compiler'],'basis_gates':tuple(self.p['compiler']['basis_gates'])})
        compiled = self.call('compile',label,lambda:compiled_wrapper(raw,profile))
        self._write_circuit(label,'compiled',compiled)
        def check():
            if any(x.operation.name not in {'rz','sx','x','cx','measure','barrier'} for x in compiled.data):
                raise ValueError('NONNATIVE_COMPILED_GATE')
            clean = lambda q:q.remove_final_measurements(inplace=False) if q.num_clbits else q
            a,b = Operator(clean(raw)).data,Operator(clean(compiled)).data
            report = component_check(b,a,self.tol['operator_component_abs'],label='P0 phase:'+label)
            report['unitarity'] = component_check(b.conj().T@b,np.eye(len(b)),self.tol['unitarity_component_abs'],label='unitarity:'+label)
            return report
        diagnostic = self.call('compile_equivalence_check',label,check)
        metrics = self.call('metrics',label,lambda:circuit_metrics(compiled))
        return dict(metrics=metrics,diagnostics=dict(compiled=diagnostic,
            reference_scope='raw-versus-P0 only; independent molecular reference not repeated'))


def independent_sequence(candidate,kernel,*,master,m,axis,shot,modes,builder=None,provider=None):
    """Materializer for small authorized checks, not the cost estimator.

    Fresh domain-separated seed per coherent occurrence. The cost path uses
    n*E[C] by linearity, not this sequence with one sampled word repeated n
    times. No long-round circuit is built by run().
    """
    from .c1_stage3_native_bridge import partial_kernel
    if candidate.family != 'PR': raise ValueError('Independent PR occurrence factory required')
    def factory(j):
        seed = connection.trajectory_seed(master,candidate.arm,candidate.id,m,axis,shot,j)
        events = connection.sample_occurrence(kernel['tail'],kernel['distribution'],candidate.r,seed)
        return partial_kernel(kernel['plan'],kernel['h'],events,rte_config=kernel['config'],builder=builder,provider=provider)
    return connection.serial_native_sequence(factory,occurrences=(1<<m)*candidate.s,modes=modes)


def aggregate_optimistic(candidate, shots, rounds):
    """N from saved optimistic input, separate from CA-gated old aggregate.

    Do not set same_accuracy=True to bypass the old scientific gate. Compose
    with the old serial-cost helper, then weight by exact integer saved N.
    Rational sums preserve the declared sample means; no ceil/shot formula.
    """
    if len(rounds) != 26: raise ValueError('ROUND_COST_COVERAGE')
    lookup = {(x['m'],x['axis']):x for x in rounds}
    if set(lookup) != {(m,a) for m in range(13) for a in ('X','Y')}: raise ValueError('ROUND_COST_IDENTITY')
    point = {k:Fraction(0) for k in accuracy.METRICS};low=point.copy();high=point.copy();qualified=True
    for m in range(13):
        for axis in ('X','Y'):
            r = lookup[m,axis];n = int(shots['rounds'][m]['axes'][axis]['optimistic_shots_decimal'])
            if (n<1 or r['target_id'] != candidate.id or not r['long_round_coverage']
                    or r['scope'] != 'serial_native_kernel_composition_without_state_preparation'
                    or r['kernel_occurrences'] != candidate.s*(1<<m)):
                raise ValueError('ROUND_COST_SCOPE')
            q = r['simultaneous_interval_qualified'] and r['lower'] is not None and r['upper'] is not None
            qualified &= q
            for key in accuracy.METRICS:
                value = r['mean'][key]
                if not math.isfinite(value) or value<0: raise ValueError('INVALID_COST')
                point[key] += n*Fraction(value)
                if q:
                    lo,hi = r['lower'][key],r['upper'][key]
                    if not 0<=lo<=value<=hi or not math.isfinite(hi): raise ValueError('COST_SUPPORT')
                    low[key] += n*Fraction(lo);high[key] += n*Fraction(hi)
    return dict(id=candidate.id,candidate=asdict(candidate),status='CONDITIONAL_OPTIMISTIC_COST',
        point_totals={k:float(v) for k,v in point.items()},point_totals_rational={k:str(v) for k,v in point.items()},
        support_ranges={k:[float(low[k]),float(high[k])] for k in point} if qualified else None,
        whole_law_cost_support_qualified=qualified,
        cost_scope='serial_native_kernel_composition_without_state_preparation',
        shots_source='saved optimistic prescription, not actual guaranteed shots',
        expected_cost_point_law='finite-bit IID calibration point estimate; ideal law transfer unqualified'
            if candidate.family=='PR' else 'deterministic P0 native metrics',
        independent_per_quantum_occurrence=True,calibration_is_not_repeated_quantum_trajectory=True,
        calibration_uncertainty='reuse same calibration at all rounds; no artificial sqrt(n) reduction',
        synthesis_signal_error=None,actual_law_signal_error=None,CA_achieved=None,
        guaranteed_total_resource_superiority=False,state_preparation_included=False)


def compare_costs(left,right):
    """Reuse interval comparison, explicitly conditioned on optimistic N."""
    def old(x):return dict(totals=x['point_totals'],intervals=x['support_ranges'],cost_scope=x['cost_scope'])
    verdict = accuracy.compare_totals(old(left),old(right))
    # The reused function calls its operands CDF/XDF. Some pairs here are
    # PR versus deterministic within one arm, so expose neutral operand names.
    verdict['verdict'] = {'CDF_LOWER_CONDITIONAL_INTERROGATION_COST':'LEFT_LOWER_CONDITIONAL_COST_AT_OPTIMISTIC_SHOTS',
        'XDF_LOWER_CONDITIONAL_INTERROGATION_COST':'RIGHT_LOWER_CONDITIONAL_COST_AT_OPTIMISTIC_SHOTS'}.get(verdict['verdict'],verdict['verdict'])
    return dict(left=left['id'],right=right['id'],point_RZ_ratio_left_over_right=
        left['point_totals']['rz_count']/right['point_totals']['rz_count'] if right['point_totals']['rz_count'] else None,
        cost_relation=verdict,interpretation='Cost-only comparison at fixed optimistic shots; not same-CA or actual-resource superiority',
        CA_achieved=None,guaranteed_total_resource_superiority=False)


def run(backend, proposal, output, *, workers=None):
    output = Path(output);output.mkdir(parents=True,exist_ok=False);(output/'candidates').mkdir()
    calls = integrated.Calls(output,call_limits(),sum(call_limits().values()))
    plan = integrated.resource_plan(task_count=8)
    if workers is not None:plan['workers'] = workers
    start,cpu = time.monotonic(),time.process_time()
    try:
        backend.execution_plan=plan;backend.prepare(calls)
        def task(c):
            try:
                kernel = backend.cost_kernel(c)
                calibration = calls('native_calibration',c.id,lambda:backend.calibrate(c,kernel,calls))
                integrated.progress(output,c.id,'native_calibration',calibration)
                rows = []
                for m in proposal['rounds']:
                    for axis in ('X','Y'):
                        row = calls('serial_cost_expansion',f'{c.id}:m{m}:{axis}',lambda m=m,axis=axis:
                            connection.compose_serial_cost(calibration['kernel'],calibration['shells'][axis],
                                m=m,substeps=c.s,target_id=c.id))
                        rows.append(row);integrated.progress(output,c.id,'round_cost',row)
                total = calls('optimistic_total_aggregate',c.id,lambda:aggregate_optimistic(c,backend.shots[c.id],rows))
                io.save(output/'candidates'/(c.id+'.json'),dict(**total,calibration=calibration,rounds=rows))
                return total
            except BaseException as error:
                calls.stop.set()
                io.save(output/'candidates'/(c.id+'.json'),dict(id=c.id,status='STOPPED_DEPENDENCY' if isinstance(error,integrated.Stopped) else 'FAIL',error=str(error)))
                raise
        from threadpoolctl import threadpool_limits
        completed = {}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures = {pool.submit(task,c):c for c in candidates()}
                try:
                    for future in as_completed(futures):completed[futures[future].id] = future.result()
                except BaseException:
                    calls.stop.set()
                    for f in futures:f.cancel()
                    raise
        comparisons = [calls('cost_comparison',a+'::'+b,lambda a=a,b=b:compare_costs(completed[a],completed[b]))
                       for a,b in comparison_pairs()]
        result = dict(status='COMPLETE_COST_ONLY',scope=SCOPE,candidates=[completed[c.id] for c in candidates()],comparisons=comparisons,
            out_of_scope_candidates=[c.id for c in integrated.candidates() if c not in candidates()],
            out_of_scope_meaning='Not selected for this pilot; no scientific NO-GO',
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),
            new_fit=0,optimizer=0,PF_RTE_recomputed=0,shots_formula_recomputed=0,retry=0,resume=0,
            synthesis_signal_error=None,actual_law_signal_error=None,CA_achieved=None,
            guaranteed_total_resource_superiority=False,mandatory_stop=True)
        io.save(output/'result.json',result);return result
    except BaseException as error:
        calls.stop.set()
        io.save(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),first_failure=calls.first_failure,call_counts=dict(calls.counts)))
        raise
    finally:
        io.save(output/'resources.json',dict(plan=plan,wall_seconds=time.monotonic()-start,
            process_cpu_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),retry=0,resume=0))


def execute(root, contract_path, input_path, output):
    v = preflight(root,contract_path,input_path);output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(v['root']/FAMILY): raise ValueError('FRESH_COST_OUTPUT_REQUIRED')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=v['root']/'.cache/stage3-cost-only'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    sys.pycache_prefix=str(v['root']/'.cache/stage3-cost-only/pycache')
    try:return run(CostBackend(v,output),v['contract']['proposal'],output)
    finally:
        if output.exists():
            io.save(output/'SOURCE.json',v['source']);io.save(output/'contract.json',v['contract']);io.save(output/'runtime.json',v['runtime'])
