"""Opt-in C2 qualification tools. Import/preflight never analyze molecular tables.

Exact finite-grid counts qualify a *declared IID random-bit model*, not PCG64
independence or the CAR/atom reference bridge. Sampled residuals are not bounds.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io

VERSION = 'C2_K0_exact_choice_grid_table_v1'
SCOPE = 'C_PR_AWARE_DF_C2B_K0_TABLE_IID_GRID_V1'
FAMILY = 'artifacts/research_c_pr_aware_df_stage_c2/2026-10-27'
FRESH = FAMILY + '/C2B-run01'
ARMS = ('baseline', 'proposed')
IID = 'IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL'
POLICY = dict(molecule='linear H4',distance_angstrom=1.,basis='STO-3G',
    representation='normal number CDF',rank=8,Nalpha=2,Nbeta=2,sector_dimension=36,
    frame_order=list(range(8)),ld=2,delta='1/10',s=4,r=16,K=0,rounds=list(range(13)),
    oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',
    atoms_per_arm=216,CA_Ha=.00159,DF_reserve_Ha=.00019875,
    native_policy='common_precision_spin_kappa_relative_head_full_frame_tail_v1')
RECORDS = dict(C1_result='artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/result.json',
    C1_STOP='artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/STOP.json',
    C1_SOURCE='artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/SOURCE.json',
    baseline_tail='artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01/baseline_tail.json',
    proposed_tail='artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01/proposed_tail.json')
LIMITS = dict(saved_load=1, identity_gate=1, table_replay=2, choice_grid_analysis=2,
              law_round=26, arm_qualification=2, output_summary=1)
MAX_CALLS = sum(LIMITS.values())


def rational(x):
    if isinstance(x, float):
        if not math.isfinite(x): raise ValueError('FINITE_RATIONAL')
        return Fraction.from_float(x)
    return Fraction(x)


def upper_float(x):
    """Serialize an exact nonnegative rational upward, including subnormals."""
    x = rational(x)
    if x < 0: raise ValueError('NONNEGATIVE_BOUND')
    f = float(x)
    if not math.isfinite(f): raise ValueError('BOUND_FLOAT_RANGE')
    return math.nextafter(f, math.inf) if Fraction.from_float(f) < x else f


def tv(p, q):
    p, q = tuple(map(rational, p)), tuple(map(rational, q))
    if not p or len(p) != len(q) or min(*p, *q) < 0 or sum(p) != 1 or sum(q) != 1:
        raise ValueError('EXACT_PROBABILITY_SIMPLEX')
    return sum((abs(a-b) for a, b in zip(p, q)), Fraction()) / 2


def choice_grid(probabilities, *, bits=53):
    """NumPy1.26.4 replace=True,p=... CDF; count grid cells without sampling.

    Search side='right': bin i contains ceil(D*cdf[i])-ceil(D*cdf[i-1])
    grid points. NumPy's own argument validator is invoked with size=0;
    no random numbers are drawn. Actual computed CDF floats are treated as
    exact rationals. No 2**53-element enumeration is performed.
    """
    import numpy as np
    if type(bits) is not int or not 1 <= bits <= 53: raise ValueError('GRID_BITS')
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all(): raise ValueError('FINITE_PROBABILITIES')
    rng = np.random.Generator(np.random.PCG64(0))
    rng.choice(len(p), size=0, p=p)  # reuse fixed library's acceptance rules
    cdf = p.cumsum(); cdf /= cdf[-1]
    if cdf[-1] != 1. or any(x < 0 or x > 1 for x in cdf) or np.any(np.diff(cdf) < 0):
        raise ValueError('INVALID_CDF')
    D = 1 << bits
    ceil = lambda x: -((-x.numerator)//x.denominator)
    ends = [ceil(D*Fraction.from_float(float(x))) for x in cdf]
    counts = [b-a for a, b in zip([0]+ends[:-1], ends)]
    assert min(counts) >= 0 and sum(counts) == D
    return dict(probabilities=[float(x) for x in p], cdf=[float(x) for x in cdf],
                counts=counts, denominator=D, probabilities_exact=[str(Fraction(n, D)) for n in counts],
                zero_random_draws=True, mapping='searchsorted_right', bits=bits)


def analyze_table(weights, stored_probabilities):
    """K0 atom probabilities only; angles/normalization/reference remain separate."""
    import numpy as np
    weights = tuple(map(rational, weights)); stored = tuple(map(float, stored_probabilities))
    if not weights or len(weights) != len(stored) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('COEFFICIENT_TABLE')
    if any(not math.isfinite(x) or x < 0 for x in stored): raise ValueError('STORED_PROBABILITIES')
    ideal = [x/sum(weights) for x in weights]
    mass = math.fsum(stored)
    if mass <= 0: raise ValueError('ZERO_STORED_MASS')
    # Identical scalar divisions/dtype to iter_sample_rte_events.
    p = np.asarray([x/mass for x in stored], dtype=float)
    grid = choice_grid(p)
    normalized = lambda xs: [x/sum(xs) for x in xs]
    first = normalized(list(map(Fraction.from_float, stored)))
    sampler = normalized([Fraction.from_float(float(x)) for x in p])
    boundaries = list(map(Fraction.from_float, grid['cdf']))
    continuous = [b-a for a, b in zip([Fraction()]+boundaries[:-1], boundaries)]
    actual_grid = list(map(Fraction, grid['probabilities_exact']))
    direct = tv(ideal, actual_grid)
    return dict(status='EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL',
        reference='exact normalized absolute saved atom coefficients;not original integrals or CAR matrix',
        ideal_probabilities=[str(x) for x in ideal], first_probabilities=[x.hex() for x in stored],
        sampler_probabilities=[float(x).hex() for x in p], cdf=[x.hex() for x in grid['cdf']],
        grid=grid, eta_exact=str(direct), eta_upper_float=upper_float(direct),
        stages_TV={name:str(tv(a, b)) for name, a, b in [
            ('ideal_to_first', ideal, first), ('first_to_sampler', first, sampler),
            ('sampler_to_continuous_CDF', sampler, continuous),
            ('continuous_CDF_to_grid', continuous, actual_grid)]},
        lost_positive_ideal_atoms=[i for i, (a, b) in enumerate(zip(ideal, actual_grid)) if a > 0 and b == 0],
        pcg64_IID_proven=False, coefficient_generation_error=None,
        reference_bridge_error=None, angle_normalization_error=None,
        actual_implemented_law_error=None)


def law_round(table, m, *, random_model=IID):
    if type(m) is not int or not 0 <= m <= 12: raise ValueError('REGISTERED_ROUND')
    if table['status'] != 'EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL':
        raise ValueError('TABLE_QUALIFICATION')
    eta = Fraction(table['eta_exact'])
    if not 0 <= eta <= 1: raise ValueError('TV_RANGE')
    bound = min(Fraction(2), 2*16*4*(1 << m)*eta)
    return dict(m=m, events=16*4*(1 << m), random_model=random_model,
        status='CONDITIONAL_IID_TABLE_BOUND' if random_model == IID else 'UNRESOLVED_RANDOMNESS_MODEL',
        conditional_raw_complex_bound_exact=str(bound) if random_model == IID else None,
        conditional_raw_complex_bound_upper=upper_float(bound) if random_model == IID else None,
        norm='raw_complex_signal', normalization='no multiplication or division by attenuation',
        assumptions=['same exact unitary trajectory map in both laws', '|matrix element|<=1',
                     'IID uniform53bit words,including discarded K0 order draws'],
        actual_implemented_law_error=None, reference_bridge_error=None,
        synthesis_error=None, CA_achieved=None, shots=None, total=None)


def synthesis_preview(factors, *, path, occurrences, shells):
    """Scoped analytic local proof INPUTS -> exact L1 telescoping preview.

    A whole-kernel P0 record must cover the final whole-kernel transform.
    Event-wise compiled residuals do not have that coverage. This is a proof
    input checker/propagator, not an automated proof verifier or an H4 bound.
    """
    if type(occurrences) is not int or occurrences < 1: raise ValueError('OCCURRENCES')
    records = [*factors, *shells.values()]
    if not factors or set(shells) != {'head','X','Y'}: raise ValueError('FACTOR_SHELL_COVERAGE')
    if any(x is None for x in records):
        return dict(status='UNRESOLVED_BOUND_INPUTS', candidate_raw_complex_bound=None)
    for record in records:
        if (record.get('kind') not in ('analytic_unitary_bound', 'enclosed_unitary_bound')
                or record.get('norm') != 'full_controlled_operator_2'
                or record.get('phase_sensitive') is not True
                or record.get('reference_error') is None or not record.get('provenance')):
            raise ValueError('UNQUALIFIED_LOCAL_BOUND_OR_RELATIVE_PHASE')
        if record.get('compiler_path') != path: raise ValueError('COMPILER_PATH_COVERAGE')
    if path == 'whole_kernel_P0' and not any(x.get('coverage') == 'all_final_whole_kernel_outputs' for x in factors):
        raise ValueError('EVENT_COMPILE_IS_NOT_WHOLE_KERNEL_PROOF')
    eps = lambda x: rational(x['epsilon']) + rational(x['reference_error'])
    if any(rational(x['epsilon']) < 0 or rational(x['reference_error']) < 0 for x in records):
        raise ValueError('NEGATIVE_ERROR')
    kernel = sum((eps(x) for x in factors), Fraction())
    axes = {a:min(Fraction(2), occurrences*kernel+eps(shells['head'])+eps(shells[a])) for a in ('X','Y')}
    # Existing round_synthesis uses 2*hypot(dX,dY). L1 >= hypot gives a
    # rational outward enclosure without promoting its floating pilot label.
    # Separate imperfect X/Y shells need not yield a unit-disc complex signal.
    # Do not apply the ideal-unitary law's cap2 to this synthesis estimate.
    bound = 2*sum(axes.values())
    return dict(status='CONDITIONAL_ON_SUPPLIED_LOCAL_PROOFS', norm='raw_complex_signal',
                candidate_raw_complex_bound=str(bound), actual_H4_qualification=None,
                proof_inputs_are_not_verified_by_schema=True)


def support_preview(record):
    """No support from observed extrema, event-wise prices or unknown loops."""
    if record is None: return dict(status='UNRESOLVED_COST_SUPPORT', support=None)
    if (record.get('kind') != 'structural_whole_kernel_P0_range'
            or record.get('all_words') is not True or not record.get('pass_iteration_proof')
            or not record.get('provenance')):
        raise ValueError('UNQUALIFIED_COST_SUPPORT')
    from .c1_stage3_accuracy import METRICS
    bounds = record['bounds']
    if set(bounds) != set(METRICS) or any(type(x) is not int or x < 0 for pair in bounds.values() for x in pair):
        raise ValueError('COST_RANGE')
    if any(len(pair) != 2 or pair[0] > pair[1] for pair in bounds.values()): raise ValueError('COST_RANGE')
    return dict(status='CONDITIONAL_ON_STRUCTURAL_RANGE_PROOF', support=bounds,
                alpha_proposal='.01/12 per arm/metric;not adopted', proof_verified_by_schema=False)


def c1_preview(law_rows, synthesis=None):
    """Prepare a schema preview, never call shots/resources or alter old C1.

    C2B only supplies a conditional categorical law bound; PCG64 independence,
    atom/CAR/angle bridge and P0 synthesis remain unknown, hence no all-law
    actual-implementation bound can enter C1's evidence lane yet.
    """
    if [r['m'] for r in law_rows] != list(range(13)): raise ValueError('ROUND_COVERAGE')
    return dict(kind='UNQUALIFIED', norm='raw_complex_signal',
        law_model='actual_implemented_law', round_errors=[dict(m=m, synthesis=None, law=None) for m in range(13)],
        conditional_law_rows=law_rows, candidate_synthesis=synthesis,
        missing=['whole-kernel P0 all-law synthesis', 'atom/CAR and angle/normalization reference bridge',
                 'PCG64 stream vs IID random-bit model'], error_evidence=None,
        guaranteed_shots=None, guaranteed_total=None, actual_CA_achieved=None)


def replay_saved_table(saved):
    """Future permitted science only: reuse the actual make_tail pack path."""
    from .pr_dictionary_validation.prototype import MixedAtom, make_tail
    atoms = tuple(MixedAtom(**{**x, 'coefficient':Fraction(x['coefficient']),
                             'origins':tuple(x['origins']), 'word':tuple(x['word'])}) for x in saved['atoms_catalog'])
    tail = make_tail(atoms, 'C2B_saved_table')
    if len(atoms) != 216 or saved['native_policy'] != 'common_precision_spin_kappa_relative_head_full_frame_tail_v1':
        raise ValueError('REGISTERED_TABLE_SCOPE')
    for component, old in zip(tail.components, saved['components'], strict=True):
        for key in ('component_id','probability','coefficient_abs','coefficient_sign','basis_id','df_fragment_id'):
            if getattr(component, key) != old[key]: raise ValueError('SAVED_COMPONENT_REPLAY')
    if (saved['configuration']['r'] != 16 or saved['configuration']['K'] != 0
            or saved['finite_distribution']['orders'] != [0]
            or saved['finite_distribution']['order_probabilities'] != [1.]):
        raise ValueError('K0_ORDER_NOT_DETERMINISTIC')
    return dict(weights=[str(abs(a.coefficient)) for a in atoms],
                probabilities=[c.probability for c in tail.components], atoms=len(atoms),
                status='SAVED_TABLE_REPLAY_EXACT', fit=0, sampling=0)


def preflight(root, path):
    """Hash/package/schema only; no saved H4 probability or cost arithmetic."""
    root=Path(root).resolve(); c=io.read(path)
    if (c['scope'] != SCOPE or c['implementation'] != VERSION or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['fresh_output'] != FRESH or c['retry'] or c['resume']
            or c['random_model'] != IID or c['physical'] != POLICY): raise ValueError('C2B_SCOPE')
    if not set(c['records'].values()) <= c['saved_inputs'].keys(): raise ValueError('UNPINNED_SAVED_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want: raise ValueError('SOURCE_OR_INPUT:'+name)
    for name,want in c['packages'].items():
        if importlib.metadata.version(name) != want: raise ValueError('PACKAGE:'+name)
    stop=io.read(root/c['records']['C1_STOP'])
    if stop['failed'] or stop['scientific_units'] != 164 or stop['retry'] or stop['resume'] or not stop['mandatory_stop']:
        raise ValueError('C1_STOP')
    return c


def execute(root,c,output,call):
    bundle=call('saved_load','C1 provenance and two saved scalar tables',
                lambda:{k:io.read(Path(root)/p) for k,p in c['records'].items()})
    def identity():
        if (bundle['C1_result']['status'] != 'C1_SAVED_SCALAR_SCENARIOS_COMPLETE'
                or bundle['C1_SOURCE']['execution_commit'] != 'b1c735da394d4b0a444cbc618ead9781ef4d9134'):
            raise ValueError('C1_SOURCE_IDENTITY')
        return dict(status='PASS',new_fit=0,new_sampling=0,build_compile=0)
    io.save(Path(output)/'identity.json',call('identity_gate','fixed C1/source',identity))
    def arm_task(arm):
        packet=call('table_replay',arm,lambda:replay_saved_table(bundle[arm+'_tail']))
        io.save(Path(output)/(arm+'_replay.json'),packet)
        analysis=call('choice_grid_analysis',arm,lambda:analyze_table(packet['weights'],packet['probabilities']))
        io.save(Path(output)/(arm+'_table.json'),analysis)
        rows=[call('law_round',arm+':m'+str(m),lambda m=m:law_round(analysis,m)) for m in range(13)]
        preview=call('arm_qualification',arm,lambda:c1_preview(rows))
        io.save(Path(output)/(arm+'_qualification.json'),preview)
        return arm,dict(table=analysis,preview=preview)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .c1_stage3_integrated import resource_plan
    plan=resource_plan(task_count=2,full_dimension=1);io.save(Path(output)/'resource_plan.json',plan)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        arms=dict(f.result() for f in as_completed([pool.submit(arm_task,a) for a in ARMS]))
    def summary():
        return dict(status='C2B_CONDITIONAL_IID_TABLE_ANALYSIS_COMPLETE',scope=SCOPE,arms=arms,
                    actual_implemented_law_error=None,whole_law_synthesis=None,cost_support=None,
                    guaranteed_shots=None,guaranteed_total=None,CA_achieved=None,
                    scientific_units=sum(call.counts.values()),new_fit=0,sampling=0,build_compile=0,
                    retry=0,resume=0,mandatory_stop=True)
    result=call('output_summary','table-only conditional scope;actual evidence unresolved',summary)
    # The summary call itself has already been charged before summary() runs.
    io.save(Path(output)/'result.json',result);return result


def run(root,path,output):
    import resource, subprocess, time
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();c=preflight(root,path);out=Path(output).resolve()
    if out != root/FRESH or out.exists(): raise ValueError('FRESH_C2B_NO_RETRY_RESUME')
    out.mkdir(parents=True);call=Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c)
        io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
