"""Limited r16 original-H validation; passive import, existing math only.

S1 reuses saved r-independent PF/DF evidence and recomputes the corrected
finite-RTE norm and attenuation. No own-DF state, sampler or compiler is used.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import io
import json
import math
import subprocess

from . import track_d_original_h_accuracy as accuracy

CARRIER = '3649b75af191916daf44abb91e14d34c43b83e8f'
PACKET = '1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
ENDPOINT = 'a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
OLD = 'artifacts/research_c_near_practical_r32/2026-11-03/run01/'
INPUT = 'artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz'
CDF = 'artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz'
SECTOR = 'artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz'
SECTOR_SHA = 'ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd'
H = Fraction(1, 10)
R = 16
SAMPLE_ID = 'r16_q4_sample00'
MASTER = 2026111016
DOMAIN = 'track_d_r16_q4_single_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixed(root, path, records, expected=None, *, arrays=False):
    data = subprocess.check_output(['git', 'show', CARRIER+':'+path], cwd=root)
    sha = digest(data)
    if expected is not None and sha != expected:
        raise ValueError('FIXED_INPUT_HASH:'+path)
    records[path] = dict(commit=CARRIER, bytes=len(data), sha256=sha)
    if arrays:
        import numpy as np
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {key: z[key].copy() for key in z.files}
    return json.loads(data)


def target():
    return accuracy.OriginalHTarget(energy_ha=.00159, base_time=.1,
        beta_total=math.pi/6, beta_stat=math.pi/12, alpha_axis=.01/26,
        alpha_total=.01, maximum_round_index=12, energy_window_ha=(-10., 10.))


def seed_plan():
    from .c1_stage3_connection import trajectory_seed
    seeds = [trajectory_seed((MASTER, DOMAIN), 'proposed', ENDPOINT,
                            2, 'R16_Q4_PRICE_ONLY', 0, i) for i in range(4)]
    return dict(sample_id=SAMPLE_ID, master=MASTER, domain=DOMAIN,
        seeds=seeds, q=4, r=R, cut_after_complete_step=2,
        sampler='existing canonical sample_occurrence; numpy PCG64',
        selection='one pre-fixed path; no outcome-dependent replacement',
        sampling_allowed_only_after_S1=True)


def prepare(root):
    import numpy as np
    from . import c_pr_aware_df_stage_b as b
    from .c1_stage3_molecular import component_check
    from .c_rc2_pr_stage_b import distribution_gate
    from .c_rte_r_retuning import normalization
    records = {}
    p = fixed(root, INPUT, records, PACKET, arrays=True)
    d = fixed(root, CDF, records, ENDPOINT, arrays=True)
    s = fixed(root, SECTOR, records, SECTOR_SHA, arrays=True)
    old_identity = fixed(root, OLD+'PR/A/identity.json', records)
    old_law = fixed(root, OLD+'PR/A/law.json', records)
    old_refs = fixed(root, OLD+'PR/B/references.json', records)
    physical = fixed(root, OLD+'shared/physical_ground.json', records)
    if (s['generators'].shape != (9, 36, 36) or s['H'].shape != (36, 36)
            or d['J'].shape != (8, 4, 4) or d['U'].shape != (8, 4, 4)
            or not all(np.array_equal(d[k], s[k]) for k in ('J', 'U'))):
        raise ValueError('FIXED_CDF8_SECTOR')
    df = old_identity['saved_DF']
    if (old_identity['candidate'] != ENDPOINT or df['candidate_sha256'] != ENDPOINT
            or df['kappa'] != 'original_g_corrected' or df['scalar'] != 'nuclear_once'
            or df['actual_rank'] != 8 or df['evidence_kind'] != 'numerical_pilot'
            or df['rigorous_enclosure'] is not None or old_law['status'] != 'PASS'
            or old_refs['candidate'] != ENDPOINT or old_refs['r'] != 32
            or [x['m'] for x in old_refs['rounds']] != list(range(13))):
        raise ValueError('SAVED_DF_PF_PREMISES')
    if (physical['status'] != 'PASS' or not -10 <= physical['energy_Ha'] <= 10
            or not 0 <= physical['residual'] <= 1e-10
            or not 0 <= physical['normalization_error'] <= 1e-12
            or not math.isclose(physical['alias_margin'], math.pi-(1+math.pi/6),
                                rel_tol=0, abs_tol=1e-14)):
        raise ValueError('SAVED_ORIGINAL_H_PRIOR')
    arm = b.prepare_arm(**{k:p[k] for k in ('h','g','constant')},
        **{k:d[k] for k in ('J','U')}, role='proposed',
        source_identity=ENDPOINT, input_identity=PACKET, ld=2)
    tail, config, distribution = b.configure(arm, H, R, 0)
    law = distribution_gate(tail, config, distribution, expected_steps=R)
    norm = normalization(tail, config, R)
    if (arm.plan.extracted_identity != old_law['identity']
            or not math.isclose(tail.lambda_r, old_law['lambda_R'], rel_tol=1e-12, abs_tol=1e-14)
            or len(arm.plan.atoms) != old_law['atoms']):
        raise ValueError('R_INDEPENDENT_TAIL_BINDING')
    gate = component_check(sum(s['generators']), s['H'], 1e-10, label='saved sector generator sum')
    # No exact evolution/ground calculation: PF errors are r-independent and
    # already saved. Only the finite-RTE corrected kernel changes with r.
    kernels = b.kernels(arm, s['generators'], H, config)
    old_kernel = fixed(root, OLD+'PR/B/sector_reference.npz', records, arrays=True)
    pf_gate = component_check(kernels['pf'], old_kernel['pf'], 1e-10,
                              label='r-independent PF kernel versus saved PF')
    law.update(r=R, K=0, h='1/10', LD=2, frame_order=list(range(8)),
        config=asdict(config), extracted_identity=arm.plan.extracted_identity,
        scalar=float(arm.model.constant), normalization=norm,
        old_r32_law_or_N_used=False)
    law_hash = digest(json.dumps(law, sort_keys=True, separators=(',', ':')).encode())
    return dict(records=records, packet=p, arm=arm, tail=tail, config=config,
        distribution=distribution, generators=s['generators'], kernels=kernels,
        df_bound=math.fsum((df['sector_error'], df['numerical_guard'])),
        pf_errors=[x['reference']['pf_operator_error'] for x in old_refs['rounds']],
        law=law, law_sha256=law_hash, gates=[gate, pf_gate])


def assess(references, *, df_bound, source, law_hash):
    from .track_d_b1_c1 import transport_round
    from .c1_stage3_accuracy import _finite
    t = target()
    binding = accuracy.SignalBinding(original_h=PACKET, cdf=ENDPOINT,
        sector='Nalpha2_Nbeta2_sector36; saved original CAR generators',
        law='number-CDF8 original-order LD2 ordered S2/s1/r16/K0 h=.1; '+law_hash,
        source=source, implementation='ideal finite-RTE IID / exact native gates')
    rows, plans = [], []
    for m, reference in enumerate(references):
        row = transport_round(reference, m=m, df_error=df_bound,
            base_time=.1, historical_target=asdict(t), finite=_finite)
        evidence = accuracy.OriginalHRoundEvidence.from_b1_row(row, target=t,
            binding=binding, implementation_signal_error=0., finite_law_signal_error=0.,
            law_model=accuracy.IDEAL, alias_branch_verified=True, iid_per_shot=True,
            operator_provenance='saved r-independent PF/DF; new r16 RTE operator norm',
            alias_provenance='fixed prior [-10,10] and saved original-H eigenpair record',
            iid_provenance='ideal fresh independent full ordered path on every quantum shot',
            implementation_provenance='conditional exact-native model; not compiled all-law guarantee',
            law_provenance='new canonical r16 finite-RTE law '+law_hash)
        plan = accuracy.assess_round(t, binding, evidence)
        rows.append(row); plans.append(plan)
    if len(rows) != 13:
        raise ValueError('ALL13_REQUIRED')
    energy = accuracy.assess_energy(t, binding, plans)
    passed = all(p['status'] == 'CONDITIONAL_ROUND_PASS' for p in plans) and energy['same_accuracy']
    return dict(status='S1_CONDITIONAL_IDEAL_PASS' if passed else 'S1_NOT_QUALIFIED_STOP',
        target=asdict(t), binding=asdict(binding), horizon=t.horizon(), rows=rows,
        round_plans=plans, energy=energy, S2_allowed=passed,
        conditional_numerical_pilot=True, rigorous_enclosure=False,
        state_vectors_used=False, own_DF_signal_or_N_used=False,
        actual_implemented_success_qualified=False,
        DF_counting='included once in operator signal error; no extra Weyl shift')


def run_s1(context, source, workers):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from trotterlib.rte import finite_rte_attenuation
    def one(m):
        q = 1 << m
        k = context['kernels']
        # Same norm/matrix-power definition as reference_round; skip its
        # r-independent exact exponential and unused own-DF signal branch.
        error = float(np.linalg.norm(np.linalg.matrix_power(k['corrected'], q)
                                    - np.linalg.matrix_power(k['pf'], q), 2))
        return dict(pf_operator_error=context['pf_errors'][m],
            rte_corrected_operator_error=error,
            attenuation=finite_rte_attenuation(context['config'], tail_evolutions=q))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        references = list(pool.map(one, range(13)))
    return assess(references, df_bound=context['df_bound'], source=source,
                  law_hash=context['law_sha256'])
