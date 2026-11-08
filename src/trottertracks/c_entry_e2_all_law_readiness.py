"""Metadata-only E2/r16 binding and missing-bound inventory; no science path.

This is deliberately not an evidence attacher. Schema acceptance cannot prove
whole-P0 coverage, a trajectory target correspondence, or a joint stream law.
Reuse the existing accuracy/bridge engines only after those arguments exist.
"""
import math
from pathlib import Path

from . import c1_stage1_execution as io

FAMILY = 'artifacts/research_c_entry_e2_all_law/2026-11-08/technical'
RECORD = FAMILY + '/readiness.json'
PARTS = ('synthesis', 'categorical', 'reference_bridge', 'reference_uncertainty', 'stream_model')
MODELS = ('IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL', 'actual_implemented_law')
PREMISES = ('whole_P0_all_words_success_and_error', 'raw_to_same_atom_target',
            'categorical_unitarity_or_norm_growth', 'actual_inverse', 'relative_phase',
            'head_X_Y_shells', 'full512_with_leakage', 'independent_coherent_occurrences',
            'independent_shots', 'actual_joint_stream')


def target_identity():
    return dict(candidate='H4_100_C0_ENTRY_g003_budget64', arm='ENTRY', r=16,
        candidate_sha256='0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5',
        J_sha256='69dac3b923847fc85cd403fa38ddf42db31385ca3bbcc973d3646f0567fc6d2d',
        U_sha256='dfbdd050719f96084d3c5cb3b5c1b9bdd86bef352246fe95ea306aa3934c1310',
        original_physical_input_sha256='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db',
        original_kappa_export_sha256='c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08',
        original_scalar_array_sha256='874b34c93c04576c14050d91ecace903d7b2693560bf77a05b2178fc48fb232b',
        relocation='tail_linear_Z_to_existing_kappa_keep_frame_identity_v1',
        model='linear H4/1.00 Angstrom/STO-3G', rank=8, ld=2, frame_order='original',
        sector=dict(nalpha=2, nbeta=2, dimension=36), controlled_dimension=512,
        formula='ordered partial S2', s=1, h='1/10', K=0, rounds=list(range(13)),
        ground='own_DF_ground_state_oracle', state_preparation_cost=False,
        accuracy=dict(CA_Ha='0.00159', df_reserve='1/8', beta_total='pi/6',
            beta_stat='pi/12', alpha_total='0.01', alpha_axis='0.01/26'),
        scalar='nuclear_once_plus_relocation_shift_keep_frame_identity',
        compiler=dict(qiskit='1.3.0', profile='P0', optimization_level=1,
            seed_transpiler=17, basis_gates=['rz','sx','x','cx'], approximation_degree=1.,
            phase='MPFR160_declared_phase_no_global_phase_fit', initially_zero=False),
        inverse='same_realized_word_actual_raw_and_compiled_inverse', shells=['head','X','Y'],
        scientific_SOURCE_commit='cfc64b70861a6482dddd042da8a110abebd6ad7a',
        code_SOURCE_sha256='6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc',
        sampling_SOURCE_sha256='bb102f16c3bf757853f9e9fc98d811dc230592d71cd6bad57bcce0363fa2046e')


def missing_packet():
    return dict(components={p:None for p in PARTS}, premises={p:False for p in PREMISES})


def inspect_packet(observed_identity, packet, *, model=MODELS[0]):
    """Check proposed metadata, retaining unknowns and refusing promotion.

    Non-null rows are *supplied claims*, not verified bounds. In particular,
    even an all-true artificial packet cannot enable the scientific pipeline.
    """
    if observed_identity != target_identity():
        raise ValueError('E2_RELOCATED_R16_SOURCE_IDENTITY')
    if model not in MODELS:
        raise ValueError('EXPLICIT_LAW_MODEL_REQUIRED')
    components, premises = packet['components'], packet['premises']
    if set(components) != set(PARTS) or set(premises) != set(PREMISES):
        raise ValueError('EXACT_ERROR_PARTITION_AND_PREMISES')
    if any(type(v) is not bool for v in premises.values()):
        raise ValueError('EXPLICIT_PREMISE_BOOLEANS')
    if model == MODELS[0] and (components['stream_model'] is not None or premises['actual_joint_stream']):
        raise ValueError('IID_GRID_IS_NOT_ACTUAL_STREAM_EVIDENCE')
    for name, record in components.items():
        if record is None:
            continue
        if record.get('identity') != observed_identity or record.get('law_model') != model:
            raise ValueError('BOUND_SOURCE_AND_MODEL_BINDING')
        if (record.get('norm') != 'raw_complex_signal' or record.get('included_terms') != [name]
                or record.get('kind') not in ('numerical_bound', 'rigorous_bound')
                or not record.get('provenance')):
            raise ValueError('RAW_UNITS_DISJOINT_PARTITION_QUALIFICATION')
        coverage = 'all_trajectories_all_occurrences_and_shells' if name in ('synthesis','stream_model') else 'model_reference_all_rounds'
        if record.get('coverage') != coverage:
            raise ValueError('FINITE_WORDS_ARE_NOT_ALL_LAW')
        rows = record.get('rows', [])
        if ([x.get('m') for x in rows] != list(range(13))
                or any(type(x['m']) is not int or type(x.get('bound')) not in (int,float)
                       or not math.isfinite(x['bound']) or x['bound'] < 0 for x in rows)):
            raise ValueError('THIRTEEN_FINITE_NONNEGATIVE_RAW_BOUNDS')
    required = PARTS if model == MODELS[1] else PARTS[:-1]
    needed = PREMISES if model == MODELS[1] else PREMISES[:-1]
    gaps = [x for x in required if components[x] is None] + [x for x in needed if not premises[x]]
    return dict(status='NOT_READY_BOUND_INPUTS' if gaps else 'SCHEMA_ONLY_METHOD_NOT_ESTABLISHED',
        law_model=model, missing=gaps, supplied_claims_verified=False,
        error_evidence=None, round_errors=[dict(m=m,**{p:None for p in PARTS}) for m in range(13)],
        all_law_bound_qualified=False, PCG64_joint_stream_qualified=False,
        RPE_accuracy='UNQUALIFIED', actual_CA_achieved=None, rigorous_CA=None,
        scientific_calls=0, scientific_execution_ready=False, mandatory_stop=True)


def preflight(root, record_path=RECORD):
    """Bytes/JSON only, including the original 16 inputs and code SOURCE."""
    root = Path(root).resolve()
    record = io.read(root/record_path)
    if record['identity'] != target_identity():
        raise ValueError('FIXED_READINESS_IDENTITY')
    checked = {}
    def check(name, want):
        path = (root/name).resolve()
        if not path.is_relative_to(root) or io.sha(path) != want:
            raise ValueError('SAVED_BYTES:'+name)
        checked[name] = want
    for name, want in record['saved_files'].items():
        check(name, want)
    for manifest_path in record['code_manifests']:
        if manifest_path not in checked:
            raise ValueError('UNBOUND_SOURCE_MANIFEST')
        for name, want in io.read(root/manifest_path)['files'].items():
            check(name, want)
    assessment = inspect_packet(record['identity'], record['bound_request'])
    if (record['case'] != 'B' or record['whole_P0_scientific_calls'] is not None
            or record['new_science_authorized'] is not False
            or any(v is not None for v in record['bound_request']['components'].values())
            or any(record['bound_request']['premises'].values())):
        raise ValueError('PREPARATION_ONLY_NULL_BOUNDS')
    return dict(status='PASSIVE_BINDING_PASS', files_checked=len(checked), arrays_decoded=0,
        assessment=assessment, case='B', mandatory_stop=True)
