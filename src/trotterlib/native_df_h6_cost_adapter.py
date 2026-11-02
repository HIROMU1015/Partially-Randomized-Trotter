"""H6 archive -> existing A/B model connection; no import-time evaluation.

Description extraction is static. Primitive production, probability arithmetic
and gate counting are separate permit-only operations. Never infer a categorical
law from realized events. Uniform A/B and literal A-prefix/B-tail are distinct.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from types import SimpleNamespace

BASE = 'b0c7a2855482175e438ffebe6c821ce36b0b2860'
PROVIDER = '87b6b9f46b7e8ba9abe132ee21478c114b6f46f7'
ARCHIVE_SHA256 = 'ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def candidate_from_index(item):
    """Normalize the documented B0I short cell schema, retaining original ID."""
    c=dict(item['cell'])
    if item['scope']=='H6_B0I_FAMILY12':
        if set(c)!={'prefix','q','formula'} or c['prefix'] not in (5,10,15) or c['formula']!='2nd':
            raise ValueError('SAVED_B0I_SHORT_SCHEMA')
        c.update(method='B0I',r=None,K=None,R=None,id=item['cell_id'])
    if c.get('method') not in ('B0','B0I','B1','B2') or c.get('q') not in (1,2,4,8):
        raise ValueError('EXISTING_H6_CANDIDATE_DOMAIN')
    expected=f"{c['method']}_p{c['prefix']}_q{c['q']}_{c['formula']}"
    if c['method']=='B2':expected+=f"_r{c['r']}_K{c['K']}"
    if c.get('id')!=expected or item['cell_id']!=expected:
        raise ValueError('SAVED_CANDIDATE_ID_MISMATCH')
    return c


def plan():
    return dict(schema='track_a_v2_h6_cost_binding_readiness_v1', base=BASE, provider=PROVIDER,
        scope=dict(system='H6', T=.8, existing_unique_candidates=104, scope_entries=132,
                   models=['A', 'B'], native_policy='A_prefix_B_closed_RTE_occurrences',
                   H8=False, new_V1_candidates=False),
        metadata=dict(archive_sha256=ARCHIVE_SHA256, new_preparations=0,
            stored_Gaussian_and_coefficients='AVAILABLE_STATIC_DESCRIPTION',
            timed_diagonal_primitives='ORIGINAL_PRODUCER_NUMERICAL_CONVERSION_PENDING',
            full_component_catalog='AVAILABLE_IN_FAITHFUL_p5_p10_p15',
            order_law='K2_AND_K4_ALL_EVEN_ORDERS_FROM_SOURCE_NOT_OBSERVED_FREQUENCIES',
            full_connection_envelopes='PENDING_RECEIPT_CONTENT_AND_VARIANT_BINDING'),
        demand=dict(new_SCF=0, new_DF=0, new_state=0, new_signal=0,
            new_sampling=0, native_full_wrapper_build=0, new_compile=0,
            timed_primitive_records_max=2048, model_candidate_axis_rows_max=416,
            expected_occurrence_evaluations_max=144, transitions_per_DP_max=1000000,
            aggregate_identity_phase_bound='0..1 P per closed occurrence'),
        resources=dict(processes=1, threads=1, AS_bytes=8*2**30, RSS_bytes=4*2**30,
            CPU_seconds=4*3600, wall_seconds=6*3600, output_bytes=256*2**20,
            stage_storage_bytes=512*2**20,
            estimate_kind='UNAPPROVED_STOP_CEILINGS_NOT_MEASURED_PREDICTIONS'),
        comparison=dict(metrics=['RZ', 'CX'], sample_scope='exploration8 and confirmation32 kept separate',
            shot_rule='existing saved epsilon/factor/accounting rows; no reallocation',
            regression='compiled_axis_cost = intercept + slope * theory_axis_cost within compatible provenance stratum',
            residual='shared-stratum affine calibration then PR/PF signed relative residuals; separate-family fits secondary',
            residual_stop_formula='abs(mean((compiled-fit)/compiled)_PR - mean((compiled-fit)/compiled)_PF) >= 0.05',
            stop_residual_PR_PF_relative_difference=.05,
            rare_orders='retain every supported k even when absent from saved trajectories',
            nonClifford='classify lowered RZ angles; no T-count or Toffoli conversion'),
        approved=False, runtime_authorization=False, allowed_cpus=[],
        absolute_launch_command=None, retry=False, resume=False,
        mandatory_stop=True, next_stage_authorized=False)


def preparation_description(graph, key):
    """Copy saved coefficient/basis DESCRIPTIONS without time multiplication."""
    if key not in graph.archive['codec']['roots']:
        raise ValueError('MISSING_FROZEN_ROOT')
    f = graph.fields(graph.archive['codec']['roots'][key], 'DFPartialS2Preparation')
    n = graph.scalar(f['num_system_qubits'])
    blocks = []
    for ref in graph.sequence(f['deterministic_blocks']):
        node = graph.node(ref); b = node['fields']; kind = node['type']
        if kind not in ('DFDeterministicOneBodySpec', 'DFDeterministicFragmentSpec'):
            raise ValueError('UNSUPPORTED_PF_BLOCK')
        basis = dict(basis_id=b['basis_id'], basis_hash=b['basis_hash'], num_system_qubits=n,
            phase_policy='native_gate_list_only', control_scope='uncontrolled',
            operations=graph.basis_description(b['basis_change_operations'], n=n,
                basis_id=b['basis_id'], basis_hash=b['basis_hash']))
        graph.compare_runtime_descriptions(b['runtime_basis_operations'], basis['operations'])
        diagonal = b['diagonal_eigenvalues'] if kind == 'DFDeterministicOneBodySpec' else b['diagonal_eta']
        coeffs = [dict(v) for v in graph.sequence(diagonal)]
        if len(coeffs) != n or any(type(graph.scalar(v)) is not float for v in coeffs):
            raise ValueError('DIAGONAL_COEFFICIENT_SCHEMA')
        blocks.append(dict(kind=kind, block_id=b['block_id'], order_index=b['order_index'],
            original_fragment_index=b['original_fragment_index'], basis=basis,
            diagonal_coefficients=coeffs, lam=b.get('lam'), original_node_sha256=ref['ref']))
    if [b['order_index'] for b in blocks] != list(range(len(blocks))):
        raise ValueError('PF_BLOCK_ORDER')
    rte = graph.fields(f['rte_preparation'], 'DFRTEEventPreparation')
    tail = graph.fields(rte['symbolic_tail'], 'SymbolicRTETail')
    components = []
    for ref in graph.sequence(tail['components']):
        c = graph.fields(ref, 'RTEComponent')
        record = {k: graph.scalar(c[k]) for k in ('component_id', 'basis_id', 'basis_hash',
            'df_fragment_id', 'coefficient_sign', 'is_identity', 'diagonal_pauli_support')}
        # Keep encoded binary64 values; normalization is explicitly a later step.
        record.update(probability=dict(c['probability']), coefficient_abs=dict(c['coefficient_abs']))
        record['basis'] = None if record['is_identity'] else dict(basis_id=record['basis_id'],
            basis_hash=record['basis_hash'], num_system_qubits=n, phase_policy='native_gate_list_only',
            control_scope='uncontrolled', operations=graph.basis_description(c['basis_change_operations'],
                n=n, basis_id=record['basis_id'], basis_hash=record['basis_hash']))
        components.append(record)
    declared = graph.fields(tail['extraction_metadata'], 'DFTailExtractionMetadata')
    if (len(components) != graph.scalar(declared['randomized_component_count']) or
            len({c['component_id'] for c in components}) != len(components)):
        raise ValueError('INCOMPLETE_COMPONENT_CATALOG')
    return dict(root=key, root_node_sha256=graph.archive['codec']['roots'][key]['ref'],
        preparation_hash=f['preparation_hash'], hamiltonian_hash=f['hamiltonian_hash'],
        n=n, prefix=f['ld'], blocks=blocks, components=components,
        constant=dict(f['constant_coefficient']), extracted_identity=dict(f['extracted_identity_coefficient']),
        lambda_r=dict(f['exact_rte_lambda_r']), primitive_lists_evaluated=False,
        probabilities_evaluated=False, costs_evaluated=False, matrix_semantics_certified=False)


def basis_object(record, model, schema):
    runtime = schema.runtime_basis_metadata(record, expected_basis_id=record['basis_id'],
                                             expected_basis_hash=record['basis_hash'])
    return model.basis_from_runtime(record['basis_id'], record['basis_hash'], runtime)


def model_inputs(description, model, schema, producer, permit):
    """Numerical conversion only AFTER an independently validated future permit.

    producer must be the source-pinned original df_trotter.ops module, never a
    rewritten polynomial. Model and registry hashes remain separate domains.
    """
    permit.require('h6_v2_numeric')
    permit.verify_producer(producer)
    bases = [basis_object(b['basis'], model, schema) for b in description['blocks']]
    cache = getattr(permit, 'primitive_cache', {})

    def primitives(index, t):
        permit.guard()
        if type(index) is not int or not 0 <= index < len(bases) or not math.isfinite(t):
            raise ValueError('PRIMITIVE_REQUEST_SCHEMA')
        key = description['preparation_hash'], index, float(t).hex()
        if key not in cache:
            permit.take('timed_primitive_records')
            b = description['blocks'][index]
            coefficients = [float.fromhex(v['float_hex']) for v in b['diagonal_coefficients']]
            cache[key] = (producer.one_body_diagonal_primitives(coefficients, t)
                if b['kind'] == 'DFDeterministicOneBodySpec' else
                producer.df_squared_diagonal_primitives(coefficients, float.fromhex(b['lam']['float_hex']), t))
        return cache[key]

    components = []
    if description['components']:
        permit.take('complete_component_law_conversion')
        weights = [Fraction(float.fromhex(c['probability']['float_hex'])) for c in description['components']]
        if any(w <= 0 for w in weights): raise ValueError('NONPOSITIVE_DECLARED_COMPONENT')
        total = sum(weights)
        if abs(total-1) > Fraction(1, 10**12): raise ValueError('SAVED_COMPONENT_MASS_MISMATCH')
        if permit.law_policy != 'EXACT_RENORMALIZATION_OF_SAVED_BINARY64_WEIGHTS':
            raise ValueError('DECLARED_PROBABILITY_POLICY_REQUIRED')
        for c, w in zip(description['components'], weights):
            basis = None if c['is_identity'] else basis_object(c['basis'], model, schema)
            components.append(model.Component(c['component_id'], w/total, basis,
                tuple(c['diagonal_pauli_support']), c['coefficient_sign']))
    return bases, primitives, components


def evaluate_candidate(description, candidate, *, model, schema, producer, permit,
                       variant, axis, scalar_phase):
    """Future theoretical call. Never called by static inspection or this task."""
    permit.require('h6_v2_numeric')
    if variant not in ('A', 'B', 'literal_native') or axis not in ('cosine', 'sine'):
        raise ValueError('MODEL_OR_AXIS')
    if candidate['prefix'] != description['prefix'] or candidate['formula'] not in ('2nd', '4th'):
        raise ValueError('CANDIDATE_PREPARATION_BINDING')
    bases, primitives, components = model_inputs(description, model, schema, producer, permit)
    ancilla = description['n']; theory_axis = 'X' if axis == 'cosine' else 'Y'
    fixed_model = 'A' if variant == 'literal_native' else variant
    permit.take('model_candidate_axis_rows')
    if candidate['method'] != 'B2':
        evolution = model.deterministic_circuit(bases, primitives, .8, candidate['q'],
            formula=candidate['formula'], policy='symmetric_directional', ancilla=ancilla,
            scalar_phase=scalar_phase, model=fixed_model)
        return model.count(model.hadamard_wrapper(evolution, ancilla, theory_axis))
    if candidate['formula'] != '2nd' or not components:
        raise ValueError('PR_REQUIRES_COMPLETE_FINITE_LAW_AND_S2')
    # The normalization/law receipt must independently bind tau, K and R=q*r.
    tau, orders = permit.order_law(candidate, description)
    if set(orders) != set(range(0, candidate['K']+1, 2)) or any(p <= 0 for p in orders.values()):
        raise ValueError('MISSING_OR_ZERO_SUPPORTED_TAYLOR_ORDER')
    prefix = model.partial_prefix_circuit(bases, primitives, .8, candidate['q'], ancilla,
        policy='symmetric_directional', scalar_phase=scalar_phase, model=fixed_model)
    event_model = 'B' if variant == 'literal_native' else variant
    event_cache = getattr(permit, 'occurrence_cache', {})
    key = (description['preparation_hash'], candidate['q'], candidate['r'],
           tau.hex(), tuple(sorted(orders.items())), event_model)
    if key not in event_cache:
        permit.take('expected_occurrence_evaluations')
        event_cache[key] = model.expected_rte_occurrence(components, orders, candidate['r'], tau, ancilla,
            model=event_model, max_transitions=1000000)
    event = event_cache[key]
    return model.expected_partial_trajectory(event, candidate['q'], prefix, ancilla, theory_axis)


def validate_full_binding(record, expected, *, schema, receipt_loader):
    """Reuse the existing envelope validator, then authenticate receipt contents."""
    result = schema.validate_connection_identity(record, expected=expected)
    for field in ('phase_receipt_sha256', 'normalization_receipt_sha256', 'shot_receipt_sha256',
                  'numeric_allowance_receipt_sha256', 'fragment_order_sha256',
                  'primitive_receipt_sha256', 'basis_registry_sha256'):
        raw = receipt_loader(field)
        if hashlib.sha256(raw).hexdigest() != record[field]:
            raise ValueError('RECEIPT_BYTES:'+field)
    # These declarations are emitted by the separate source-bound receipt reader;
    # this API does not accept a hash alone as a proof of numerical semantics.
    if record.get('receipt_semantics_verified') is not True:
        raise ValueError('INDEPENDENT_RECEIPT_CONTENT_VALIDATION_REQUIRED')
    if record['family'] == 'PR':
        for field in ('law_receipt_sha256', 'component_law_sha256'):
            if hashlib.sha256(receipt_loader(field)).hexdigest() != record[field]:
                raise ValueError('LAW_RECEIPT_BYTES:'+field)
    return dict(result, status='SOURCE_BOUND_RECEIPTS_VALIDATED', receipt_contents_certified=False)


def paired_compiled_rows(rows, *, expected):
    if len(rows) != 2 or {r['axis'] for r in rows} != {'cosine', 'sine'}:
        raise ValueError('EXACTLY_TWO_PAIRED_AXES')
    for r in rows:
        for key in ('cell_id', 'execution_source', 'dataset_seed_source', 'input_sha256',
                    'compiler_hash', 'profile_hash', 'sample_scope', 'replica', 'seed', 'event_digest'):
            if expected.get(key) is None or r.get(key) != expected[key]:
                raise ValueError('MISSING_OR_CHANGED_COMPILED_BINDING:'+key)
        if r.get('status') != 'MEASURED_COMPLETE' or any(type(r.get(m)) is not int or r[m] < 0 for m in ('RZ', 'CX')):
            raise ValueError('COMPILED_MEASUREMENT_NOT_COMPLETE')
    return True


def rotation_classes(circuit, model, *, angle_tolerance=1e-12):
    """Future/model-test diagnostic of LOWERED angles. Never an FT gate count."""
    if not math.isfinite(angle_tolerance) or angle_tolerance<0:
        raise ValueError('FINITE_NONNEGATIVE_ANGLE_TOLERANCE_REQUIRED')
    result = dict(identity_RZ=0, Clifford_RZ=0, nonClifford_RZ=0, tolerance=angle_tolerance,
                  T_count=None, Toffoli_count=None, tolerance_classification_certified=False)
    for gate in model.lower(circuit).operations:
        if gate.name != 'rz': continue
        x = math.remainder(gate.angle, 2*math.pi)
        key = ('identity_RZ' if abs(x) <= angle_tolerance else 'Clifford_RZ'
               if abs(x-round(x/(math.pi/2))*(math.pi/2)) <= angle_tolerance else 'nonClifford_RZ')
        result[key] += 1
    return result


def affine_correspondence(rows):
    """Separate saved/model axis rows; no implicit PR/PF pooling or zero fill."""
    if len(rows) < 3 or any(r['stratum'] != rows[0]['stratum'] or r['status'] != 'MATCHED_COMPLETE' for r in rows):
        raise ValueError('THREE_MATCHED_ROWS_PER_STRATUM_REQUIRED')
    xs, ys = [Fraction(r['theory']) for r in rows], [Fraction(r['compiled']) for r in rows]
    n = len(xs); denominator = n*sum(x*x for x in xs)-sum(xs)**2
    if denominator == 0: raise ValueError('DEGENERATE_AFFINE_CORRESPONDENCE')
    slope = (n*sum(x*y for x, y in zip(xs, ys))-sum(xs)*sum(ys))/denominator
    intercept = (sum(ys)-slope*sum(xs))/n
    residuals = [(y-intercept-slope*x)/y if y else None for x, y in zip(xs, ys)]
    if any(v is None for v in residuals): raise ValueError('ZERO_COMPILED_RESIDUAL_DENOMINATOR')
    return dict(intercept=intercept, slope=slope, relative_residuals=residuals,
                calibrated_with_saved_values=True, population_or_FT_guarantee=False)


def compare_method_residuals(rows):
    fit=affine_correspondence(rows)
    means={}
    for method in ('PR','PF'):
        selected=[value for row,value in zip(rows,fit['relative_residuals']) if row['family']==method]
        if not selected:raise ValueError('BOTH_FAMILIES_REQUIRED_FOR_RESIDUAL_COMPARISON')
        means[method]=sum(selected)/len(selected)
    difference=abs(means['PR']-means['PF'])
    if difference>=Fraction(1,20):raise ValueError('PR_PF_RESIDUAL_DIFFERENCE_GE_5_PERCENT')
    return dict(fit=fit,mean_signed_relative_residuals=means,difference=difference,
                scientific_guarantee=False)
