"""Full-only matched native programs and independent price-cohort statistics."""
from dataclasses import asdict
from fractions import Fraction
import io
import json
from math import sqrt
from types import SimpleNamespace as NS

from qiskit import qpy

from . import matched_partition_stage3 as K
from .rte_compiled_cost import CompiledMetricAccumulator

METRICS = dict(RZ='rz_count', CX='cx_count', depth='total_depth',
               rz_depth='rz_depth', cx_depth='cx_depth', size='circuit_size')
SEEDS = {'P': tuple(range(320201, 320233)), 'DF': tuple(range(320301, 320333))}


def restored_circuit(decoded, *, fingerprint=None):
    circuits = qpy.load(io.BytesIO(decoded))
    if len(circuits) != 1:
        raise ValueError('ONE_SAVED_CIRCUIT_REQUIRED')
    circuit = circuits[0]
    if fingerprint is not None and K.canonical_qiskit_circuit_fingerprint(circuit) != fingerprint:
        raise ValueError('SAVED_CIRCUIT_FINGERPRINT')
    return circuit


def basis_ops(circuit):
    # Registry accepts gates, so never discard a separate circuit global phase.
    if float(circuit.global_phase) != 0. or circuit.num_clbits:
        raise ValueError('BASIS_PHASE_OR_CLASSICAL_BITS')
    ops = tuple((item.operation, tuple(circuit.find_bit(q).index for q in item.qubits))
                for item in circuit.data)
    if any(g.name == 'unitary' or g.num_qubits > 2 for g, _ in ops):
        raise ValueError('SAVED_NATIVE_FAMILY')
    return ops


def bind_components(dictionaries, normalizations, bases, saved_laws, *, n=8):
    bundles = {arm: K.component_inputs(arm, dictionaries[arm], normalizations[arm], bases, n)
               for arm in ('P', 'DF')}
    for arm, bundle in bundles.items():
        law = saved_laws[arm]
        actual = json.loads(json.dumps([asdict(x) for x in bundle.registry.metadata()]))
        if actual != law['basis_metadata'] or normalizations[arm] != law['saved_Stage2_normalization']:
            raise ValueError('SAVED_LAW_METADATA:' + arm)
        for component, row in zip(bundle.components, law['components'], strict=True):
            if (component.component_id, component.probability, component.basis_id, component.basis_hash) != (
                    row['id'], row['probability'], row['basis_id'], row['basis_hash']):
                raise ValueError('SAVED_COMPONENT_BINDING:' + arm)
    return bundles


def full_only(inputs, events, retained, angle, *, r=K.R):
    if len(events) != retained.q * r:
        raise ValueError('EVENT_CARDINALITY')
    body = K.QuantumCircuit(inputs.n + 1)
    sidecar = []
    for outer in range(retained.q):
        residual, meta = K.residual_sequence(inputs, events[outer*r:(outer+1)*r])
        sidecar.append(meta)
        body.compose(retained.controlled_half, inplace=True)
        body.compose(residual, inplace=True)
        body.compose(retained.controlled_half, inplace=True)
    body.p(-float(angle), inputs.n)
    return body, sidecar


def price_statistics(prices, *, n=32):
    expected = {f'{arm}_{i:02d}/F/{axis}' for arm in ('P', 'DF')
                for i in range(n) for axis in ('cosine', 'sine')}
    if n < 2 or set(prices) != expected:
        raise ValueError('COMPLETE_INDEPENDENT_COHORT_REQUIRED')
    arms = {}
    for arm in ('P', 'DF'):
        axis_stats, exact_means = {}, {}
        for axis in ('cosine', 'sine'):
            rows = [prices[f'{arm}_{i:02d}/F/{axis}'] for i in range(n)]
            accumulator = CompiledMetricAccumulator(weighted=False)
            for row in rows:
                accumulator.update(NS(**{field: row[m] for m, field in METRICS.items()}))
            result = dict(accumulator.finalize())
            axis_stats[axis] = {m: asdict(result[field]) for m, field in METRICS.items()}
            exact_means[axis] = {m: Fraction(sum(row[m] for row in rows), n) for m in METRICS}
            for m in METRICS:
                axis_stats[axis][m]['mean_exact'] = str(exact_means[axis][m])
        covariance = {m: sum((Fraction(prices[f'{arm}_{i:02d}/F/cosine'][m]) - exact_means['cosine'][m]) *
                            (Fraction(prices[f'{arm}_{i:02d}/F/sine'][m]) - exact_means['sine'][m])
                            for i in range(n)) / (n - 1) for m in METRICS}
        arms[arm] = dict(n=n, axes=axis_stats, cosine_sine_sample_covariance_exact={
            m: str(value) for m, value in covariance.items()}, population_mean_upper=None)
    differences = {}
    for axis in ('cosine', 'sine'):
        differences[axis] = {}
        for m in METRICS:
            p, df = (arms[a]['axes'][axis][m] for a in ('P', 'DF'))
            differences[axis][m] = dict(P_minus_DF=p['mean'] - df['mean'],
                independent_standard_error=sqrt((p['unbiased_sample_variance'] +
                                                 df['unbiased_sample_variance']) / n))
    return dict(arms=arms, independent_differences=differences, old_pilot_pooled=False,
                cross_arm_paired=False, nonasymptotic_coverage_claimed=False,
                population_tail_bound=None, new_N_G=None)
